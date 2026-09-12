"""
cVAE Fire Reconstruction — Conditional Variational Autoencoder
================================================================
LOOCV pipeline for wildfire channel reconstruction using a convolutional
Conditional VAE with learned prior.

Architecture Overview:
  Condition Encoder : 42ch (40 cond + 2 fire) → multi-scale features + 8×8 bottleneck
  Target Encoder    : 1ch ground-truth fire → downsampled features at 8×8
  Recognition Head  : bottleneck + target_feat → μ_q, logσ_q  (spatial latent, 16ch @ 8×8)
  Prior Head        : bottleneck alone       → μ_p, logσ_p  (spatial latent, 16ch @ 8×8)
  Decoder           : z + bottleneck + skip connections → 1ch logits @ 64×64

Data Flow:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  TRAINING                                                          │
  │                                                                    │
  │  cond(40ch) + fire_in(2ch) ──→ Condition Encoder ──→ skips[0,1,2] │
  │           42ch input                    │            + bottleneck   │
  │                                         │              (256ch,8×8) │
  │                                         ├──→ Prior Head ─→ μ_p,σ_p│
  │                                         │                         │
  │  target(1ch) ──→ Target Encoder ──→ tgt_feat (256ch,8×8)          │
  │                                         │                         │
  │  cat(bottleneck, tgt_feat) ──→ Recognition Head ──→ μ_q, σ_q     │
  │                                                                    │
  │  z = μ_q + σ_q * ε   (reparameterization trick, ε ~ N(0,1))      │
  │                                                                    │
  │  cat(z, bottleneck) ──→ Decoder (+ skip connections) ──→ logits   │
  │                                                                    │
  │  Loss = Focal(logits, target, masked_only) + β·KL(q ‖ p)         │
  │          ↑ reconstruction loss                  ↑ regularization   │
  │                                                                    │
  │  KL is between q(z|x,c) and p(z|c), both learned distributions.  │
  │  This lets the prior learn to predict latents from condition alone.│
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │  INFERENCE (no target available)                                   │
  │                                                                    │
  │  cond + fire_in ──→ Condition Encoder ──→ skips + bottleneck       │
  │                                │                                   │
  │                     bottleneck ──→ Prior Head ──→ μ_p, σ_p         │
  │                                                                    │
  │  z = μ_p   (use prior mean, deterministic — best single estimate) │
  │                                                                    │
  │  cat(z, bottleneck) ──→ Decoder (+ skips) ──→ logits ──→ sigmoid  │
  │                                                                    │
  │  Merge: visible pixels keep original, masked pixels get prediction │
  └─────────────────────────────────────────────────────────────────────┘

Why cVAE for this task:
  - The learned prior p(z|c) encodes "what fire pattern is likely given
    these 40 environmental channels + partial fire observations"
  - The latent z captures uncertainty: multiple valid reconstructions
    may exist for heavily masked inputs
  - At test time, the prior mean gives a principled "best guess"

Training : Focal loss (masked region) + β·KL, AMP fp16, cosine LR
Testing  : data_challenge_{blockwise,pixelwise}, all categories × difficulties
Output   : Metrics only (DICE for fire categories, FPR for no-fire categories)

Estimated parameters: ~15M
"""

import os
import glob
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ────────────────────────────── Configuration ──────────────────────────────

DATA_ROOT = "/pub/cyang27/data"
TRAIN_DIR = os.path.join(DATA_ROOT, "data_mixed_challenge_train")
SAVE_DIR  = os.path.join(DATA_ROOT, "cvae_results")

MASK_TYPES   = ["blockwise", "pixelwise"]
YEARS        = [2018, 2019, 2020, 2021]
CATEGORIES   = ["Fire_Continues", "Fire_Extinguished", "NewFire_NoHistory", "NoFire"]
DICE_CATS    = {"Fire_Continues", "Fire_Extinguished"}
FPR_CATS     = {"NewFire_NoHistory", "NoFire"}
DIFFICULTIES = [10, 20, 30, 40, 50, 60, 70, 80]
FIRE_CH      = 42
DET_CH       = 41

# Architecture
IMG_SIZE   = 64
BASE_CH    = 64
CH_MULTS   = [1, 2, 4]     # → [64, 128, 256], 3 downsample levels (64→32→16→8)
LATENT_CH  = 16             # spatial latent: 16 channels @ 8×8 = 1024 latent dims
ATTN_HEADS = 8
DROPOUT    = 0.1

# Training
SEED         = 42
BATCH_SIZE   = 64
LR           = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS       = 50
NUM_WORKERS  = 8
PREFETCH     = 4
FOCAL_ALPHA  = 0.75
FOCAL_GAMMA  = 2.0
KL_WEIGHT    = 1e-3         # β for KL term (annealed from 0 during warmup)
KL_WARMUP    = 10           # linearly anneal β over first N epochs

# Inference
TEST_BATCH = 40


# ────────────────────────────── Utilities ──────────────────────────────────

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def src_dir(mt):
    return os.path.join(DATA_ROOT, f"data_challenge_{mt}")


def drop_det(x43):
    """(43,H,W) → (42,H,W): drop detection channel (index 41)."""
    return torch.cat([x43[:DET_CH], x43[DET_CH + 1:]], dim=0)


def get_num_groups(channels, target=32):
    """Find the largest number of groups <= target that divides channels evenly.

    GroupNorm requires num_channels % num_groups == 0.  When channel counts
    come from sums like bot_ch + latent_ch (256 + 16 = 272), 32 may not divide
    evenly.  This helper gracefully falls back to smaller group counts.
    """
    for g in [target, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


# ────────────────────────────── Dataset ────────────────────────────────────

class FireDataset(Dataset):
    """Returns (cond_40ch, fire_2ch, binary_target, visibility_mask)."""
    def __init__(self, files):
        self.files = files
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        d = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        x = drop_det(d["x"].float())
        y = (d["y"].float() > 0).float()
        return x[:40], x[40:], y, x[40]


def gather_train_files():
    out = {}
    for yr in YEARS:
        d = os.path.join(TRAIN_DIR, str(yr))
        out[yr] = sorted(glob.glob(os.path.join(d, "*.pt"))) if os.path.isdir(d) else []
    return out


# ────────────────────────────── Model Components ───────────────────────────

class ResBlock(nn.Module):
    """GroupNorm → SiLU → Conv3×3 → GroupNorm → SiLU → Dropout2d → Conv3×3 + skip."""
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(get_num_groups(in_ch), in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(get_num_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class SelfAttention2D(nn.Module):
    """Multi-head self-attention with GroupNorm pre-norm and residual."""
    def __init__(self, ch, heads=8):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(get_num_groups(ch), ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, 1, bias=False)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        d = C // self.heads
        qkv = self.qkv(self.norm(x)).reshape(B, 3, self.heads, d, H * W)
        q, k, v = [t.permute(0, 1, 3, 2) for t in qkv.unbind(1)]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(out)


# ────────────────────────────── cVAE Model ─────────────────────────────────

class cVAEFire(nn.Module):
    """
    Conditional VAE for fire channel reconstruction.

    Components:
      1. Condition Encoder: extracts multi-scale features from 42ch input
         (40 environment + 2 fire channels: visibility mask + visible values)
      2. Target Encoder: lightweight CNN to downsample ground-truth fire to 8×8
      3. Recognition Head: maps (bottleneck + target_feat) → μ_q, logσ_q
         Only used during training when ground truth is available.
      4. Prior Head: maps bottleneck → μ_p, logσ_p
         Used at test time to predict latent without seeing ground truth.
      5. Decoder: upsamples (z + bottleneck) using skip connections → 1ch logits

    The spatial latent z has shape (B, LATENT_CH, 8, 8), preserving spatial
    structure rather than collapsing to a flat vector. This helps the decoder
    know WHERE to place fire pixels, not just IF there is fire.
    """

    def __init__(self, in_ch=42, base_ch=BASE_CH, ch_mults=CH_MULTS,
                 latent_ch=LATENT_CH, attn_heads=ATTN_HEADS, dropout=DROPOUT):
        super().__init__()
        channels = [base_ch * m for m in ch_mults]  # [64, 128, 256]
        bot_ch = channels[-1]  # 256 (bottleneck channel dim)

        # ── Condition Encoder (42ch → skips + bottleneck at 8×8) ──
        self.cond_stem = nn.Conv2d(in_ch, channels[0], 3, padding=1, bias=False)
        self.cond_enc  = nn.ModuleList()
        self.cond_down = nn.ModuleList()
        prev = channels[0]
        for ch in channels:
            self.cond_enc.append(nn.Sequential(
                ResBlock(prev, ch, dropout), ResBlock(ch, ch, dropout)))
            self.cond_down.append(Downsample(ch))
            prev = ch
        self.cond_bottleneck = nn.Sequential(
            ResBlock(bot_ch, bot_ch, dropout),
            SelfAttention2D(bot_ch, attn_heads),
            ResBlock(bot_ch, bot_ch, dropout),
        )

        # ── Target Encoder (1ch → 256ch @ 8×8, lightweight) ──
        # Mirrors condition encoder's spatial reduction: 64→32→16→8
        self.tgt_enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(inplace=True),     # 32×32
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(inplace=True),    # 16×16
            nn.Conv2d(128, bot_ch, 3, stride=2, padding=1), nn.SiLU(inplace=True), # 8×8
            nn.GroupNorm(get_num_groups(bot_ch), bot_ch),
        )

        # ── Recognition Head: q(z|x,c) ──
        # Input: cat(bottleneck, target_feat) = 2×bot_ch → latent params
        self.recog_head = nn.Sequential(
            ResBlock(bot_ch * 2, bot_ch, dropout),
            nn.Conv2d(bot_ch, latent_ch * 2, 1),  # → μ_q, logσ_q
        )

        # ── Prior Head: p(z|c) ──
        # Input: bottleneck alone = bot_ch → latent params
        self.prior_head = nn.Sequential(
            ResBlock(bot_ch, bot_ch, dropout),
            nn.Conv2d(bot_ch, latent_ch * 2, 1),  # → μ_p, logσ_p
        )

        # ── Decoder (z + bottleneck + skips → 1ch logits) ──
        self.dec_fuse = ResBlock(bot_ch + latent_ch, bot_ch, dropout)

        self.dec_up    = nn.ModuleList()
        self.dec_block = nn.ModuleList()
        prev = bot_ch
        for ch in reversed(channels):
            self.dec_up.append(Upsample(prev))
            self.dec_block.append(nn.Sequential(
                ResBlock(prev + ch, ch, dropout),
                ResBlock(ch, ch, dropout),
            ))
            prev = ch

        self.head = nn.Sequential(
            nn.GroupNorm(get_num_groups(prev), prev),
            nn.SiLU(inplace=True),
            nn.Conv2d(prev, 1, 1),
        )

    @staticmethod
    def reparameterize(mu, logvar):
        """Sample z = μ + σ·ε, ε ~ N(0,1)."""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    @staticmethod
    def kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
        """KL(q ‖ p) for two diagonal Gaussians. Returns scalar (mean over batch)."""
        # KL = 0.5 * Σ [ logσ²_p - logσ²_q + (σ²_q + (μ_q-μ_p)²) / σ²_p - 1 ]
        kl = 0.5 * (logvar_p - logvar_q
                     + torch.exp(logvar_q - logvar_p)
                     + (mu_q - mu_p).pow(2) / torch.exp(logvar_p)
                     - 1.0)
        # Sum over latent dims (C,H,W), mean over batch
        return kl.sum(dim=[1, 2, 3]).mean()

    def _encode_condition(self, cond, fire_in):
        """Run condition encoder, return (skips, bottleneck)."""
        x = self.cond_stem(torch.cat([cond, fire_in], dim=1))
        skips = []
        for enc, down in zip(self.cond_enc, self.cond_down):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.cond_bottleneck(x)
        return skips, x

    def _decode(self, z, bottleneck, skips):
        """Fuse z with bottleneck and decode through skip connections."""
        x = self.dec_fuse(torch.cat([z, bottleneck], dim=1))
        for up, dec, skip in zip(self.dec_up, self.dec_block, reversed(skips)):
            x = torch.cat([up(x), skip], dim=1)
            x = dec(x)
        return self.head(x)

    def forward(self, cond, fire_in, target=None):
        """
        Training (target provided):
          Returns (logits, kl_loss)
        Inference (target=None):
          Returns logits only
        """
        skips, bottleneck = self._encode_condition(cond, fire_in)

        # Prior: always computed
        prior_params = self.prior_head(bottleneck)
        mu_p, logvar_p = prior_params.chunk(2, dim=1)

        if target is not None:
            # Recognition: encode target → posterior
            tgt_feat = self.tgt_enc(target.unsqueeze(1) if target.dim() == 3 else target)
            recog_params = self.recog_head(torch.cat([bottleneck, tgt_feat], dim=1))
            mu_q, logvar_q = recog_params.chunk(2, dim=1)

            # Clamp log-variance for stability
            logvar_q = logvar_q.clamp(-10, 10)
            logvar_p = logvar_p.clamp(-10, 10)

            z = self.reparameterize(mu_q, logvar_q)
            kl = self.kl_divergence(mu_q, logvar_q, mu_p, logvar_p)
            logits = self._decode(z, bottleneck, skips)
            return logits, kl
        else:
            # Inference: use prior mean (deterministic, best single estimate)
            z = mu_p
            logits = self._decode(z, bottleneck, skips)
            return logits

    @torch.no_grad()
    def predict_merged(self, cond, fire_in):
        """Visible keeps original; masked gets sigmoid(pred). Uses prior mean."""
        with autocast():
            logits = self.forward(cond, fire_in, target=None)
        prob = torch.sigmoid(logits.float()).squeeze(1)
        vis = fire_in[:, 0]
        return vis * fire_in[:, 1] + (1 - vis) * prob


# ────────────────────────────── Loss & Metrics ─────────────────────────────

def focal_loss_masked(logits, target, vis_mask, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
    occ = 1 - vis_mask
    logits, target = logits.float(), target.float()
    pt  = target * torch.sigmoid(logits) + (1 - target) * (1 - torch.sigmoid(logits))
    at  = target * alpha + (1 - target) * (1 - alpha)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (at * (1 - pt) ** gamma * bce * occ).sum() / occ.sum().clamp(min=1)


def dice_on_mask(pred_bin, target, vis_mask):
    occ = 1 - vis_mask
    p, t = pred_bin * occ, target * occ
    inter, denom = (p * t).sum(), p.sum() + t.sum()
    return (2.0 * inter / (denom + 1e-6)).item() if denom > 0 else 1.0


def fpr_on_mask(pred_bin, target, vis_mask):
    occ = (1 - vis_mask).bool()
    p, t = pred_bin[occ], target[occ]
    fp = ((p == 1) & (t == 0)).float().sum()
    tn = ((p == 0) & (t == 0)).float().sum()
    return (fp / (fp + tn + 1e-6)).item() if (fp + tn) > 0 else 0.0


# ────────────────────────────── Training ───────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, device, epoch, total_epochs):
    """Training with AMP. Loss = focal(masked) + β·KL with linear warmup."""
    model.train()
    total_loss, total_recon, total_kl, count = 0., 0., 0., 0

    # β annealing: 0 → KL_WEIGHT linearly over first KL_WARMUP epochs
    beta = KL_WEIGHT * min(1.0, epoch / max(KL_WARMUP, 1))

    pbar = tqdm(loader, desc=f"    Train (β={beta:.1e})", leave=False)
    for cond, fire, target, vis in pbar:
        cond   = cond.to(device, non_blocking=True)
        fire   = fire.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        vis    = vis.to(device, non_blocking=True)

        with autocast():
            logits, kl = model(cond, fire, target=target)
            logits = logits.squeeze(1)
        recon = focal_loss_masked(logits, target, vis)
        loss  = recon + beta * kl

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = cond.shape[0]
        total_loss  += loss.item() * bs
        total_recon += recon.item() * bs
        total_kl    += kl.item() * bs
        count += bs
        pbar.set_postfix(loss=f"{total_loss/count:.4f}",
                         recon=f"{total_recon/count:.4f}",
                         kl=f"{total_kl/count:.2f}")

    n = max(count, 1)
    return total_loss / n, total_recon / n, total_kl / n


@torch.no_grad()
def validate(model, loader, device):
    """Dice on validation set using prior mean (inference mode)."""
    model.eval()
    scores = []
    for cond, fire, target, vis in tqdm(loader, desc="    Val  ", leave=False):
        cond   = cond.to(device, non_blocking=True)
        fire   = fire.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        vis    = vis.to(device, non_blocking=True)

        with autocast():
            logits = model(cond, fire, target=None)  # inference: use prior
            logits = logits.squeeze(1)
        pred = (torch.sigmoid(logits.float()) >= 0.5).float()
        for b in range(cond.shape[0]):
            scores.append(dice_on_mask(pred[b], target[b], vis[b]))

    return np.mean(scores) if scores else 0.0


# ────────────────────────────── Testing ────────────────────────────────────

def _run_batch(model, device, conds, fires, metas,
               metrics, dice_all, cat, diff, is_dice):
    """Run inference batch, compute per-sample metrics."""
    cb = torch.stack(conds).to(device, non_blocking=True)
    fb = torch.stack(fires).to(device, non_blocking=True)
    merged = model.predict_merged(cb, fb).cpu()
    pred_bin = (merged >= 0.5).float()

    for i, m in enumerate(metas):
        score_fn = dice_on_mask if is_dice else fpr_on_mask
        metrics[(cat, diff)].append(score_fn(pred_bin[i], m["tgt"], m["vm"]))
        dice_all[diff].append(dice_on_mask(pred_bin[i], m["tgt"], m["vm"]))


@torch.no_grad()
def test_metrics_only(model, test_year, mask_type, device):
    """Evaluate all categories × difficulties, return metrics (no file saving)."""
    model.eval()
    root = src_dir(mask_type)
    metrics  = defaultdict(list)
    dice_all = defaultdict(list)

    for cat in CATEGORIES:
        clean_dir = os.path.join(root, f"{test_year}_{cat}", "difficulty_0")
        if not os.path.isdir(clean_dir):
            continue

        fnames  = sorted(f for f in os.listdir(clean_dir) if f.endswith(".pt"))
        is_dice = cat in DICE_CATS

        # Pre-load clean fire channel as binary uint8
        clean_fire = {}
        for fn in tqdm(fnames, desc=f"    Load {cat}", leave=False):
            x = torch.load(os.path.join(clean_dir, fn),
                           map_location="cpu", weights_only=False)["x"]
            clean_fire[fn] = (x[:, FIRE_CH] > 0).to(torch.uint8)

        for diff in tqdm(DIFFICULTIES, desc=f"    {mask_type}/{cat}", leave=False):
            diff_dir = os.path.join(root, f"{test_year}_{cat}", f"difficulty_{diff}")
            if not os.path.isdir(diff_dir):
                continue

            batch_c, batch_f, batch_m = [], [], []

            for fn in fnames:
                path = os.path.join(diff_dir, fn)
                if not os.path.exists(path):
                    continue

                deg = torch.load(path, map_location="cpu", weights_only=False)
                T = deg["x"].shape[0]

                for t in range(T):
                    xt = drop_det(deg["x"][t].float())
                    batch_c.append(xt[:40])
                    batch_f.append(xt[40:])
                    batch_m.append({"vm": xt[40], "tgt": clean_fire[fn][t].float()})

                    if len(batch_c) >= TEST_BATCH:
                        _run_batch(model, device, batch_c, batch_f, batch_m,
                                   metrics, dice_all, cat, diff, is_dice)
                        batch_c, batch_f, batch_m = [], [], []

            if batch_c:
                _run_batch(model, device, batch_c, batch_f, batch_m,
                           metrics, dice_all, cat, diff, is_dice)

    return metrics, dice_all


# ────────────────────────────── LOOCV Pipeline ─────────────────────────────

def run_loocv():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Print model size
    tmp = cVAEFire()
    n_params = sum(p.numel() for p in tmp.parameters())
    print(f"cVAE params: {n_params / 1e6:.2f}M  ({n_params:,})")
    del tmp

    files_by_year = gather_train_files()
    for yr, fl in files_by_year.items():
        print(f"  Year {yr}: {len(fl)} samples")
    os.makedirs(SAVE_DIR, exist_ok=True)

    all_metrics  = {mt: defaultdict(list) for mt in MASK_TYPES}
    all_dice_all = {mt: defaultdict(list) for mt in MASK_TYPES}

    for test_yr in YEARS:
        print(f"\n{'=' * 60}")
        print(f"  FOLD: test_year = {test_yr}")
        print(f"{'=' * 60}")

        train_files = [f for yr in YEARS if yr != test_yr for f in files_by_year[yr]]
        val_files   = files_by_year[test_yr]
        if not val_files:
            print("  No val data, skipping.")
            continue
        print(f"  Train: {len(train_files)} | Val: {len(val_files)}")

        loader_kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                         pin_memory=True, persistent_workers=True, prefetch_factor=PREFETCH)
        train_loader = DataLoader(FireDataset(train_files), shuffle=True,  **loader_kw)
        val_loader   = DataLoader(FireDataset(val_files),   shuffle=False, **loader_kw)

        model     = cVAEFire().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler    = GradScaler()
        best_dice, best_state = 0.0, None

        print(f"\n  Training ({EPOCHS} epochs, KL warmup={KL_WARMUP})...")
        for ep in range(EPOCHS):
            loss, recon, kl = train_one_epoch(
                model, train_loader, optimizer, scaler, device, ep, EPOCHS)
            scheduler.step()

            if (ep + 1) % 5 == 0 or ep == EPOCHS - 1:
                vd = validate(model, val_loader, device)
                star = " *" if vd > best_dice else ""
                print(f"  Ep {ep+1:>3}/{EPOCHS}  loss={loss:.4f} "
                      f"(recon={recon:.4f} kl={kl:.2f})  "
                      f"val_DICE={vd:.4f}  lr={scheduler.get_last_lr()[0]:.2e}{star}")
                if vd > best_dice:
                    best_dice = vd
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                print(f"  Ep {ep+1:>3}/{EPOCHS}  loss={loss:.4f} "
                      f"(recon={recon:.4f} kl={kl:.2f})")

        model.load_state_dict(best_state)
        model.to(device)
        torch.save(best_state, os.path.join(SAVE_DIR, f"model_test{test_yr}.pt"))
        print(f"  Best val DICE: {best_dice:.4f}")

        # Test both mask types (metrics only, no file reconstruction)
        for mt in MASK_TYPES:
            print(f"\n  Testing {mt} {test_yr}...")
            fold_m, fold_d = test_metrics_only(model, test_yr, mt, device)
            for k, v in fold_m.items():
                all_metrics[mt][k].extend(v)
            for k, v in fold_d.items():
                all_dice_all[mt][k].extend(v)

    # ── Print & save final results ──
    _print_results(all_metrics, all_dice_all)


def _print_results(all_metrics, all_dice_all):
    print(f"\n{'=' * 70}")
    print("  FINAL RESULTS — cVAE (all 4 test folds pooled)")
    print(f"{'=' * 70}")

    summary = {}
    for mt in MASK_TYPES:
        print(f"\n  +--- {mt.upper()} ---")
        summary[mt] = {}

        for cat in CATEGORIES:
            mn = "DICE" if cat in DICE_CATS else "FPR"
            print(f"  |  {cat} ({mn})")
            summary[mt][cat] = {}
            for diff in DIFFICULTIES:
                vals = all_metrics[mt].get((cat, diff), [])
                if not vals:
                    continue
                m, s = np.mean(vals), np.std(vals)
                summary[mt][cat][diff] = {"mean": m, "std": s, "n": len(vals)}
                print(f"  |    diff={diff:>2}%: {mn}={m:.4f} +/- {s:.4f} (n={len(vals)})")

        print(f"  |  ALL SCENARIOS (DICE)")
        summary[mt]["ALL"] = {}
        for diff in DIFFICULTIES:
            vals = all_dice_all[mt].get(diff, [])
            if not vals:
                continue
            m, s = np.mean(vals), np.std(vals)
            summary[mt]["ALL"][diff] = {"mean": m, "std": s, "n": len(vals)}
            print(f"  |    diff={diff:>2}%: DICE={m:.4f} +/- {s:.4f} (n={len(vals)})")
        print(f"  +---")

    result_path = os.path.join(SAVE_DIR, "loocv_results.pt")
    torch.save({
        "summary":      summary,
        "raw_per_cat":  {mt: {str(k): v for k, v in d.items()} for mt, d in all_metrics.items()},
        "raw_all_dice": {mt: {str(k): v for k, v in d.items()} for mt, d in all_dice_all.items()},
    }, result_path)

    print(f"\n{'=' * 70}")
    print(f"Results → {result_path}")
    print(f"Models  → {SAVE_DIR}/model_test{{year}}.pt")


if __name__ == "__main__":
    run_loocv()