"""
D3PM Fire Reconstruction — Discrete Denoising Diffusion for Binary Fire Maps
=============================================================================
LOOCV pipeline using conditional D3PM with binary uniform diffusion.

Key idea:
  - Fire maps are binary {0,1}. D3PM diffuses by randomly flipping pixels.
  - A U-Net learns to predict the clean fire map x0 from any noisy version xt.
  - At inference, iteratively denoise from pure noise to reconstruct masked pixels.

Training:  corrupt target -> U-Net predicts x0 -> focal loss on masked pixels
Inference: start from random bits -> iterate T steps -> final prediction
"""

import os, glob, math, random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ══════════════════════════════ Config ══════════════════════════════════════

DATA_ROOT = "/pub/cyang27/data"
TRAIN_DIR = os.path.join(DATA_ROOT, "data_mixed_challenge_train")
SAVE_DIR  = os.path.join(DATA_ROOT, "d3pm_results")

MASK_TYPES   = ["blockwise", "pixelwise"]
YEARS        = [2018, 2019, 2020, 2021]
CATEGORIES   = ["Fire_Continues", "Fire_Extinguished", "NewFire_NoHistory", "NoFire"]
DICE_CATS    = {"Fire_Continues", "Fire_Extinguished"}
FPR_CATS     = {"NewFire_NoHistory", "NoFire"}
DIFFICULTIES = [10, 20, 30, 40, 50, 60, 70, 80]
FIRE_CH, DET_CH = 42, 41

# Architecture
IMG_SIZE, BASE_CH = 64, 64
CH_MULTS   = [1, 2, 4]       # encoder/decoder channel multipliers
ATTN_HEADS = 8
DROPOUT    = 0.1
TIME_DIM   = 256              # timestep embedding size

# Diffusion
T_DIFF      = 100             # total diffusion steps
INFER_STEPS = 10              # strided steps at inference (10 steps ≈ 11 forward passes)

# Training
SEED, BATCH_SIZE, EPOCHS = 42, 64, 50
LR, WEIGHT_DECAY = 1e-4, 1e-5
NUM_WORKERS, PREFETCH = 8, 4
FOCAL_ALPHA, FOCAL_GAMMA = 0.75, 2.0
TEST_BATCH = 40


# ══════════════════════════════ Utilities ═══════════════════════════════════

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def src_dir(mt):
    return os.path.join(DATA_ROOT, f"data_challenge_{mt}")

def drop_det(x43):
    """(43,H,W) -> (42,H,W): remove detection channel at index 41."""
    return torch.cat([x43[:DET_CH], x43[DET_CH+1:]], dim=0)

def num_groups(ch, target=32):
    """Largest divisor of ch that is <= target, for GroupNorm."""
    for g in [target, 16, 8, 4, 2, 1]:
        if ch % g == 0: return g
    return 1


# ══════════════════════════════ Noise Schedule ═════════════════════════════

def cosine_beta_schedule(T):
    """Cosine schedule: smooth beta curve, avoids too-small early betas."""
    s = 0.008
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * (math.pi / 2)) ** 2
    alpha_bar = f / f[0]
    betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-6, 0.999)
    return betas.float()


# ══════════════════════════════ Dataset ═════════════════════════════════════

class FireDataset(Dataset):
    """Returns (cond_40ch, fire_2ch, binary_target, vis_mask)."""
    def __init__(self, files):
        self.files = files
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        d = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        x = drop_det(d["x"].float())
        y = (d["y"].float() > 0).float()
        return x[:40], x[40:], y, x[40]  # cond, fire_in, target, vis_mask

def gather_train_files():
    out = {}
    for yr in YEARS:
        d = os.path.join(TRAIN_DIR, str(yr))
        out[yr] = sorted(glob.glob(os.path.join(d, "*.pt"))) if os.path.isdir(d) else []
    return out


# ══════════════════════════════ Building Blocks ════════════════════════════

class SinusoidalTimeEmb(nn.Module):
    """Integer timestep -> sinusoidal encoding -> MLP -> dense vector."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.SiLU(), nn.Linear(dim*4, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        emb = torch.cat([(t[:,None].float() * freqs[None]).sin(),
                          (t[:,None].float() * freqs[None]).cos()], dim=-1)
        return self.mlp(emb)


class AdaGNResBlock(nn.Module):
    """ResBlock with Adaptive GroupNorm -- timestep modulates scale & shift."""
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.drop  = nn.Dropout2d(dropout)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch * 2))

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        # Adaptive norm: timestep controls scale & shift of second norm
        h = self.norm2(h)
        scale, shift = self.time_proj(t_emb)[:, :, None, None].chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


class SelfAttention2D(nn.Module):
    def __init__(self, ch, heads=8):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(num_groups(ch), ch)
        self.qkv  = nn.Conv2d(ch, ch*3, 1, bias=False)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C//self.heads, H*W)
        q, k, v = [t.permute(0, 1, 3, 2) for t in qkv.unbind(1)]
        out = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj(out.permute(0, 1, 3, 2).reshape(B, C, H, W))


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.conv(x)

class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x): return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# ══════════════════════════════ D3PM Model ═════════════════════════════════

class D3PMFire(nn.Module):
    """
    Conditional D3PM for binary fire reconstruction.

    Input:  43ch = 40 cond + 2 fire_in (vis_mask, vis_values) + 1 noisy_fire
    Output: 1ch logits -> sigmoid -> P(fire) per pixel

    The network always predicts x0 (clean fire map), NOT noise.
    At inference, x0 prediction + Bayes posterior -> sample x_{t-1} iteratively.
    """

    def __init__(self, in_ch=43, base_ch=BASE_CH, ch_mults=CH_MULTS,
                 time_dim=TIME_DIM, T=T_DIFF):
        super().__init__()
        self.T = T
        chs = [base_ch * m for m in ch_mults]  # [64, 128, 256]
        bot = chs[-1]

        # --- Diffusion schedule (registered as buffers -> auto move to GPU) ---
        # Convention: t=0 is clean, t=T is fully noisy
        # betas[1..T] are noise rates, alpha_bar[0]=1.0 (clean), length T+1
        betas_raw = cosine_beta_schedule(T)                       # length T
        betas = F.pad(betas_raw, (1, 0), value=0.0)              # [0, β1, ..., βT], length T+1
        alpha_bar = torch.cumprod(1.0 - betas, dim=0)            # [1.0, ᾱ1, ..., ᾱT], length T+1
        self.register_buffer("betas", betas)                     # index 0..T
        self.register_buffer("alpha_bar", alpha_bar)              # index 0..T

        # --- Timestep embedding ---
        self.time_emb = SinusoidalTimeEmb(time_dim)

        # --- U-Net Encoder ---
        self.stem = nn.Conv2d(in_ch, chs[0], 3, padding=1, bias=False)
        self.enc, self.downs = nn.ModuleList(), nn.ModuleList()
        prev = chs[0]
        for ch in chs:
            self.enc.append(nn.ModuleList([
                AdaGNResBlock(prev, ch, time_dim, DROPOUT),
                AdaGNResBlock(ch, ch, time_dim, DROPOUT),
            ]))
            self.downs.append(Down(ch))
            prev = ch

        # --- Bottleneck ---
        self.bot = nn.ModuleList([
            AdaGNResBlock(bot, bot, time_dim, DROPOUT),
            SelfAttention2D(bot, ATTN_HEADS),
            AdaGNResBlock(bot, bot, time_dim, DROPOUT),
        ])

        # --- U-Net Decoder ---
        self.ups, self.dec = nn.ModuleList(), nn.ModuleList()
        prev = bot
        for ch in reversed(chs):
            self.ups.append(Up(prev))
            self.dec.append(nn.ModuleList([
                AdaGNResBlock(prev + ch, ch, time_dim, DROPOUT),
                AdaGNResBlock(ch, ch, time_dim, DROPOUT),
            ]))
            prev = ch

        # --- Output ---
        self.head = nn.Sequential(
            nn.GroupNorm(num_groups(prev), prev), nn.SiLU(), nn.Conv2d(prev, 1, 1))

    # -------------------- Forward diffusion --------------------

    def q_sample(self, x0, t):
        """
        Corrupt binary x0 at timestep t (1-indexed: t=1..T).
        alpha_bar[0]=1 (clean), alpha_bar[T]≈0 (pure noise).
        """
        ab = self.alpha_bar[t][:, None, None]       # (B,1,1)
        keep_prob = (1.0 + ab) / 2.0
        keep = torch.bernoulli(keep_prob.expand_as(x0))
        rand = torch.bernoulli(torch.full_like(x0, 0.5))
        return keep * x0 + (1 - keep) * rand

    # -------------------- Reverse posterior --------------------

    def q_posterior_logits(self, x_t, x0_logits, t, t_prev):
        """
        Compute log q(x_{t_prev} | x_t, x0_hat) via Bayes rule.
        Supports arbitrary step sizes (t -> t_prev) for strided inference.

        Key: the effective transition from t_prev to t has
          effective_alpha = alpha_bar[t] / alpha_bar[t_prev]
          effective_beta  = 1 - effective_alpha
        This replaces single-step betas[t] to handle stride correctly.

        Returns: (B, 2, H, W) log-probabilities for x_{t_prev} in {0, 1}
        """
        x0_prob = torch.sigmoid(x0_logits)            # P(x0=1)

        ab_t    = self.alpha_bar[t][:, None, None]     # ᾱ_t
        ab_prev = self.alpha_bar[t_prev][:, None, None] # ᾱ_{t_prev}

        # Effective transition rate from t_prev to t
        eff_alpha = (ab_t / ab_prev.clamp(min=1e-10)).clamp(max=1.0)
        eff_beta  = 1.0 - eff_alpha                    # prob of resampling in this gap

        # log q(x_t | x_{t_prev}=v): transition from t_prev to t
        log_same = torch.log((1 - eff_beta/2).clamp(min=1e-10))  # stay same
        log_flip = torch.log((eff_beta/2).clamp(min=1e-10))      # flip state

        log_xt_if_1 = x_t * log_same + (1-x_t) * log_flip    # x_{t_prev}=1
        log_xt_if_0 = (1-x_t) * log_same + x_t * log_flip    # x_{t_prev}=0

        # log q(x_{t_prev}=v | x0): marginal at t_prev
        p1 = x0_prob * (1+ab_prev)/2 + (1-x0_prob) * (1-ab_prev)/2
        p0 = 1 - p1

        # Combine and normalize
        log_unnorm = torch.stack([
            log_xt_if_0 + torch.log(p0.clamp(min=1e-10)),     # v=0
            log_xt_if_1 + torch.log(p1.clamp(min=1e-10)),     # v=1
        ], dim=1)  # (B,2,H,W)
        return log_unnorm - torch.logsumexp(log_unnorm, dim=1, keepdim=True)

    # -------------------- U-Net denoiser --------------------

    def _denoise(self, x_in, t):
        """U-Net: (B,43,H,W) + timestep -> (B,1,H,W) x0 logits."""
        te = self.time_emb(t)

        # Encoder
        h = self.stem(x_in)
        skips = []
        for blocks, down in zip(self.enc, self.downs):
            for b in blocks: h = b(h, te)
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.bot[0](h, te)
        h = self.bot[1](h)          # self-attention (no time cond)
        h = self.bot[2](h, te)

        # Decoder
        for up, blocks, skip in zip(self.ups, self.dec, reversed(skips)):
            h = torch.cat([up(h), skip], dim=1)
            for b in blocks: h = b(h, te)

        return self.head(h)

    # -------------------- Training forward --------------------

    def forward(self, cond, fire_in, target, vis_mask):
        """
        Training pass:
          1. Sample random t per sample
          2. Corrupt target at masked pixels -> x_t
          3. Build 43ch input = [cond(40), fire_in(2), noisy_fire(1)]
          4. U-Net predicts x0 logits
        Returns: (B,1,H,W) logits for loss computation
        """
        B, device = cond.shape[0], cond.device
        t = torch.randint(1, self.T + 1, (B,), device=device)  # t in {1, ..., T}

        # Corrupt target, but only at masked pixels
        x_t = self.q_sample(target, t)
        occ = 1.0 - vis_mask
        noisy_fire = vis_mask * fire_in[:, 1] + occ * x_t     # (B,H,W)

        x_in = torch.cat([cond, fire_in, noisy_fire.unsqueeze(1)], dim=1)  # (B,43,H,W)
        return self._denoise(x_in, t)

    # -------------------- Inference --------------------

    @torch.no_grad()
    def predict_merged(self, cond, fire_in, steps=INFER_STEPS):
        """
        Iterative reverse diffusion to reconstruct masked pixels.
        Visible pixels are always kept as-is.
        Returns: (B,H,W) soft predictions in [0,1].
        """
        B, _, H, W = cond.shape
        device = cond.device
        vis = fire_in[:, 0]                 # visibility mask
        val = fire_in[:, 1]                 # visible fire values
        occ = 1.0 - vis

        # Strided timestep schedule: T -> ... -> 1 -> 0
        # t=0 is clean, so the last denoising step goes from t=stride to t=0
        stride = max(1, self.T // steps)
        ts = list(range(self.T, 0, -stride))    # e.g. [100, 96, ..., 4]
        if ts[-1] != 1: ts.append(1)            # make sure we reach t=1

        # Start from pure noise at masked pixels
        x_t = vis * val + occ * torch.bernoulli(torch.full((B,H,W), 0.5, device=device))
        x0_prob = None

        for i, t_val in enumerate(ts):
            t_batch = torch.full((B,), t_val, dtype=torch.long, device=device)

            # Predict clean fire map
            x_in = torch.cat([cond, fire_in, x_t.unsqueeze(1)], dim=1)
            with autocast():
                logits = self._denoise(x_in, t_batch).float().squeeze(1)
            x0_prob = torch.sigmoid(logits)

            # Determine t_prev (where we're jumping to)
            if i + 1 < len(ts):
                t_prev_val = ts[i + 1]
            else:
                t_prev_val = 0  # final step: jump to clean

            if t_prev_val == 0:
                break  # last step: use x0 prediction directly

            # Sample x_{t_prev} from posterior at masked pixels
            t_prev_batch = torch.full((B,), t_prev_val, dtype=torch.long, device=device)
            log_p = self.q_posterior_logits(occ * x_t, logits, t_batch, t_prev_batch)
            gumbel = -torch.log(-torch.log(torch.rand_like(log_p).clamp(1e-10)).clamp(1e-10))
            x_prev = (log_p + gumbel).argmax(dim=1).float()
            x_t = vis * val + occ * x_prev

        return vis * val + occ * x0_prob

    @torch.no_grad()
    def predict_merged_fast(self, cond, fire_in):
        """Few-step iterative denoising for validation (5 steps ≈ 6 forward passes)."""
        return self.predict_merged(cond, fire_in, steps=5)


# ══════════════════════════════ Loss & Metrics ═════════════════════════════

def focal_loss_masked(logits, target, vis_mask):
    """Focal BCE on masked (occluded) pixels only."""
    occ = 1 - vis_mask
    logits, target = logits.float(), target.float()
    pt  = target * torch.sigmoid(logits) + (1 - target) * (1 - torch.sigmoid(logits))
    at  = target * FOCAL_ALPHA + (1 - target) * (1 - FOCAL_ALPHA)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (at * (1 - pt)**FOCAL_GAMMA * bce * occ).sum() / occ.sum().clamp(min=1)

def dice_on_mask(pred_bin, target, vis_mask):
    occ = 1 - vis_mask
    p, t = pred_bin * occ, target * occ
    inter = (p * t).sum(); denom = p.sum() + t.sum()
    return (2 * inter / (denom + 1e-6)).item() if denom > 0 else 1.0

def fpr_on_mask(pred_bin, target, vis_mask):
    occ = (1 - vis_mask).bool()
    p, t = pred_bin[occ], target[occ]
    fp = ((p==1)&(t==0)).float().sum(); tn = ((p==0)&(t==0)).float().sum()
    return (fp / (fp + tn + 1e-6)).item() if (fp+tn) > 0 else 0.0


# ══════════════════════════════ Train / Val / Test ═════════════════════════

def train_one_epoch(model, loader, opt, scaler, device):
    model.train()
    total, n = 0.0, 0
    for cond, fire, tgt, vis in tqdm(loader, desc="    Train", leave=False):
        cond, fire = cond.to(device, non_blocking=True), fire.to(device, non_blocking=True)
        tgt, vis   = tgt.to(device, non_blocking=True),  vis.to(device, non_blocking=True)

        with autocast():
            logits = model(cond, fire, tgt, vis).squeeze(1)
        loss = focal_loss_masked(logits, tgt, vis)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()

        total += loss.item() * cond.shape[0]; n += cond.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, device):
    """Validation with 5-step denoising (6 forward passes per sample)."""
    model.eval()
    scores = []
    for cond, fire, tgt, vis in tqdm(loader, desc="    Val  ", leave=False):
        cond, fire = cond.to(device, non_blocking=True), fire.to(device, non_blocking=True)
        tgt, vis   = tgt.to(device, non_blocking=True),  vis.to(device, non_blocking=True)
        pred = (model.predict_merged_fast(cond, fire) >= 0.5).float()
        for b in range(cond.shape[0]):
            scores.append(dice_on_mask(pred[b], tgt[b], vis[b]))
    return np.mean(scores) if scores else 0.0


def _run_batch(model, device, conds, fires, metas, metrics, dice_all, cat, diff, is_dice):
    merged = model.predict_merged(
        torch.stack(conds).to(device), torch.stack(fires).to(device)).cpu()
    pred = (merged >= 0.5).float()
    for i, m in enumerate(metas):
        fn = dice_on_mask if is_dice else fpr_on_mask
        metrics[(cat, diff)].append(fn(pred[i], m["tgt"], m["vm"]))
        dice_all[diff].append(dice_on_mask(pred[i], m["tgt"], m["vm"]))


@torch.no_grad()
def test_metrics(model, year, mask_type, device):
    model.eval()
    root = src_dir(mask_type)
    metrics, dice_all = defaultdict(list), defaultdict(list)

    for cat in CATEGORIES:
        clean_dir = os.path.join(root, f"{year}_{cat}", "difficulty_0")
        if not os.path.isdir(clean_dir): continue
        fnames = sorted(f for f in os.listdir(clean_dir) if f.endswith(".pt"))
        is_dice = cat in DICE_CATS

        # Load ground truth
        gt = {}
        for fn in tqdm(fnames, desc=f"    Load {cat}", leave=False):
            x = torch.load(os.path.join(clean_dir, fn), map_location="cpu", weights_only=False)["x"]
            gt[fn] = (x[:, FIRE_CH] > 0).to(torch.uint8)

        for diff in tqdm(DIFFICULTIES, desc=f"    {mask_type}/{cat}", leave=False):
            diff_dir = os.path.join(root, f"{year}_{cat}", f"difficulty_{diff}")
            if not os.path.isdir(diff_dir): continue
            bc, bf, bm = [], [], []

            for fn in fnames:
                path = os.path.join(diff_dir, fn)
                if not os.path.exists(path): continue
                deg = torch.load(path, map_location="cpu", weights_only=False)
                for t in range(deg["x"].shape[0]):
                    xt = drop_det(deg["x"][t].float())
                    bc.append(xt[:40]); bf.append(xt[40:])
                    bm.append({"vm": xt[40], "tgt": gt[fn][t].float()})
                    if len(bc) >= TEST_BATCH:
                        _run_batch(model, device, bc, bf, bm, metrics, dice_all, cat, diff, is_dice)
                        bc, bf, bm = [], [], []
            if bc:
                _run_batch(model, device, bc, bf, bm, metrics, dice_all, cat, diff, is_dice)

    return metrics, dice_all


# ══════════════════════════════ LOOCV Pipeline ═════════════════════════════

def run_loocv():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # Model info
    tmp = D3PMFire(); print(f"D3PM params: {sum(p.numel() for p in tmp.parameters())/1e6:.2f}M"); del tmp

    files = gather_train_files()
    for yr, fl in files.items(): print(f"  Year {yr}: {len(fl)} samples")
    os.makedirs(SAVE_DIR, exist_ok=True)

    all_met  = {mt: defaultdict(list) for mt in MASK_TYPES}
    all_dice = {mt: defaultdict(list) for mt in MASK_TYPES}

    for test_yr in YEARS:
        print(f"\n{'='*60}\n  FOLD: test_year = {test_yr}\n{'='*60}")
        train_f = [f for yr in YEARS if yr != test_yr for f in files[yr]]
        val_f   = files[test_yr]
        if not val_f: print("  No val data."); continue
        print(f"  Train: {len(train_f)} | Val: {len(val_f)}")

        kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                  pin_memory=True, persistent_workers=True, prefetch_factor=PREFETCH)
        train_dl = DataLoader(FireDataset(train_f), shuffle=True,  **kw)
        val_dl   = DataLoader(FireDataset(val_f),   shuffle=False, **kw)

        model = D3PMFire().to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        scaler = GradScaler()
        best_dice, best_state = 0.0, None

        for ep in range(EPOCHS):
            loss = train_one_epoch(model, train_dl, opt, scaler, device)
            sched.step()

            if (ep+1) % 5 == 0 or ep == EPOCHS-1:
                vd = validate(model, val_dl, device)
                star = " *" if vd > best_dice else ""
                print(f"  Ep {ep+1:>3}/{EPOCHS}  loss={loss:.4f}  dice={vd:.4f}  "
                      f"lr={sched.get_last_lr()[0]:.2e}{star}")
                if vd > best_dice:
                    best_dice = vd
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                print(f"  Ep {ep+1:>3}/{EPOCHS}  loss={loss:.4f}")

        model.load_state_dict(best_state); model.to(device)
        torch.save(best_state, os.path.join(SAVE_DIR, f"model_test{test_yr}.pt"))
        print(f"  Best val DICE: {best_dice:.4f}")

        for mt in MASK_TYPES:
            print(f"\n  Testing {mt} {test_yr}...")
            m, d = test_metrics(model, test_yr, mt, device)
            for k, v in m.items(): all_met[mt][k].extend(v)
            for k, v in d.items(): all_dice[mt][k].extend(v)

    # Print results
    print(f"\n{'='*70}\n  FINAL RESULTS - D3PM\n{'='*70}")
    summary = {}
    for mt in MASK_TYPES:
        print(f"\n  --- {mt.upper()} ---")
        summary[mt] = {}
        for cat in CATEGORIES:
            mn = "DICE" if cat in DICE_CATS else "FPR"
            print(f"  {cat} ({mn})")
            summary[mt][cat] = {}
            for diff in DIFFICULTIES:
                vals = all_met[mt].get((cat, diff), [])
                if not vals: continue
                m, s = np.mean(vals), np.std(vals)
                summary[mt][cat][diff] = {"mean": m, "std": s, "n": len(vals)}
                print(f"    diff={diff:>2}%: {mn}={m:.4f} +/- {s:.4f} (n={len(vals)})")
        print(f"  ALL (DICE)")
        summary[mt]["ALL"] = {}
        for diff in DIFFICULTIES:
            vals = all_dice[mt].get(diff, [])
            if not vals: continue
            m, s = np.mean(vals), np.std(vals)
            summary[mt]["ALL"][diff] = {"mean": m, "std": s, "n": len(vals)}
            print(f"    diff={diff:>2}%: DICE={m:.4f} +/- {s:.4f} (n={len(vals)})")

    torch.save({"summary": summary,
                "raw_cat":  {mt: {str(k):v for k,v in d.items()} for mt,d in all_met.items()},
                "raw_dice": {mt: {str(k):v for k,v in d.items()} for mt,d in all_dice.items()}},
               os.path.join(SAVE_DIR, "loocv_results.pt"))
    print(f"\nResults -> {SAVE_DIR}")


if __name__ == "__main__":
    run_loocv()