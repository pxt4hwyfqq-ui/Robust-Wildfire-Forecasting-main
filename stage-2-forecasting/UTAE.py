#!/usr/bin/env python3
"""
U-TAE Fire Prediction — Leave-One-Year-Out CV
==============================================
Input:  T-day satellite sequences (41ch, drop ch40/ch41)
Output: Next-day binary fire map

Evaluation:
  - Fire_Continues / NewFire_NoHistory  → AP  (per sample)
  - Fire_Extinguished / NoFire          → FPR (per sample, threshold=0.5)
  - Combined (all scenarios)            → AP  (per sample)

Key design choices:
  - Focal Loss (gamma=2, alpha=[0.75, 0.25])
  - Fixed threshold 0.5 for both validation model selection and test
  - 4-fold Leave-One-Year-Out CV, train on difficulty_0 only
"""

import os, random, gc, json, copy
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, f1_score
from tqdm import tqdm
import warnings

os.environ["PYTHONUNBUFFERED"] = "1"
warnings.filterwarnings("ignore")

# ========================= CONFIG =========================

DATA_ROOT   = Path("/pub/cyang27/data")
OUTPUT_ROOT = Path("/pub/cyang27/UTAE_Challenge_Results")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MASK_TYPES   = ["pixelwise", "blockwise"]
YEARS        = [2018, 2019, 2020, 2021]
SCENARIOS    = ["Fire_Continues", "Fire_Extinguished", "NewFire_NoHistory", "NoFire"]
AP_SCENARIOS = {"Fire_Continues", "NewFire_NoHistory"}   # evaluated by AP
FPR_SCENARIOS = {"Fire_Extinguished", "NoFire"}           # evaluated by FPR
DIFFICULTIES = list(range(0, 81, 10))                     # 0, 10, ..., 80

# Channel selection: 43 total, drop ch40 (valid_mask) and ch41 (detection_time)
KEEP_CHS  = [i for i in range(43) if i not in {40, 41}]
INPUT_DIM = len(KEEP_CHS)  # 41

# Training hyperparams
SEED        = 42
BATCH_SIZE  = 8
EPOCHS      = 10
PATIENCE    = 3
LR          = 1e-3
WD          = 1e-4
NUM_WORKERS = max(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)) - 2, 4)
TEST_BATCH  = 16

# Fixed settings
THRESHOLD    = 0.5                                  # for val F1 and test FPR
CLASS_WEIGHT = torch.tensor([0.25, 0.75])           # [background, fire]
FOCAL_GAMMA  = 2.0


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def data_dir(mask_type):
    return DATA_ROOT / f"data_challenge_{mask_type}"


# ========================= FOCAL LOSS =========================

class FocalLoss(nn.Module):
    """
    Focal Loss:  FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Reduces contribution of easy examples, focuses on hard ones.
    """

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else torch.ones(2))

    def forward(self, logits, targets):
        # logits: [B, C, H, W],  targets: [B, H, W]
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        p_t = F.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        return ((1.0 - p_t) ** self.gamma * ce).mean()


# ========================= U-TAE MODEL =========================
# Lightweight Temporal Attention Encoder + U-Net decoder

class PositionalEncoder(nn.Module):
    def __init__(self, d, T=1000, repeat=None, offset=0):
        super().__init__()
        self.d, self.T, self.repeat = d, T, repeat
        self.denom = torch.pow(T, 2 * (torch.arange(offset, offset + d).float() // 2) / d)
        self._moved = False

    def forward(self, bp):
        if not self._moved:
            self.denom = self.denom.to(bp.device)
            self._moved = True
        tab = bp[:, :, None] / self.denom[None, None, :]
        tab[:, :, 0::2] = torch.sin(tab[:, :, 0::2])
        tab[:, :, 1::2] = torch.cos(tab[:, :, 1::2])
        return torch.cat([tab] * self.repeat, dim=-1) if self.repeat else tab


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, pad_mask=None):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)) / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        attn = self.dropout(F.softmax(attn, dim=2))
        return torch.matmul(attn, v), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head, self.d_k, self.d_in = n_head, d_k, d_in
        self.Q = nn.Parameter(torch.zeros(n_head, d_k))
        nn.init.normal_(self.Q, 0, np.sqrt(2.0 / d_k))
        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, 0, np.sqrt(2.0 / d_k))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None):
        B, T, _ = v.size()
        n, dk, d = self.n_head, self.d_k, self.d_in

        q = self.Q.unsqueeze(1).expand(-1, B, -1).reshape(-1, dk)
        k = self.fc1_k(v).view(B, T, n, dk).permute(2, 0, 1, 3).reshape(-1, T, dk)
        v2 = torch.stack(v.split(d // n, dim=-1)).reshape(n * B, T, -1)
        if pad_mask is not None:
            pad_mask = pad_mask.repeat(n, 1)

        output, attn = self.attention(q, k, v2, pad_mask=pad_mask)
        attn   = attn.view(n, B, 1, T).squeeze(2)
        output = output.view(n, B, 1, d // n).squeeze(2)
        return output, attn


class LTAE2d(nn.Module):
    """Lightweight Temporal Attention Encoder (2D spatial)."""

    def __init__(self, in_channels=128, n_head=16, d_k=4, mlp_dims=[256, 128],
                 dropout=0.2, d_model=256, T=1000):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model

        self.inconv = nn.Conv1d(in_channels, d_model, 1) if d_model else None
        self.pos_enc = PositionalEncoder(d_model // n_head, T=T, repeat=n_head)
        self.attn = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=d_model)
        self.in_norm = nn.GroupNorm(n_head, in_channels)
        self.out_norm = nn.GroupNorm(n_head, mlp_dims[-1])

        layers = []
        dims = [d_model] + mlp_dims[1:]
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.BatchNorm1d(dims[i + 1]), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None):
        B, T, C, H, W = x.shape

        # Reshape to (B*H*W, T, C) for temporal processing
        if pad_mask is not None:
            pad_mask = (pad_mask[:, :, None, None]
                        .expand(-1, -1, H, W)
                        .permute(0, 2, 3, 1)
                        .reshape(B * H * W, T))

        out = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        # Positional encoding
        bp = (batch_positions[:, :, None, None]
              .expand(-1, -1, H, W)
              .permute(0, 2, 3, 1)
              .reshape(B * H * W, T))
        out = out + self.pos_enc(bp)

        # Temporal attention
        out, attn = self.attn(out, pad_mask=pad_mask)
        out = out.permute(1, 0, 2).reshape(B * H * W, -1)
        out = self.out_norm(self.dropout(self.mlp(out)))
        out = out.view(B, H, W, -1).permute(0, 3, 1, 2)
        attn = attn.view(self.n_head, B, H, W, T).permute(0, 1, 4, 2, 3)
        return out, attn


# --------------- U-Net building blocks ---------------

class TemporallySharedBlock(nn.Module):
    """Apply a 2D block independently to each time step."""

    def __init__(self, pad_value=None):
        super().__init__()
        self.pad_value = pad_value
        self._out_shape = None

    def smart_forward(self, x):
        if x.dim() == 4:
            return self.forward(x)
        B, T, C, H, W = x.shape
        if self.pad_value is not None and self._out_shape is None:
            # Only compute output shape once, then cache it
            with torch.no_grad():
                self._out_shape = self.forward(torch.zeros(1, C, H, W, device=x.device)).shape
        flat = x.view(B * T, C, H, W)
        if self.pad_value is not None:
            pad_mask = (flat == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
            if pad_mask.any():
                _, C2, H2, W2 = self._out_shape
                out = torch.ones(B * T, C2, H2, W2, device=x.device) * self.pad_value
                out[~pad_mask] = self.forward(flat[~pad_mask])
            else:
                out = self.forward(flat)
        else:
            out = self.forward(flat)
        _, C2, H2, W2 = out.shape
        return out.view(B, T, C2, H2, W2)


class ConvLayer(nn.Module):
    def __init__(self, dims, norm="batch", k=3, s=1, p=1, n_groups=4,
                 last_relu=True, padding_mode="reflect"):
        super().__init__()
        norm_fn = {"batch": nn.BatchNorm2d,
                   "instance": nn.InstanceNorm2d,
                   "group": lambda c: nn.GroupNorm(n_groups, c)}.get(norm)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Conv2d(dims[i], dims[i + 1], k, stride=s, padding=p,
                                    padding_mode=padding_mode))
            if norm_fn:
                layers.append(norm_fn(dims[i + 1]))
            if last_relu or i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class ConvBlock(TemporallySharedBlock):
    def __init__(self, dims, pad_value=None, norm="batch", last_relu=True, padding_mode="reflect"):
        super().__init__(pad_value=pad_value)
        self.conv = ConvLayer(dims, norm=norm, last_relu=last_relu, padding_mode=padding_mode)

    def forward(self, x):
        return self.conv(x)


class DownConvBlock(TemporallySharedBlock):
    def __init__(self, d_in, d_out, k, s, p, pad_value=None, norm="batch", padding_mode="reflect"):
        super().__init__(pad_value=pad_value)
        self.down  = ConvLayer([d_in, d_in], norm=norm, k=k, s=s, p=p, padding_mode=padding_mode)
        self.conv1 = ConvLayer([d_in, d_out], norm=norm, padding_mode=padding_mode)
        self.conv2 = ConvLayer([d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, x):
        out = self.down(x)
        out = self.conv1(out)
        return out + self.conv2(out)


class UpConvBlock(nn.Module):
    def __init__(self, d_in, d_out, k, s, p, norm="batch", d_skip=None, padding_mode="reflect"):
        super().__init__()
        d = d_skip or d_out
        self.skip_conv = nn.Sequential(nn.Conv2d(d, d, 1), nn.BatchNorm2d(d), nn.ReLU())
        self.up = nn.Sequential(nn.ConvTranspose2d(d_in, d_out, k, stride=s, padding=p),
                                nn.BatchNorm2d(d_out), nn.ReLU())
        self.conv1 = ConvLayer([d_out + d, d_out], norm=norm, padding_mode=padding_mode)
        self.conv2 = ConvLayer([d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, x, skip):
        out = self.up(x)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        return out + self.conv2(out)


class TemporalAggregator(nn.Module):
    """Aggregate temporal features using attention weights from LTAE."""

    def __init__(self, mode="att_group"):
        super().__init__()
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):
        if self.mode == "att_group":
            return self._att_group(x, pad_mask, attn_mask)
        elif self.mode == "att_mean":
            return self._att_mean(x, pad_mask, attn_mask)
        else:
            return self._mean(x, pad_mask)

    def _att_group(self, x, pad_mask, attn_mask):
        n_heads, B, T, h, w = attn_mask.shape
        attn = attn_mask.view(n_heads * B, T, h, w)
        if x.shape[-2] > w:
            attn = F.interpolate(attn, size=x.shape[-2:], mode="bilinear", align_corners=False)
        else:
            attn = F.avg_pool2d(attn, kernel_size=w // x.shape[-2])
        attn = attn.view(n_heads, B, T, *x.shape[-2:])
        if pad_mask is not None:
            attn = attn * (~pad_mask).float()[None, :, :, None, None]
        out = torch.stack(x.chunk(n_heads, dim=2))
        out = (attn[:, :, :, None, :, :] * out).sum(dim=2)
        return torch.cat(list(out), dim=1)

    def _att_mean(self, x, pad_mask, attn_mask):
        attn = attn_mask.mean(dim=0)
        attn = F.interpolate(attn, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if pad_mask is not None:
            attn = attn * (~pad_mask).float()[:, :, None, None]
        return (x * attn[:, :, None, :, :]).sum(dim=1)

    def _mean(self, x, pad_mask):
        if pad_mask is not None and pad_mask.any():
            mask = (~pad_mask).float()[:, :, None, None, None]
            return (x * mask).sum(dim=1) / mask.sum(dim=1)
        return x.mean(dim=1)


# --------------- Main U-TAE ---------------

class UTAE(nn.Module):
    """
    U-Net with Temporal Attention Encoder.
    Encoder processes each time step independently, LTAE fuses temporally
    at the bottleneck, decoder upsamples with skip connections.
    """

    def __init__(self, input_dim, encoder_widths=[64, 64, 64, 128],
                 decoder_widths=[32, 32, 64, 128], out_conv=[32, 2],
                 str_conv_k=4, str_conv_s=2, str_conv_p=1,
                 agg_mode="att_group", encoder_norm="group",
                 n_head=16, d_model=256, d_k=4, pad_value=0,
                 padding_mode="reflect"):
        super().__init__()
        n_stages = len(encoder_widths)
        self.pad_value = pad_value

        # Encoder
        self.in_conv = ConvBlock(
            [input_dim] + [encoder_widths[0]] * 2,
            pad_value=pad_value, norm=encoder_norm, padding_mode=padding_mode)
        self.down_blocks = nn.ModuleList([
            DownConvBlock(encoder_widths[i], encoder_widths[i + 1],
                          str_conv_k, str_conv_s, str_conv_p,
                          pad_value=pad_value, norm=encoder_norm, padding_mode=padding_mode)
            for i in range(n_stages - 1)])

        # Temporal attention at bottleneck
        self.temporal_encoder = LTAE2d(
            in_channels=encoder_widths[-1], d_model=d_model, n_head=n_head,
            mlp_dims=[d_model, encoder_widths[-1]], d_k=d_k)
        self.temporal_aggregator = TemporalAggregator(mode=agg_mode)

        # Decoder
        self.up_blocks = nn.ModuleList([
            UpConvBlock(decoder_widths[i], decoder_widths[i - 1],
                        str_conv_k, str_conv_s, str_conv_p,
                        norm="batch", d_skip=encoder_widths[i - 1], padding_mode=padding_mode)
            for i in range(n_stages - 1, 0, -1)])

        # Output head (raw logits)
        self.out_conv = nn.Sequential(
            ConvBlock([decoder_widths[0], out_conv[0]], padding_mode=padding_mode),
            nn.Conv2d(out_conv[0], out_conv[1], 1))

    def forward(self, x, batch_positions=None):
        # x: [B, T, C, H, W]
        pad_mask = (x == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)

        if batch_positions is None:
            batch_positions = (torch.arange(x.shape[1], device=x.device)
                               .unsqueeze(0).expand(x.shape[0], -1).float())

        # Encoder: multi-scale features per time step
        out = self.in_conv.smart_forward(x)
        feature_maps = [out]
        for down in self.down_blocks:
            out = down.smart_forward(feature_maps[-1])
            feature_maps.append(out)

        # Temporal fusion at bottleneck
        out, attn = self.temporal_encoder(feature_maps[-1], batch_positions=batch_positions,
                                          pad_mask=pad_mask)

        # Decoder with skip connections
        for i, up in enumerate(self.up_blocks):
            skip = self.temporal_aggregator(feature_maps[-(i + 2)],
                                           pad_mask=pad_mask, attn_mask=attn)
            out = up(out, skip)

        return self.out_conv(out)  # [B, 2, H, W]


# ========================= DATASET =========================

class FireDataset(Dataset):
    def __init__(self, file_list):
        self.files = [str(f) for f in file_list]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            data = torch.load(self.files[idx], map_location="cpu", weights_only=False)
            x = data["x"].float()[:, KEEP_CHS]  # [T, 41, H, W]
            y = data["y"].long()                 # [H, W]
            return x, y
        except Exception as e:
            print(f"[Warning] Load failed: {self.files[idx]}: {e}")
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])


# ========================= FILE DISCOVERY =========================

def get_files(root, year, scenario, difficulty):
    d = root / f"{year}_{scenario}" / f"difficulty_{difficulty}"
    return sorted(d.glob("*.pt")) if d.exists() else []


def collect_train_files(root, years):
    """Gather all difficulty_0 files across given years and scenarios."""
    files = []
    for yr in years:
        for sc in SCENARIOS:
            files.extend(get_files(root, yr, sc, 0))
    return files


# ========================= TRAINING =========================

def train_model(root, train_years, device):
    """Train U-TAE on difficulty_0 data, select best model by val F1 (threshold=0.5)."""

    # Split train/val
    all_files = collect_train_files(root, train_years)
    random.shuffle(all_files)
    n_val = max(int(0.15 * len(all_files)), 1)
    val_files, train_files = all_files[:n_val], all_files[n_val:]
    print(f"    Train: {len(train_files)},  Val: {len(val_files)}")

    loader_kw = dict(num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
                     prefetch_factor=4 if NUM_WORKERS > 0 else None, persistent_workers=False)
    train_dl = DataLoader(FireDataset(train_files), batch_size=BATCH_SIZE,
                          shuffle=True, drop_last=True, **loader_kw)
    val_dl   = DataLoader(FireDataset(val_files), batch_size=BATCH_SIZE * 2,
                          shuffle=False, **loader_kw)

    # Model
    model = UTAE(
        input_dim=INPUT_DIM,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[32, 2],
        agg_mode="att_group", encoder_norm="group",
        n_head=16, d_model=256, d_k=4, pad_value=0,
    ).to(device)
    print(f"    Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Optimizer & loss
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    criterion = FocalLoss(alpha=CLASS_WEIGHT.to(device), gamma=FOCAL_GAMMA)

    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    best_f1, best_state, no_improve = 0.0, None, 0

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in tqdm(train_dl, desc=f"      Ep {epoch}/{EPOCHS}", leave=False):
            if batch is None:
                continue
            X, y = batch
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = criterion(model(X), y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss = criterion(model(X), y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            total_loss += loss.item()
            n_batches += 1
            del X, y, loss

        avg_loss = total_loss / max(n_batches, 1)

        # ---- Validate (fixed threshold=0.5) ----
        model.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for batch in val_dl:
                if batch is None:
                    continue
                X, y = batch
                X = X.to(device, non_blocking=True)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = model(X)
                else:
                    logits = model(X)
                probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy().ravel()
                all_probs.append(probs)
                all_targets.append(y.numpy().ravel())
                del X, y, logits

        if all_probs:
            yt = np.concatenate(all_targets)
            yp = np.concatenate(all_probs)
            val_f1 = f1_score(yt, (yp > THRESHOLD).astype(int), zero_division=0)
            val_ap = average_precision_score(yt, yp) if len(np.unique(yt)) > 1 else 0.0
            print(f"    Ep {epoch}: loss={avg_loss:.4f}  F1={val_f1:.4f}  AP={val_ap:.4f}")
            sched.step(val_f1)

            if val_f1 > best_f1:
                best_f1, no_improve = val_f1, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"    Early stop at epoch {epoch}")
                    break
            del yt, yp
        else:
            print(f"    Ep {epoch}: loss={avg_loss:.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to(device)


# ========================= PER-SAMPLE METRICS =========================

def ap_score(probs, targets):
    """AP for a single sample (handles all-zero / all-one edge cases)."""
    n_pos = targets.sum()
    if n_pos == 0:
        return 1.0 - float(probs.max())     # penalize any predicted fire
    if n_pos == len(targets):
        return 1.0
    return float(average_precision_score(targets, probs))


def fpr_score(probs, targets):
    """FPR for a single sample (fixed threshold=0.5)."""
    preds = (probs > THRESHOLD).astype(int)
    fp = ((preds == 1) & (targets == 0)).sum()
    tn = ((preds == 0) & (targets == 0)).sum()
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


# ========================= TESTING =========================

@torch.no_grad()
def test_year(model, root, year, device):
    """
    Evaluate one test year across all scenarios & difficulties.
    Returns:
        cat_scores:  {scenario: {difficulty: [score, ...]}}
        comb_scores: {difficulty: [ap_score, ...]}
    """
    model.eval()
    use_amp = device.type == "cuda"

    cat_scores  = {}
    comb_scores = defaultdict(list)

    for scenario in SCENARIOS:
        is_ap = scenario in AP_SCENARIOS
        scenario_scores = {}

        for diff in DIFFICULTIES:
            files = get_files(root, year, scenario, diff)
            if not files:
                continue

            loader = DataLoader(FireDataset(files), batch_size=TEST_BATCH, shuffle=False,
                                num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

            scores_diff, comb_diff = [], []
            for batch in loader:
                if batch is None:
                    continue
                X, y = batch
                X = X.to(device, non_blocking=True)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = model(X)
                else:
                    logits = model(X)
                probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
                y_np  = y.numpy()

                for b in range(probs.shape[0]):
                    p, t = probs[b].ravel(), y_np[b].ravel()
                    scores_diff.append(ap_score(p, t) if is_ap else fpr_score(p, t))
                    comb_diff.append(ap_score(p, t))
                del X, y, logits

            if scores_diff:
                scenario_scores[diff] = scores_diff
            comb_scores[diff].extend(comb_diff)

        cat_scores[scenario] = scenario_scores

    return cat_scores, dict(comb_scores)


# ========================= MAIN: LOOCV =========================

def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'=' * 70}")
    print(f"  U-TAE Fire Prediction — Leave-One-Year-Out CV")
    print(f"  Focal Loss (gamma={FOCAL_GAMMA}, alpha={CLASS_WEIGHT.tolist()})")
    print(f"  Fixed threshold = {THRESHOLD}")
    print(f"{'=' * 70}")
    print(f"Device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"Input: {INPUT_DIM} channels")

    all_results = {}

    for mask_type in MASK_TYPES:
        print(f"\n{'=' * 70}")
        print(f"  MASK TYPE: {mask_type.upper()}")
        print(f"{'=' * 70}")
        root = data_dir(mask_type)

        # Pool per-sample scores across all 4 folds
        cat_pool  = {sc: defaultdict(list) for sc in SCENARIOS}
        comb_pool = defaultdict(list)

        for test_yr in YEARS:
            train_yrs = [y for y in YEARS if y != test_yr]
            print(f"\n  --- Fold: test={test_yr}, train={train_yrs} ---")

            # Train
            model = train_model(root, train_yrs, device)
            torch.save(model.state_dict(), OUTPUT_ROOT / f"model_{mask_type}_test{test_yr}.pt")

            # Test
            print(f"  Testing year {test_yr}...")
            cat_scores, comb_scores = test_year(model, root, test_yr, device)

            # Pool scores
            for sc in SCENARIOS:
                for diff, scores in cat_scores.get(sc, {}).items():
                    cat_pool[sc][diff].extend(scores)
            for diff, scores in comb_scores.items():
                comb_pool[diff].extend(scores)

            # Print fold summary
            for sc in SCENARIOS:
                metric = "AP" if sc in AP_SCENARIOS else "FPR"
                vals = cat_scores.get(sc, {})
                if vals:
                    s = "  ".join(f"d{d}={np.mean(v):.4f}(n={len(v)})"
                                 for d, v in sorted(vals.items()))
                    print(f"    {sc} ({metric}): {s}")
            if comb_scores:
                s = "  ".join(f"d{d}={np.mean(v):.4f}(n={len(v)})"
                             for d, v in sorted(comb_scores.items()))
                print(f"    Combined (AP): {s}")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # ---- Aggregate: mean ± std across all 4 years ----
        summary = {"per_scenario": {}, "combined": {}}

        for sc in SCENARIOS:
            summary["per_scenario"][sc] = {}
            for diff in DIFFICULTIES:
                vals = cat_pool[sc].get(diff, [])
                if vals:
                    summary["per_scenario"][sc][str(diff)] = {
                        "mean": float(np.mean(vals)),
                        "std":  float(np.std(vals)),
                        "n":    len(vals),
                    }

        for diff in DIFFICULTIES:
            vals = comb_pool.get(diff, [])
            if vals:
                summary["combined"][str(diff)] = {
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals)),
                    "n":    len(vals),
                }

        all_results[mask_type] = summary

    # ========================= PRINT FINAL =========================
    print(f"\n{'=' * 70}")
    print("  FINAL RESULTS (all 4 folds pooled)")
    print(f"{'=' * 70}")

    for mt in MASK_TYPES:
        print(f"\n  ┌─── {mt.upper()} ───")
        for sc in SCENARIOS:
            metric = "AP" if sc in AP_SCENARIOS else "FPR"
            print(f"  │")
            print(f"  │ {sc} ({metric})")
            for diff in DIFFICULTIES:
                d = all_results[mt]["per_scenario"].get(sc, {}).get(str(diff))
                if d:
                    print(f"  │   diff={diff:>2}%:  {metric} = {d['mean']:.4f} ± {d['std']:.4f}  (n={d['n']})")
        print(f"  │")
        print(f"  │ ALL SCENARIOS (AP)")
        for diff in DIFFICULTIES:
            d = all_results[mt]["combined"].get(str(diff))
            if d:
                print(f"  │   diff={diff:>2}%:  AP = {d['mean']:.4f} ± {d['std']:.4f}  (n={d['n']})")
        print(f"  └───")

    # Save JSON
    out_path = OUTPUT_ROOT / "challenge_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()