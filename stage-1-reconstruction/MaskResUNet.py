"""
Residual U-Net for Wildfire Reconstruction
============================================

This script trains a Residual U-Net with bottleneck self-attention to
reconstruct masked (occluded) fire pixels from multi-channel satellite imagery.

Training:   Leave-One-Year-Out Cross-Validation (LOOCV) on mixed-challenge
            training data with focal loss computed only over masked pixels.
Testing:    Evaluates on blockwise and pixelwise masked challenge sets across
            all fire categories and difficulty levels.
Outputs:    1. Reconstructed datasets  -> data_reconstructed_{blockwise,pixelwise}
            2. Per-category metrics (DICE / FPR) and pooled DICE per difficulty

Architecture
------------
  Input:       42 channels = 40 context + 2 fire (visibility mask + fire value)
  Encoder:     4 levels, each with 2x ResBlock + strided-conv downsample
               Channel progression: 64 -> 128 -> 256 -> 512
  Bottleneck:  ResBlock -> Multi-Head Self-Attention -> ResBlock (4x4, 16 tokens)
  Decoder:     4 levels with skip connections, nearest-upsample + conv
  Output:      1-channel raw logits (binary fire probability)

  ResBlock:    GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Dropout2d -> Conv3x3
               with a 1x1 skip projection when channel dimensions differ.

Channel Layout (43 channels in raw data)
-----------------------------------------
  Channels  0-39: Environmental context features
  Channel     40: Visibility mask (1 = observed, 0 = masked/occluded)
  Channel     41: Detection flag (dropped before model input)
  Channel     42: Fire channel (binary ground truth)

Performance Optimizations
--------------------------
  - AMP (float16) mixed-precision training and inference
  - Asynchronous file saving via ThreadPoolExecutor
  - Batched inference across multiple files during testing
  - Persistent DataLoader workers with prefetching
  - cuDNN auto-tuner and TF32 enabled for Ampere GPUs (A100)

Usage
-----
    python resunet_fire_reconstruction.py

Requires: PyTorch >= 2.0, NumPy, tqdm
"""

import os
import glob
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =============================================================================
#  Configuration
# =============================================================================

# -- Paths --
DATA_ROOT = "/pub/cyang27/data"
TRAIN_DIR = os.path.join(DATA_ROOT, "data_mixed_challenge_train")
SAVE_DIR = os.path.join(DATA_ROOT, "resunet_results")

# -- Dataset constants --
MASK_TYPES = ["blockwise", "pixelwise"]
YEARS = [2018, 2019, 2020, 2021]
CATEGORIES = [
    "Fire_Continues",
    "Fire_Extinguished",
    "NewFire_NoHistory",
    "NoFire",
]
DICE_CATEGORIES = {"Fire_Continues", "Fire_Extinguished"}
FPR_CATEGORIES = {"NewFire_NoHistory", "NoFire"}
DIFFICULTIES = [10, 20, 30, 40, 50, 60, 70, 80]

# -- Channel indices --
FIRE_CHANNEL = 42       # Fire presence (binary ground truth)
DETECTION_CHANNEL = 41  # Detection flag (excluded from model input)
NUM_CONTEXT_CHANNELS = 40
NUM_FIRE_INPUT_CHANNELS = 2  # [visibility_mask, fire_observed]
MODEL_INPUT_CHANNELS = NUM_CONTEXT_CHANNELS + NUM_FIRE_INPUT_CHANNELS  # 42

# -- Model hyperparameters --
IMAGE_SIZE = 64
BASE_CHANNELS = 64
CHANNEL_MULTIPLIERS = [1, 2, 4, 8]  # -> [64, 128, 256, 512]
ATTN_HEADS = 8
DROPOUT = 0.1

# -- Training hyperparameters --
SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 50
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0

# -- DataLoader --
NUM_WORKERS = 8
PREFETCH_FACTOR = 4

# -- Inference / I/O --
TEST_BATCH_SIZE = 40
IO_WORKERS = 4


# =============================================================================
#  Utility Functions
# =============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Note: This does NOT set torch.backends.cudnn.deterministic = True, because
    run_loocv() intentionally enables cudnn.benchmark for speed on fixed-size
    inputs. This trades bitwise reproducibility for ~10-20% faster training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def source_dir(mask_type: str) -> str:
    """Return the challenge data directory for a given mask type."""
    return os.path.join(DATA_ROOT, f"data_challenge_{mask_type}")


def reconstructed_dir(mask_type: str) -> str:
    """Return the output directory for reconstructed data."""
    return os.path.join(DATA_ROOT, f"data_reconstructed_{mask_type}")


def drop_detection_channel(x: torch.Tensor) -> torch.Tensor:
    """Remove the detection channel (index 41) from 43-channel input.

    Args:
        x: Tensor of shape (43, H, W).

    Returns:
        Tensor of shape (42, H, W) with channel 41 removed.
        After removal: channels 0-39 = context, 40 = visibility mask, 41 = fire.
    """
    return torch.cat([x[:DETECTION_CHANNEL], x[DETECTION_CHANNEL + 1:]], dim=0)


# =============================================================================
#  Dataset
# =============================================================================

class FireReconstructionDataset(Dataset):
    """Dataset for fire reconstruction training.

    Each sample yields:
        context:         (40, H, W)  environmental context features
        fire_input:      (2, H, W)   [visibility_mask, fire_observed]
        target:          (H, W)      binary fire ground truth
        visibility_mask: (H, W)      1 = observed, 0 = masked
    """

    def __init__(self, file_paths: list[str]):
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int):
        # NOTE: weights_only=False is required because training data contains
        # non-tensor objects. Only use with trusted data sources.
        data = torch.load(self.file_paths[index], map_location="cpu",
                          weights_only=False)
        x = drop_detection_channel(data["x"].float())  # (42, H, W)

        context = x[:NUM_CONTEXT_CHANNELS]               # (40, H, W)
        fire_input = x[NUM_CONTEXT_CHANNELS:]             # (2, H, W)
        visibility_mask = x[NUM_CONTEXT_CHANNELS]         # (H, W)
        target = (data["y"].float() > 0).float()          # (H, W) — binary fire

        return context, fire_input, target, visibility_mask


def collect_train_files_by_year() -> dict[int, list[str]]:
    """Gather sorted .pt file paths grouped by year."""
    files_by_year = {}
    for year in YEARS:
        year_dir = os.path.join(TRAIN_DIR, str(year))
        if os.path.isdir(year_dir):
            files_by_year[year] = sorted(
                glob.glob(os.path.join(year_dir, "*.pt"))
            )
        else:
            files_by_year[year] = []
    return files_by_year


# =============================================================================
#  Model Components
# =============================================================================

class ResBlock(nn.Module):
    """Pre-activation residual block with spatial dropout.

    Structure: GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Dropout2d -> Conv3x3
    A 1x1 convolution is used for the skip connection when input and output
    channel dimensions differ; otherwise an identity mapping is used.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
        dropout:      Dropout probability (applied as Dropout2d to drop entire
                      feature maps, which is more effective for spatial data).
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(min(32, in_channels), in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class Downsample(nn.Module):
    """2x spatial downsample via stride-2 convolution.

    Using a learned convolution instead of pooling preserves more spatial
    information and avoids the information loss of max/average pooling.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """2x spatial upsample via nearest-neighbor interpolation + convolution.

    Nearest-neighbor interpolation avoids checkerboard artifacts that can
    occur with transposed convolutions.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class SelfAttention2D(nn.Module):
    """Multi-head self-attention on 2D spatial features.

    Applied at the bottleneck (4x4 spatial = 16 tokens) to capture global
    dependencies across the entire spatial extent. Uses pre-normalization
    with GroupNorm and a residual connection.

    The implementation uses PyTorch's scaled_dot_product_attention (SDPA),
    which automatically selects the most efficient kernel (FlashAttention,
    memory-efficient attention, or standard) based on hardware and input size.

    Args:
        channels:  Number of input/output channels.
        num_heads: Number of attention heads. channels must be divisible by num_heads.
    """

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            (B, C, H, W) with global self-attention applied.
        """
        B, C, H, W = x.shape
        head_dim = C // self.num_heads

        # Joint Q/K/V projection and reshape for multi-head attention
        qkv = self.qkv(self.norm(x))  # (B, 3*C, H, W)
        qkv = qkv.reshape(B, 3, self.num_heads, head_dim, H * W)
        q, k, v = qkv.unbind(dim=1)  # each: (B, heads, head_dim, N)
        q = q.permute(0, 1, 3, 2)    # (B, heads, N, head_dim)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        # Scaled dot-product attention (uses flash/memory-efficient kernels)
        out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, N, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        return x + self.proj(out)


# =============================================================================
#  Full Model
# =============================================================================

class ResUNet(nn.Module):
    """Residual U-Net with bottleneck self-attention for fire reconstruction.

    Spatial flow (64x64 input):
        Encoder:     64 -> 32 -> 16 -> 8   (4 levels, 2x ResBlock + Downsample)
        Bottleneck:  4x4                    (ResBlock + SelfAttention + ResBlock)
        Decoder:     8 -> 16 -> 32 -> 64    (Upsample + skip concat + 2x ResBlock)
        Output:      GroupNorm -> SiLU -> 1x1 Conv -> 1-channel logits

    Args:
        in_channels:         Total input channels (context + fire input).
        base_channels:       Number of channels at the first encoder level.
        channel_multipliers: Multipliers for each encoder level.
        attn_heads:          Number of attention heads in the bottleneck.
        dropout:             Dropout probability for ResBlocks.
    """

    def __init__(
        self,
        in_channels: int = MODEL_INPUT_CHANNELS,
        base_channels: int = BASE_CHANNELS,
        channel_multipliers: list[int] = CHANNEL_MULTIPLIERS,
        attn_heads: int = ATTN_HEADS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        channels = [base_channels * m for m in channel_multipliers]

        # Stem: project concatenated input to feature space
        self.stem = nn.Conv2d(in_channels, channels[0], 3, padding=1, bias=False)

        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        prev_ch = channels[0]
        for ch in channels:
            self.encoder_blocks.append(nn.Sequential(
                ResBlock(prev_ch, ch, dropout),
                ResBlock(ch, ch, dropout),
            ))
            self.downsample_layers.append(Downsample(ch))
            prev_ch = ch

        # Bottleneck with self-attention at 4x4 spatial resolution
        self.bottleneck = nn.Sequential(
            ResBlock(prev_ch, prev_ch, dropout),
            SelfAttention2D(prev_ch, attn_heads),
            ResBlock(prev_ch, prev_ch, dropout),
        )

        # Decoder path (mirrors encoder)
        self.upsample_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for ch in reversed(channels):
            self.upsample_layers.append(Upsample(prev_ch))
            # After concatenating skip connection, input channels double
            self.decoder_blocks.append(nn.Sequential(
                ResBlock(prev_ch + ch, ch, dropout),
                ResBlock(ch, ch, dropout),
            ))
            prev_ch = ch

        # Output head
        self.head = nn.Sequential(
            nn.GroupNorm(min(32, prev_ch), prev_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(prev_ch, 1, 1),
        )

    def forward(self, context: torch.Tensor,
                fire_input: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict fire logits for all pixels.

        Args:
            context:    (B, 40, H, W) environmental context features.
            fire_input: (B, 2, H, W)  [visibility_mask, fire_observed].

        Returns:
            logits: (B, 1, H, W) raw logits before sigmoid.
        """
        x = self.stem(torch.cat([context, fire_input], dim=1))

        # Encoder: save features before each downsample for skip connections
        skips = []
        for encoder, downsample in zip(self.encoder_blocks, self.downsample_layers):
            x = encoder(x)
            skips.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder: upsample -> concat skip -> refine
        for upsample, decoder, skip in zip(
            self.upsample_layers, self.decoder_blocks, reversed(skips)
        ):
            x = torch.cat([upsample(x), skip], dim=1)
            x = decoder(x)

        return self.head(x)

    @torch.no_grad()
    def predict_merged(self, context: torch.Tensor,
                       fire_input: torch.Tensor) -> torch.Tensor:
        """Predict fire probabilities, keeping observed pixels intact.

        For observed pixels (visibility_mask=1), the original fire value is
        preserved. For masked pixels (visibility_mask=0), the model's sigmoid
        prediction is used.

        Args:
            context:    (B, 40, H, W)
            fire_input: (B, 2, H, W)

        Returns:
            merged: (B, H, W) — merged fire prediction map.
        """
        with torch.amp.autocast("cuda"):
            logits = self.forward(context, fire_input)
        probs = torch.sigmoid(logits.float()).squeeze(1)  # (B, H, W)

        visibility_mask = fire_input[:, 0]   # (B, H, W)
        observed_fire = fire_input[:, 1]     # (B, H, W)

        return visibility_mask * observed_fire + (1 - visibility_mask) * probs


# =============================================================================
#  Loss Function and Metrics
# =============================================================================

def focal_loss_masked(logits: torch.Tensor, target: torch.Tensor,
                      visibility_mask: torch.Tensor,
                      alpha: float = FOCAL_ALPHA,
                      gamma: float = FOCAL_GAMMA) -> torch.Tensor:
    """Focal loss computed only over masked (occluded) pixels.

    Focal loss down-weights well-classified examples, focusing training on
    hard negatives / positives. Here it is applied exclusively to occluded
    pixels where the model must predict unseen fire activity.

    Args:
        logits:          (B, H, W) raw model output (before sigmoid).
        target:          (B, H, W) binary ground truth.
        visibility_mask: (B, H, W) 1 = observed, 0 = masked.
        alpha:           Weighting factor for the positive class.
        gamma:           Focusing parameter (higher = more focus on hard examples).

    Returns:
        Scalar focal loss averaged over masked pixels.
    """
    occlusion_mask = 1.0 - visibility_mask

    logits_f = logits.float()
    target_f = target.float()

    prob = torch.sigmoid(logits_f)
    pt = target_f * prob + (1 - target_f) * (1 - prob)
    alpha_t = target_f * alpha + (1 - target_f) * (1 - alpha)
    bce = F.binary_cross_entropy_with_logits(logits_f, target_f, reduction="none")

    loss = alpha_t * (1 - pt) ** gamma * bce * occlusion_mask
    return loss.sum() / occlusion_mask.sum().clamp(min=1)


def dice_score_masked(pred_binary: torch.Tensor, target: torch.Tensor,
                      visibility_mask: torch.Tensor) -> float:
    """Compute Dice coefficient over masked (occluded) pixels only.

    Returns 1.0 when both prediction and target are empty in the masked region
    (i.e., the model correctly predicted no fire where there is none).
    """
    occlusion_mask = 1.0 - visibility_mask
    pred_masked = pred_binary * occlusion_mask
    target_masked = target * occlusion_mask

    intersection = (pred_masked * target_masked).sum()
    denominator = pred_masked.sum() + target_masked.sum()

    if denominator > 0:
        return (2.0 * intersection / (denominator + 1e-6)).item()
    return 1.0


def false_positive_rate_masked(pred_binary: torch.Tensor, target: torch.Tensor,
                               visibility_mask: torch.Tensor) -> float:
    """Compute False Positive Rate over masked (occluded) pixels only.

    FPR = FP / (FP + TN). Used for NoFire and NewFire_NoHistory categories
    where the primary concern is avoiding false alarms.
    """
    occlusion_mask = (1.0 - visibility_mask).bool()
    pred_occ = pred_binary[occlusion_mask]
    target_occ = target[occlusion_mask]

    false_positives = ((pred_occ == 1) & (target_occ == 0)).float().sum()
    true_negatives = ((pred_occ == 0) & (target_occ == 0)).float().sum()
    denominator = false_positives + true_negatives

    if denominator > 0:
        return (false_positives / (denominator + 1e-6)).item()
    return 0.0


# =============================================================================
#  Training and Validation
# =============================================================================

def train_one_epoch(model: nn.Module, dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler,
                    device: torch.device) -> float:
    """Train for one epoch with AMP mixed precision.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_samples = 0

    progress = tqdm(dataloader, desc="    Train", leave=False)
    for context, fire_input, target, vis_mask in progress:
        context = context.to(device, non_blocking=True)
        fire_input = fire_input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        vis_mask = vis_mask.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = model(context, fire_input).squeeze(1)

        loss = focal_loss_masked(logits, target, vis_mask)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = context.shape[0]
        total_loss += loss.item() * batch_size
        num_samples += batch_size
        progress.set_postfix(loss=f"{total_loss / num_samples:.4f}")

    return total_loss / max(num_samples, 1)


@torch.no_grad()
def validate(model: nn.Module, dataloader: DataLoader,
             device: torch.device) -> float:
    """Validate and return mean Dice score over masked regions.

    Returns:
        Mean Dice score across all validation samples.
    """
    model.eval()
    dice_scores = []

    for context, fire_input, target, vis_mask in tqdm(dataloader, desc="    Val  ", leave=False):
        context = context.to(device, non_blocking=True)
        fire_input = fire_input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        vis_mask = vis_mask.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = model(context, fire_input).squeeze(1)

        pred_binary = (torch.sigmoid(logits.float()) >= 0.5).float()
        for b in range(context.shape[0]):
            dice_scores.append(
                dice_score_masked(pred_binary[b], target[b], vis_mask[b])
            )

    return np.mean(dice_scores) if dice_scores else 0.0


# =============================================================================
#  Testing and Reconstruction
# =============================================================================

def _save_file_worker(args: tuple) -> None:
    """Worker function for ThreadPoolExecutor-based async file saving."""
    data, path = args
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path)


def _process_test_batch(
    model: nn.Module,
    device: torch.device,
    contexts: list[torch.Tensor],
    fire_inputs: list[torch.Tensor],
    metadata: list[dict],
    per_category_metrics: dict,
    pooled_dice: dict,
    category: str,
    difficulty: int,
    use_dice: bool,
    pending_files: dict,
) -> None:
    """Run model inference on a batch and compute metrics.

    Updates per_category_metrics, pooled_dice, and pending_files in place.
    """
    context_batch = torch.stack(contexts).to(device, non_blocking=True)
    fire_batch = torch.stack(fire_inputs).to(device, non_blocking=True)

    merged = model.predict_merged(context_batch, fire_batch).cpu()
    pred_binary = (merged >= 0.5).float()

    for i, meta in enumerate(metadata):
        target = meta["target"]
        vis_mask = meta["visibility_mask"]
        fname = meta["fname"]
        t = meta["t"]

        # Per-category metric (Dice for fire categories, FPR for non-fire)
        if use_dice:
            score = dice_score_masked(pred_binary[i], target, vis_mask)
        else:
            score = false_positive_rate_masked(pred_binary[i], target, vis_mask)
        per_category_metrics[(category, difficulty)].append(score)

        # Pooled Dice across all categories
        # NOTE: For NoFire samples, Dice is trivially 1.0 when both prediction
        # and target are empty. This inflates pooled scores. Interpret with care.
        pooled_dice[difficulty].append(
            dice_score_masked(pred_binary[i], target, vis_mask)
        )

        # Store merged output for file reconstruction
        if fname in pending_files:
            pending_files[fname]["outs"][t] = merged[i]


@torch.no_grad()
def test_and_reconstruct(
    model: nn.Module,
    test_year: int,
    mask_type: str,
    device: torch.device,
    save_pool: ThreadPoolExecutor,
    save_futures: list,
) -> tuple[dict, dict]:
    """Run batched inference on the test set and reconstruct masked fire pixels.

    Accumulates time steps from multiple files into larger GPU batches for
    efficiency. Reconstructed files are saved asynchronously in background
    threads.

    Args:
        model:        Trained ResUNet model.
        test_year:    Year to evaluate on.
        mask_type:    "blockwise" or "pixelwise".
        device:       CUDA device.
        save_pool:    ThreadPoolExecutor for async file saving.
        save_futures: List to collect futures for error checking.

    Returns:
        per_category_metrics: dict[(category, difficulty)] -> list of metric values
        pooled_dice:          dict[difficulty] -> list of Dice values (all categories)
    """
    model.eval()
    root = source_dir(mask_type)
    dst = reconstructed_dir(mask_type)

    per_category_metrics = defaultdict(list)
    pooled_dice = defaultdict(list)

    for category in CATEGORIES:
        clean_dir = os.path.join(root, f"{test_year}_{category}", "difficulty_0")
        if not os.path.isdir(clean_dir):
            continue

        filenames = sorted(f for f in os.listdir(clean_dir) if f.endswith(".pt"))
        use_dice = category in DICE_CATEGORIES

        # Copy clean (difficulty_0) reference files to output directory
        clean_dst = os.path.join(dst, f"{test_year}_{category}", "difficulty_0")
        os.makedirs(clean_dst, exist_ok=True)
        for fname in filenames:
            src_path = os.path.join(clean_dir, fname)
            dst_path = os.path.join(clean_dst, fname)
            if not os.path.exists(dst_path):
                save_futures.append(
                    save_pool.submit(shutil.copy2, src_path, dst_path)
                )

        # Preload only the fire channel from clean references to save memory.
        # Storing as uint8 uses ~43x less memory than keeping full float32 data.
        clean_fire_cache = {}
        for fname in tqdm(filenames, desc=f"    Load clean {category}", leave=False):
            data = torch.load(
                os.path.join(clean_dir, fname), map_location="cpu",
                weights_only=False,
            )
            clean_fire_cache[fname] = (
                (data["x"][:, FIRE_CHANNEL] > 0).to(torch.uint8)  # (T, H, W)
            )

        # Evaluate each difficulty level
        for difficulty in tqdm(DIFFICULTIES, desc=f"    {mask_type}/{category}", leave=False):
            diff_dir = os.path.join(
                root, f"{test_year}_{category}", f"difficulty_{difficulty}"
            )
            if not os.path.isdir(diff_dir):
                continue

            # Track partial reconstructions: fname -> {deg_path, T, outs: {t: tensor}}
            pending_files = {}

            # Accumulate samples for batched GPU inference
            batch_contexts = []
            batch_fire_inputs = []
            batch_metadata = []

            for fname in filenames:
                degraded_path = os.path.join(diff_dir, fname)
                if not os.path.exists(degraded_path):
                    continue

                degraded_data = torch.load(
                    degraded_path, map_location="cpu", weights_only=False
                )
                degraded_x = degraded_data["x"]
                clean_fire = clean_fire_cache[fname]
                num_timesteps = degraded_x.shape[0]

                pending_files[fname] = {
                    "deg_path": degraded_path,
                    "T": num_timesteps,
                    "outs": {},
                }

                for t in range(num_timesteps):
                    xt = drop_detection_channel(degraded_x[t].float())
                    target_fire = clean_fire[t].float()

                    batch_contexts.append(xt[:NUM_CONTEXT_CHANNELS])
                    batch_fire_inputs.append(xt[NUM_CONTEXT_CHANNELS:])
                    batch_metadata.append({
                        "fname": fname,
                        "t": t,
                        "visibility_mask": xt[NUM_CONTEXT_CHANNELS],
                        "target": target_fire,
                    })

                    # Process batch when full
                    if len(batch_contexts) >= TEST_BATCH_SIZE:
                        _process_test_batch(
                            model, device, batch_contexts, batch_fire_inputs,
                            batch_metadata, per_category_metrics, pooled_dice,
                            category, difficulty, use_dice, pending_files,
                        )
                        batch_contexts, batch_fire_inputs, batch_metadata = [], [], []

            # Flush remaining samples
            if batch_contexts:
                _process_test_batch(
                    model, device, batch_contexts, batch_fire_inputs,
                    batch_metadata, per_category_metrics, pooled_dice,
                    category, difficulty, use_dice, pending_files,
                )

            # Save all fully reconstructed files for this (category, difficulty)
            save_prefix = os.path.join(
                dst, f"{test_year}_{category}", f"difficulty_{difficulty}"
            )
            for fname, pdata in pending_files.items():
                if len(pdata["outs"]) != pdata["T"]:
                    continue  # Skip incomplete files

                # Reload full data only at save time (not kept in memory)
                deg_data = torch.load(
                    pdata["deg_path"], map_location="cpu", weights_only=False
                )
                recon_x = deg_data["x"].clone()
                for t_idx, merged_output in pdata["outs"].items():
                    recon_x[t_idx, FIRE_CHANNEL] = (merged_output >= 0.5).half()

                save_data = {**deg_data, "x": recon_x.half()}
                os.makedirs(save_prefix, exist_ok=True)
                save_futures.append(
                    save_pool.submit(
                        _save_file_worker,
                        (save_data, os.path.join(save_prefix, fname)),
                    )
                )

    return per_category_metrics, pooled_dice


# =============================================================================
#  Leave-One-Year-Out Cross-Validation (LOOCV)
# =============================================================================

def run_loocv() -> None:
    """Main training and evaluation loop using LOOCV.

    For each fold:
        1. Train on 3 years, validate on the held-out year
        2. Evaluate on both blockwise and pixelwise challenge sets
        3. Reconstruct masked fire pixels and save to disk
        4. Compute per-category and pooled metrics
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Enable performance optimizations for Ampere GPUs (A100)
    # NOTE: cudnn.benchmark=True + TF32 trade bitwise reproducibility for
    # ~10-20% faster training. Input sizes are fixed (64x64), so the
    # auto-tuner overhead is amortized after the first iteration.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Print model size
    tmp_model = ResUNet()
    num_params = sum(p.numel() for p in tmp_model.parameters()) / 1e6
    print(f"ResUNet parameters: {num_params:.2f}M")
    del tmp_model

    files_by_year = collect_train_files_by_year()
    for year, files in files_by_year.items():
        print(f"  Year {year}: {len(files)} samples")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Accumulate metrics across all folds
    all_category_metrics = {mt: defaultdict(list) for mt in MASK_TYPES}
    all_pooled_dice = {mt: defaultdict(list) for mt in MASK_TYPES}

    for test_year in YEARS:
        print(f"\n{'=' * 60}")
        print(f"  FOLD: test_year = {test_year}")
        print(f"{'=' * 60}")

        # Split: train on all other years, validate on test_year
        train_files = [
            f for yr in YEARS if yr != test_year
            for f in files_by_year[yr]
        ]
        val_files = files_by_year[test_year]

        if not val_files:
            print("  No validation data, skipping fold.")
            continue
        print(f"  Train: {len(train_files)} | Val: {len(val_files)}")

        # DataLoaders with persistent workers and prefetching
        loader_kwargs = dict(
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=PREFETCH_FACTOR,
        )
        train_loader = DataLoader(
            FireReconstructionDataset(train_files), shuffle=True, **loader_kwargs
        )
        val_loader = DataLoader(
            FireReconstructionDataset(val_files), shuffle=False, **loader_kwargs
        )

        # Initialize model, optimizer, scheduler, and AMP scaler
        model = ResUNet().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )
        scaler = torch.amp.GradScaler("cuda")

        best_dice = 0.0
        best_state = None

        # ---- Training loop ----
        print(f"\n  Training ({EPOCHS} epochs)...")
        for epoch in range(EPOCHS):
            loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
            scheduler.step()

            # Validate every 5 epochs and on the final epoch
            if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
                val_dice = validate(model, val_loader, device)
                is_best = val_dice > best_dice
                marker = " *" if is_best else ""
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  Ep {epoch + 1:>3}/{EPOCHS}  loss={loss:.4f}  "
                    f"val_DICE={val_dice:.4f}  lr={lr:.2e}{marker}"
                )
                if is_best:
                    best_dice = val_dice
                    best_state = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
            else:
                print(f"  Ep {epoch + 1:>3}/{EPOCHS}  loss={loss:.4f}")

        # Load best checkpoint
        model.load_state_dict(best_state)
        model.to(device)
        checkpoint_path = os.path.join(SAVE_DIR, f"model_test{test_year}.pt")
        torch.save(best_state, checkpoint_path)
        print(f"  Best val DICE: {best_dice:.4f}")

        # ---- Test & reconstruct both mask types ----
        with ThreadPoolExecutor(max_workers=IO_WORKERS) as save_pool:
            save_futures = []

            for mask_type in MASK_TYPES:
                print(f"\n  Testing {mask_type} {test_year}...")
                fold_metrics, fold_dice = test_and_reconstruct(
                    model, test_year, mask_type, device, save_pool, save_futures
                )

                for key, values in fold_metrics.items():
                    all_category_metrics[mask_type][key].extend(values)
                for key, values in fold_dice.items():
                    all_pooled_dice[mask_type][key].extend(values)

            # Wait for all async saves and report errors
            num_failed = 0
            for future in save_futures:
                try:
                    future.result()
                except Exception as e:
                    num_failed += 1
                    print(f"  [SAVE ERROR] {e}")

            total_saves = len(save_futures)
            if num_failed:
                print(f"  Saved {total_saves - num_failed}/{total_saves} "
                      f"files ({num_failed} failed)")
            else:
                print(f"  All {total_saves} files saved OK")

    # =========================================================================
    #  Print and Save Results
    # =========================================================================
    _print_and_save_results(all_category_metrics, all_pooled_dice)


def _print_and_save_results(all_category_metrics: dict,
                            all_pooled_dice: dict) -> None:
    """Print aggregated cross-validation results and save to disk."""
    print(f"\n{'=' * 70}")
    print("  FINAL RESULTS — ResUNet (all 4 test folds pooled)")
    print(f"{'=' * 70}")

    summary = {}
    for mask_type in MASK_TYPES:
        print(f"\n  --- {mask_type.upper()} ---")
        summary[mask_type] = {}

        # Per-category metrics
        for category in CATEGORIES:
            metric_name = "DICE" if category in DICE_CATEGORIES else "FPR"
            print(f"\n  {category} ({metric_name})")
            summary[mask_type][category] = {}

            for difficulty in DIFFICULTIES:
                values = all_category_metrics[mask_type].get(
                    (category, difficulty), []
                )
                if not values:
                    continue
                mean_val = np.mean(values)
                std_val = np.std(values)
                summary[mask_type][category][difficulty] = {
                    "mean": mean_val, "std": std_val, "n": len(values),
                }
                print(
                    f"    diff={difficulty:>2}%: {metric_name}="
                    f"{mean_val:.4f} +/- {std_val:.4f} (n={len(values)})"
                )

        # Pooled Dice across all categories
        print(f"\n  ALL SCENARIOS (DICE)")
        summary[mask_type]["ALL"] = {}
        for difficulty in DIFFICULTIES:
            values = all_pooled_dice[mask_type].get(difficulty, [])
            if not values:
                continue
            mean_val = np.mean(values)
            std_val = np.std(values)
            summary[mask_type]["ALL"][difficulty] = {
                "mean": mean_val, "std": std_val, "n": len(values),
            }
            print(
                f"    diff={difficulty:>2}%: DICE="
                f"{mean_val:.4f} +/- {std_val:.4f} (n={len(values)})"
            )

    print(f"\n{'=' * 70}")

    # Save to disk
    results_path = os.path.join(SAVE_DIR, "loocv_results.pt")
    torch.save(
        {
            "summary": summary,
            "raw_per_category": {
                mt: {str(k): v for k, v in d.items()}
                for mt, d in all_category_metrics.items()
            },
            "raw_pooled_dice": {
                mt: {str(k): v for k, v in d.items()}
                for mt, d in all_pooled_dice.items()
            },
        },
        results_path,
    )
    print(f"Results   -> {results_path}")
    print(f"Models    -> {SAVE_DIR}/model_test{{year}}.pt")
    print(f"Reconstr. -> {reconstructed_dir('blockwise')}")
    print(f"             {reconstructed_dir('pixelwise')}")


if __name__ == "__main__":
    run_loocv()