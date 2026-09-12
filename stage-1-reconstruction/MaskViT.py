"""
Cross-Attention Vision Transformer for Wildfire Reconstruction
===============================================================

This script trains a patch-based Vision Transformer with cross-attention to
reconstruct masked (occluded) fire pixels from satellite imagery. The model
takes multi-channel environmental context (40 channels) and a 2-channel fire
input (visibility mask + observed fire) and predicts fire presence in the
masked regions.

Training:   Leave-One-Year-Out Cross-Validation (LOOCV) on mixed-challenge
            training data with focal loss computed only over masked pixels.
Testing:    Evaluates on blockwise and pixelwise masked challenge sets across
            all fire categories and difficulty levels.
Outputs:    1. Reconstructed datasets  -> data_reconstructed_{blockwise,pixelwise}
            2. Per-category metrics (DICE / FPR) and pooled DICE per difficulty

Architecture
------------
- PatchEmbed: Conv2d-based patch embedding with LayerNorm
- CrossAttentionBlock: Self-attention on fire tokens, cross-attention to
  context tokens, followed by a feed-forward network
- Unpatchify head: Linear projection back to pixel space

Performance Optimizations
-------------------------
- AMP (float16) mixed-precision training and inference
- Asynchronous file saving via ThreadPoolExecutor
- Batched inference across multiple files during testing
- Persistent DataLoader workers with prefetching
- Non-blocking CUDA memory transfers

Channel Layout (43 channels total)
-----------------------------------
- Channels  0-39: Environmental context features
- Channel     40: Visibility mask (1 = observed, 0 = masked/occluded)
- Channel     41: Detection flag (dropped before model input)
- Channel     42: Fire channel (binary ground truth)

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
SAVE_DIR = os.path.join(DATA_ROOT, "crossattn_vit_results")

# -- Dataset constants --
MASK_TYPES = ["blockwise", "pixelwise"]
YEARS = [2018, 2019, 2020, 2021]
CATEGORIES = ["Fire_Continues", "Fire_Extinguished", "NewFire_NoHistory", "NoFire"]
DICE_CATEGORIES = ["Fire_Continues", "Fire_Extinguished"]
FPR_CATEGORIES = ["NewFire_NoHistory", "NoFire"]
DIFFICULTIES = [10, 20, 30, 40, 50, 60, 70, 80]

# -- Channel indices --
FIRE_CHANNEL = 42       # Fire presence (binary ground truth)
DETECTION_CHANNEL = 41  # Detection flag (excluded from model input)
NUM_CONTEXT_CHANNELS = 40
NUM_FIRE_INPUT_CHANNELS = 2  # [visibility_mask, fire_observed]

# -- Model hyperparameters --
SEED = 42
IMAGE_SIZE = 64
PATCH_SIZE = 4
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
MLP_RATIO = 4
DROPOUT = 0.1

# -- Training hyperparameters --
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 50
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0

# -- DataLoader --
NUM_WORKERS = 8
PREFETCH_FACTOR = 4

# -- Test / I/O --
TEST_BATCH_SIZE = 40  # Accumulate days from multiple files into GPU batches
IO_WORKERS = 4        # Threads for asynchronous file saving


# =============================================================================
#  Utility Functions
# =============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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

        context = x[:NUM_CONTEXT_CHANNELS]              # (40, H, W)
        fire_input = x[NUM_CONTEXT_CHANNELS:]            # (2, H, W)
        visibility_mask = x[NUM_CONTEXT_CHANNELS]        # (H, W) — channel 40
        target = (data["y"].float() > 0).float()         # (H, W) — binary fire

        return context, fire_input, target, visibility_mask


def collect_train_files_by_year() -> dict[int, list[str]]:
    """Gather sorted .pt file paths grouped by year."""
    files_by_year = {}
    for year in YEARS:
        year_dir = os.path.join(TRAIN_DIR, str(year))
        if os.path.isdir(year_dir):
            files_by_year[year] = sorted(glob.glob(os.path.join(year_dir, "*.pt")))
        else:
            files_by_year[year] = []
    return files_by_year


# =============================================================================
#  Model Components
# =============================================================================

class PatchEmbedding(nn.Module):
    """Convert image patches into embedding tokens via Conv2d.

    Args:
        in_channels:  Number of input channels.
        embed_dim:    Dimension of output embeddings.
        patch_size:   Spatial size of each patch.
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            tokens:     (B, num_patches, embed_dim)
            height_patches: Number of patches along height
            width_patches:  Number of patches along width
        """
        x = self.projection(x)  # (B, embed_dim, Hp, Wp)
        B, D, Hp, Wp = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, Hp*Wp, embed_dim)
        return self.norm(tokens), Hp, Wp


class CrossAttentionBlock(nn.Module):
    """Transformer block with self-attention, cross-attention, and FFN.

    The fire tokens first attend to themselves (self-attention), then attend
    to the context tokens (cross-attention), and finally pass through a
    feed-forward network.

    Args:
        embed_dim:  Token embedding dimension.
        num_heads:  Number of attention heads.
        mlp_ratio:  FFN hidden dimension multiplier.
        dropout:    Dropout probability.
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()

        # Self-attention on fire tokens
        self.norm_self = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )

        # Cross-attention: fire tokens attend to context tokens
        self.norm_fire_cross = nn.LayerNorm(embed_dim)
        self.norm_context = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )

        # Feed-forward network
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, fire_tokens: torch.Tensor,
                context_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fire_tokens:    (B, N, D) — tokens to reconstruct
            context_tokens: (B, N, D) — conditioning tokens

        Returns:
            Updated fire tokens: (B, N, D)
        """
        # Self-attention
        normed = self.norm_self(fire_tokens)
        fire_tokens = fire_tokens + self.self_attn(
            normed, normed, normed, need_weights=False
        )[0]

        # Cross-attention
        normed_fire = self.norm_fire_cross(fire_tokens)
        normed_ctx = self.norm_context(context_tokens)
        fire_tokens = fire_tokens + self.cross_attn(
            normed_fire, normed_ctx, normed_ctx, need_weights=False
        )[0]

        # Feed-forward
        fire_tokens = fire_tokens + self.ffn(self.norm_ffn(fire_tokens))

        return fire_tokens


# =============================================================================
#  Full Model
# =============================================================================

class CrossAttentionViT(nn.Module):
    """Cross-Attention Vision Transformer for fire pixel reconstruction.

    Architecture overview:
        1. Patch-embed context channels (40ch) and fire input (2ch) separately
        2. Add learnable positional embeddings
        3. Process through N CrossAttentionBlocks
        4. Project back to pixel space via linear head + unpatchify

    Args:
        image_size:  Spatial resolution of input images.
        patch_size:  Size of each patch.
        embed_dim:   Transformer embedding dimension.
        num_heads:   Number of attention heads per block.
        num_layers:  Number of transformer blocks.
        mlp_ratio:   FFN hidden dimension multiplier.
        dropout:     Dropout probability.
    """

    def __init__(
        self,
        image_size: int = IMAGE_SIZE,
        patch_size: int = PATCH_SIZE,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        mlp_ratio: int = MLP_RATIO,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2

        # Separate patch embeddings for context and fire input
        self.context_embed = PatchEmbedding(NUM_CONTEXT_CHANNELS, embed_dim, patch_size)
        self.fire_embed = PatchEmbedding(NUM_FIRE_INPUT_CHANNELS, embed_dim, patch_size)

        # Learnable positional embeddings shared by both streams
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, embed_dim) * 0.02
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)
        self.prediction_head = nn.Linear(embed_dim, patch_size * patch_size)

    def forward(self, context: torch.Tensor,
                fire_input: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict fire logits for all pixels.

        Args:
            context:    (B, 40, H, W) environmental context features.
            fire_input: (B, 2, H, W)  [visibility_mask, fire_observed].

        Returns:
            logits: (B, 1, H, W) raw logits for fire prediction.
        """
        batch_size = context.shape[0]

        # Patch embedding
        ctx_tokens, Hp, Wp = self.context_embed(context)   # (B, N, D)
        fire_tokens, _, _ = self.fire_embed(fire_input)     # (B, N, D)

        # Add positional embeddings
        pos = self.pos_embed[:, :ctx_tokens.shape[1], :]
        ctx_tokens = ctx_tokens + pos
        fire_tokens = fire_tokens + pos

        # Transformer blocks
        for block in self.blocks:
            fire_tokens = block(fire_tokens, ctx_tokens)

        # Project to pixel space and unpatchify
        out = self.prediction_head(self.final_norm(fire_tokens))  # (B, N, ps*ps)
        out = out.transpose(1, 2)  # (B, ps*ps, N)
        ps = self.patch_size
        out = out.reshape(batch_size, 1, ps, ps, Hp, Wp)
        out = out.permute(0, 1, 4, 2, 5, 3)  # (B, 1, Hp, ps, Wp, ps)
        logits = out.reshape(batch_size, 1, Hp * ps, Wp * ps)

        return logits

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
            logits = self.forward(context, fire_input).squeeze(1)  # (B, H, W)
        predicted_probs = torch.sigmoid(logits.float())

        visibility_mask = fire_input[:, 0]   # (B, H, W)
        observed_fire = fire_input[:, 1]     # (B, H, W)

        # Merge: keep observed values where visible, use prediction elsewhere
        merged = visibility_mask * observed_fire + (1 - visibility_mask) * predicted_probs
        return merged


# =============================================================================
#  Loss Functions and Metrics
# =============================================================================

def focal_loss_masked(logits: torch.Tensor, target: torch.Tensor,
                      visibility_mask: torch.Tensor,
                      alpha: float = FOCAL_ALPHA,
                      gamma: float = FOCAL_GAMMA) -> torch.Tensor:
    """Focal loss computed only over masked (occluded) pixels.

    Args:
        logits:          (B, H, W) raw model output (before sigmoid).
        target:          (B, H, W) binary ground truth.
        visibility_mask: (B, H, W) 1 = observed, 0 = masked.
        alpha:           Weighting factor for positive class.
        gamma:           Focusing parameter.

    Returns:
        Scalar focal loss averaged over masked pixels.
    """
    occlusion_mask = 1.0 - visibility_mask  # 1 where masked

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
    """Compute Dice score over masked (occluded) pixels only.

    Returns 1.0 when both prediction and target are empty in masked region.
    """
    occlusion_mask = 1.0 - visibility_mask
    pred_masked = pred_binary * occlusion_mask
    target_masked = target * occlusion_mask

    intersection = (pred_masked * target_masked).sum()
    denominator = pred_masked.sum() + target_masked.sum()

    if denominator > 0:
        return (2.0 * intersection / (denominator + 1e-6)).item()
    return 1.0  # Both empty in masked region


def false_positive_rate_masked(pred_binary: torch.Tensor, target: torch.Tensor,
                               visibility_mask: torch.Tensor) -> float:
    """Compute False Positive Rate over masked (occluded) pixels only."""
    occlusion_mask = (1 - visibility_mask).bool()
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
#  Asynchronous File Saving
# =============================================================================

def _save_file_worker(args: tuple) -> None:
    """Worker function for ThreadPoolExecutor-based async file saving."""
    data, path = args
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path)


# =============================================================================
#  Testing and Reconstruction
# =============================================================================

@torch.no_grad()
def test_and_reconstruct(
    model: nn.Module,
    test_year: int,
    mask_type: str,
    device: torch.device,
    save_pool: ThreadPoolExecutor,
    save_futures: list,
) -> tuple[dict, dict]:
    """Run batched inference on test set and reconstruct masked fire pixels.

    Accumulates time steps from multiple files into larger GPU batches for
    efficiency. Reconstructed files are saved asynchronously.

    Args:
        model:        Trained CrossAttentionViT model.
        test_year:    Year to evaluate on.
        mask_type:    "blockwise" or "pixelwise".
        device:       CUDA device.
        save_pool:    ThreadPoolExecutor for async file saving.
        save_futures: List to collect futures for error checking.

    Returns:
        per_category_metrics: dict[(category, difficulty)] -> list of metric values
        pooled_dice:          dict[difficulty] -> list of Dice values across all categories
    """
    model.eval()
    root = source_dir(mask_type)
    dst = reconstructed_dir(mask_type)

    per_category_metrics = defaultdict(list)
    pooled_dice = defaultdict(list)  # NOTE: Includes NoFire samples where Dice is trivially 1.0

    for category in CATEGORIES:
        clean_dir = os.path.join(root, f"{test_year}_{category}", "difficulty_0")
        if not os.path.isdir(clean_dir):
            continue

        filenames = sorted(f for f in os.listdir(clean_dir) if f.endswith(".pt"))
        use_dice = category in DICE_CATEGORIES

        # Copy clean (difficulty_0) files to output directory
        clean_dst = os.path.join(dst, f"{test_year}_{category}", "difficulty_0")
        os.makedirs(clean_dst, exist_ok=True)
        for fname in filenames:
            src_path = os.path.join(clean_dir, fname)
            dst_path = os.path.join(clean_dst, fname)
            if not os.path.exists(dst_path):
                save_futures.append(
                    save_pool.submit(shutil.copy2, src_path, dst_path)
                )

        # Preload only fire channel from clean references to save memory
        # Full 43-channel data would use ~43x more RAM
        clean_fire_cache = {}
        for fname in tqdm(filenames, desc=f"    Load clean {category}", leave=False):
            data = torch.load(
                os.path.join(clean_dir, fname), map_location="cpu",
                weights_only=False,
            )
            # Binary fire ground truth: (T, H, W) as uint8
            clean_fire_cache[fname] = (data["x"][:, FIRE_CHANNEL] > 0).to(torch.uint8)

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
                clean_fire = clean_fire_cache[fname]  # (T, H, W) uint8
                num_timesteps = degraded_x.shape[0]

                pending_files[fname] = {
                    "deg_path": degraded_path,
                    "T": num_timesteps,
                    "outs": {},
                }

                for t in range(num_timesteps):
                    xt = drop_detection_channel(degraded_x[t].float())  # (42, H, W)
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

                save_data = dict(deg_data)
                save_data["x"] = recon_x.half()
                os.makedirs(save_prefix, exist_ok=True)
                save_futures.append(
                    save_pool.submit(
                        _save_file_worker,
                        (save_data, os.path.join(save_prefix, fname)),
                    )
                )

    return per_category_metrics, pooled_dice


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
        # NOTE: For NoFire samples, Dice is trivially 1.0 when both pred and
        # target are empty. This is by design to capture overall reconstruction
        # quality, but users should interpret pooled scores with this in mind.
        pooled_dice[difficulty].append(
            dice_score_masked(pred_binary[i], target, vis_mask)
        )

        # Store merged output for file reconstruction
        if fname in pending_files:
            pending_files[fname]["outs"][t] = merged[i]


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

    files_by_year = collect_train_files_by_year()
    for year, files in files_by_year.items():
        print(f"  Train data {year}: {len(files)} samples")
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
        model = CrossAttentionViT().to(device)
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

            if num_failed:
                print(f"  WARNING: {num_failed} files failed to save")
            else:
                print(f"  All {len(save_futures)} files saved OK")

    # =========================================================================
    #  Print and Save Results
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("  FINAL RESULTS (all 4 test folds pooled)")
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
                    f"    diff={difficulty:>2}%:  {metric_name} = "
                    f"{mean_val:.4f} +/- {std_val:.4f}  (n={len(values)})"
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
                f"    diff={difficulty:>2}%:  DICE = "
                f"{mean_val:.4f} +/- {std_val:.4f}  (n={len(values)})"
            )

    print(f"\n{'=' * 70}")

    # Save results
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
    print(f"            {reconstructed_dir('pixelwise')}")


if __name__ == "__main__":
    run_loocv()