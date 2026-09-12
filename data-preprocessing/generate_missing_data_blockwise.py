#!/usr/bin/env python3
"""
Fire Channel Challenge Data Generator (Region-Based Masking)

Generates challenge datasets by masking rectangular regions in satellite imagery.
Masks ENTIRE REGIONS with a valid_mask channel indicating observable areas.

Channel Layout (43 channels):
    0-17:  Base features (18 ch)
    18-34: Landcover one-hot (17 ch)
    35-39: Weather features (5 ch)
    40:    Valid mask (1=observable, 0=occluded)
    41:    Detection time (masked)
    42:    Fire mask (masked)

Challenge Levels: 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%
"""

import random
import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class Config:
    """Global configuration."""
    data_root: Path = Path("/pub/cyang27/data/data_pt")
    output_root: Path = Path("/pub/cyang27/data")
    batch_size: int = 32
    num_workers: int = 8
    random_seed: int = 42
    challenge_levels: Tuple[int, ...] = (0, 10, 20, 30, 40, 50, 60, 70, 80)
    
    # Channel indices
    valid_mask_channel: int = 40
    detection_time_channel: int = 41
    fire_mask_channel: int = 42
    landcover_channel: int = 16
    num_landcover_classes: int = 17
    
    # Angle transform config (raw channels)
    current_degree_features: Tuple[int, ...] = (7, 13)  # -> sin, cos
    future_degree_features: Tuple[int, ...] = (19,)     # -> sin only
    
    # Block masking params
    block_min_size: int = 4
    block_max_size: int = 16
    block_size_power: float = 2.0
    
    @property
    def blockwise_output(self) -> Path:
        return self.output_root / "data_challenge_blockwise"
    
    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


CONFIG = Config()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BlockSizeSampler:
    """Sample block sizes from power-law distribution: P(s) ∝ s^(-power)"""
    
    def __init__(self, min_size: int, max_size: int, power: float = 2.0):
        self.sizes = np.arange(min_size, max_size + 1)
        weights = self.sizes.astype(np.float64) ** (-power)
        self.probs = weights / weights.sum()
    
    def sample_pair(self) -> Tuple[int, int]:
        h = np.random.choice(self.sizes, p=self.probs)
        w = np.random.choice(self.sizes, p=self.probs)
        return int(h), int(w)


def generate_region_mask(
    H: int, W: int, mask_ratio: float, size_sampler: BlockSizeSampler
) -> np.ndarray:
    """
    Generate occlusion mask covering EXACTLY mask_ratio of the image.
    
    Algorithm:
        1. Sample block centers from visible pixels only
        2. Generate full blocks until close to target
        3. Final block: only occlude exactly the remaining needed pixels (preserves block structure)
    
    Returns:
        valid_mask: [H, W] where 1=observable, 0=occluded
    """
    if mask_ratio <= 0:
        return np.ones((H, W), dtype=np.float32)
    if mask_ratio >= 1:
        return np.zeros((H, W), dtype=np.float32)
    
    valid_mask = np.ones((H, W), dtype=np.float32)
    total_pixels = H * W
    target_occluded = int(total_pixels * mask_ratio)
    max_iterations = max(target_occluded * 2, 1000)
    
    for _ in range(max_iterations):
        current_occluded = int((valid_mask == 0).sum())
        remaining = target_occluded - current_occluded
        if remaining <= 0:
            break
        
        # Sample center from visible pixels
        visible_y, visible_x = np.where(valid_mask == 1)
        if len(visible_y) == 0:
            break
        
        idx = np.random.randint(len(visible_y))
        cy, cx = visible_y[idx], visible_x[idx]
        bh, bw = size_sampler.sample_pair()
        
        # Compute block boundaries
        y0 = max(0, cy - bh // 2)
        y1 = min(H, y0 + bh)
        x0 = max(0, cx - bw // 2)
        x1 = min(W, x0 + bw)
        
        # Count how many NEW pixels this block would occlude
        block_visible = valid_mask[y0:y1, x0:x1]
        new_occluded = int(block_visible.sum())
        
        if new_occluded <= remaining:
            # Full block fits within budget
            valid_mask[y0:y1, x0:x1] = 0
        else:
            # Partial block: only occlude exactly 'remaining' pixels within this block
            block_vis_y, block_vis_x = np.where(block_visible == 1)
            select_idx = np.random.choice(len(block_vis_y), size=remaining, replace=False)
            valid_mask[y0 + block_vis_y[select_idx], x0 + block_vis_x[select_idx]] = 0
            break
    
    return valid_mask


def apply_mask_to_channels(
    fire_mask: torch.Tensor,
    detection_time: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero out occluded regions in fire and detection time channels."""
    occluded = (valid_mask == 0)
    fire_out = fire_mask.clone()
    time_out = detection_time.clone()
    fire_out[occluded] = 0
    time_out[occluded] = 0
    return fire_out, time_out


def generate_temporal_masks(
    T: int, H: int, W: int, mask_ratio: float, 
    size_sampler: BlockSizeSampler, independent_per_frame: bool = True
) -> torch.Tensor:
    """Generate occlusion masks for all timesteps."""
    if independent_per_frame:
        masks = [generate_region_mask(H, W, mask_ratio, size_sampler) for _ in range(T)]
        return torch.from_numpy(np.stack(masks, axis=0)).float()
    else:
        mask = generate_region_mask(H, W, mask_ratio, size_sampler)
        return torch.from_numpy(mask).float().unsqueeze(0).expand(T, -1, -1)


def compute_statistics(pt_files: List[Path], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute normalization statistics for angle-transformed channels.
    
    Raw (23 ch) -> Angle Transform -> 25 channels:
        ch 7, 13: -> sin, cos (2 each)
        ch 19: -> sin only
        others: identity
    
    Returns:
        (means, stds) with 25 elements each
    """
    num_raw = 23
    sums = torch.zeros(num_raw, dtype=torch.float64, device=device)
    sq_sums = torch.zeros(num_raw, dtype=torch.float64, device=device)
    counts = torch.zeros(num_raw, dtype=torch.float64, device=device)
    
    print(f"📊 Computing statistics ({len(pt_files)} files)...")
    for f in tqdm(pt_files, desc="Statistics"):
        try:
            x = torch.load(f, map_location=device)['x'].float()
            x_flat = x.permute(1, 0, 2, 3).reshape(num_raw, -1)
            valid = ~torch.isnan(x_flat)
            vals = torch.where(valid, x_flat, torch.zeros_like(x_flat))
            sums += vals.sum(dim=1)
            sq_sums += (vals ** 2).sum(dim=1)
            counts += valid.sum(dim=1).float()
        except Exception as e:
            print(f"⚠️ Skipping {f.name}: {e}")
    
    counts = counts.clamp(min=1.0)
    means = sums / counts
    stds = torch.sqrt((sq_sums / counts) - (means ** 2) + 1e-6)
    
    # Build 25-channel statistics
    exp_means, exp_stds = [], []
    for ch in range(num_raw):
        if ch in CONFIG.current_degree_features:
            exp_means.extend([0.0, 0.0])
            exp_stds.extend([1.0, 1.0])
        elif ch in CONFIG.future_degree_features:
            exp_means.append(0.0)
            exp_stds.append(1.0)
        elif ch == CONFIG.landcover_channel:
            exp_means.append(0.0)
            exp_stds.append(1.0)
        else:
            exp_means.append(means[ch].item())
            exp_stds.append(stds[ch].item())
    
    assert len(exp_means) == 25
    print(f"✓ Statistics: {len(exp_means)} channels")
    return torch.tensor(exp_means, dtype=torch.float32), torch.tensor(exp_stds, dtype=torch.float32)


def apply_angle_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Apply sin/cos transform to angle features.
    Input: [B, T, 23, H, W] -> Output: [B, T, 25, H, W]
    """
    channels = []
    for ch in range(x.shape[2]):
        data = x[:, :, ch:ch+1]
        if ch in CONFIG.current_degree_features:
            rad = torch.deg2rad(data)
            channels.extend([torch.sin(rad), torch.cos(rad)])
        elif ch in CONFIG.future_degree_features:
            channels.append(torch.sin(torch.deg2rad(data)))
        else:
            channels.append(data)
    return torch.cat(channels, dim=2)


def preprocess_batch(
    files: List[Path], means: torch.Tensor, stds: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
    """
    Preprocess batch: angle transform -> normalize -> one-hot landcover.
    
    Returns:
        x_base: [B, T, 40, H, W] (18 base + 17 landcover + 5 weather)
        fire_mask: [B, T, H, W]
        detection_time: [B, T, H, W]
        y: [B, H, W]
        metadata: List[Dict]
    """
    x_list, y_list, meta_list = [], [], []
    for f in files:
        data = torch.load(f, map_location="cpu")
        x_list.append(data['x'])
        y_raw = data['y'] if isinstance(data['y'], torch.Tensor) else torch.tensor(data['y'])
        y_list.append(y_raw)
        meta_list.append(_extract_metadata(data['x'], y_raw, f))
    
    x = torch.stack(x_list).to(device).float()  # [B, T, 23, H, W]
    y = torch.stack(y_list).to(device).float()
    
    # Angle transform (23 -> 25)
    x_tf = apply_angle_transform(x)
    
    # Normalize
    m = means.view(1, 1, -1, 1, 1).to(device)
    s = stds.view(1, 1, -1, 1, 1).to(device)
    x_norm = torch.nan_to_num((x_tf - m) / s, nan=0.0)
    
    # One-hot landcover
    lc = torch.clamp(x[:, :, CONFIG.landcover_channel].long(), 0, CONFIG.num_landcover_classes - 1)
    lc_oh = torch.nn.functional.one_hot(lc, CONFIG.num_landcover_classes).permute(0, 1, 4, 2, 3).float()
    
    # Extract fire/detection
    detection_time = torch.nan_to_num(x_tf[:, :, -1], nan=0.0).clamp(0, 23) / 23.0
    fire_mask = (x[:, :, -1] > 0).float()
    
    # Build x_base: [0:18] + lc_oh + [19:24] = 40 channels
    x_base = torch.cat([x_norm[:, :, :18], lc_oh, x_norm[:, :, 19:24]], dim=2)
    y_final = (torch.nan_to_num(y, nan=0.0) > 0).byte()
    
    return x_base, fire_mask, detection_time, y_final, meta_list


def _extract_metadata(x: torch.Tensor, y: torch.Tensor, filepath: Path) -> Dict:
    """Extract sample metadata for categorization."""
    has_hist = (torch.nan_to_num(x[:, -1], nan=0.0) > 0).any().item()
    has_future = (torch.nan_to_num(y, nan=0.0) > 0).any().item()
    
    if has_hist and has_future:
        situation = "Fire_Continues"
    elif has_hist:
        situation = "Fire_Extinguished"
    elif has_future:
        situation = "NewFire_NoHistory"
    else:
        situation = "NoFire"
    
    year = 2020
    for yr in [2018, 2019, 2020, 2021]:
        if str(yr) in filepath.stem:
            year = yr
            break
    
    return {"situation": situation, "year": year, "filename": filepath.name}


def save_sample(x: torch.Tensor, y: torch.Tensor, meta: Dict, level: int, output_dir: Path) -> None:
    """Save processed sample to disk."""
    save_dir = output_dir / f"{meta['year']}_{meta['situation']}" / f"difficulty_{level}"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'x': x.half().cpu(), 'y': y.cpu(),
        'difficulty': level, 'mask_type': 'blockwise_region'
    }, save_dir / meta['filename'])


def _save_worker(args: Tuple) -> None:
    save_sample(*args)


def process_and_save_batch(
    x_base: torch.Tensor, fire_mask: torch.Tensor, detection_time: torch.Tensor,
    y_batch: torch.Tensor, meta_list: List[Dict],
    size_sampler: BlockSizeSampler, executor: ThreadPoolExecutor
) -> None:
    """Apply all challenge levels and save results."""
    B, T, _, H, W = x_base.shape
    tasks = []
    
    for b in range(B):
        x_b, fire_b, time_b = x_base[b], fire_mask[b], detection_time[b]
        y, meta = y_batch[b], meta_list[b]
        
        for level in CONFIG.challenge_levels:
            ratio = level / 100.0
            valid_mask = generate_temporal_masks(T, H, W, ratio, size_sampler).to(x_base.device)
            fire_masked, time_masked = apply_mask_to_channels(fire_b, time_b, valid_mask)
            
            # Assemble: [base(40) + valid_mask(1) + det_time(1) + fire(1)] = 43
            x_final = torch.cat([
                x_b, valid_mask.unsqueeze(1), 
                time_masked.unsqueeze(1), fire_masked.unsqueeze(1)
            ], dim=1)
            tasks.append((x_final, y, meta, level, CONFIG.blockwise_output))
    
    list(executor.map(_save_worker, tasks))


def main():
    set_seed(CONFIG.random_seed)
    
    print("=" * 60)
    print("🔥 Fire Channel Challenge Data Generator")
    print("=" * 60)
    print(f"Input:  {CONFIG.data_root}")
    print(f"Output: {CONFIG.blockwise_output}")
    print(f"Device: {CONFIG.device}")
    print(f"Levels: {CONFIG.challenge_levels}")
    print("=" * 60)
    
    pt_files = sorted(CONFIG.data_root.glob("*.pt"))
    if not pt_files:
        print("❌ No .pt files found!")
        return
    print(f"📁 Found {len(pt_files)} files")
    
    CONFIG.blockwise_output.mkdir(parents=True, exist_ok=True)
    means, stds = compute_statistics(pt_files, CONFIG.device)
    size_sampler = BlockSizeSampler(CONFIG.block_min_size, CONFIG.block_max_size, CONFIG.block_size_power)
    
    stats = {'total': 0, 'by_situation': {}, 'challenge_levels': CONFIG.challenge_levels,
             'mask_type': 'blockwise_region', 'num_channels': 43}
    
    print(f"\n🚀 Processing (batch_size={CONFIG.batch_size})...")
    with ThreadPoolExecutor(max_workers=CONFIG.num_workers) as executor:
        pbar = tqdm(range(0, len(pt_files), CONFIG.batch_size), desc="Batches")
        for i in pbar:
            batch_files = pt_files[i:i + CONFIG.batch_size]
            x_base, fire_mask, detection_time, y_batch, meta_list = preprocess_batch(
                batch_files, means, stds, CONFIG.device
            )
            process_and_save_batch(x_base, fire_mask, detection_time, y_batch, meta_list, size_sampler, executor)
            
            stats['total'] += len(batch_files)
            for m in meta_list:
                stats['by_situation'][m['situation']] = stats['by_situation'].get(m['situation'], 0) + 1
            pbar.set_postfix({'processed': stats['total']})
            
            del x_base, fire_mask, detection_time, y_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Save metadata
    with open(CONFIG.blockwise_output / "stats.pkl", 'wb') as f:
        pickle.dump(stats, f)
    with open(CONFIG.blockwise_output / "preprocessing_params.pkl", 'wb') as f:
        pickle.dump({'means': means.numpy(), 'stds': stds.numpy()}, f)
    
    print("\n" + "=" * 60)
    print("✅ Complete!")
    print(f"Samples: {stats['total']} | Files: {stats['total'] * len(CONFIG.challenge_levels)}")
    print(f"Output: {CONFIG.blockwise_output}")
    for sit, count in sorted(stats['by_situation'].items()):
        print(f"  {sit}: {count}")
    print("=" * 60)


if __name__ == '__main__':
    main()