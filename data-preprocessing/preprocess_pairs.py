# =========================
# Wildfire HDF5 -> Cropped Pairs (.pt)
# Store all the 23 channels (multi-modal) not only the active fire channel
# Past T days (default 5) -> next day
# Crop center: largest fire blob center on day t (fallback to earlier days)
# Save all pairs into one folder (no train/val/test split)
# =========================

import os
from pathlib import Path
import numpy as np
import torch
import h5py
from scipy import ndimage
from collections import Counter

# -------------------------
# User config (edit here)
# -------------------------
DATA_ROOT = Path("../data_HDF5")  # year/*.hdf5
OUT_ROOT  = Path("../data_pt")  # save all .pt here
T = 5                      # past days
CROP_SIZE = 64             # crop size
MAX_PAIRS = None           # None => process all; or set int like 20000 for quick test
SAVE_FLOAT16 = False       # True can shrink disk
FIRE_CHANNEL_IDX = -1      # last channel is active fire (time); binary is ( >0 )

# -------------------------
# Helpers
# -------------------------
def largest_cc_center(mask_2d: np.ndarray):
    """Return (cy,cx) center of largest connected component in a binary mask. None if empty."""
    labeled, n = ndimage.label(mask_2d.astype(np.uint8))
    if n == 0:
        return None
    sizes = ndimage.sum(mask_2d, labeled, index=np.arange(1, n+1))
    largest = int(np.argmax(sizes) + 1)
    coords = np.argwhere(labeled == largest)  # [N,2] (y,x)
    cy, cx = coords.mean(axis=0)
    return int(round(cy)), int(round(cx))

def get_crop_top_left(center_yx, img_2d: np.ndarray, crop: int):
    """Crop [crop,crop] around center, clamped to image bounds. Returns (top,left)."""
    H, W = img_2d.shape
    cy, cx = center_yx
    half = crop // 2
    top  = max(0, min(cy - half, H - crop))
    left = max(0, min(cx - half, W - crop))
    return (top, left)

def decode_dates(arr):
    # HDF5 attrs can be bytes
    out = []
    for x in arr:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode())
        else:
            out.append(str(x))
    return out

# -------------------------
# Main
# -------------------------
OUT_ROOT.mkdir(parents=True, exist_ok=True)
print(f"[info] DATA_ROOT={DATA_ROOT}")
print(f"[info] OUT_ROOT ={OUT_ROOT}")
print(f"[info] T={T}, CROP_SIZE={CROP_SIZE}, MAX_PAIRS={MAX_PAIRS}")
print(f"[info] STORE ALL 23 CHANNELS (MULTI-MODAL)\n")

pair_count = 0
stats = Counter()

# iterate years
year_dirs = [p for p in sorted(DATA_ROOT.iterdir()) if p.is_dir() and p.name.isdigit()]
if not year_dirs:
    raise RuntimeError(f"No year folders found under {DATA_ROOT}")

for year_dir in year_dirs:
    year = int(year_dir.name)
    h5_files = sorted(year_dir.glob("*.hdf5"))
    if not h5_files:
        continue

    print(f"[year {year}] found {len(h5_files)} HDF5 files")

    for h5_path in h5_files:
        fire_id = h5_path.stem

        with h5py.File(h5_path, "r") as f:
            data = f["data"][:]  # [days, channels, H, W]
            dates = f["data"].attrs.get("img_dates", None)

        n_days, n_ch, H, W = data.shape
        print(f"  {fire_id}: shape={data.shape} (days={n_days}, channels={n_ch}, H={H}, W={W})")

        if dates is None:
            dates = [f"{year}-day{d:03d}" for d in range(n_days)]
        else:
            dates = decode_dates(dates)

        # binary fire per day (use active fire detection time channel > 0)
        fire_binary = (data[:, FIRE_CHANNEL_IDX] > 0).astype(np.uint8)  # [days,H,W]

        # iterate timepoints (need T past days and 1 future)
        for t_last in range(T-1, n_days-1):
            y_day = t_last + 1

            # find crop center from day t_last (k=0) back to day t_last-(T-1) (k=T-1)
            center = None
            used_day = -1
            # k = 0 means "use day t_last"; k = 4 means "use day t_last-4"
            for k in range(0, T):
                d = t_last - k
                if fire_binary[d].sum() > 0:
                    center = largest_cc_center(fire_binary[d])
                    used_day = k
                    break

            if center is None:
                center = (H // 2, W // 2)  # fallback: image center

            # crop past T days with 23 channels as x, and next day as y
            crop_top, crop_left = get_crop_top_left(center, fire_binary[y_day], CROP_SIZE)
            
            x = data[t_last - (T-1): t_last + 1, :, crop_top: crop_top+CROP_SIZE, crop_left: crop_left+CROP_SIZE] # [T,C,H,W]
            y = fire_binary[y_day][crop_top: crop_top+CROP_SIZE, crop_left: crop_left+CROP_SIZE]  # [H,W]
           
            # to torch
            x_t = torch.tensor(x, dtype=torch.float16 if SAVE_FLOAT16 else torch.float32) 
            y_t = torch.tensor(y, dtype=torch.long)

            x_last_sum = int(x[-1,-1].sum())
            y_sum = int(y.sum())

            sample = {
                "x": x_t, # [T,C,H,W]
                "y": y_t, # [H,W]
                "meta": {
                    "year": year,
                    "fire_id": fire_id,
                    "t_last": int(t_last),
                    "used_day": int(used_day),   
                    "crop": int(CROP_SIZE),
                    "T": int(T),
                    "n_channels": int(n_ch),
                    "x_last_sum": x_last_sum,
                    "y_sum": y_sum,
                    "dates_x": dates[t_last-(T-1):t_last+1],
                    "date_y": dates[y_day],
                    "source_hdf5": str(h5_path),
                }
            }

            out_name = f"{year}_{fire_id}_day{t_last:03d}.pt"
            torch.save(sample, OUT_ROOT / out_name)

            # stats
            pair_count += 1
            stats["pairs"] += 1
            stats["y_sum==0"] += int(y_sum == 0)
            stats["x_last_sum==0"] += int(x_last_sum == 0)
            stats[f"used_day={used_day}"] += 1

            if pair_count % 500 == 0:
                print(f"[progress] pairs={pair_count:,}  fire={year}/{fire_id} t_last={t_last} y_sum={y_sum} used_day={used_day}")

            if MAX_PAIRS is not None and pair_count >= MAX_PAIRS:
                break

        if MAX_PAIRS is not None and pair_count >= MAX_PAIRS:
            break

    if MAX_PAIRS is not None and pair_count >= MAX_PAIRS:
        break

print("\n" + "="*70)
print("✅ Done - 23 CHANNEL MULTI-MODAL DATA")
print("="*70)
print(f"Saved pairs: {pair_count:,}")
print(f"Data format: x = [T={T}, C=channels, H={CROP_SIZE}, W={CROP_SIZE}]")
print(f"             y = [H={CROP_SIZE}, W={CROP_SIZE}]")


if pair_count > 0:
    print(f"\n Data statistics:")
    print(f"y_sum==0: {stats['y_sum==0']:,} ({100*stats['y_sum==0']/pair_count:.2f}%)")
    print(f"x_last_sum==0: {stats['x_last_sum==0']:,} ({100*stats['x_last_sum==0']/pair_count:.2f}%)")
    print("used_day distribution:")
    for k in sorted([k for k in stats.keys() if str(k).startswith("used_day=")]):
        print(f"  {k}: {stats[k]:,} ({100*stats[k]/pair_count:.2f}%)")

print(f"\n [info] Example output file: {next(iter(OUT_ROOT.glob('*.pt')), None)}")