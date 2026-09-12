"""
Baseline Models for Fire Reconstruction — Random & Dilation
============================================================
No training needed. Same LOOCV test pipeline as D3PM.

Random:   50% chance of fire at each masked pixel
Dilation: dilate visible fire by radius=5 box, predict fire in masked region
"""

import os, glob, math, random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ══════════════════════════════ Config ══════════════════════════════════════

DATA_ROOT = "/pub/cyang27/data"
SAVE_DIR  = os.path.join(DATA_ROOT, "baseline_results")

MASK_TYPES   = ["blockwise", "pixelwise"]
YEARS        = [2018, 2019, 2020, 2021]
CATEGORIES   = ["Fire_Continues", "Fire_Extinguished", "NewFire_NoHistory", "NoFire"]
DICE_CATS    = {"Fire_Continues", "Fire_Extinguished"}
FPR_CATS     = {"NewFire_NoHistory", "NoFire"}
DIFFICULTIES = [10, 20, 30, 40, 50, 60, 70, 80]
FIRE_CH, DET_CH = 42, 41

SEED = 42
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


# ══════════════════════════════ Baseline Models ════════════════════════════

def predict_random(fire_in):
    """
    Random baseline: 50% fire probability at each masked pixel.
    fire_in: (B, 2, H, W) — ch0=vis_mask, ch1=vis_fire_values
    Returns: (B, H, W) binary prediction
    """
    vis = fire_in[:, 0]   # (B, H, W)
    val = fire_in[:, 1]   # (B, H, W)
    occ = 1.0 - vis
    rand = torch.bernoulli(torch.full_like(vis, 0.5))
    return vis * val + occ * rand


def predict_dilation(fire_in, radius=5):
    """
    Dilation baseline: dilate visible fire pixels by a box of radius=5,
    then use that as prediction in masked region.
    fire_in: (B, 2, H, W) — ch0=vis_mask, ch1=vis_fire_values
    Returns: (B, H, W) binary prediction
    """
    vis = fire_in[:, 0]   # (B, H, W)
    val = fire_in[:, 1]   # (B, H, W)
    occ = 1.0 - vis

    # Visible fire pixels only
    vis_fire = (vis * val).unsqueeze(1)  # (B, 1, H, W)

    # Dilation via max-pool with kernel = 2*radius+1
    kernel_size = 2 * radius + 1
    dilated = F.max_pool2d(vis_fire, kernel_size=kernel_size, stride=1,
                           padding=radius)  # (B, 1, H, W)
    dilated = dilated.squeeze(1)  # (B, H, W)

    # In masked region, predict fire if dilated region says fire
    pred_occ = (dilated > 0.5).float()
    return vis * val + occ * pred_occ


# ══════════════════════════════ Metrics ═════════════════════════════════════

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


# ══════════════════════════════ Test Pipeline ══════════════════════════════

def _run_batch(predict_fn, conds, fires, metas, metrics, dice_all, cat, diff, is_dice):
    fire_in = torch.stack(fires)
    pred = (predict_fn(fire_in) >= 0.5).float()
    for i, m in enumerate(metas):
        fn = dice_on_mask if is_dice else fpr_on_mask
        metrics[(cat, diff)].append(fn(pred[i], m["tgt"], m["vm"]))
        dice_all[diff].append(dice_on_mask(pred[i], m["tgt"], m["vm"]))


def test_metrics(predict_fn, year, mask_type):
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
                        _run_batch(predict_fn, bc, bf, bm, metrics, dice_all, cat, diff, is_dice)
                        bc, bf, bm = [], [], []
            if bc:
                _run_batch(predict_fn, bc, bf, bm, metrics, dice_all, cat, diff, is_dice)

    return metrics, dice_all


# ══════════════════════════════ Main ═══════════════════════════════════════

def run_baselines():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    baselines = {
        "Random":   predict_random,
        "Dilation": predict_dilation,
    }

    for name, predict_fn in baselines.items():
        print(f"\n{'='*70}")
        print(f"  BASELINE: {name}")
        print(f"{'='*70}")

        all_met  = {mt: defaultdict(list) for mt in MASK_TYPES}
        all_dice = {mt: defaultdict(list) for mt in MASK_TYPES}

        for test_yr in YEARS:
            for mt in MASK_TYPES:
                print(f"\n  Testing {name} | {mt} | {test_yr}...")
                m, d = test_metrics(predict_fn, test_yr, mt)
                for k, v in m.items(): all_met[mt][k].extend(v)
                for k, v in d.items(): all_dice[mt][k].extend(v)

        # Print results
        print(f"\n{'='*70}")
        print(f"  FINAL RESULTS - {name}")
        print(f"{'='*70}")
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
                   os.path.join(SAVE_DIR, f"loocv_results_{name.lower()}.pt"))
        print(f"\n  Results saved -> {SAVE_DIR}/loocv_results_{name.lower()}.pt")


if __name__ == "__main__":
    run_baselines()