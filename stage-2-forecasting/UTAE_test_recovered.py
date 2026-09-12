#!/usr/bin/env python3
"""
Evaluate trained U-TAE models on RECONSTRUCTED data.
Each fold's model predicts its held-out year on reconstructed data.
Metrics: AP (Fire_Continues, NewFire_NoHistory), FPR (Fire_Extinguished, NoFire), Combined AP.
"""

import os, json, gc
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))
from UTAE import UTAE 

# ========================= CONFIG =========================

DATA_ROOT       = Path("/pub/cyang27/data")
MODEL_ROOT      = Path("/pub/cyang27/UTAE_Challenge_Results")
OUTPUT_JSON     = MODEL_ROOT / "reconstructed_results.json"

MASK_TYPES  = ["pixelwise", "blockwise"]
YEARS       = [2018, 2019, 2020, 2021]
SCENARIOS   = ["Fire_Continues", "Fire_Extinguished", "NewFire_NoHistory", "NoFire"]
AP_SCENARIOS  = {"Fire_Continues", "NewFire_NoHistory"}
FPR_SCENARIOS = {"Fire_Extinguished", "NoFire"}
DIFFICULTIES  = list(range(0, 81, 10))

KEEP_CHS  = [i for i in range(43) if i not in {40, 41}]
INPUT_DIM = len(KEEP_CHS)
THRESHOLD = 0.5
BATCH_SIZE = 16
NUM_WORKERS = max(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)) - 2, 4)


# ========================= DATASET =========================

class FireDataset(Dataset):
    def __init__(self, file_list):
        self.files = [str(f) for f in file_list]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            data = torch.load(self.files[idx], map_location="cpu", weights_only=False)
            x = data["x"].float()[:, KEEP_CHS]
            y = data["y"].long()
            return x, y
        except Exception as e:
            print(f"[Warning] Load failed: {self.files[idx]}: {e}")
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])


# ========================= METRICS =========================

def ap_score(probs, targets):
    n_pos = targets.sum()
    if n_pos == 0:
        return 1.0 - float(probs.max())
    if n_pos == len(targets):
        return 1.0
    return float(average_precision_score(targets, probs))


def fpr_score(probs, targets):
    preds = (probs > THRESHOLD).astype(int)
    fp = ((preds == 1) & (targets == 0)).sum()
    tn = ((preds == 0) & (targets == 0)).sum()
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


# ========================= MODEL =========================

def load_model(checkpoint_path, device):
    model = UTAE(
        input_dim=INPUT_DIM, encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128], out_conv=[32, 2],
        agg_mode="att_group", encoder_norm="group",
        n_head=16, d_model=256, d_k=4, pad_value=0,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    model.eval()
    return model


# ========================= INFERENCE =========================

def get_files(recon_root, year, scenario, difficulty):
    d = recon_root / f"{year}_{scenario}" / f"difficulty_{difficulty}"
    return sorted(d.glob("*.pt")) if d.exists() else []


@torch.no_grad()
def evaluate_year(model, recon_root, year, device):
    """Evaluate one year, return per-scenario and combined scores."""
    use_amp = device.type == "cuda"
    cat_scores = {}
    comb_scores = defaultdict(list)

    for scenario in SCENARIOS:
        is_ap = scenario in AP_SCENARIOS
        scenario_scores = {}

        for diff in DIFFICULTIES:
            files = get_files(recon_root, year, scenario, diff)
            if not files:
                continue

            loader = DataLoader(FireDataset(files), batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

            scores_diff, comb_diff = [], []
            for batch in tqdm(loader, desc=f"      {scenario} d={diff}%", leave=False):
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
                y_np = y.numpy()

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


# ========================= MAIN =========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"{'='*70}")
    print(f"  Evaluate U-TAE on RECONSTRUCTED data")
    print(f"{'='*70}")

    all_results = {}

    for mask_type in MASK_TYPES:
        print(f"\n{'='*70}")
        print(f"  MASK TYPE: {mask_type.upper()}")
        print(f"{'='*70}")

        recon_root = DATA_ROOT / f"data_reconstructed_{mask_type}"
        if not recon_root.exists():
            print(f"  [SKIP] {recon_root} does not exist")
            continue

        # Pool scores across 4 folds
        cat_pool = {sc: defaultdict(list) for sc in SCENARIOS}
        comb_pool = defaultdict(list)

        for test_yr in YEARS:
            ckpt = MODEL_ROOT / f"model_{mask_type}_test{test_yr}.pt"
            if not ckpt.exists():
                print(f"  [SKIP] Checkpoint not found: {ckpt}")
                continue

            print(f"\n  --- Year {test_yr} (model: {ckpt.name}) ---")
            model = load_model(ckpt, device)

            cat_scores, comb_scores = evaluate_year(model, recon_root, test_yr, device)

            # Pool
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

        # ---- Aggregate ----
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

    # ========================= PRINT =========================
    print(f"\n{'='*70}")
    print("  FINAL RESULTS — RECONSTRUCTED DATA (all 4 folds pooled)")
    print(f"{'='*70}")

    for mt in MASK_TYPES:
        if mt not in all_results:
            continue
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
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()