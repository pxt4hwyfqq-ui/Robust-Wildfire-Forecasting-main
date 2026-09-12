"""
Mixed Fire Challenge Dataset Generator (Per-Day Pairs, All Files, Fire_Continues only)

For EVERY .pt file in Fire_Continues across all years:
  For each temporal day t, independently flip coin (blockwise/pixelwise)
  and draw difficulty (10-80). Output ONE .pt file PER DAY with:
    'x': (43, H, W) float16 — day t from the degraded source
    'y': (H, W) — clean binary fire map (channel 42 from difficulty_0's x[t])
"""

import os
import random
import re
import torch

# ========================= CONFIG =========================
DATA_ROOT  = "/pub/cyang27/data"
OUTPUT_DIR = "/pub/cyang27/data/data_mixed_challenge_train"
SEED       = 42

FIRE_CHANNEL = 42

MASK_TYPES   = ["blockwise", "pixelwise"]
DIFFICULTIES = [10, 20, 30, 40, 50, 60, 70, 80]
YEARS        = [2018, 2019, 2020, 2021]
CATEGORY     = "Fire_Continues"
FNAME_RE     = re.compile(r"^(\d{4})_fire_(\d+)_day(\d+)\.pt$")


def resolve_path(mask_type, year, diff, fname):
    return os.path.join(DATA_ROOT, f"data_challenge_{mask_type}",
                        f"{year}_{CATEGORY}", f"difficulty_{diff}", fname)


def discover_all_files():
    """Scan difficulty_0 (pixelwise) Fire_Continues for all years."""
    files = []
    for year in YEARS:
        d0 = os.path.join(DATA_ROOT, "data_challenge_pixelwise",
                          f"{year}_{CATEGORY}", "difficulty_0")
        if not os.path.isdir(d0):
            continue
        for f in sorted(os.listdir(d0)):
            if FNAME_RE.match(f):
                files.append((year, f))
    return files


def generate():
    rng = random.Random(SEED)
    torch.manual_seed(SEED)

    all_files = discover_all_files()
    total_files = len(all_files)
    print(f"Found {total_files} .pt files in {CATEGORY}.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    saved = 0
    for i, (year, fname) in enumerate(all_files):
        stem = fname.replace(".pt", "")

        clean_path = resolve_path("pixelwise", year, 0, fname)
        try:
            clean_x = torch.load(clean_path, map_location="cpu", weights_only=False)['x']
        except FileNotFoundError:
            print(f"[SKIP] Clean not found: {clean_path}")
            continue

        T = clean_x.shape[0]

        day_choices = [(rng.choice(MASK_TYPES), rng.choice(DIFFICULTIES)) for _ in range(T)]

        cache = {}
        for mt, df in set(day_choices):
            p = resolve_path(mt, year, df, fname)
            try:
                cache[(mt, df)] = torch.load(p, map_location="cpu", weights_only=False)['x']
            except FileNotFoundError:
                print(f"[SKIP] Degraded not found: {p}")
                cache[(mt, df)] = None

        out_dir = os.path.join(OUTPUT_DIR, str(year))
        os.makedirs(out_dir, exist_ok=True)

        for t in range(T):
            mt, df = day_choices[t]
            deg_x = cache.get((mt, df))
            if deg_x is None:
                continue

            torch.save({
                'x': deg_x[t].half(),
                'y': clean_x[t, FIRE_CHANNEL],
                'mask_type': mt,
                'difficulty': df,
                'source_day': t,
            }, os.path.join(out_dir, f"{stem}_t{t}.pt"))
            saved += 1

        del cache
        if (i + 1) % 200 == 0 or (i + 1) == total_files:
            print(f"[{i+1}/{total_files}] files processed, {saved} pairs saved")

    print(f"\nDone. {saved} total pairs -> {OUTPUT_DIR}")


if __name__ == "__main__":
    generate()