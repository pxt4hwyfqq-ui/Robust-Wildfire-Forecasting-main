from pathlib import Path
import h5py
import numpy as np
import warnings
from tqdm import tqdm
import multiprocessing
from functools import partial

# ================= Configuration =================
# structure of directory  E:/data_hdf5/2018, E:/data_hdf5/2019 ...
DATA_DIR = Path("E:/data_hdf5") 
YEARS = ["2018", "2019", "2020", "2021"]  
KEY = "data"
NUM_WORKERS = 8
# =================================================

channel_book = {
    1: 'band M11', 2: 'band I2', 3: 'band I1', 4: 'NDVI', 5: 'EVI2',
    6: 'total precipitation', 7: 'wind speed', 8: 'wind direction',
    9: 'min temperature', 10: 'max temperature', 11: 'energy release',
    12: 'specific humidity', 13: 'slope', 14: 'aspect', 15: 'elevation',
    16: 'drought index', 17: 'land cover', 18: 'fcst total precipitation',
    19: 'fcst wind speed', 20: 'fcst wind direction', 21: 'fcst temperature',
    22: 'fcst specific humidity', 23: 'active fire'
}

def safe_mean(sum_, cnt_):
    out = sum_ / np.maximum(cnt_, 1)
    out[cnt_ == 0] = np.nan
    return out

def process_single_file(fp):
    """
    Single-file processing function for use with multiprocessing
    Return: (min_per_channel, max_per_channel, sum_per_channel, cnt_per_channel)
    """
    try:
        with h5py.File(fp, "r") as f:
            if KEY not in f:
                return None
            ds = f[KEY]

            # store data into array   
            ds_arr = np.asarray(ds, dtype=np.float64)             # (T, C, H, W)

        # first calculate vaild elements in each channel (deal with all-NaN channel)
        cnt_per_channel = np.sum(~np.isnan(ds_arr), axis=(0, 2, 3))
        valid = cnt_per_channel > 0

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            
            # per-channel min/max（ignore NaN）
            min_per_channel = np.nanmin(ds_arr, axis=(0, 2, 3))   # (C, )
            max_per_channel = np.nanmax(ds_arr, axis=(0, 2, 3))   # (C, )
            
            # key: all-NaN channel do not pollute global min/max
            min_per_channel[~valid] = np.inf
            max_per_channel[~valid] = -np.inf
            
            # per-channel sum（ignore NaN）
            sum_per_channel = np.nansum(ds_arr, axis=(0, 2, 3))    # (C, )

        return (min_per_channel, max_per_channel, sum_per_channel, cnt_per_channel)
    
    except Exception as e:
        print(f"Error processing {fp}: {e}")
        return None
    
def main():
    # 1. Collect file dir of all years
    all_files = []
    for year in YEARS:
        year_dir = DATA_DIR / year
        if not year_dir.exists():
            print(f"Warning: Directory not found: {year_dir}")
            continue
        files = sorted(year_dir.glob("*.hdf5"))
        print(f"Year {year}: Found {len(files)} files.")
        all_files.extend(files)

    if not all_files:
        raise FileNotFoundError("No .hdf5 files found in any specified directories.")

    print(f"\nStarting processing {len(all_files)} files with {NUM_WORKERS} processes...")

    # 2. Initialize global statistics variables
    C = len(channel_book) 
    glb_min = np.full(C, np.inf, dtype=np.float64)
    glb_max = np.full(C, -np.inf, dtype=np.float64)
    glb_sum = np.zeros(C, dtype=np.float64)
    glb_cnt = np.zeros(C, dtype=np.int64)

    # 3. Parallel processing with multiprocessing
    # Using imap_unordered makes result handling smoother and works well with tqdm to show progress
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        # Wrap the iterator with tqdm
        iterator = tqdm(pool.imap_unordered(process_single_file, all_files), 
                        total=len(all_files), unit="file", miniters=10)
        
        for res in iterator:
            if res is None:
                continue
            
            local_min, local_max, local_sum, local_cnt = res
            
            # Update global statistics (reducer step)
            glb_min = np.minimum(glb_min, local_min)
            glb_max = np.maximum(glb_max, local_max)
            glb_sum += local_sum
            glb_cnt += local_cnt

    # 4. ✅ Stats of all files（global)
    glb_mean = safe_mean(glb_sum, glb_cnt)
    glb_min[np.isinf(glb_min)] = np.nan
    glb_max[np.isinf(glb_max)] = np.nan

    # 5. Print result
    print("\n================= GLOBAL STATS (2018-2021) =================")
    print(f"{'ID':<4} {'Name':<25} {'Min':<12} {'Max':<12} {'Mean':<12}")
    print("-" * 70)
    for i in range(len(glb_mean)):
        name = channel_book.get(i + 1, f"Channel_{i+1}")
        print(f"{i+1:<4d} {name:<25s} "
              f"{glb_min[i]:<12.4f} {glb_max[i]:<12.4f} {glb_mean[i]:<12.4f}")

if __name__ == '__main__':
    main()