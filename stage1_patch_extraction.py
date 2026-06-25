# %%
import sys
print(sys.executable) 

import tiatoolbox
print(tiatoolbox.__version__)


"""
stage1: Patch Extraction
                 
"""

import csv
import logging
import random
import time
from multiprocessing import Pool, current_process
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from tiatoolbox.tools import patchextraction, stainnorm
from tiatoolbox.wsicore.wsireader import WSIReader

# Configuration
WSI_DIR           = Path("/media/user/Expansion/WSIs_156")
OUTPUT_DIR        = Path("outputs/2026_03_23_224px_025mpp_patches_h5")
REF_PATCH_PATH    = Path("outputs/reference_patch.npy")

PATCH_SIZE        = (224, 224)  
STRIDE            = (224, 224) 
RESOLUTION        = 0.5         
UNITS             = "mpp"

TISSUE_THRESH     = 0.5    
LAP_THRESH        = 25       
MEAN_THRESH       = 235    
PEN_THRESH        = 0.05    

MAX_PATCHES_PER_SLIDE = 7500
RANDOM_SEED           = 42

NUM_WORKERS = 4

WSI_EXTENSIONS = {".svs", ".ndpi", ".tiff", ".tif", ".mrxs", ".scn"}

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_normalizer = None


def _init_worker(ref_patch_path: str) -> None:
    global _normalizer
    pid = current_process().pid
    try:
        ref_arr     = np.load(ref_patch_path)
        _normalizer = stainnorm.MacenkoNormalizer()
        _normalizer.fit(ref_arr)
        logger.debug(f"[PID {pid}] Macenko fitted ✓")
    except Exception as e:
        logger.error(f"[PID {pid}] Macenko init failed: {e}")
        _normalizer = None


# QC helpers

def tissue_fraction(arr: np.ndarray) -> float:
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float((mask > 0).mean())


def laplacian_variance(arr: np.ndarray) -> float:
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_pen_mark(arr: np.ndarray) -> bool:
    hsv   = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    green = cv2.inRange(hsv, (40,  150, 50), (80,  255, 255))
    blue  = cv2.inRange(hsv, (100, 150, 50), (130, 255, 255))
    red1  = cv2.inRange(hsv, (0,   150, 50), (10,  255, 255))
    red2  = cv2.inRange(hsv, (170, 150, 50), (180, 255, 255))
    pen   = cv2.bitwise_or(green, blue)
    pen   = cv2.bitwise_or(pen,   red1)
    pen   = cv2.bitwise_or(pen,   red2)
    return bool((pen > 0).mean() > PEN_THRESH)


# Per-slide worker

def process_wsi(wsi_path: Path) -> dict:
    slide_id = wsi_path.stem
    out_path = OUTPUT_DIR / f"{slide_id}.h5"

    if out_path.exists():
        logger.info(f"[{slide_id}] SKIP")
        return {"slide_id": slide_id, "status": "skipped", "n_patches": 0}

    try:
        # Patch extractor with Otsu tissue mask
        # input_mask="otsu" + min_mask_ratio for tissue filtering
         extractor = patchextraction.get_patch_extractor(
            input_img=str(wsi_path),
            method_name="slidingwindow",
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            resolution=RESOLUTION,
            units=UNITS,
            input_mask="otsu",
            min_mask_ratio=TISSUE_THRESH,
        )

        # numpy iteration
        locs  = extractor.locations_df[["x", "y"]].values.astype(int)
        n_loc = len(locs)
        logger.info(f"[{slide_id}] {n_loc} tissue locations (Otsu)")

        if n_loc == 0:
            return {"slide_id": slide_id, "status": "no_tissue", "n_patches": 0}

        # Random sampling
        all_indices = list(range(n_loc))
        rng = random.Random(RANDOM_SEED)
        sampled = (
            rng.sample(all_indices, MAX_PATCHES_PER_SLIDE)
            if n_loc > MAX_PATCHES_PER_SLIDE
            else all_indices
        )
        logger.info(f"[{slide_id}] Sampling {len(sampled)} / {n_loc}")

        # Extract + QC + Macenko
        patches_out = []
        coords_out  = []

        for iloc_idx in sampled:
            try:
                patch_arr = np.array(extractor[iloc_idx])
                x, y      = int(locs[iloc_idx][0]), int(locs[iloc_idx][1])

                # QC filters
                if patch_arr.mean() > MEAN_THRESH:              continue
                if laplacian_variance(patch_arr) < LAP_THRESH:  continue
                if is_pen_mark(patch_arr):                      continue

                # Macenko
                if _normalizer is not None:
                    try:
                        patch_arr = _normalizer.transform(patch_arr.copy())
                    except Exception:
                        pass

                patches_out.append(patch_arr.astype(np.uint8))
                coords_out.append((x, y))

            except Exception as pe:
                logger.info(f"[{slide_id}] iloc={iloc_idx} EXCEPTION: {pe}")
                continue

        if not patches_out:
            return {"slide_id": slide_id, "status": "no_valid_patches", "n_patches": 0}

        patches_arr = np.stack(patches_out, axis=0)        
        coords_arr  = np.array(coords_out, dtype=np.int32) 

        # Save HDF5
        with h5py.File(out_path, "w") as f:
            f.create_dataset(
                "patches", data=patches_arr,
                compression="gzip", compression_opts=4,
                chunks=(1, *PATCH_SIZE, 3),
            )
            f.create_dataset("coords", data=coords_arr)
            f.attrs["slide_id"]       = slide_id
            f.attrs["n_patches"]      = len(patches_arr)
            f.attrs["resolution_mpp"] = RESOLUTION
            f.attrs["patch_size_wh"]  = list(PATCH_SIZE)
            f.attrs["macenko"]        = (_normalizer is not None)

        logger.info(f"[{slide_id}] ✓  {len(patches_arr)} patches → {out_path.name}")
        return {"slide_id": slide_id, "status": "success", "n_patches": len(patches_arr)}

    except Exception as e:
        logger.error(f"[{slide_id}] FAILED: {e}", exc_info=True)
        if out_path.exists():
            out_path.unlink()
        return {"slide_id": slide_id, "status": "error", "error": str(e), "n_patches": 0}


# Timed single-slide diagnostic

def run_timing_test(wsi_path: Path) -> None:
    """
    Run before full pipeline to confirm per-stage timings.
    Reports time for each stage and extrapolations to 1000 patches.
    """
    print(f"\n{'='*60}")
    print(f"TIMING TEST: {wsi_path.name[:55]}")
    print(f"{'='*60}")

    t0 = time.time()
    WSIReader.open(wsi_path)
    print(f"WSIReader.open        : {time.time()-t0:.2f}s", flush=True)

    t0 = time.time()
    extractor = patchextraction.get_patch_extractor(
        input_img=str(wsi_path),
        method_name="slidingwindow",
        patch_size=PATCH_SIZE, stride=STRIDE,
        resolution=RESOLUTION, units=UNITS,
        input_mask="otsu",
        min_mask_ratio=TISSUE_THRESH,
    )
    n_locs = len(extractor.locations_df)
    print(f"get_patch_extractor   : {time.time()-t0:.2f}s  ({n_locs} locations)", flush=True)

    if n_locs == 0:
        print("No tissue locations found — check slide or TISSUE_THRESH")
        return

    locs = extractor.locations_df[["x", "y"]].values.astype(int)

    # Time 20 patch reads
    n_test = min(20, n_locs)
    test_indices = random.sample(range(n_locs), n_test)

    t0 = time.time()
    patches = []
    for i in test_indices:
        patches.append(np.array(extractor[i]))
    elapsed = time.time() - t0
    per_patch = elapsed / n_test
    print(f"patch read ({n_test} patches) : {elapsed:.2f}s  ({per_patch:.3f}s/patch)", flush=True)
    print(f"  → extrapolated 1000 patches: {per_patch*1000:.1f}s  ({per_patch*1000/60:.1f} min)")

    # Time QC
    t0 = time.time()
    qc_pass = 0
    for p in patches:
        if p.mean() > MEAN_THRESH:              continue
        if tissue_fraction(p) < TISSUE_THRESH:  continue
        if laplacian_variance(p) < LAP_THRESH:  continue
        if is_pen_mark(p):                      continue
        qc_pass += 1
    elapsed = time.time() - t0
    print(f"QC ({n_test} patches)        : {elapsed:.2f}s  ({qc_pass}/{n_test} pass QC)", flush=True)

    # Time Macenko
    if _normalizer is not None:
        qc_patches = [p for p in patches
                      if p.mean() <= MEAN_THRESH
                      and tissue_fraction(p) >= TISSUE_THRESH][:5]
        if qc_patches:
            t0 = time.time()
            for p in qc_patches:
                try: _normalizer.transform(p.copy())
                except Exception: pass
            elapsed = time.time() - t0
            per_m = elapsed / len(qc_patches)
            print(f"Macenko ({len(qc_patches)} patches)    : {elapsed:.2f}s  ({per_m:.3f}s/patch)", flush=True)
            print(f"  → extrapolated 1000 patches: {per_m*1000:.1f}s  ({per_m*1000/60:.1f} min)")

    print(f"\nEstimated total per slide:")
    total = 5 + per_patch * 1000 + (per_m if _normalizer else 0) * 1000
    print(f"  ~{total:.0f}s  ({total/60:.1f} min)")
    print(f"  × 155 slides / {NUM_WORKERS} workers = ~{total*155/NUM_WORKERS/3600:.1f} hours")
    print(f"{'='*60}\n")


# Main

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not REF_PATCH_PATH.exists():
        raise FileNotFoundError(
            f"Reference patch not found: {REF_PATCH_PATH}\n"
            "Run stage0_macenko_fit.py first."
        )

    wsi_paths = sorted(
        p for p in WSI_DIR.rglob("*")
        if p.suffix.lower() in WSI_EXTENSIONS
    )
    logger.info(f"Found {len(wsi_paths)} WSIs")
    logger.info(f"Already done: {len(list(OUTPUT_DIR.glob('*.h5')))} (will skip)")

    logger.info(f"Launching Pool with {NUM_WORKERS} workers …")
    with Pool(
        processes=NUM_WORKERS,
        initializer=_init_worker,
        initargs=(str(REF_PATCH_PATH),),
    ) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_wsi, wsi_paths),
                total=len(wsi_paths),
                desc="Extracting patches",
            )
        )

    status_counts = {}
    total_patches = 0
    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
        total_patches += r.get("n_patches", 0)

    logger.info("─" * 60)
    logger.info("Extraction complete.")
    for status, count in sorted(status_counts.items()):
        logger.info(f"  {status:<28}: {count}")
    logger.info(f"  {'total patches':<28}: {total_patches:,}")
    logger.info("─" * 60)

    log_path = OUTPUT_DIR / "extraction_log.tsv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["slide_id", "status", "n_patches", "error"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Log saved → {log_path}")


if __name__ == "__main__":
    main()





