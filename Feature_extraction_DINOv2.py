#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
stage2: Feature Extraction using DINOv2
Extracts DINOv2 ViT-B/14 feature embeddings for each patch.

  - Backbone: DINOv2 ViT-B/14 (vit_base_patch14_dinov2.lvd142m) via timm
  - Feature dim: 768 (mean-pooled over 256 patch tokens)
  - Input: 224×224 patches → 16×16 = 256 tokens
"""

import logging
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Configuration
H5_DIR     = Path("/media/vvlab/Expansion/hypoxia/2026_03_27_hypoxia_redone/2026_03_23_224px_05mpp_patches_h5")
FEAT_DIR   = Path("/media/vvlab/Expansion/hypoxia/2026_03_27_hypoxia_redone/outputs/2026_04_22_224px_05mpp_new_features")
FEAT_DIM   = 768
BATCH_SIZE = 128
NUM_THREADS = 32

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

torch.set_num_threads(NUM_THREADS)
logger.info(f"PyTorch using {torch.get_num_threads()} CPU threads")
logger.info(f"CUDA available : {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device         : {DEVICE}")
logger.info(f"Feature dim    : {FEAT_DIM}")
logger.info(f"Output dir     : {FEAT_DIR}")


# Dataset
class PatchDataset(Dataset):
    _transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    def __init__(self, patches: np.ndarray):
        self.patches = patches   # (N, 224, 224, 3) uint8

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._transform(self.patches[idx])


# DINOv2 Extractor

class DINOv2Extractor(nn.Module):

    def __init__(self):
        super().__init__()
        model_name = "vit_base_patch14_dinov2.lvd142m"
        logger.info(f"Loading {model_name} from timm ...")
        t0 = time.time()
        self.vit = timm.create_model(
            model_name,
            pretrained       = True,
            num_classes      = 0,
            dynamic_img_size = True,
        )
        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad = False
        n_params = sum(p.numel() for p in self.vit.parameters()) / 1e6
        logger.info(
            f"Loaded {model_name} in {time.time()-t0:.1f}s  |  "
            f"params={n_params:.0f}M  |  embed_dim={self.vit.embed_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.vit.forward_features(x)

        if isinstance(out, dict):
            tokens = out["x"]       
        elif out.ndim == 2:
            raise RuntimeError(
                "forward_features returned, timm already pooled internally. "
                "Upgrade timm instead."
            )
        else:
            tokens = out

        tokens = tokens[:, 1:, :]
        return tokens.mean(dim=1)


# Feature extraction

def extract_features_one_slide(
    h5_path   : Path,
    model     : nn.Module,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        patches = f["patches"][:]   
        coords  = f["coords"][:]   

    loader = DataLoader(
        PatchDataset(patches),
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = DEVICE.type == "cuda",
    )

    chunks = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch.to(DEVICE))
            chunks.append(out.cpu().numpy())

    features = np.concatenate(chunks, axis=0).astype(np.float32) 
    return features, coords


# Main

def main():
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(H5_DIR.glob("*.h5"))
    logger.info(f"Found {len(h5_files)} HDF5 patch files in {H5_DIR}")

    if not h5_files:
        logger.error("No .h5 files found. Run stage1_extract_patches.py first.")
        return

    already_done = len(list(FEAT_DIR.glob("*.npy")))
    logger.info(f"Already extracted: {already_done} (will skip)")

    model = DINOv2Extractor().to(DEVICE)

    records    = []
    total_time = 0.0

    for h5_path in tqdm(h5_files, desc="DINOv2 feature extraction"):
        slide_id = h5_path.stem
        npy_path = FEAT_DIR / f"{slide_id}.npy"

        if npy_path.exists():
            existing = np.load(npy_path, mmap_mode="r")
            logger.info(f"[{slide_id[:50]}] SKIP ({existing.shape[0]} patches cached)")
            records.append({
                "slide_id" : slide_id,
                "npy_path" : str(npy_path),
                "n_patches": existing.shape[0],
                "feat_dim" : existing.shape[1],
                "status"   : "cached",
            })
            continue

        t0 = time.time()
        try:
            features, coords = extract_features_one_slide(h5_path, model, BATCH_SIZE)

            assert features.shape[1] == FEAT_DIM, (
                f"Expected {FEAT_DIM}-dim features, got {features.shape[1]}"
            )

            np.save(npy_path, features)
            np.save(FEAT_DIR / f"{slide_id}_coords.npy", coords)

            elapsed     = time.time() - t0
            total_time += elapsed

            logger.info(
                f"[{slide_id[:50]}]  "
                f"({features.shape[0]}, {features.shape[1]})  [{elapsed:.1f}s]"
            )
            records.append({
                "slide_id" : slide_id,
                "npy_path" : str(npy_path),
                "n_patches": features.shape[0],
                "feat_dim" : features.shape[1],
                "status"   : "success",
            })

        except Exception as e:
            logger.error(f"[{slide_id[:50]}] FAILED: {e}", exc_info=True)
            if npy_path.exists():
                npy_path.unlink()
            records.append({
                "slide_id" : slide_id,
                "npy_path" : str(npy_path),
                "n_patches": 0,
                "feat_dim" : 0,
                "status"   : f"error: {e}",
            })

    manifest_path = FEAT_DIR / "extraction_manifest.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)
    logger.info(f"Manifest saved → {manifest_path}")

    df            = pd.DataFrame(records)
    n_success     = (df["status"].isin(["success", "cached"])).sum()
    n_failed      = (df["status"].str.startswith("error")).sum()
    total_patches = df["n_patches"].sum()

    logger.info("─" * 60)
    logger.info("Feature extraction complete.")
    logger.info(f"  Device         : {DEVICE}")
    logger.info(f"  Success        : {n_success}")
    logger.info(f"  Failed         : {n_failed}")
    logger.info(f"  Total patches  : {total_patches:,}")
    logger.info(f"  Feature dim    : {FEAT_DIM}")
    logger.info(f"  Total time     : {total_time/60:.1f} min")
    logger.info(f"  Features saved : {FEAT_DIR}")
    logger.info("─" * 60)


if __name__ == "__main__":
    main()

