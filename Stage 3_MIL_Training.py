#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
stage3_mil_training
Supported backbones:
  - "dinov2_vitb14" : DINOv2 ViT-B/14  - 768-dim features
  - "resnet50"      : ResNet50         - 2048-dim features
  - "vgg19"         : VGG19             - 4096-dim features

This runs independently for each of three signatures: winter, west, buffa.
Pipeline:
  1. Loading the model
  2. Matching the slides to ssGSEA scores (winter/west/ buffa)
  3. StratifiedKFold (by score quartile, patient-level split)
  4. Per epoch: random tile subsample → gated attention MIL → regression
"""

import gc
import logging
import os
import random
import warnings
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Configuration

BACKBONE = "dinov2_vitb14"  

_BACKBONE_REGISTRY = {
    "dinov2_vitb14" : "dinov2vitb14",
    "resnet50"      : "resnet50",
    "vgg19"         : "vgg19",
}
if BACKBONE not in _BACKBONE_REGISTRY:
    raise ValueError(
        f"Unknown backbone '{BACKBONE}'. "
        f"Choose from: {list(_BACKBONE_REGISTRY.keys())}"
    )
_dir_suffix = _BACKBONE_REGISTRY[BACKBONE]


class Config:
    # Paths
    FEAT_DIR    = Path(f"2026_04_26_luad_224px_05mpp_features_dinov2vitb14/")  
    SCORES_CSV  = Path("Hypoxia_ssGSEA_scores_luad_tum_only.csv") 
    SAVE_DIR    = Path(f"outputs/2026_03_16_mil_results_{_dir_suffix}")

    SIGNATURES = ["composite_score", "winter_score", "west_score", "buffa_score"]

    MAX_TILES   = 512 

    # Model architecture
    PROJ_DIM    = 256       
    ATTN_DIM    = 128       
    DROPOUT_PROJ = 0.3
    DROPOUT_REG  = 0.2

    # Training
    N_FOLDS      = 5
    EPOCHS       = 60
    LR           = 1e-4
    WEIGHT_DECAY = 1e-5
    PATIENCE     = 12    
    SEED         = 42
    STRAT_BINS   = 4   
    BATCH_SIZE   = 1        
    NUM_WORKERS  = 0
    DEVICE = torch.device("cpu")


cfg = Config()

log.info(f"Backbone   : {BACKBONE}")
log.info(f"Feat dir   : {cfg.FEAT_DIR}")
log.info(f"Save dir   : {cfg.SAVE_DIR}")

random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)


# Utilities

def normalise_tcga_id(raw: str) -> str:
    s = str(raw).strip()
    for ext in (".svs", ".ndpi", ".tif", ".tiff", ".mrxs"):
        if s.lower().endswith(ext):
            s = s[:-len(ext)]
    # Handle both dot and dash separators
    s = s.replace("_", "-").upper()
    # Normalise to dash-separated
    parts = s.replace(".", "-").split("-")
    # Return first 3 parts: TCGA-XX-XXXX
    return "-".join(parts[:3]) if len(parts) >= 3 else s


# Data loading

def load_scores(csv_path: Path, signature: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=None, engine="python", dtype=str)

    id_col = next(
        (c for c in df.columns
         if c.lower() in ("slide_id", "sample_id", "id")
         or "sample" in c.lower()),
        df.columns[0],
    )

    if signature not in df.columns:
        available = [c for c in df.columns if c != id_col]
        raise ValueError(
            f"Signature '{signature}' not found.\n"
            f"Available: {available}"
        )

    df = df[[id_col, signature]].copy()
    df.columns = ["raw_id", "score"]
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"]).reset_index(drop=True)

    def is_primary_tumour(raw_id):
        parts = str(raw_id).replace(".", "-").split("-")
        return parts[3].startswith("01") if len(parts) >= 4 else True

    n_before = len(df)
    df = df[df["raw_id"].apply(is_primary_tumour)].copy()
    log.info(f"[{signature}] Tumour filter: {n_before} → {len(df)} samples")

    df["patient_id"] = df["raw_id"].apply(normalise_tcga_id)

    df = df.drop_duplicates(subset="patient_id", keep="first")

    log.info(
        f"[{signature}] {len(df)} unique tumour samples | "
        f"range=[{df.score.min():.3f}, {df.score.max():.3f}] | "
        f"mean={df.score.mean():.3f}"
    )
    return df[["patient_id", "score"]]


def build_manifest(
    feat_dir  : Path,
    scores_df : pd.DataFrame,
    signature : str,
) -> tuple[pd.DataFrame, int]:

    records = []
    for npy_path in sorted(feat_dir.glob("*.npy")):
        if "_coords" in npy_path.stem:
            continue   # skip coordinate files
        slide_id = npy_path.stem
        pid      = normalise_tcga_id(slide_id)
        try:
            arr    = np.load(npy_path, mmap_mode="r")
            n_tiles = arr.shape[0]
        except Exception as e:
            log.warning(f"Cannot read {npy_path}: {e}")
            continue
        records.append({
            "patient_id": pid,
            "npy_path"  : str(npy_path),
            "n_tiles"   : n_tiles,
        })

    feat_df = pd.DataFrame(records)
    log.info(f"[{signature}] Found {len(feat_df)} .npy files")

    merged = scores_df.merge(feat_df, on="patient_id", how="inner")
    log.info(
        f"[{signature}] Matched {len(merged)} / {len(scores_df)} slides"
    )

    if len(merged) == 0:
        raise RuntimeError(
            f"No .npy files matched scores for signature '{signature}'.\n"
            "Check FEAT_DIR and SCORES_CSV patient ID formats."
        )
    if merged["patient_id"].duplicated().any():
        n_before = len(merged)
        merged = (
            merged
            .sort_values("score", key=abs, ascending=False)
            .drop_duplicates(subset="patient_id", keep="first")
            .reset_index(drop=True)
        )
        log.info(f"[{signature}] Deduped {n_before} → {len(merged)} patients")
    
    MIN_PATCHES = 100
    n_before = len(merged)
    merged = merged[merged["n_tiles"] >= MIN_PATCHES].copy()
    if len(merged) < n_before:
        log.info(
            f"[{signature}] Excluded {n_before - len(merged)} slides "
            f"with < {MIN_PATCHES} patches"
        )
  
    sample = np.load(merged["npy_path"].iloc[0], mmap_mode="r")
    feat_dim = sample.shape[1]
    log.info(f"[{signature}] Feature dim = {feat_dim}")

    return merged.reset_index(drop=True), feat_dim


# Dataset

class FeatureSlideDataset(Dataset):

    def __init__(self, manifest: pd.DataFrame, max_tiles: int = None):
        self.df        = manifest.reset_index(drop=True)
        self.max_tiles = max_tiles or cfg.MAX_TILES

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        score = float(row["score"])
        pid   = str(row["patient_id"])

        arr = np.load(row["npy_path"], mmap_mode="r")  # [N, D]
        N   = arr.shape[0]

        if N > self.max_tiles:
            sel = np.random.choice(N, self.max_tiles, replace=False)
            sel.sort()
            arr = arr[sel]

        features = torch.from_numpy(arr.copy()).float()
        label    = torch.tensor(score, dtype=torch.float32)
        return features, label, pid


def mil_collate(batch):
    """Custom collate: each slide has different number of patches."""
    feat_bags, labels, pids = zip(*batch)
    return list(feat_bags), torch.stack(labels), list(pids)


# Model
class GatedAttentionPooling(nn.Module):
    """
    Gated attention MIL pooling.
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.V = nn.Linear(in_dim, hidden_dim)   # tanh branch
        self.U = nn.Linear(in_dim, hidden_dim)   # sigmoid branch
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: torch.Tensor):
        gate      = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))  
        raw       = self.w(gate)                                
        attn      = torch.softmax(raw, dim=0)                       
        slide_emb = (attn * h).sum(dim=0, keepdim=True)              
        return slide_emb, attn.squeeze(1)


class MILRegressor(nn.Module):
    """
    MIL regression on pre-extracted DINOv2 features.
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, cfg.PROJ_DIM),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT_PROJ),
        )
        self.attention = GatedAttentionPooling(cfg.PROJ_DIM, cfg.ATTN_DIM)
        self.regressor = nn.Sequential(
            nn.Linear(cfg.PROJ_DIM, 64),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT_REG),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor):
        h         = self.projection(features)          
        emb, attn = self.attention(h)                  
        pred      = self.regressor(emb).squeeze()      
        return pred, attn


# Metrics

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse       = mean_squared_error(y_true, y_pred)
    r2        = r2_score(y_true, y_pred)
    pear, pp  = pearsonr(y_true, y_pred)
    spea, sp  = spearmanr(y_true, y_pred)
    return {
        "MSE"        : round(float(mse),  4),
        "R2"         : round(float(r2),   4),
        "Pearson_r"  : round(float(pear), 4),
        "Pearson_p"  : round(float(pp),   6),
        "Spearman_r" : round(float(spea), 4),
        "Spearman_p" : round(float(sp),   6),
    }


# Training loop

def run_epoch(model, loader, optimizer, criterion, train: bool):
    model.train(train)
    total_loss = 0.0
    all_true, all_pred = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for feat_bags, labels, _ in loader:
            labels = labels.to(cfg.DEVICE)
            preds  = torch.stack([model(bag.to(cfg.DEVICE))[0] for bag in feat_bags])
            loss   = criterion(preds, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            all_true.extend(labels.detach().cpu().numpy())
            all_pred.extend(preds.detach().cpu().numpy())

    return total_loss / max(len(loader), 1), np.array(all_true), np.array(all_pred)


def train_fold(fold_idx, train_df, val_df, feat_dim, signature, save_dir):
    log.info(f"\n{'─'*60}")
    log.info(f"[{signature}] Fold {fold_idx+1}/{cfg.N_FOLDS}  "
             f"train={len(train_df)}  val={len(val_df)}")

    train_loader = DataLoader(
        FeatureSlideDataset(train_df), batch_size=cfg.BATCH_SIZE,
        shuffle=True, num_workers=cfg.NUM_WORKERS, collate_fn=mil_collate,
    )
    val_loader = DataLoader(
        FeatureSlideDataset(val_df), batch_size=cfg.BATCH_SIZE,
        shuffle=False, num_workers=cfg.NUM_WORKERS, collate_fn=mil_collate,
    )

    model     = MILRegressor(feat_dim).to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR,
                                 weight_decay=cfg.WEIGHT_DECAY)
    criterion = nn.SmoothL1Loss(beta=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=1e-6
    )

    best_loss, patience_count, best_state = float("inf"), 0, None
    history = {"train_loss": [], "val_loss": []}

    ckpt_path = save_dir / f"{signature}_fold{fold_idx+1}_best.pt"

    for epoch in range(1, cfg.EPOCHS + 1):
        tr_loss, _, _              = run_epoch(model, train_loader, optimizer, criterion, True)
        val_loss, y_true, y_pred   = run_epoch(model, val_loader,   optimizer, criterion, False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        m = compute_metrics(y_true, y_pred)

        log.info(
            f"  Ep {epoch:3d}  train={tr_loss:.4f}  val={val_loss:.4f}  "
            f"r={m['Pearson_r']:.3f}  ρ={m['Spearman_r']:.3f}  R²={m['R2']:.3f}"
        )

        if val_loss < best_loss - 1e-5:
            best_loss      = val_loss
            best_state     = deepcopy(model.state_dict())
            patience_count = 0
            torch.save(
                {"epoch": epoch, "model": best_state,
                 "val_loss": best_loss, "feat_dim": feat_dim},
                ckpt_path,
            )
        else:
            patience_count += 1
            if patience_count >= cfg.PATIENCE:
                log.info(f"  Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    _, y_true_f, y_pred_f = run_epoch(
        model, val_loader, optimizer, criterion, False
    )
    final_metrics = compute_metrics(y_true_f, y_pred_f)
    log.info(
        f"  Best: r={final_metrics['Pearson_r']:.4f}  "
        f"ρ={final_metrics['Spearman_r']:.4f}  "
        f"R²={final_metrics['R2']:.4f}"
    )

    del model
    gc.collect()

    return {
        "fold"   : fold_idx + 1,
        "metrics": final_metrics,
        "y_true" : y_true_f,
        "y_pred" : y_pred_f,
        "pids"   : val_df["patient_id"].tolist(),
        "history": history,
    }


# Cross-validation

def run_cv(manifest, feat_dim, signature, save_dir):
    patients = manifest["patient_id"].unique()
    pat_df   = (
        manifest[["patient_id", "score"]]
        .drop_duplicates("patient_id")
        .set_index("patient_id")
        .loc[patients]
        .reset_index()
    )

    kbd = KBinsDiscretizer(
        n_bins=cfg.STRAT_BINS, encode="ordinal", strategy="quantile"
    )
    strat = kbd.fit_transform(
        pat_df["score"].values.reshape(-1, 1)
    ).ravel().astype(int)

    skf     = StratifiedKFold(cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    results = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(patients, strat)):
        tr_pats  = patients[tr_idx]
        val_pats = patients[val_idx]
        train_df = manifest[manifest["patient_id"].isin(tr_pats)].copy()
        val_df   = manifest[manifest["patient_id"].isin(val_pats)].copy()
        results.append(
            train_fold(fold_idx, train_df, val_df, feat_dim, signature, save_dir)
        )

    return results


# Visualisation

def plot_scatter(y_true, y_pred, fold, metrics, signature, save_dir):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, alpha=0.65, edgecolors="k",
               linewidths=0.4, s=50, color="#4C72B0", zorder=3)

    m, b   = np.polyfit(y_true, y_pred, 1)
    x_line = np.linspace(y_true.min(), y_true.max(), 200)
    ax.plot(x_line, m * x_line + b, color="firebrick", lw=2.0, label="Fit")

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1.2, alpha=0.7, label="y=x")

    ax.set_xlabel(f"Actual {signature} ssGSEA score", fontsize=12)
    ax.set_ylabel("Predicted score", fontsize=12)
    ax.set_title(
        f"{signature} · Fold {fold}  |  "
        f"r={metrics['Pearson_r']:.3f}  ρ={metrics['Spearman_r']:.3f}  "
        f"R²={metrics['R2']:.3f}",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = save_dir / f"{signature}_scatter_fold{fold}.png"
    plt.savefig(path, dpi=150)
    plt.close()


def plot_loss_curve(history, fold, signature, save_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs  = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train", lw=1.8)
    ax.plot(epochs, history["val_loss"],   label="Val",   lw=1.8, ls="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SmoothL1 Loss")
    ax.set_title(f"{signature} · Loss Curve Fold {fold}")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = save_dir / f"{signature}_loss_fold{fold}.png"
    plt.savefig(path, dpi=150)
    plt.close()


def plot_summary(fold_results, signature, save_dir):
    keys = ["Pearson_r", "Spearman_r", "R2", "MSE"]
    data = {k: [r["metrics"][k] for r in fold_results] for k in keys}

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, key in zip(axes, keys):
        vals = data[key]
        ax.boxplot(vals, widths=0.45, patch_artist=True,
                   boxprops=dict(facecolor="#AED6F1", color="steelblue"),
                   medianprops=dict(color="firebrick", lw=2.0))
        ax.scatter([1] * len(vals), vals, color="steelblue", zorder=5, s=30)
        ax.axhline(np.mean(vals), color="firebrick", ls=":", lw=1.5,
                   label=f"μ={np.mean(vals):.3f}")
        ax.set_title(key)
        ax.set_xticks([])
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle(f"{signature} — {len(fold_results)}-Fold CV Summary", fontsize=12)
    plt.tight_layout()
    path = save_dir / f"{signature}_summary_boxplots.png"
    plt.savefig(path, dpi=150)
    plt.close()


# Save results

def save_results(fold_results, signature, save_dir):
    
    metrics_df = pd.DataFrame([
        {"fold": r["fold"], **r["metrics"]} for r in fold_results
    ])
    metrics_df.to_csv(save_dir / f"{signature}_fold_metrics.csv", index=False)

    rows = []
    for r in fold_results:
        for pid, yt, yp in zip(r["pids"], r["y_true"], r["y_pred"]):
            rows.append({
                "patient_id": pid, "fold": r["fold"],
                "y_true": yt, "y_pred": yp, "signature": signature,
            })
    oof_df = pd.DataFrame(rows)
    oof_df.to_csv(save_dir / f"{signature}_oof_predictions.csv", index=False)

    return metrics_df, oof_df


def print_summary(metrics_df, oof_df, signature):
    print(f"\n{'═'*64}")
    print(f"  {signature.upper()} — CROSS-VALIDATION SUMMARY")
    print(f"{'═'*64}")
    print(metrics_df.to_string(index=False))
    print(f"\n  Mean ± Std across {len(metrics_df)} folds:")
    for col in ["Pearson_r", "Spearman_r", "R2", "MSE"]:
        v = metrics_df[col].values
        print(f"    {col:<14}  {v.mean():.4f} ± {v.std():.4f}")

    
    overall = compute_metrics(oof_df["y_true"].values, oof_df["y_pred"].values)
    print(f"\n  Pooled OOF (all slides combined):")
    for k, v in overall.items():
        if "p" not in k.lower():
            print(f"    {k:<14}  {v:.4f}")
    print(f"{'═'*64}")


# Main

def main():
    cfg.SAVE_DIR.mkdir(parents=True, exist_ok=True)

    npy_files = [f for f in cfg.FEAT_DIR.glob("*.npy")
              if "_coords" not in f.stem]
    log.info(f"Found {len(npy_files)} feature files in {cfg.FEAT_DIR}")

    if not npy_files:
        log.error("No .npy files found. Run stage2_extract_features.py first.")
        return

    all_signature_results = {}

    for signature in cfg.SIGNATURES:
        log.info(f"\n{'='*64}")
        log.info(f"  SIGNATURE: {signature.upper()}")
        log.info(f"{'='*64}")

        sig_dir = cfg.SAVE_DIR / signature
        sig_dir.mkdir(exist_ok=True)

        scores_df         = load_scores(cfg.SCORES_CSV, signature)
        manifest, feat_dim = build_manifest(cfg.FEAT_DIR, scores_df, signature)

        log.info(
            f"  Slides    : {len(manifest)}\n"
            f"  Feat dim  : {feat_dim}\n"
            f"  MAX_TILES : {cfg.MAX_TILES}\n"
            f"  Score range: [{manifest.score.min():.3f}, {manifest.score.max():.3f}]"
        )

        manifest.to_csv(sig_dir / "manifest.csv", index=False)

        fold_results = run_cv(manifest, feat_dim, signature, sig_dir)

        for r in fold_results:
            plot_scatter(r["y_true"], r["y_pred"], r["fold"],
                        r["metrics"], signature, sig_dir)
            plot_loss_curve(r["history"], r["fold"], signature, sig_dir)
        plot_summary(fold_results, signature, sig_dir)

        metrics_df, oof_df = save_results(fold_results, signature, sig_dir)
        print_summary(metrics_df, oof_df, signature)

        all_signature_results[signature] = {
            "metrics_df": metrics_df,
            "oof_df"    : oof_df,
        }

    # Cross-signature comparison
    print(f"\n{'═'*64}")
    print("  CROSS-SIGNATURE COMPARISON (mean Pearson r / Spearman ρ)")
    print(f"{'═'*64}")
    print(f"  {'Signature':<10}  {'Pearson r':>10}  {'Spearman ρ':>10}  {'R²':>8}")
    print(f"  {'─'*9}  {'─'*10}  {'─'*10}  {'─'*8}")
    for sig, res in all_signature_results.items():
        m = res["metrics_df"]
        print(
            f"  {sig:<10}  "
            f"{m['Pearson_r'].mean():>10.4f}  "
            f"{m['Spearman_r'].mean():>10.4f}  "
            f"{m['R2'].mean():>8.4f}"
        )
    print(f"{'═'*64}")
    print(f"\n All outputs saved under: {cfg.SAVE_DIR}\n")


if __name__ == "__main__":
    main()

