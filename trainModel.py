from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torch
import torch.nn as nn
import torch.distributed as dist
from models import *
from datasets import *
import platform
from metrics1 import *
import copy
import numpy as np
from torch.utils.data import Dataset, IterableDataset, DataLoader
from milesial_unet_model import UNet1, APAU_Net
from leejunhyun_unet_models import U_Net, R2U_Net, AttU_Net, R2AttU_Net, AttU_Net_S
from CellularAutomataPostprocessing import UNet1_CA
from CellularAutomataAllfeaturespostporcessing import U2Net_CA
# from transformerCA import TransformerCA_Seg
import pickle
import random
import os
import json

# ---------- NEW: plotting imports (no seaborn) ----------
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt
from matplotlib.patches import Patch  # legend handles

# --------- Defaults (no CLI needed) ----------
MC_PASSES = int(os.environ.get("MC_DROPOUT_PASSES", "30"))
LAMBDA_FN = 1

# ---------- helpers to avoid NumPy scalars in checkpoints + safe loading ----------
def to_py_float(x):
    """Coerce tensors/NumPy scalars/py floats to native Python float."""
    import numpy as _np
    import torch as _torch
    if isinstance(x, _torch.Tensor):
        return float(x.item())
    if isinstance(x, (_np.floating,)):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    # last resort
    try:
        return float(x)
    except Exception:
        return float(_np.asarray(x).item())

def safe_load_checkpoint(path, map_location="cpu"):
    """
    Robust loader for PyTorch 2.6+:
    - Try default (weights_only=True)
    - If blocked by NumPy scalar, allow-list it and retry
    - As a last resort (ONLY if you trust the file), disable weights_only
    """
    try:
        return torch.load(path, map_location=map_location)  # default weights_only=True in 2.6
    except pickle.UnpicklingError:
        # allow-list numpy scalar used by older saves
        import numpy as _np
        from torch.serialization import safe_globals
        with safe_globals([_np._core.multiarray.scalar]):
            return torch.load(path, map_location=map_location)
    except TypeError:
        # fallback (trusted source only)
        return torch.load(path, map_location=map_location, weights_only=False)
# -------------------------------------------------------------------------------

# -------------------- Bayesian (MC) utilities — default, no CLI needed -------------------- #
@torch.no_grad()
def _enable_mc_dropout(model: torch.nn.Module):
    """Enable dropout at inference time by setting only Dropout layers to train()."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.train()

def _model_has_dropout(model: torch.nn.Module) -> bool:
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            return True
    return False

@torch.no_grad()
def mc_dropout_collect_mean_var(model, dataloader, n_passes=MC_PASSES):
    """
    Run MC passes with stochasticity:
      - If the model has Dropout, enable it (MC Dropout).
      - Otherwise, add small Gaussian noise to inputs each pass.

    Returns:
      mean_probs: Tensor [N, 1, H, W]
      var_probs:  Tensor [N, 1, H, W]
      all_labels: Tensor [N, 1, H, W]
    """
    model.eval()

    has_do = _model_has_dropout(model)
    if has_do:
        _enable_mc_dropout(model)
        input_noise_sigma = 0.0  # pure MC Dropout
        print("[Bayesian Eval] Using MC Dropout (dropout layers found).")
    else:
        input_noise_sigma = 0.03  # tiny input noise (for models without dropout)
        print(f"[Bayesian Eval] No dropout found; using input Gaussian noise sigma={input_noise_sigma}.")

    pass_sums = None
    pass_sq_sums = None
    all_labels_list = []

    for pass_idx in range(n_passes):
        probs_pass = []
        batch_labels = []

        for images, labels in dataloader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            if input_noise_sigma > 0.0:
                images = images + torch.randn_like(images) * input_noise_sigma

            logits = model(images)
            probs = torch.sigmoid(logits)  # [B,1,H,W]
            probs_pass.append(probs)

            if pass_idx == 0:
                batch_labels.append(labels)

        probs_pass = torch.cat(probs_pass, dim=0)  # [N,1,H,W]

        if pass_idx == 0:
            pass_sums = probs_pass.clone()
            pass_sq_sums = probs_pass.pow(2)
            all_labels_list = torch.cat(batch_labels, dim=0) if batch_labels else None
        else:
            pass_sums.add_(probs_pass)
            pass_sq_sums.add_(probs_pass.pow(2))

    mean_probs = pass_sums / float(n_passes)
    var_probs = pass_sq_sums / float(n_passes) - mean_probs.pow(2)
    return mean_probs, var_probs, all_labels_list

# >>> PAPER RULE (Bayesian): choose t maximizing the *median* Dice across images
@torch.no_grad()
def find_best_threshold_from_probs_max_median_dice(mean_probs, all_labels, thresholds=np.linspace(0.001, 0.99, 120)):
    """
    Choose threshold that maximizes the median Dice across images (paper's rule).
    Operates on precomputed probability maps.
    """
    N = all_labels.size(0)
    best_t, best_med = 0.5, -1.0
    for t in thresholds:
        bins = (mean_probs > t).float()
        dices = []
        for i in range(N):
            dices.append(to_py_float(dice_score(all_labels[i], bins[i])))
        med = float(np.median(dices)) if len(dices) else 0.0
        if med > best_med:
            best_med = med
            best_t = float(t)
    return best_t, best_med

def find_best_threshold_from_probs(mean_probs, all_labels, thresholds=np.linspace(0.001, 0.99, 120), lambda_fn=LAMBDA_FN):
    """
    Global risk minimisation on precomputed probability maps.
    """
    y = all_labels.view(-1)
    best = {"thresh": 0.5, "risk": float("inf"), "dice": -1.0, "prec": 0.0, "rec": 0.0}
    with torch.no_grad():
        for t in thresholds:
            p = (mean_probs > t).float().view(-1)

            tp = torch.sum((p == 1) & (y == 1)).item()
            fp = torch.sum((p == 1) & (y == 0)).item()
            fn = torch.sum((p == 0) & (y == 1)).item()
            tn = torch.sum((p == 0) & (y == 0)).item()

            pos = tp + fn
            neg = fp + tn
            fn_rate = 0.0 if pos == 0 else (fn / pos)
            fp_rate = 0.0 if neg == 0 else (fp / neg)

            prec = 0.0 if (tp + fp) == 0 else (tp / (tp + fp))
            rec  = 0.0 if (tp + fn) == 0 else (tp / (tp + fn))
            denom = (2*tp + fp + fn)
            dice = 0.0 if denom == 0 else (2*tp / denom)
            risk = lambda_fn * fn_rate + fp_rate

            if risk < best["risk"]:
                best = {"thresh": float(t), "risk": float(risk), "dice": float(dice),
                        "prec": float(prec), "rec": float(rec)}
    return best["thresh"], best["dice"]

def compute_metrics_from_probs(mean_probs, all_labels, threshold=0.5):
    """
    Compute aggregate metrics from precomputed probability maps.
    Returns: (avg_loss, avg_iou, avg_accuracy, avg_f1, avg_auc, avg_dice, avg_precision, avg_recall)
    Note: 'loss' here is set to 0.0 (no criterion on probs here).
    """
    preds = (mean_probs > threshold).float()
    total_iou = 0.0
    total_accuracy = 0.0
    total_f1 = 0.0
    total_auc = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0

    with torch.no_grad():
        for i in range(all_labels.size(0)):
            y = all_labels[i]
            p_bin = preds[i]
            p_prob = mean_probs[i]

            total_iou += to_py_float(mean_iou(y, p_bin))
            total_accuracy += to_py_float(accuracy(y, p_bin))
            total_f1 += to_py_float(f1_score(y, p_bin))
            total_auc += to_py_float(auc_score(y, p_prob))
            total_dice += to_py_float(dice_score(y, p_bin))
            pr, rc = precision_recall(y, p_bin)
            total_precision += to_py_float(pr)
            total_recall += to_py_float(rc)

    n = float(all_labels.size(0))
    avg_loss = 0.0
    return (
        float(avg_loss),
        float(total_iou / n),
        float(total_accuracy / n),
        float(total_f1 / n),
        float(total_auc / n),
        float(total_dice / n),
        float(total_precision / n),
        float(total_recall / n),
    )
# ------------------------------------------------------------------------ #

# =================== NEW: plotting helpers (non-invasive) =================== #
PREV_FIRE_IDX = 7  # in your feature order

@torch.no_grad()
def _collect_fire_samples(model, loader, threshold=0.5, max_cols=10):
    """Grab up to max_cols samples (Prev, GT, Pred) from the first few batches."""
    model.eval()
    prev_list, gt_list, pred_list = [], [], []
    for images, labels in loader:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        prev = images[:, PREV_FIRE_IDX]  # [B,H,W]
        prev = (prev > 0.5).float()

        for i in range(images.size(0)):
            prev_list.append(prev[i].detach().cpu().numpy())
            gt_list.append(labels[i, 0].detach().cpu().numpy())
            pred_list.append(preds[i, 0].detach().cpu().numpy())
            if len(prev_list) >= max_cols:
                return prev_list, gt_list, pred_list
    return prev_list, gt_list, pred_list

def _comparison_rgb(gt, pred):
    """RGB overlay: TP→green, FP→red, FN→blue; TN/background→black."""
    gt = (gt > 0.5).astype(np.uint8)
    pred = (pred > 0.5).astype(np.uint8)
    tp = (gt == 1) & (pred == 1)
    fp = (gt == 0) & (pred == 1)
    fn = (gt == 1) & (pred == 0)

    H, W = gt.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[..., 1] = tp.astype(np.float32)  # green
    rgb[..., 0] = fp.astype(np.float32)  # red
    rgb[..., 2] = fn.astype(np.float32)  # blue
    return rgb

def save_fire_grid(prev_list, gt_list, pred_list, save_path, title=None):
    """Save a 4xN grid: Prev | GT | Pred | Comparison (rows) with a bottom legend."""
    assert len(prev_list) == len(gt_list) == len(pred_list)
    N = len(prev_list)
    if N == 0:
        print(f"[Plot] No samples to plot at {save_path}")
        return

    fig = plt.figure(figsize=(1.8 * N, 7.6))
    if title:
        fig.suptitle(title, y=0.98, fontsize=12)

    rows = 4
    last_comp_ax = None
    for c in range(N):
        ax = plt.subplot(rows, N, c + 1)
        ax.imshow(prev_list[c], cmap="gray", interpolation="nearest")
        if c == 0: ax.set_ylabel("Previous\nfire mask", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        ax = plt.subplot(rows, N, N + c + 1)
        ax.imshow(gt_list[c], cmap="gray", interpolation="nearest")
        if c == 0: ax.set_ylabel("True\nfire mask", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        ax = plt.subplot(rows, N, 2*N + c + 1)
        ax.imshow(pred_list[c], cmap="gray", interpolation="nearest")
        if c == 0: ax.set_ylabel("Predicted\nfire mask", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        comp = _comparison_rgb(gt_list[c], pred_list[c])
        ax = plt.subplot(rows, N, 3*N + c + 1)
        ax.imshow(comp, interpolation="nearest")
        if c == 0: ax.set_ylabel("Prediction\ncomparison", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        last_comp_ax = ax  # save for legend

    # Existing in-plot legend (kept, but you can delete this block if you don't want it)
    if last_comp_ax is not None:
        legend_elements = [
            Patch(facecolor='green', edgecolor='green', label='TP (green)'),
            Patch(facecolor='red', edgecolor='red', label='FP (red)'),
            Patch(facecolor='blue', edgecolor='blue', label='FN (blue)'),
            Patch(facecolor='black', edgecolor='black', label='TN / background (black)'),
        ]
        last_comp_ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.6)

    # NEW: bottom, centered legend strip like the paper figure
    bottom_legend = [
        Patch(facecolor='green', edgecolor='green', label='True Positive'),
        Patch(facecolor='red', edgecolor='red', label='False Positive'),
        Patch(facecolor='blue', edgecolor='blue', label='False Negative'),
        Patch(facecolor='black', edgecolor='black', label='No Fire'),
    ]
    # Give a little extra bottom space for the legend bar
    plt.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.legend(
        handles=bottom_legend,
        loc='lower center',
        ncol=4,
        frameon=True,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01)
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] Saved fire grid → {save_path}")


def save_training_curves(train_loss_hist, val_metrics_hist, save_dir):
    """
    Save PNGs:
      - loss_curves.png: train loss vs epoch, val loss vs epoch
      - f1_dice_curves.png: validation F1 and Dice vs epoch
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = np.arange(1, len(train_loss_hist) + 1)

    if len(val_metrics_hist) > 0:
        val_loss = np.array([m[0] for m in val_metrics_hist], dtype=float)
        val_f1   = np.array([m[3] for m in val_metrics_hist], dtype=float)
        val_dice = np.array([m[5] for m in val_metrics_hist], dtype=float)
    else:
        val_loss = np.zeros_like(epochs, dtype=float)
        val_f1   = np.zeros_like(epochs, dtype=float)
        val_dice = np.zeros_like(epochs, dtype=float)

    fig1 = plt.figure(figsize=(7, 4.2))
    plt.plot(epochs, train_loss_hist, label="Train loss")
    plt.plot(epochs, val_loss, label="Val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training/Validation Loss")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    path1 = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(path1, dpi=200); plt.close(fig1)
    print(f"[Plot] Saved {path1}")

    fig2 = plt.figure(figsize=(7, 4.2))
    plt.plot(epochs, val_f1, label="Val F1")
    plt.plot(epochs, val_dice, label="Val Dice")
    plt.xlabel("Epoch"); plt.ylabel("Score"); plt.title("Validation F1 & Dice")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    path2 = os.path.join(save_dir, "f1_dice_curves.png")
    plt.savefig(path2, dpi=200); plt.close(fig2)
    print(f"[Plot] Saved {path2}")
# ============================================================================ #

TRAIN = 'train'
VAL = 'validation'
MASTER_RANK = 0
SAVE_INTERVAL = 1

DATASET_PATH = 'data/next-day-wildfire-spread'
SAVE_MODEL_PATH = 'savedModels'

# loss_functions = ['WBCE+DICE', 'FOCAL', 'FOCAL+DICE']
loss_functions = ['FOCAL+DICE']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--master', default='sardine', help='master node')
    parser.add_argument('-p', '--port', default='30437', help='master node')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    args = parser.parse_args()
    print(f'initializing training on single GPU')

    for loss_name in loss_functions:
        print(f"\n====== Training with loss: {loss_name} ======")
        print("******************************************************************************")
        train(0, args, loss_name)  # uses default MC_PASSES
        print(f"====== Finished training with loss: {loss_name} ======\n")
        print("******************************************************************************")

def create_data_loaders(rank, gpu, world_size, selected_features=None):
    batch_size = 64

    ALL_FEATURES = [
        'elevation', 'fws', 'population', 'pdsi', 'pr', 'sph', 'slope', 'PrevFireMask', 
        'erc', 'NDVI', 'fpr', 'ftemp', 'th', 'EVI', 'vs', 'tmmx', 'fwd', 
        'aspect', 'tmmn'
    ]

    if selected_features is not None:
        feature_indices = [ALL_FEATURES.index(feature) for feature in selected_features]
    else:
        feature_indices = list(range(len(ALL_FEATURES)))
        selected_features = ALL_FEATURES

    dataset_obj = {
        TRAIN: RotatedWildfireDataset(f"{DATASET_PATH}/{TRAIN}.data", 
                                      f"{DATASET_PATH}/{TRAIN}.labels", 
                                      features=feature_indices, crop_size=64),
        VAL: WildfireDataset(f"{DATASET_PATH}/{VAL}.data", 
                             f"{DATASET_PATH}/{VAL}.labels", 
                             features=feature_indices, crop_size=64)
    }

    dataLoaders = {
        TRAIN: torch.utils.data.DataLoader(dataset=dataset_obj[TRAIN], batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2, pin_memory=True),
        VAL: torch.utils.data.DataLoader(dataset=dataset_obj[VAL], batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() // 2, pin_memory=True)
    }

    return dataLoaders

# -------------------- Threshold selection (per-image vs global) -------------------- #
def find_best_threshold_per_image(model, dataloader, loss_name,
                                  thresholds=np.linspace(0.001, 0.99, 120),
                                  lambda_fn=LAMBDA_FN):
    """Your original per-image risk rule (averages risk across images)."""
    model.eval()
    with torch.no_grad():
        all_probs, all_labels = [], []
        for images, labels in dataloader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs)    # [B,1,H,W]
            all_labels.append(labels)  # [B,1,H,W]
        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

    def conf_counts(y, p):
        yb = y.view(-1)
        pb = p.view(-1)
        tp = torch.sum((pb == 1) & (yb == 1)).item()
        fp = torch.sum((pb == 1) & (yb == 0)).item()
        fn = torch.sum((pb == 0) & (yb == 1)).item()
        tn = torch.sum((pb == 0) & (yb == 0)).item()
        return tp, fp, fn, tn

    best = {"thresh": 0.5, "risk": float("inf"), "dice": -1.0, "prec": 0.0, "rec": 0.0}
    for t in thresholds:
        preds = (all_probs > t).float()
        risks, dice_vals, prec_vals, rec_vals = [], [], [], []
        for i in range(all_labels.size(0)):
            y = all_labels[i]
            p = preds[i]
            tp, fp, fn, tn = conf_counts(y, p)

            pos = tp + fn
            neg = fp + tn
            fn_rate = 0.0 if pos == 0 else (fn / pos)
            fp_rate = 0.0 if neg == 0 else (fp / neg)

            prec = 0.0 if (tp + fp) == 0 else (tp / (tp + fp))
            rec  = 0.0 if (tp + fn) == 0 else (tp / (tp + fn))
            denom = (2*tp + fp + fn)
            dice = 0.0 if denom == 0 else (2*tp / denom)

            risks.append(lambda_fn * fn_rate + fp_rate)
            dice_vals.append(dice)
            prec_vals.append(prec)
            rec_vals.append(rec)

        avg_risk = float(np.mean(risks)) if risks else float("inf")
        avg_dice = float(np.mean(dice_vals)) if dice_vals else 0.0
        avg_prec = float(np.mean(prec_vals)) if prec_vals else 0.0
        avg_rec  = float(np.mean(rec_vals)) if rec_vals else 0.0

        if avg_risk < best["risk"]:
            best = {"thresh": float(t), "risk": avg_risk, "dice": avg_dice,
                    "prec": avg_prec, "rec": avg_rec}
    print(f"[Threshold Selection / per-image] λ={lambda_fn} → t={best['thresh']:.3f}, "
          f"risk={best['risk']:.6f}, prec={best['prec']:.4f}, rec={best['rec']:.4f}, dice={best['dice']:.4f}")
    return best["thresh"], best["dice"]

def find_best_threshold_global(model, dataloader, loss_name,
                               thresholds=np.linspace(0.001, 0.99, 120),
                               lambda_fn=LAMBDA_FN):
    """NEW: Global risk rule—aggregate TP/FP/FN/TN over the whole set for each t."""
    model.eval()
    with torch.no_grad():
        all_probs, all_labels = [], []
        for images, labels in dataloader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs)    # [B,1,H,W]
            all_labels.append(labels)  # [B,1,H,W]
        probs = torch.cat(all_probs, dim=0).view(-1)
        labels = torch.cat(all_labels, dim=0).view(-1)

    best = {"thresh": 0.5, "risk": float("inf"), "dice": -1.0, "prec": 0.0, "rec": 0.0}
    for t in thresholds:
        p = (probs > t).float()

        tp = torch.sum((p == 1) & (labels == 1)).item()
        fp = torch.sum((p == 1) & (labels == 0)).item()
        fn = torch.sum((p == 0) & (labels == 1)).item()
        tn = torch.sum((p == 0) & (labels == 0)).item()

        pos = tp + fn
        neg = fp + tn
        fn_rate = 0.0 if pos == 0 else (fn / pos)
        fp_rate = 0.0 if neg == 0 else (fp / neg)

        prec = 0.0 if (tp + fp) == 0 else (tp / (tp + fp))
        rec  = 0.0 if (tp + fn) == 0 else (tp / (tp + fn))
        denom = (2*tp + fp + fn)
        dice = 0.0 if denom == 0 else (2*tp / denom)
        risk = lambda_fn * fn_rate + fp_rate

        if risk < best["risk"]:
            best = {"thresh": float(t), "risk": float(risk), "dice": float(dice),
                    "prec": float(prec), "rec": float(rec)}
    print(f"[Threshold Selection / GLOBAL] λ={lambda_fn} → t={best['thresh']:.3f}, "
          f"risk={best['risk']:.6f}, prec={best['prec']:.4f}, rec={best['rec']:.4f}, dice={best['dice']:.4f}")
    return best["thresh"], best["dice"]

# >>> PAPER RULE: maximize *median* Dice on the TRAIN set
def find_best_threshold_max_median_dice_on_loader(model, dataloader, thresholds=np.linspace(0.001, 0.99, 120)):
    """
    Paper's segmentation threshold rule:
      Choose t that maximizes the *median* Dice across images of the given loader (TRAIN).
    """
    model.eval()
    with torch.no_grad():
        best_t, best_med = 0.5, -1.0
        for t in thresholds:
            per_img_dice = []
            for images, labels in dataloader:
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                logits = model(images)
                probs = torch.sigmoid(logits)
                preds = (probs > t).float()
                for i in range(images.size(0)):
                    per_img_dice.append(to_py_float(dice_score(labels[i], preds[i])))
            median_d = float(np.median(per_img_dice)) if len(per_img_dice) else 0.0
            if median_d > best_med:
                best_med = median_d
                best_t = float(t)
    print(f"[Threshold Selection / PAPER (max median Dice on TRAIN)] t={best_t:.3f}, median Dice={best_med:.4f}")
    return best_t, best_med

def find_best_threshold(model, dataloader, loss_name, mode="global",
                        thresholds=np.linspace(0.001, 0.99, 120), lambda_fn=LAMBDA_FN):
    if mode == "per_image":
        return find_best_threshold_per_image(model, dataloader, loss_name, thresholds, lambda_fn)
    elif mode == "paper_max_median_dice":
        return find_best_threshold_max_median_dice_on_loader(model, dataloader, thresholds)
    else:
        return find_best_threshold_global(model, dataloader, loss_name, thresholds, lambda_fn)
# ------------------------------------------------------------------------------------- #

def perform_validation(model, loader, loss_name, return_per_image_metrics=False, threshold=0.5):
    """
    VALIDATION FIXES:
      - loss computed on logits (not thresholded masks)
      - AUC computed on probabilities
      - IoU/F1/Dice/Precision/Recall on thresholded predictions
    """
    model.eval()

    per_image_dice = []
    per_image_iou = []
    per_image_f1 = []

    total_loss = 0.0
    total_iou = 0.0
    total_accuracy = 0.0
    total_f1 = 0.0
    total_auc = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            logits = model(images)                     # raw logits
            probs  = torch.sigmoid(logits)             # probabilities
            bins   = (probs > threshold).float()       # hard mask for thresholded metrics

            # ✅ loss on logits (your loss() handled logits in training)
            total_loss += to_py_float(loss(labels, logits, loss_name))

            # ✅ AUC on probabilities
            total_auc += to_py_float(auc_score(labels, probs))

            # Thresholded metrics
            total_iou += to_py_float(mean_iou(labels, bins))
            total_accuracy += to_py_float(accuracy(labels, bins))
            total_f1 += to_py_float(f1_score(labels, bins))
            total_dice += to_py_float(dice_score(labels, bins))
            precision, recall = precision_recall(labels, bins)
            total_precision += to_py_float(precision)
            total_recall += to_py_float(recall)

            for j in range(images.size(0)):
                label = labels[j]
                pred = bins[j]
                per_image_dice.append(to_py_float(dice_score(label, pred)))
                per_image_iou.append(to_py_float(mean_iou(label, pred)))
                per_image_f1.append(to_py_float(f1_score(label, pred)))

    n_batches = len(loader)
    avg_loss = total_loss / n_batches
    avg_iou = total_iou / n_batches
    avg_accuracy = total_accuracy / n_batches
    avg_f1 = total_f1 / n_batches
    avg_auc = total_auc / n_batches
    avg_dice = total_dice / n_batches
    avg_precision = total_precision / n_batches
    avg_recall = total_recall / n_batches

    print(f"Validation - Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, Accuracy: {avg_accuracy:.4f}")
    print(f"F1 Score: {avg_f1:.4f}, AUC: {avg_auc:.4f}, Dice: {avg_dice:.4f}")
    print(f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")

    if return_per_image_metrics:
        return {
            "avg_metrics": {
                "loss": float(avg_loss),
                "iou": float(avg_iou),
                "accuracy": float(avg_accuracy),
                "f1": float(avg_f1),
                "auc": float(avg_auc),
                "dice": float(avg_dice),
                "precision": float(avg_precision),
                "recall": float(avg_recall),
            },
            "per_image_metrics": {
                "dice": [float(v) for v in per_image_dice],
                "iou":  [float(v) for v in per_image_iou],
                "f1":   [float(v) for v in per_image_f1]
            }
        }

    return (
        float(avg_loss),
        float(avg_iou),
        float(avg_accuracy),
        float(avg_f1),
        float(avg_auc),
        float(avg_dice),
        float(avg_precision),
        float(avg_recall),
    )

def train(gpu, args, loss_name):
    rank = args.nr * args.gpus + gpu
    validate = True
    print("Current GPU", gpu, "\n RANK: ", rank)

    dataLoaders = create_data_loaders(rank, gpu, args.gpus * args.nodes)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = U2Net_CA(19, 1).cuda()
    # model = UNet1_CA(19, 1).cuda()
    # model = U_Net(19, 1).cuda()
    # model = TransformerCA_Seg(in_ch=19, img_size=64, out_ch=1, patch_size=4).cuda()
    
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.002, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    start = datetime.now()
    print(f'TRAINING ON: {platform.node()}, Starting at: {datetime.now()}')

    total_step = len(dataLoaders[TRAIN])
    best_epoch = 0
    best_f1_score = -float("inf")

    train_loss_history = []
    val_metrics_history = []

    for epoch in range(args.epochs):
        dataLoaders[TRAIN].dataset.reseed_crops(seed=epoch)
        model.train()
        loss_train = 0.0

        for i, (images, labels) in enumerate(dataLoaders[TRAIN]):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(images)
            loss_value = loss(labels, outputs, loss_name) 
            loss_train += to_py_float(loss_value)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            if i % 20 == 0:
                print('Epoch [{}/{}], Steps [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i, total_step, to_py_float(loss_value)))

        train_loss_history.append(float(loss_train / len(dataLoaders[TRAIN])))

        if validate:
            metrics = perform_validation(model, dataLoaders[VAL], loss_name)
            metrics = tuple(to_py_float(x) for x in metrics)
            val_metrics_history.append(metrics)

            curr_avg_loss_val, _, _, curr_f1_score, _, _, _, _ = metrics

            if best_f1_score < curr_f1_score:
                print("Saving model...")
                best_epoch = epoch
                best_f1_score = curr_f1_score
                filename_weights = f'model-{model.__class__.__name__}-{loss_name}-bestF1Score-Rank-{rank}.weights'
                os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
                torch.save(model.state_dict(), f'{SAVE_MODEL_PATH}/{filename_weights}')
                print("Model weights saved!")

                clean_train_hist = [float(v) for v in train_loss_history]
                clean_val_hist = [tuple(float(x) for x in m) for m in val_metrics_history]

                checkpoint = {
                    'epoch': int(best_epoch),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss_history': clean_train_hist,
                    'val_metrics_history': clean_val_hist
                }
                filename_ckpt = f'model-{model.__class__.__name__}-{loss_name}-bestF1Score-Rank-{rank}.pt'
                torch.save(checkpoint, os.path.join(SAVE_MODEL_PATH, filename_ckpt))
                print("Checkpoint saved!")
            else:
                print("Model is not being saved")

    # ---------- NEW: save curves after training ----------
    save_training_curves(train_loss_history, val_metrics_history, SAVE_MODEL_PATH)

    pickle.dump([float(v) for v in train_loss_history], open(f"{SAVE_MODEL_PATH}/train_loss_history.pkl", "wb"))
    pickle.dump([tuple(float(x) for x in m) for m in val_metrics_history], open(f"{SAVE_MODEL_PATH}/val_metrics_history.pkl", "wb"))

    if gpu == 0:
        end_time = datetime.now()
        elapsed = end_time - start
        elapsed_seconds = elapsed.total_seconds()
        print(f"Training completed in {int(elapsed_seconds // 3600)}h {int((elapsed_seconds % 3600) // 60)}m {int(elapsed_seconds % 60)}s")
        print(f"Best epoch: {best_epoch + 1}")
        print(f"Best F1 score: {best_f1_score}")
        
        # --- Print all metrics from the epoch where F1 was best (weights saved) ---
        if 0 <= best_epoch < len(val_metrics_history):
            (best_val_loss,
            best_val_iou,
            best_val_acc,
            best_val_f1,
            best_val_auc,
            best_val_dice,
            best_val_prec,
            best_val_rec) = val_metrics_history[best_epoch]

            print("\nValidation metrics at best-F1 epoch:")
            print(f"  Epoch: {best_epoch + 1}")
            print(f"  Loss: {best_val_loss:.4f}, IoU: {best_val_iou:.4f}, Accuracy: {best_val_acc:.4f}")
            print(f"  F1: {best_val_f1:.4f}, AUC: {best_val_auc:.4f}, Dice: {best_val_dice:.4f}")
            print(f"  Precision: {best_val_prec:.4f}, Recall: {best_val_rec:.4f}")

            if 0 <= best_epoch < len(train_loss_history):
                print(f"  Train loss (same epoch): {train_loss_history[best_epoch]:.4f}")
        else:
            print("\n[Warn] Could not retrieve metrics for best epoch.")

        model_path = f"{SAVE_MODEL_PATH}/model-{model.__class__.__name__}-{loss_name}-bestF1Score-Rank-{rank}.weights"
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)

        # ==================== Deterministic evaluation (PAPER threshold rule) ====================
        # Pick threshold on TRAIN by maximizing *median* Dice, then apply to VAL
        paper_best_threshold, paper_med_dice = find_best_threshold(
            model, dataLoaders[TRAIN], loss_name, mode="paper_max_median_dice",
            thresholds=np.unique(np.r_[0.001, 0.003, 0.005, np.linspace(0.01, 0.99, 99)])
        )
        results = perform_validation(model, dataLoaders[VAL], loss_name, return_per_image_metrics=True, threshold=paper_best_threshold)
        val_dice_at_t = float(results["avg_metrics"]["dice"])

        # ---------- qualitative grid on validation set ----------
        try:
            prev_list, gt_list, pred_list = _collect_fire_samples(model, dataLoaders[VAL], threshold=paper_best_threshold, max_cols=12)
            grid_title = f"{model.__class__.__name__} — {loss_name} — t={paper_best_threshold:.2f} (TRAIN median Dice)"
            grid_path = os.path.join(SAVE_MODEL_PATH, "qualitative_fire_grid.png")
            save_fire_grid(prev_list, gt_list, pred_list, grid_path, title=grid_title)
        except Exception as e:
            print(f"[Plot] Skipped qualitative grid due to error: {e}")

        # --- SAVE deterministic threshold + results ---
        ckpt_path = os.path.join(SAVE_MODEL_PATH, f"model-{model.__class__.__name__}-{loss_name}-bestF1Score-Rank-{rank}.pt")
        if os.path.exists(ckpt_path):
            ckpt = safe_load_checkpoint(ckpt_path, map_location="cpu")
        else:
            ckpt = {}

        ckpt["best_threshold"] = float(paper_best_threshold)
        ckpt["threshold_selection_rule"] = {
            "type": "paper_max_median_dice",
            "picked_on": "TRAIN",
            "sweep": "linspace incl {0.001,0.003,0.005} + 0.01..0.99",
            "note": "Threshold maximizes the median Dice across TRAIN images; applied to VAL/TEST."
        }
        ckpt["best_threshold_train_median_dice"] = float(paper_med_dice)
        ckpt["best_threshold_val_dice"] = float(val_dice_at_t)  # keep for compatibility
        torch.save(ckpt, ckpt_path)

        os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
        thresh_file = os.path.join(SAVE_MODEL_PATH, "best_thresholds.json")
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_id": f"{model.__class__.__name__}-{loss_name}",
            "best_threshold": float(paper_best_threshold),
            "train_median_dice_at_best_threshold": float(paper_med_dice),
            "val_dice_at_best_threshold": float(val_dice_at_t),
            "rule": ckpt["threshold_selection_rule"],
        }
        try:
            with open(thresh_file, "r") as f:
                db = json.load(f)
        except Exception:
            db = {}
        db[entry["model_id"]] = entry
        with open(thresh_file, "w") as f:
            json.dump(db, f, indent=4)

        model_id = f"{model.__class__.__name__}-{loss_name}"
        json_file = f"{SAVE_MODEL_PATH}/per_image_metrics.json"
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}
        all_metrics[model_id] = results
        with open(json_file, "w") as f:
            json.dump(all_metrics, f, indent=4)
        print(f"[Deterministic] Logged per-image + avg metrics to {json_file} for: {model_id}")

        # ==================== Bayesian evaluation (MC / input noise) ====================
        print(f"\n[Bayesian Eval] Running MC with n_passes={MC_PASSES} ...")

        # PAPER RULE for Bayesian: pick t on TRAIN probability maps by max median Dice, apply to VAL
        mean_probs_train, var_probs_train, labels_train = mc_dropout_collect_mean_var(model, dataLoaders[TRAIN], n_passes=MC_PASSES)
        bayes_paper_t, bayes_train_med_dice = find_best_threshold_from_probs_max_median_dice(
            mean_probs_train, labels_train,
            thresholds=np.unique(np.r_[0.001, 0.003, 0.005, np.linspace(0.01, 0.99, 99)])
        )

        mean_probs_val, var_probs_val, labels_val = mc_dropout_collect_mean_var(model, dataLoaders[VAL], n_passes=MC_PASSES)
        bayes_metrics_tuple = compute_metrics_from_probs(mean_probs_val, labels_val, threshold=bayes_paper_t)
        bayes_avg_loss, bayes_iou, bayes_acc, bayes_f1, bayes_auc, bayes_dice, bayes_prec, bayes_rec = bayes_metrics_tuple

        avg_variance = float(var_probs_val.mean().item())
        max_variance = float(var_probs_val.max().item())

        bayes_results = {
            "settings": {
                "mc_passes": int(MC_PASSES),
                "threshold_rule": {
                    "type": "paper_max_median_dice",
                    "picked_on": "TRAIN",
                    "sweep": "linspace incl {0.001,0.003,0.005} + 0.01..0.99",
                }
            },
            "best_threshold": float(bayes_paper_t),
            "train_median_dice_at_t": float(bayes_train_med_dice),
            "val_metrics_at_t": {
                "loss": float(bayes_avg_loss),
                "iou": float(bayes_iou),
                "accuracy": float(bayes_acc),
                "f1": float(bayes_f1),
                "auc": float(bayes_auc),
                "dice": float(bayes_dice),
                "precision": float(bayes_prec),
                "recall": float(bayes_rec),
            },
            "uncertainty_summary": {
                "predictive_variance_mean": float(avg_variance),
                "predictive_variance_max": float(max_variance)
            }
        }

        if os.path.exists(ckpt_path):
            ckpt = safe_load_checkpoint(ckpt_path, map_location="cpu")
        else:
            ckpt = {}
        ckpt.setdefault("bayesian_eval", {})
        ckpt["bayesian_eval"][loss_name] = bayes_results
        torch.save(ckpt, ckpt_path)

        bayes_thresh_file = os.path.join(SAVE_MODEL_PATH, "best_thresholds_bayesian.json")
        bayes_entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_id": f"{model.__class__.__name__}-{loss_name}",
            "mc_passes": int(MC_PASSES),
            "best_threshold": float(bayes_paper_t),
            "train_median_dice_at_best_threshold": float(bayes_train_med_dice),
            "val_dice_at_best_threshold": float(bayes_dice)
        }
        try:
            with open(bayes_thresh_file, "r") as f:
                db_b = json.load(f)
        except Exception:
            db_b = {}
        db_b[bayes_entry["model_id"]] = bayes_entry
        with open(bayes_thresh_file, "w") as f:
            json.dump(db_b, f, indent=4)

        bayes_metrics_file = os.path.join(SAVE_MODEL_PATH, "bayesian_eval_summary.json")
        try:
            with open(bayes_metrics_file, "r") as f:
                bayes_all = json.load(f)
        except Exception:
            bayes_all = {}
        bayes_all[model_id] = bayes_results
        with open(bayes_metrics_file, "w") as f:
            json.dump(bayes_all, f, indent=4)

        print(f"[Bayesian] Saved MC eval + threshold to {bayes_metrics_file}")
    print("Training complete!")

if __name__ == '__main__':
    main()
