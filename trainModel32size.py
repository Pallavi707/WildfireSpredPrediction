from datetime import datetime
import argparse
import torch
import torch.multiprocessing as mp
import torchvision
import torch.nn as nn
import torch.distributed as dist
from models import *
from datasets import *
import platform
from metrics1 import *
import copy
import numpy as np
from torch.utils.data import Dataset, IterableDataset, DataLoader, WeightedRandomSampler
from milesial_unet_model import UNet1, APAU_Net
from leejunhyun_unet_models import U_Net, R2U_Net, AttU_Net, R2AttU_Net, AttU_Net_S
from CellularAutomataPostprocessing import UNet1_CA
from CellularAutomataAllfeaturespostporcessing import U2Net_CA
import pickle
import random
import os
import json
import warnings
warnings.filterwarnings("ignore", message="networkx backend defined more than once")

# NEW: safer AMP import + temp file utils
from torch import amp
import tempfile, shutil

# new dataset module
from datasetsAreaResize import WildfireDataset, make_has_fire_weights

# ------------- NEW: plotting imports (headless backend) ------------- #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- Constants / Defaults (no CLI for these) -------------------- #
TRAIN = "train"
VAL = "validation"
LOSS_FUNCTIONS = ["FOCAL+DICE"]

# training defaults
BATCH_SIZE = 64
LR = 1e-3
SEED = 0
RANDOM_CROP = False
FIRE_AWARE_MIN_PROP = None
ROTATE = False
USE_SAMPLER = False
GRAD_ACCUM = 1
USE_AMP = True

# eval defaults
MC_PASSES = int(os.environ.get("MC_DROPOUT_PASSES", "30"))  # Bayesian passes
LAMBDA_FN = 30  # cost(FN)/cost(FP) for thresholding

# -------------------------------------------------------------------------------- #

# ---------- helpers to avoid NumPy scalars in checkpoints ---------- #
def to_py_float(x):
    """Coerce tensors/NumPy scalars/py floats to native Python float."""
    import numpy as _np
    if isinstance(x, torch.Tensor):
        return float(x.item())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(x)
    except Exception:
        return float(_np.asarray(x).item())

# -------------------- Bayesian (MC) utilities -------------------- #
@torch.no_grad()
def _enable_mc_dropout(model: torch.nn.Module):
    """
    Enable dropout at inference time by setting only Dropout layers to train(),
    leaving the rest of the model in eval() mode.
    """
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.train()

def _model_has_dropout(model: torch.nn.Module) -> bool:
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            return True
    return False

@torch.no_grad()
def mc_dropout_collect_mean_var(model, dataloader, device, n_passes=MC_PASSES):
    """
    Run MC passes with stochasticity:
      - If the model has Dropout, we enable it at inference (MC Dropout).
      - Otherwise, we add small Gaussian noise to the *inputs* each pass.
    Returns:
      mean_probs: Tensor [N, 1, H, W]
      var_probs:  Tensor [N, 1, H, W]
      all_labels: Tensor [N, 1, H, W]
    """
    model.eval()
    has_do = _model_has_dropout(model)
    if has_do:
        _enable_mc_dropout(model)
        input_noise_sigma = 0.0
        print("[Bayesian Eval] Using MC Dropout (dropout layers found).")
    else:
        input_noise_sigma = 0.03
        print(f"[Bayesian Eval] No dropout found; using input Gaussian noise sigma={input_noise_sigma}.")

    pass_sums = None
    pass_sq_sums = None
    all_labels = None
    first = True

    for pass_idx in range(n_passes):
        probs_pass = []
        labels_pass = []

        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if input_noise_sigma > 0.0:
                images = images + torch.randn_like(images) * input_noise_sigma

            logits = model(images)
            probs = torch.sigmoid(logits)  # [B,1,H,W]
            probs_pass.append(probs)
            if first:
                labels_pass.append(labels)

        probs_pass = torch.cat(probs_pass, dim=0)  # [N,1,H,W]
        if first:
            all_labels = torch.cat(labels_pass, dim=0)
            pass_sums = probs_pass.clone()
            pass_sq_sums = probs_pass.pow(2)
            first = False
        else:
            pass_sums.add_(probs_pass)
            pass_sq_sums.add_(probs_pass.pow(2))

    mean_probs = pass_sums / float(n_passes)
    var_probs = pass_sq_sums / float(n_passes) - mean_probs.pow(2)
    return mean_probs, var_probs, all_labels

# -------------------- Cost-sensitive threshold selection -------------------- #
def _conf_counts(y, p):
    yb = y.view(-1)
    pb = p.view(-1)
    tp = torch.sum((pb == 1) & (yb == 1)).item()
    fp = torch.sum((pb == 1) & (yb == 0)).item()
    fn = torch.sum((pb == 0) & (yb == 1)).item()
    tn = torch.sum((pb == 0) & (yb == 0)).item()
    return tp, fp, fn, tn

def find_best_threshold_cost_sensitive_from_probs(mean_probs, all_labels, thresholds=np.linspace(0.01, 0.99, 99), lambda_fn=LAMBDA_FN):
    best = {"thresh": 0.5, "risk": float("inf"), "dice": -1.0, "prec": 0.0, "rec": 0.0}
    with torch.no_grad():
        for t in thresholds:
            preds = (mean_probs > t).float()
            risks, dice_vals, prec_vals, rec_vals = [], [], [], []
            for i in range(all_labels.size(0)):
                y = all_labels[i]
                p = preds[i]
                tp, fp, fn, tn = _conf_counts(y, p)
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
                best = {"thresh": float(t), "risk": avg_risk, "dice": avg_dice, "prec": avg_prec, "rec": avg_rec}
    print(f"[Threshold Selection] Cost-sensitive (λ={lambda_fn}) → t={best['thresh']:.2f}, risk={best['risk']:.6f}, dice={best['dice']:.4f}, prec={best['prec']:.4f}, rec={best['rec']:.4f}")
    return best["thresh"], best["dice"]

def find_best_threshold_cost_sensitive(model, dataloader, device, thresholds=np.linspace(0.01, 0.99, 99), lambda_fn=LAMBDA_FN):
    model.eval()
    with torch.no_grad():
        all_probs, all_labels = [], []
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs)
            all_labels.append(labels)
        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
    return find_best_threshold_cost_sensitive_from_probs(all_probs, all_labels, thresholds, lambda_fn)

# -------------------- Data loaders -------------------- #
def create_data_loaders(tile_size, batch_size=BATCH_SIZE, data_root=None, seed=SEED,
                        random_crop=RANDOM_CROP, fire_aware_min_prop=FIRE_AWARE_MIN_PROP,
                        rotate=ROTATE, use_sampler=USE_SAMPLER, selected_features=None):
    tile = tile_size
    data_root = data_root or f"data{tile}/next-day-wildfire-spread"

    data_files = {
        TRAIN: (os.path.join(data_root, f"{TRAIN}.data"), os.path.join(data_root, f"{TRAIN}.labels")),
        VAL:   (os.path.join(data_root, f"{VAL}.data"),   os.path.join(data_root,   f"{VAL}.labels")),
    }

    ALL_FEATURES = [
        "elevation", "fws", "population", "pdsi", "pr", "sph", "slope", "PrevFireMask",
        "erc", "NDVI", "fpr", "ftemp", "th", "EVI", "vs", "tmmx", "fwd",
        "aspect", "tmmn"
    ]
    if selected_features is not None:
        feature_indices = [ALL_FEATURES.index(feat) for feat in selected_features]
    else:
        feature_indices = list(range(len(ALL_FEATURES)))

    ds_train = WildfireDataset(
        data_filename=data_files[TRAIN][0],
        labels_filename=data_files[TRAIN][1],
        features=feature_indices,
        crop_size=tile if not random_crop else tile,  # AreaResize dataset respects crop internally
        random_crop=bool(random_crop),
        fire_aware_min_prop=fire_aware_min_prop,
        rotate=bool(rotate),
        seed=seed,
    )

    ds_val = WildfireDataset(
        data_filename=data_files[VAL][0],
        labels_filename=data_files[VAL][1],
        features=feature_indices,
        crop_size=tile,
        random_crop=False,
        fire_aware_min_prop=None,
        rotate=False,
        seed=seed,
    )

    sampler = None
    if use_sampler:
        weights = make_has_fire_weights(ds_train.labels)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    num_workers = max(1, os.cpu_count() // 2)

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return {TRAIN: train_loader, VAL: val_loader}

# -------------------- Validation -------------------- #
def perform_validation(model, loader, loss_name, return_per_image_metrics=False, threshold=0.5, device="cpu"):
    model.eval()
    per_image_dice, per_image_iou, per_image_f1 = [], [], []

    total_loss = 0.0
    total_iou = 0.0
    total_accuracy = 0.0
    total_f1 = 0.0
    total_auc = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            # compute validation loss on logits (or probs), not hard preds
            total_loss += float(loss(labels, logits, loss_name))
            total_iou += float(mean_iou(labels, preds))
            total_accuracy += float(accuracy(labels, preds))
            total_f1 += float(f1_score(labels, preds))
            total_auc += float(auc_score(labels, preds))
            total_dice += float(dice_score(labels, preds))
            pr, rc = precision_recall(labels, preds)
            total_precision += float(pr)
            total_recall += float(rc)

            if return_per_image_metrics:
                for j in range(images.size(0)):
                    y = labels[j]
                    p = preds[j]
                    per_image_dice.append(float(dice_score(y, p)))
                    per_image_iou.append(float(mean_iou(y, p)))
                    per_image_f1.append(float(f1_score(y, p)))

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
                "loss": avg_loss,
                "iou": avg_iou,
                "accuracy": avg_accuracy,
                "f1": avg_f1,
                "auc": avg_auc,
                "dice": avg_dice,
                "precision": avg_precision,
                "recall": avg_recall,
            },
            "per_image_metrics": {
                "dice": per_image_dice,
                "iou": per_image_iou,
                "f1": per_image_f1,
            },
        }

    return avg_loss, avg_iou, avg_accuracy, avg_f1, avg_auc, avg_dice, avg_precision, avg_recall

# -------------------- NEW: Plotting helpers -------------------- #
def _plot_training_curves(train_loss_history, val_metrics_history, out_dir, run_tag):
    """Save train/val curves (loss, F1, Dice) as PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, len(train_loss_history) + 1))

    # Extract val metrics by position matching perform_validation return
    val_loss = [m[0] for m in val_metrics_history]
    val_f1   = [m[3] for m in val_metrics_history]
    val_dice = [m[5] for m in val_metrics_history]

    # Train vs Val Loss
    plt.figure(figsize=(7,5))
    plt.plot(epochs, train_loss_history, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves (run {run_tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"loss_curves_{run_tag}.png"))
    plt.close()

    # Val F1
    plt.figure(figsize=(7,5))
    plt.plot(epochs, val_f1, label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title(f"Validation F1 (run {run_tag})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"val_f1_{run_tag}.png"))
    plt.close()

    # Val Dice
    plt.figure(figsize=(7,5))
    plt.plot(epochs, val_dice, label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title(f"Validation Dice (run {run_tag})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"val_dice_{run_tag}.png"))
    plt.close()

def _save_firemask_predictions(model, loader, device, threshold, out_dir, run_tag, max_images=16):
    """
    Save a grid of GT vs probability heatmap vs thresholded prediction for up to max_images from VAL.
    Produces: firemask_examples_{run_tag}.png
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    saved = 0
    panels = []  # each is (GT, Prob, Pred)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            b = images.size(0)
            for i in range(b):
                gt = labels[i, 0].detach().cpu().numpy()
                pr = probs[i, 0].detach().cpu().numpy()
                pd = preds[i, 0].detach().cpu().numpy()
                panels.append((gt, pr, pd))
                saved += 1
                if saved >= max_images:
                    break
            if saved >= max_images:
                break

    if not panels:
        print("[Plot] No validation images available to save fire-mask predictions.")
        return

    # Build a figure with rows = samples, cols = 3 (GT, Prob, Pred)
    rows = len(panels)
    cols = 3
    fig_h = max(2, int(rows * 1.8))
    fig_w = 9
    plt.figure(figsize=(fig_w, fig_h))
    for r, (gt, pr, pd) in enumerate(panels, start=1):
        # GT
        ax = plt.subplot(rows, cols, (r-1)*cols + 1)
        ax.imshow(gt, interpolation="nearest")
        ax.set_title("GT")
        ax.axis("off")

        # Prob heatmap
        ax = plt.subplot(rows, cols, (r-1)*cols + 2)
        im = ax.imshow(pr, interpolation="nearest")
        ax.set_title("Prob")
        ax.axis("off")

        # Pred (binary)
        ax = plt.subplot(rows, cols, (r-1)*cols + 3)
        ax.imshow(pd, interpolation="nearest")
        ax.set_title(f"Pred (t={threshold:.2f})")
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"firemask_examples_{run_tag}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved fire-mask examples to {out_path}")

# NEW: robust, atomic checkpoint save with legacy serializer fallback
def _atomic_torch_save(obj, path, use_legacy_fallback=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmpf = None
    try:
        fd, tmpf = tempfile.mkstemp(prefix="ckpt_", suffix=".pt", dir=os.path.dirname(path))
        os.close(fd)
        try:
            torch.save(obj, tmpf)
        except Exception as e:
            if use_legacy_fallback:
                print(f"⚠️ torch.save failed with new zip format: {e}. Retrying legacy serializer...")
                torch.save(obj, tmpf, _use_new_zipfile_serialization=False)
            else:
                raise
        shutil.move(tmpf, path)
    finally:
        if tmpf and os.path.exists(tmpf):
            try:
                os.remove(tmpf)
            except Exception:
                pass

# -------------------- Train one loss -------------------- #
def train_one_loss(epochs, tile_size):
    # paths/tags based solely on tile
    save_model_path = f"savedModels{tile_size}"
    run_tag = f"{tile_size}"

    # device & seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    loaders = create_data_loaders(tile_size)

    # model
    model = U2Net_CA(19, 1).to(device)
    model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, momentum=0.9)

    use_amp = bool(USE_AMP) and (device.type == "cuda")
    scaler = amp.GradScaler('cuda', enabled=use_amp)

    best_epoch = 0
    best_f1_score = -float("inf")

    train_loss_history = []
    val_metrics_history = []

    start = datetime.now()
    print(f"TRAINING ON: {platform.node()}, Starting at: {datetime.now()} (tag={run_tag})")
    print(f"Device: {device} | AMP: {use_amp} | GradAccum: {GRAD_ACCUM}")

    total_step = len(loaders[TRAIN])

    for epoch in range(epochs):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        for i, (images, labels) in enumerate(loaders[TRAIN]):
            images = images.to(device, non_blocking=True)
            if images.dim() == 4:
                images = images.to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)

            try:
                with amp.autocast('cuda', enabled=use_amp):
                    logits = model(images)
                    loss_value = loss(labels, logits, LOSS_FUNCTIONS[0]) / max(1, GRAD_ACCUM)

                if use_amp:
                    scaler.scale(loss_value).backward()
                else:
                    loss_value.backward()

                if (i + 1) % GRAD_ACCUM == 0:
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                running += float(loss_value) * max(1, GRAD_ACCUM)

                if i % 20 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Steps [{i}/{total_step}], "
                          f"Loss: {float(loss_value)*max(1,GRAD_ACCUM):.4f}")

            except torch.cuda.OutOfMemoryError:
                print("⚠️ CUDA OOM: clearing cache, skipping this batch.")
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

        epoch_train_loss = running / len(loaders[TRAIN])
        train_loss_history.append(epoch_train_loss)

        metrics = perform_validation(model, loaders[VAL], LOSS_FUNCTIONS[0], device=device)
        val_metrics_history.append(metrics)
        _, _, _, curr_f1, _, _, _, _ = metrics

        if curr_f1 > best_f1_score:
            best_f1_score = curr_f1
            best_epoch = epoch
            print("Saving model...")

            os.makedirs(save_model_path, exist_ok=True)
            wpath = os.path.join(
                save_model_path,
                f"model-{model.__class__.__name__}-{LOSS_FUNCTIONS[0]}-bestF1Score-{run_tag}.weights",
            )
            cpath = os.path.join(
                save_model_path,
                f"model-{model.__class__.__name__}-{LOSS_FUNCTIONS[0]}-bestF1Score-{run_tag}.pt",
            )

            # build a lean checkpoint on CPU
            cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            ckpt = {
                "epoch": int(best_epoch),
                "model_state_dict": cpu_state,
                "train_loss_history": train_loss_history,
                "val_metrics_history": val_metrics_history,
            }

            # save weights (fallback)
            try:
                torch.save(cpu_state, wpath)
            except Exception as e:
                print(f"⚠️ weights save failed with new zip format: {e}. Retrying legacy...")
                torch.save(cpu_state, wpath, _use_new_zipfile_serialization=False)

            # save checkpoint atomically (fallback)
            try:
                _atomic_torch_save(ckpt, cpath, use_legacy_fallback=True)
                print("Checkpoint saved!")
            except Exception as e:
                print(f"❌ Failed to save checkpoint even after fallback: {e}")
        else:
            print("Model is not being saved")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Persist histories separately (tile-suffixed)
    os.makedirs(save_model_path, exist_ok=True)
    pickle.dump(
        train_loss_history,
        open(os.path.join(save_model_path, f"train_loss_history_{run_tag}.pkl"), "wb"),
    )
    pickle.dump(
        val_metrics_history,
        open(os.path.join(save_model_path, f"val_metrics_history_{run_tag}.pkl"), "wb"),
    )

    # -------------------- NEW: save training curves -------------------- #
    plots_dir = os.path.join(save_model_path, "plots")
    _plot_training_curves(train_loss_history, val_metrics_history, plots_dir, run_tag)

    print("\n=== Final evaluation on VAL (Deterministic + Bayesian) ===")
    # Load best weights
    model.load_state_dict(
        torch.load(
            os.path.join(
                save_model_path,
                f"model-{model.__class__.__name__}-{LOSS_FUNCTIONS[0]}-bestF1Score-{run_tag}.weights",
            ),
            map_location=device,
        )
    )

    # ---------------- Deterministic, cost-sensitive threshold ---------------- #
    best_t_det, best_d_det = find_best_threshold_cost_sensitive(model, loaders[VAL], device=device, lambda_fn=LAMBDA_FN)
    results_det = perform_validation(
        model, loaders[VAL], LOSS_FUNCTIONS[0], return_per_image_metrics=True, threshold=best_t_det, device=device
    )

    # -------------------- NEW: save fire-mask example images -------------------- #
    _save_firemask_predictions(model, loaders[VAL], device, threshold=best_t_det, out_dir=plots_dir, run_tag=run_tag, max_images=16)

    # Save deterministic per-image metrics (tile-suffixed file)
    model_id = f"{model.__class__.__name__}-{LOSS_FUNCTIONS[0]}-{run_tag}"
    json_file = os.path.join(save_model_path, f"per_image_metrics_{run_tag}.json")
    try:
        with open(json_file, "r") as f:
            all_metrics = json.load(f)
    except Exception:
        all_metrics = {}
    all_metrics[model_id] = results_det
    with open(json_file, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"[Deterministic] Logged per-image + avg metrics to {json_file} for: {model_id}")

    # Save deterministic threshold summary (tile-suffixed)
    thresh_file = os.path.join(save_model_path, f"best_thresholds_{run_tag}.json")
    det_entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_id": model_id,
        "best_threshold": float(best_t_det),
        "val_dice_at_best_threshold": float(best_d_det),
        "rule": {
            "type": "cost_sensitive",
            "lambda_fn": float(LAMBDA_FN),
            "risk": "lambda*FN_rate + FP_rate",
            "sweep": "linspace(0.01, 0.99, 99)"
        },
    }
    try:
        with open(thresh_file, "r") as f:
            db = json.load(f)
    except Exception:
        db = {}
    db[model_id] = det_entry
    with open(thresh_file, "w") as f:
        json.dump(db, f, indent=4)

    # Embed threshold into checkpoint
    cpath = os.path.join(
        save_model_path,
        f"model-{model.__class__.__name__}-{LOSS_FUNCTIONS[0]}-bestF1Score-{run_tag}.pt",
    )
    try:
        ckpt = torch.load(cpath, map_location="cpu")
    except Exception:
        ckpt = {}
    ckpt["best_threshold"] = float(best_t_det)
    ckpt["threshold_selection_rule"] = det_entry["rule"]
    ckpt["best_threshold_val_dice"] = float(best_d_det)
    torch.save(ckpt, cpath)

    # ---------------- Bayesian (MC) evaluation ---------------- #
    print(f"\n[Bayesian Eval] Running MC with n_passes={MC_PASSES} ...")
    mean_probs, var_probs, all_labels = mc_dropout_collect_mean_var(model, loaders[VAL], device=device, n_passes=MC_PASSES)

    bayes_best_t, bayes_best_dice = find_best_threshold_cost_sensitive_from_probs(
        mean_probs, all_labels, thresholds=np.linspace(0.01, 0.99, 99), lambda_fn=LAMBDA_FN
    )

    # Compute aggregate metrics at that threshold (from probs)
    preds_b = (mean_probs > bayes_best_t).float()
    total_iou = total_acc = total_f1 = total_auc = total_dice = total_prec = total_rec = 0.0
    with torch.no_grad():
        for i in range(all_labels.size(0)):
            y = all_labels[i]
            p = preds_b[i]
            total_iou += to_py_float(mean_iou(y, p))
            total_acc += to_py_float(accuracy(y, p))
            total_f1 += to_py_float(f1_score(y, p))
            total_auc += to_py_float(auc_score(y, p))
            total_dice += to_py_float(dice_score(y, p))
            pr, rc = precision_recall(y, p)
            total_prec += to_py_float(pr)
            total_rec  += to_py_float(rc)
    n_imgs = float(all_labels.size(0))
    bayes_metrics = {
        "loss": 0.0,
        "iou": float(total_iou / n_imgs),
        "accuracy": float(total_acc / n_imgs),
        "f1": float(total_f1 / n_imgs),
        "auc": float(total_auc / n_imgs),
        "dice": float(total_dice / n_imgs),
        "precision": float(total_prec / n_imgs),
        "recall": float(total_rec / n_imgs),
    }

    avg_variance = float(var_probs.mean().item())
    max_variance = float(var_probs.max().item())

    bayes_results = {
        "settings": {
            "mc_passes": int(MC_PASSES),
            "threshold_rule": {
                "type": "cost_sensitive",
                "lambda_fn": float(LAMBDA_FN),
                "risk": "lambda*FN_rate + FP_rate",
                "sweep": "linspace(0.01, 0.99, 99)"
            }
        },
        "best_threshold": float(bayes_best_t),
        "val_metrics_at_t": bayes_metrics,
        "uncertainty_summary": {
            "predictive_variance_mean": float(avg_variance),
            "predictive_variance_max": float(max_variance)
        }
    }

    # Save Bayesian summaries (tile-suffixed)
    bayes_thresh_file = os.path.join(save_model_path, f"best_thresholds_bayesian_{run_tag}.json")
    try:
        with open(bayes_thresh_file, "r") as f:
            dbb = json.load(f)
    except Exception:
        dbb = {}
    dbb[model_id] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_id": model_id,
        "mc_passes": int(MC_PASSES),
        "best_threshold": float(bayes_best_t),
        "val_dice_at_best_threshold": float(bayes_metrics["dice"]),
        "uncertainty_mean_var": float(avg_variance),
    }
    with open(bayes_thresh_file, "w") as f:
        json.dump(dbb, f, indent=4)

    bayes_metrics_file = os.path.join(save_model_path, f"bayesian_eval_summary_{run_tag}.json")
    try:
        with open(bayes_metrics_file, "r") as f:
            bayes_all = json.load(f)
    except Exception:
        bayes_all = {}
    bayes_all[model_id] = bayes_results
    with open(bayes_metrics_file, "w") as f:
        json.dump(bayes_all, f, indent=4)
    print(f"[Bayesian] Saved MC eval + threshold to {bayes_metrics_file}")

    # also embed Bayesian block into checkpoint
    try:
        ckpt = torch.load(cpath, map_location="cpu")
    except Exception:
        ckpt = {}
    ckpt.setdefault("bayesian_eval", {})
    ckpt["bayesian_eval"][LOSS_FUNCTIONS[0]] = bayes_results
    torch.save(ckpt, cpath)

    elapsed = datetime.now() - start
    print(f"\nCompleted {LOSS_FUNCTIONS[0]} in {elapsed}. Best epoch: {best_epoch+1}, Best F1: {best_f1_score:.4f}")
    print(f"Deterministic metrics saved to: {json_file}")
    print(f"Deterministic thresholds saved to: {thresh_file}")
    print(f"Bayesian summaries saved to: {bayes_metrics_file}")

# -------------------- Main -------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10, help="Total epochs to run")
    p.add_argument("--tile_size", type=int, default=32, help="Tile size: e.g., 64, 32, 18")
    args = p.parse_args()

    print(f"initializing training (tile={args.tile_size}, epochs={args.epochs})")
    # single loss as per LOSS_FUNCTIONS list
    train_one_loss(args.epochs, args.tile_size)

if __name__ == "__main__":
    main()
