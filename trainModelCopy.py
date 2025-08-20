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
from transformerCA import TransformerCA_Seg
import pickle
import random
import os
import json

# --------- Device config ----------
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
PIN_MEMORY = True if USE_CUDA else False

# --------- Defaults (no CLI needed) ----------
MC_PASSES = int(os.environ.get("MC_DROPOUT_PASSES", "30"))
LAMBDA_FN = 30

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
def mc_dropout_collect_mean_var(model, dataloader, n_passes=MC_PASSES):
    """
    Run MC passes with stochasticity:
      - If the model has Dropout, we enable it at inference (MC Dropout).
      - Otherwise, we add small Gaussian noise to the *inputs* each pass (keeps training unchanged).

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
        input_noise_sigma = 0.03  # default tiny input noise (for models without dropout)
        print(f"[Bayesian Eval] No dropout found; using input Gaussian noise sigma={input_noise_sigma}.")

    pass_sums = None
    pass_sq_sums = None
    all_labels_list = []

    for pass_idx in range(n_passes):
        probs_pass = []
        batch_labels = []

        for images, labels in dataloader:
            images = images.to(DEVICE, non_blocking=PIN_MEMORY)
            labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)

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

def find_best_threshold_from_probs(mean_probs, all_labels, thresholds=np.linspace(0.01, 0.99, 99), lambda_fn=LAMBDA_FN):
    """
    (Kept for compatibility) Cost-sensitive risk minimisation on precomputed probability maps.
    """
    def conf_counts(y, p):
        yb = y.view(-1)
        pb = p.view(-1)
        tp = torch.sum((pb == 1) & (yb == 1)).item()
        fp = torch.sum((pb == 1) & (yb == 0)).item()
        fn = torch.sum((pb == 0) & (yb == 1)).item()
        tn = torch.sum((pb == 0) & (yb == 0)).item()
        return tp, fp, fn, tn

    best = {"thresh": 0.5, "risk": float("inf"), "dice": -1.0, "prec": 0.0, "rec": 0.0}
    with torch.no_grad():
        for t in thresholds:
            preds = (mean_probs > t).float()
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
                best = {
                    "thresh": float(t),
                    "risk":  avg_risk,
                    "dice":  avg_dice,
                    "prec":  avg_prec,
                    "rec":   avg_rec
                }
    return best["thresh"], best["dice"]

def compute_metrics_from_probs(mean_probs, all_labels, threshold=0.5):
    """
    Compute the same aggregate metrics as perform_validation(),
    but from precomputed probability maps instead of running the model.
    Returns tuple: (avg_loss, avg_iou, avg_accuracy, avg_f1, avg_auc, avg_dice, avg_precision, avg_recall)
    Note: 'loss' here is set to 0.0 because we don't recompute criterion on binarised outputs.
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
            p = preds[i]
            total_iou += to_py_float(mean_iou(y, p))
            total_accuracy += to_py_float(accuracy(y, p))
            total_f1 += to_py_float(f1_score(y, p))
            total_auc += to_py_float(auc_score(y, p))
            total_dice += to_py_float(dice_score(y, p))
            pr, rc = precision_recall(y, p)
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

TRAIN = 'train'
VAL = 'validation'
MASTER_RANK = 0
SAVE_INTERVAL = 1

DATASET_PATH = 'data/next-day-wildfire-spread'
SAVE_MODEL_PATH = 'savedModels'

#loss_functions = ['WBCE+DICE', 'FOCAL', 'FOCAL+DICE']
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
    print(f'initializing training on single {"GPU" if USE_CUDA else "CPU"}')

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

    datasets = {
        TRAIN: RotatedWildfireDataset(f"{DATASET_PATH}/{TRAIN}.data", 
                                      f"{DATASET_PATH}/{TRAIN}.labels", 
                                      features=feature_indices, crop_size=64),
        VAL: WildfireDataset(f"{DATASET_PATH}/{VAL}.data", 
                             f"{DATASET_PATH}/{VAL}.labels", 
                             features=feature_indices, crop_size=64)
    }

    dataLoaders = {
        TRAIN: torch.utils.data.DataLoader(dataset=datasets[TRAIN], batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2, pin_memory=PIN_MEMORY),
        VAL: torch.utils.data.DataLoader(dataset=datasets[VAL], batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() // 2, pin_memory=PIN_MEMORY)
    }

    return dataLoaders

# -------------------- Cost-sensitive threshold selection (deterministic) -------------------- #
def find_best_threshold(
    model,
    dataloader,
    loss_name,
    thresholds=np.linspace(0.01, 0.99, 99),
    lambda_fn=LAMBDA_FN  # cost(FN)/cost(FP)
):
    """
    Cost-sensitive thresholding for imbalanced wildfire masks.
    """
    model.eval()
    with torch.no_grad():
        all_probs, all_labels = [], []
        for images, labels in dataloader:
            images = images.to(DEVICE, non_blocking=PIN_MEMORY)
            labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs)    # [B,1,H,W]
            all_labels.append(labels)  # [B,1,H,W]

        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

    def conf_counts(y, p):
        # y, p: [1,H,W] with {0,1}
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

            # Safeguards for images with no positives/negatives
            fn_rate = 0.0 if pos == 0 else (fn / pos)
            fp_rate = 0.0 if neg == 0 else (fp / neg)

            # Precision/recall for logging
            prec = 0.0 if (tp + fp) == 0 else (tp / (tp + fp))
            rec  = 0.0 if (tp + fn) == 0 else (tp / (tp + fn))

            # Dice for logging
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
            best = {
                "thresh": float(t),
                "risk": avg_risk,
                "dice": avg_dice,
                "prec": avg_prec,
                "rec":  avg_rec
            }

    print(f"\n[Threshold Selection] Cost-sensitive (lambda={lambda_fn:.1f}) "
          f"→ t={best['thresh']:.2f}, risk={best['risk']:.6f}, "
          f"prec={best['prec']:.4f}, rec={best['rec']:.4f}, dice={best['dice']:.4f}")
    return best["thresh"], best["dice"]
# ------------------------------------------------------------------------------------- #

# -------------------- NEW: Dice-optimal threshold on TRAIN (median per image) -------------------- #
def _dice_of_pair(y, p):
    # y, p: [1,H,W] with {0,1}
    yb = y.view(-1)
    pb = p.view(-1)
    tp = torch.sum((pb == 1) & (yb == 1)).item()
    fp = torch.sum((pb == 1) & (yb == 0)).item()
    fn = torch.sum((pb == 0) & (yb == 1)).item()
    denom = (2*tp + fp + fn)
    return 0.0 if denom == 0 else (2*tp / denom)

@torch.no_grad()
def find_threshold_max_median_dice(model, dataloader, thresholds=np.linspace(0.01, 0.99, 99)):
    """
    Choose threshold that maximizes the *median* Dice across TRAIN images.
    """
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in dataloader:
        images = images.to(DEVICE, non_blocking=PIN_MEMORY)
        labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)
        logits = model(images)
        probs = torch.sigmoid(logits)
        all_probs.append(probs)    # [B,1,H,W]
        all_labels.append(labels)  # [B,1,H,W]
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    best_t, best_med = 0.5, -1.0
    for t in thresholds:
        preds = (all_probs > t).float()
        per_img = []
        for i in range(all_labels.size(0)):
            per_img.append(_dice_of_pair(all_labels[i], preds[i]))
        med = float(np.median(per_img)) if per_img else 0.0
        if med > best_med:
            best_t, best_med = float(t), med

    print(f"[Threshold Selection] Max median Dice on TRAIN → t={best_t:.2f}, median_dice={best_med:.4f}")
    return best_t, best_med

@torch.no_grad()
def find_threshold_max_median_dice_from_probs(mean_probs, all_labels, thresholds=np.linspace(0.01, 0.99, 99)):
    """
    Same as above but for precomputed mean probability maps (Bayesian path).
    """
    best_t, best_med = 0.5, -1.0
    for t in thresholds:
        preds = (mean_probs > t).float()
        per_img = []
        for i in range(all_labels.size(0)):
            per_img.append(_dice_of_pair(all_labels[i], preds[i]))
        med = float(np.median(per_img)) if per_img else 0.0
        if med > best_med:
            best_t, best_med = float(t), med
    print(f"[Bayesian Threshold] Max median Dice on TRAIN → t={best_t:.2f}, median_dice={best_med:.4f}")
    return best_t, best_med
# --------------------------------------------------------------------------------------------- #

def perform_validation(model, loader, loss_name, return_per_image_metrics=False, threshold=0.5):
    model.eval()

    per_image_dice = []
    per_image_iou = []
    per_image_f1 = []

    # make all accumulators native floats
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
            images = images.to(DEVICE, non_blocking=PIN_MEMORY)
            labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            outputs = (probs > threshold).float()

            total_loss += to_py_float(loss(labels, outputs, loss_name))
            total_iou += to_py_float(mean_iou(labels, outputs))
            total_accuracy += to_py_float(accuracy(labels, outputs))
            total_f1 += to_py_float(f1_score(labels, outputs))
            total_auc += to_py_float(auc_score(labels, outputs))
            total_dice += to_py_float(dice_score(labels, outputs))

            precision, recall = precision_recall(labels, outputs)
            total_precision += to_py_float(precision)
            total_recall += to_py_float(recall)

            for j in range(images.size(0)):
                label = labels[j]
                pred = outputs[j]
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
    if USE_CUDA:
        torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    if USE_CUDA:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = U2Net_CA(19, 1).to(DEVICE)
    # model = UNet1_CA(19, 1).to(DEVICE)
    # model = U_Net(19, 1).to(DEVICE)
    # model = TransformerCA_Seg(in_ch=19, img_size=64, out_ch=1, patch_size=4).to(DEVICE)
    
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    start = datetime.now()
    print(f'TRAINING ON: {platform.node()}, Starting at: {datetime.now()} (device={DEVICE})')

    total_step = len(dataLoaders[TRAIN])
    best_epoch = 0
    best_f1_score = -float("inf")

    train_loss_history = []
    val_metrics_history = []

    for epoch in range(args.epochs):
        model.train()
        loss_train = 0.0

        for i, (images, labels) in enumerate(dataLoaders[TRAIN]):
            images = images.to(DEVICE, non_blocking=PIN_MEMORY)
            labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)

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

            # (optional) also show the train loss for that same epoch
            if 0 <= best_epoch < len(train_loss_history):
                print(f"  Train loss (same epoch): {train_loss_history[best_epoch]:.4f}")
        else:
            print("\n[Warn] Could not retrieve metrics for best epoch.")

        model_path = f"{SAVE_MODEL_PATH}/model-{model.__class__.__name__}-{loss_name}-bestF1Score-Rank-{rank}.weights"
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)

        # ==================== Deterministic evaluation (UPDATED policy) ====================
        # Choose threshold on TRAIN by maximizing median Dice
        best_threshold, best_train_med_dice = find_threshold_max_median_dice(model, dataLoaders[TRAIN])

        # Evaluate on VAL using that fixed threshold
        results = perform_validation(model, dataLoaders[VAL], loss_name, return_per_image_metrics=True, threshold=best_threshold)

        # --- SAVE deterministic threshold + results ---
        ckpt_path = os.path.join(SAVE_MODEL_PATH, f"model-{model.__class__.__name__}-{loss_name}-bestF1Score-Rank-{rank}.pt")
        if os.path.exists(ckpt_path):
            ckpt = safe_load_checkpoint(ckpt_path, map_location="cpu")
        else:
            ckpt = {}

        ckpt["best_threshold"] = float(best_threshold)
        ckpt["threshold_selection_rule"] = {
            "type": "max_median_dice_on_train",
            "sweep": "linspace(0.01, 0.99, 99)"
        }
        ckpt["best_threshold_train_median_dice"] = float(best_train_med_dice)
        torch.save(ckpt, ckpt_path)

        os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
        thresh_file = os.path.join(SAVE_MODEL_PATH, "best_thresholds.json")
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_id": f"{model.__class__.__name__}-{loss_name}",
            "best_threshold": float(best_threshold),
            "train_median_dice_at_best_threshold": float(best_train_med_dice),
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
        # IMPORTANT: For Bayesian thresholding, compute mean_probs on TRAIN (policy match)
        mean_probs_train, var_probs_train, all_labels_train = mc_dropout_collect_mean_var(model, dataLoaders[TRAIN], n_passes=MC_PASSES)
        bayes_best_t, bayes_train_med_dice = find_threshold_max_median_dice_from_probs(
            mean_probs_train, all_labels_train, thresholds=np.linspace(0.01, 0.99, 99)
        )

        # Now compute metrics on VAL using that fixed Bayesian threshold
        mean_probs_val, var_probs_val, all_labels_val = mc_dropout_collect_mean_var(model, dataLoaders[VAL], n_passes=MC_PASSES)
        bayes_metrics_tuple = compute_metrics_from_probs(mean_probs_val, all_labels_val, threshold=bayes_best_t)
        bayes_avg_loss, bayes_iou, bayes_acc, bayes_f1, bayes_auc, bayes_dice, bayes_prec, bayes_rec = bayes_metrics_tuple

        # summarise uncertainty (on VAL)
        avg_variance = float(var_probs_val.mean().item())
        max_variance = float(var_probs_val.max().item())

        bayes_results = {
            "settings": {
                "mc_passes": int(MC_PASSES),
                "threshold_rule": {
                    "type": "max_median_dice_on_train_probs",
                    "sweep": "linspace(0.01, 0.99, 99)"
                }
            },
            "best_threshold": float(bayes_best_t),
            "train_median_dice_at_best_threshold": float(bayes_train_med_dice),
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

        # save into checkpoint under separate bayesian namespace
        if os.path.exists(ckpt_path):
            ckpt = safe_load_checkpoint(ckpt_path, map_location="cpu")
        else:
            ckpt = {}
        ckpt.setdefault("bayesian_eval", {})
        ckpt["bayesian_eval"][loss_name] = bayes_results
        torch.save(ckpt, ckpt_path)

        # save a separate bayesian thresholds json
        bayes_thresh_file = os.path.join(SAVE_MODEL_PATH, "best_thresholds_bayesian.json")
        bayes_entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_id": f"{model.__class__.__name__}-{loss_name}",
            "mc_passes": int(MC_PASSES),
            "best_threshold": float(bayes_best_t),
            "train_median_dice_at_best_threshold": float(bayes_train_med_dice),
            "val_dice_at_best_threshold": float(bayes_dice),
            "uncertainty_mean_var": float(avg_variance)
        }
        try:
            with open(bayes_thresh_file, "r") as f:
                db_b = json.load(f)
        except Exception:
            db_b = {}
        db_b[bayes_entry["model_id"]] = bayes_entry
        with open(bayes_thresh_file, "w") as f:
            json.dump(db_b, f, indent=4)

        # per-image metrics file (Bayesian) — here we store only aggregate + uncertainty summary
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
