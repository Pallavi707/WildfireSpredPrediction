import os
import json
import torch
import numpy as np
from metrics1 import *
from models import *
from datasets import *
from leejunhyun_unet_models import U_Net
from milesial_unet_model import APAU_Net
from CellularAutomataAllfeaturespostporcessing import U2Net_CA
from CellularAutomataPostprocessing import UNet1_CA
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.checkpoint import checkpoint
import random
import math

# ----------------- Reproducibility ------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ----------------- Config ------------------
SAVE_MODEL_PATH = "savedModels"
DATASET_PATH = "data/next-day-wildfire-spread"
RESULT_JSON = os.path.join(SAVE_MODEL_PATH, "test_results.json")
BEST_THRESHOLDS_FALLBACK = os.path.join(SAVE_MODEL_PATH, "best_thresholds.json")
BAYES_SUMMARY_JSON = os.path.join(SAVE_MODEL_PATH, "bayesian_eval_summary.json")  # validation-time bayes

# --- NEW: test-time MC config/paths ---
MC_PASSES = int(os.environ.get("MC_DROPOUT_PASSES", "30"))
DO_TEST_CONFIDENCE = True
TEST_BAYES_SUMMARY_JSON = os.path.join(SAVE_MODEL_PATH, "test_bayesian_eval_summary.json")

# --- NEW: compact metrics summary (for easy plotting) ---
TEST_METRICS_SUMMARY_JSON = os.path.join(SAVE_MODEL_PATH, "test_metrics_summary.json")

# --- NEW: qualitative plots (test) ---
TEST_PLOTS_DIR = os.path.join(SAVE_MODEL_PATH, "test_plots")

# --- NEW: feature index for previous fire mask ---
PREV_FIRE_IDX = 7  # <- matches your feature order in ALL_FEATURES

MODELS = {
    # "U_Net": U_Net,
    # "APAU_Net": APAU_Net,
    # "UNet1_CA": UNet1_CA,
    "U2Net_CA": U2Net_CA,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = torch.cuda.is_available()
BATCH_SIZE = 8

# --------------- Dataset -------------------
ALL_FEATURES = [
    'elevation', 'fws', 'population', 'pdsi', 'pr', 'sph', 'slope', 'PrevFireMask',
    'erc', 'NDVI', 'fpr', 'ftemp', 'th', 'EVI', 'vs', 'tmmx', 'fwd', 'aspect', 'tmmn'
]

test_dataset = WildfireDataset(
    f"{DATASET_PATH}/test.data",
    f"{DATASET_PATH}/test.labels",
    features=list(range(len(ALL_FEATURES))),
    crop_size=64
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=PIN_MEMORY
)

# ----------------- (tiny plotting util) -----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _save_test_examples(model, dataloader, threshold, out_dir, model_id, max_images=16):
    """
    Save a grid of GT / probability / binary prediction for up to max_images from TEST.
    Produces: {out_dir}/{model_id}_test_examples.png
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    panels = []
    saved = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
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
        print("[Plot] No test images available to save examples.")
        return

    rows = len(panels)
    cols = 3
    fig_h = max(2, int(rows * 1.8))
    fig_w = 9
    plt.figure(figsize=(fig_w, fig_h))
    for r, (gt, pr, pd) in enumerate(panels, start=1):
        ax = plt.subplot(rows, cols, (r-1)*cols + 1)
        ax.imshow(gt, interpolation="nearest")
        ax.set_title("GT")
        ax.axis("off")

        ax = plt.subplot(rows, cols, (r-1)*cols + 2)
        ax.imshow(pr, interpolation="nearest")
        ax.set_title("Prob")
        ax.axis("off")

        ax = plt.subplot(rows, cols, (r-1)*cols + 3)
        ax.imshow(pd, interpolation="nearest")
        ax.set_title(f"Pred (t={threshold:.2f})")
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{model_id}_test_examples.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved test examples → {out_path}")

# --- NEW: helpers to create the 4-row grid like your training figure ---
def _comparison_rgb(gt, pred):
    """RGB overlay: TP→green, FP→red, FN→blue."""
    gt = (gt > 0.5).astype(np.uint8)
    pred = (pred > 0.5).astype(np.uint8)
    tp = (gt == 1) & (pred == 1)
    fp = (gt == 0) & (pred == 1)
    fn = (gt == 1) & (pred == 0)

    H, W = gt.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[..., 1] = tp.astype(np.float32)  # G
    rgb[..., 0] = fp.astype(np.float32)  # R
    rgb[..., 2] = fn.astype(np.float32)  # B
    return rgb

def _collect_prev_gt_pred(model, dataloader, threshold, max_cols=12):
    """Collect PrevFireMask | GT | Pred samples for the grid."""
    model.eval()
    prev_list, gt_list, pred_list = [], [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            # previous fire mask channel
            prev = images[:, PREV_FIRE_IDX]  # [B,H,W]
            prev = (prev > 0.5).float()

            for i in range(images.size(0)):
                prev_list.append(prev[i].detach().cpu().numpy())
                gt_list.append(labels[i, 0].detach().cpu().numpy())
                pred_list.append(preds[i, 0].detach().cpu().numpy())
                if len(prev_list) >= max_cols:
                    return prev_list, gt_list, pred_list
    return prev_list, gt_list, pred_list

def _save_test_fire_grid(model, dataloader, threshold, out_dir, title, max_cols=12):
    """Save a 4×N grid: Prev | GT | Pred | Comparison, like the reference figure."""
    os.makedirs(out_dir, exist_ok=True)
    prev_list, gt_list, pred_list = _collect_prev_gt_pred(model, dataloader, threshold, max_cols=max_cols)
    N = len(prev_list)
    if N == 0:
        print("[Plot] No test samples to plot fire grid.")
        return

    fig = plt.figure(figsize=(1.8 * N, 7.2))
    if title:
        fig.suptitle(title, y=0.98, fontsize=12)

    rows = 4
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

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = os.path.join(out_dir, "TEMP_grid_name.png")  # overwritten by caller with a full name
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

# -------------- Helpers ------------------
def logit(p: float) -> float:
    p = float(p)
    eps = 1e-6
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))

def load_saved_threshold(model_name: str, loss_name: str, weights_path: str) -> float:
    """
    Resolve the exact threshold saved at training time for this .weights file.
    Priority:
      1) matching .pt checkpoint sitting next to the .weights
      2) best_thresholds.json entry for model_id
      3) fallback 0.5
    """
    base = os.path.splitext(os.path.basename(weights_path))[0]
    ckpt_path = os.path.join(SAVE_MODEL_PATH, base.replace(".weights", "") + ".pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.splitext(weights_path)[0] + ".pt"

    # 1) Try exact checkpoint
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict) and "best_threshold" in ckpt:
                return float(ckpt["best_threshold"])
        except Exception as e:
            print(f"Warning: failed to read checkpoint threshold from {ckpt_path}: {e}")

    # 2) Fallback JSON registry
    model_id = f"{model_name}-{loss_name}"
    if os.path.exists(BEST_THRESHOLDS_FALLBACK):
        try:
            with open(BEST_THRESHOLDS_FALLBACK, "r") as f:
                db = json.load(f)
            if model_id in db and "best_threshold" in db[model_id]:
                return float(db[model_id]["best_threshold"])
        except Exception as e:
            print(f"Warning: failed to read {BEST_THRESHOLDS_FALLBACK}: {e}")

    # 3) Default
    print(f"Note: using default threshold 0.5 for {model_id} (no saved threshold found).")
    return 0.5

# ---- Bayesian confidence (validation) ----------------------------------------
def load_bayesian_summary():
    if not os.path.exists(BAYES_SUMMARY_JSON):
        print(f"[Confidence] {BAYES_SUMMARY_JSON} not found — confidence will be omitted.")
        return {}
    try:
        with open(BAYES_SUMMARY_JSON, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Confidence] Failed to read {BAYES_SUMMARY_JSON}: {e}")
        return {}

def confidence_from_var(var_mean: float) -> float:
    # Lower variance → higher confidence (monotonic mapping)
    return 1.0 / (1.0 + float(var_mean))

def extract_confidence(bayes_db: dict, model_id: str):
    """
    Returns None if missing; else a dict with:
      mean_var, max_var, confidence, bayes_best_t, bayes_val_dice, bayes_val_iou, bayes_val_auc
    """
    entry = bayes_db.get(model_id)
    if not isinstance(entry, dict):
        return None
    try:
        u = entry["uncertainty_summary"]
        m = entry.get("val_metrics_at_t", {})
        mean_var = float(u.get("predictive_variance_mean", float("nan")))
        max_var = float(u.get("predictive_variance_max", float("nan")))
        conf = confidence_from_var(mean_var)
        bayes_best_t = float(entry.get("best_threshold", float("nan")))
        bayes_val_dice = float(m.get("dice", float("nan")))
        bayes_val_iou = float(m.get("iou", float("nan")))
        bayes_val_auc = float(m.get("auc", float("nan")))
        return {
            "mean_var": mean_var,
            "max_var": max_var,
            "confidence": conf,
            "source": "validation",
            "bayes_best_threshold": bayes_best_t,
            "bayes_val_dice": bayes_val_dice,
            "bayes_val_iou": bayes_val_iou,
            "bayes_val_auc": bayes_val_auc,
        }
    except Exception:
        return None

# ---- NEW: Test-time MC Dropout helpers ---------------------------------------
@torch.no_grad()
def _enable_mc_dropout(model: torch.nn.Module):
    # Enable dropout layers during inference for MC Dropout
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.train()

def _model_has_dropout(model: torch.nn.Module) -> bool:
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            return True
    return False

@torch.no_grad()
def mc_dropout_collect_mean_var_test(model, dataloader, n_passes=MC_PASSES):
    """
    Test-time MC passes for uncertainty (predictive variance).
    Returns mean_probs [N,1,H,W] and var_probs [N,1,H,W].
    """
    model.eval()
    has_do = _model_has_dropout(model)
    if has_do:
        _enable_mc_dropout(model)
        input_noise_sigma = 0.0
    else:
        input_noise_sigma = 0.03  # tiny input-noise fallback

    pass_sums = None
    pass_sq_sums = None

    for _ in range(n_passes):
        probs_pass = []
        for images, _labels in dataloader:  # labels unused for uncertainty
            images = images.to(DEVICE, non_blocking=True)
            if input_noise_sigma > 0.0:
                images = images + torch.randn_like(images) * input_noise_sigma
            logits = model(images)
            probs = torch.sigmoid(logits)  # [B,1,H,W]
            probs_pass.append(probs)

        probs_pass = torch.cat(probs_pass, dim=0)  # [N,1,H,W]
        if pass_sums is None:
            pass_sums = probs_pass.clone()
            pass_sq_sums = probs_pass.pow(2)
        else:
            pass_sums.add_(probs_pass)
            pass_sq_sums.add_(probs_pass.pow(2))

    mean_probs = pass_sums / float(n_passes)
    var_probs = pass_sq_sums / float(n_passes) - mean_probs.pow(2)
    return mean_probs, var_probs
# -------------------------------------------------------------------------------

# -------------- Evaluation ------------------
def evaluate_model(model, dataloader, loss_name, threshold: float):
    """
    Evaluate using exactly the saved threshold t*.

    Trick: your metrics threshold at 'pred_mask > 0' on the logits.
    To make that equivalent to 'sigmoid(logits) > t*', we pass 'logits_shift = logits - logit(t*)'
    into the metrics that binarize. For AUC (and distance) which expect probabilities,
    we still pass the unshifted logits so their internal sigmoid() yields the true probs.
    """
    model.eval()

    t_logit = logit(threshold)

    per_image_dice = []
    per_image_iou = []
    per_image_f1 = []
    per_image_accuracy = []

    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    total_auc = 0.0
    total_dice = 0.0
    total_prec = 0.0
    total_rec = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits = model(images)

            # Use original logits for: loss, auc_score
            total_loss += loss(labels, logits, loss_name).item()
            total_auc += auc_score(labels, logits)

            # Use shifted logits for all binarizing metrics
            logits_shift = logits - t_logit

            total_iou += mean_iou(labels, logits_shift)
            total_acc += accuracy(labels, logits_shift)
            total_f1 += f1_score(labels, logits_shift)
            total_dice += dice_score(labels, logits_shift)
            prec, rec = precision_recall(labels, logits_shift)
            total_prec += prec
            total_rec += rec

            for j in range(images.size(0)):
                per_image_dice.append(float(dice_score(labels[j], logits_shift[j])))
                per_image_iou.append(float(mean_iou(labels[j], logits_shift[j])))
                per_image_f1.append(float(f1_score(labels[j], logits_shift[j])))
                per_image_accuracy.append(float(accuracy(labels[j], logits_shift[j])))

    n_batches = len(dataloader)
    return {
        "threshold_used": float(threshold),
        "avg_metrics": {
            "loss": float(total_loss / n_batches),
            "iou": float(total_iou / n_batches),
            "accuracy": float(total_acc / n_batches),
            "f1": float(total_f1 / n_batches),
            "auc": float(total_auc / n_batches),
            "dice": float(total_dice / n_batches),
            "precision": float(total_prec / n_batches),
            "recall": float(total_rec / n_batches),
        },
        "per_image_metrics": {
            "dice": per_image_dice,
            "iou": per_image_iou,
            "f1": per_image_f1,
            "accuracy": per_image_accuracy
        }
    }

# -------------- Main ------------------------
if __name__ == "__main__":
    # Load Bayesian summary once for confidence lookups (validation-based)
    bayes_db = load_bayesian_summary()

    all_results = {}
    best_model_id = None
    best_f1 = -1.0  # initialize with very low value

    confidence_ranking = []  # (model_id, mean_var, confidence, bayes_val_dice) from VALIDATION bayes

    # --- NEW: compact summary holder for quick plotting across models ---
    compact_summary = {}  # model_id -> metrics

    for file in os.listdir(SAVE_MODEL_PATH):
        if not file.endswith(".weights"):
            continue

        model_file = os.path.join(SAVE_MODEL_PATH, file)
        print('--------------------------------------------------------------------')
        print(f"\nEvaluating model file: {file}")

        # Extract model name and loss type from filename
        base = os.path.splitext(file)[0]
        parts = base.split("-")  # e.g. ['model','U_Net','WBCE+DICE','bestF1Score','Rank','0']

        if len(parts) < 4 or parts[0] != "model":
            print(f"Skipping invalid filename: {file}")
            continue

        model_name = parts[1]
        loss_name = "-".join(parts[2:-3])  # handle losses with hyphens/plus
        model_id = f"{model_name}-{loss_name}"

        if model_name not in MODELS:
            print(f"Skipping: Unknown model '{model_name}' in file {file}")
            continue

        try:
            # Build model and load weights
            model = MODELS[model_name](19, 1).to(DEVICE)
            state = torch.load(model_file, map_location=DEVICE)
            model.load_state_dict(state)

            # Load EXACT threshold that was saved for this model/loss
            t_star = load_saved_threshold(model_name, loss_name, model_file)

            # Evaluate with that threshold
            result = evaluate_model(model, test_loader, loss_name, threshold=t_star)

            # ---- Attach VALIDATION Bayesian confidence if available ----
            conf_info = extract_confidence(bayes_db, model_id)
            if conf_info is not None:
                result["bayesian"] = conf_info
                confidence_ranking.append(
                    (model_id, conf_info["mean_var"], conf_info["confidence"], conf_info["bayes_val_dice"])
                )
                print(f"[Val Confidence] ≈{conf_info['confidence']:.6f} "
                      f"(mean_var={conf_info['mean_var']:.6g}) | "
                      f"Bayes t*={conf_info['bayes_best_threshold']:.2f} | "
                      f"Bayes Dice@t*={conf_info['bayes_val_dice']:.4f}")
            else:
                print("[Val Confidence] N/A (no Bayesian entry found)")

            # ---- NEW: Test-time confidence (MC on test set; reporting only) ----
            if DO_TEST_CONFIDENCE:
                print("[Test Confidence] Running MC on test set ...")
                mean_probs, var_probs = mc_dropout_collect_mean_var_test(model, test_loader, n_passes=MC_PASSES)
                test_mean_var = float(var_probs.mean().item())
                test_max_var  = float(var_probs.max().item())
                test_conf     = confidence_from_var(test_mean_var)

                result["test_bayesian"] = {
                    "mc_passes": MC_PASSES,
                    "predictive_variance_mean": test_mean_var,
                    "predictive_variance_max":  test_max_var,
                    "confidence": test_conf,
                    "source": "test"
                }

                print(f"[Test Confidence] ≈{test_conf:.6f} "
                      f"(mean_var={test_mean_var:.6g}, max_var={test_max_var:.6g})")

                # Persist compact summary per model_id
                try:
                    if os.path.exists(TEST_BAYES_SUMMARY_JSON):
                        with open(TEST_BAYES_SUMMARY_JSON, "r") as f:
                            test_bayes_db = json.load(f)
                    else:
                        test_bayes_db = {}
                    test_bayes_db[model_id] = result["test_bayesian"]
                    with open(TEST_BAYES_SUMMARY_JSON, "w") as f:
                        json.dump(test_bayes_db, f, indent=4)
                except Exception as e:
                    print(f"[Test Confidence] Failed to write {TEST_BAYES_SUMMARY_JSON}: {e}")

            all_results[model_id] = result

            print(f"Evaluation complete for: {model_id}")
            print(f"Threshold used (deterministic): {t_star:.4f}")
            print(json.dumps(result["avg_metrics"], indent=2))

            # --- Save the simple 3-column test image (unchanged) ---
            try:
                _save_test_examples(model, test_loader, threshold=t_star, out_dir=TEST_PLOTS_DIR, model_id=model_id, max_images=16)
            except Exception as e:
                print(f"[Plot] Skipped saving test examples for {model_id} due to error: {e}")

            # --- NEW: Save the 4-row fire grid like your reference image ---
            try:
                title = f"{model_name} — {loss_name} — best t={t_star:.2f}"
                temp_path = _save_test_fire_grid(model, test_loader, threshold=t_star,
                                                 out_dir=TEST_PLOTS_DIR, title=title, max_cols=12)
                # rename to a stable file name per model
                final_path = os.path.join(TEST_PLOTS_DIR, f"{model_id}_test_fire_grid.png")
                if temp_path and os.path.exists(temp_path):
                    os.replace(temp_path, final_path)
                print(f"[Plot] Saved fire grid → {final_path}")
            except Exception as e:
                print(f"[Plot] Skipped saving fire grid for {model_id} due to error: {e}")

            # --- Fill compact summary for plotting ---
            m = result["avg_metrics"]
            compact_entry = {
                "model_id": model_id,
                "threshold": float(result.get("threshold_used", t_star)),
                "f1": float(m.get("f1", 0.0)),
                "dice": float(m.get("dice", 0.0)),
                "iou": float(m.get("iou", 0.0)),
                "precision": float(m.get("precision", 0.0)),
                "recall": float(m.get("recall", 0.0)),
                "auc": float(m.get("auc", 0.0)),
                "accuracy": float(m.get("accuracy", 0.0)),
            }
            if "bayesian" in result and isinstance(result["bayesian"], dict):
                compact_entry.update({
                    "val_confidence": float(result["bayesian"].get("confidence", float("nan"))),
                    "val_mean_var": float(result["bayesian"].get("mean_var", float("nan")))
                })
            if "test_bayesian" in result and isinstance(result["test_bayesian"], dict):
                compact_entry.update({
                    "test_confidence": float(result["test_bayesian"].get("confidence", float("nan"))),
                    "test_mean_var": float(result["test_bayesian"].get("predictive_variance_mean", float("nan")))
                })
            compact_summary[model_id] = compact_entry

            # Track best model based on F1-score
            current_f1 = result["avg_metrics"]["f1"]
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_model_id = model_id

        except Exception as e:
            print(f"Error evaluating {file}: {e}")

    # --- Persist the detailed results (unchanged behavior) ---
    with open(RESULT_JSON, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nAll test results saved to {RESULT_JSON}")

    # --- Persist compact summary for plotting across models ---
    try:
        with open(TEST_METRICS_SUMMARY_JSON, "w") as f:
            json.dump(compact_summary, f, indent=4)
        print(f"Compact test metrics summary saved to {TEST_METRICS_SUMMARY_JSON}")
    except Exception as e:
        print(f"Failed to write compact summary JSON: {e}")

    # ----------- Print Best Model (by F1) -----------------
    if best_model_id:
        print(f"\nBest Performing Model (by F1): {best_model_id}")
        print(f"F1-score: {best_f1:.4f}")
        print(f"Metrics: {json.dumps(all_results[best_model_id]['avg_metrics'], indent=2)}")
        # show both val and test confidence if present
        if "bayesian" in all_results[best_model_id]:
            b = all_results[best_model_id]["bayesian"]
            print(f"[Val Confidence] ≈{b['confidence']:.6f} "
                  f"(mean_var={b['mean_var']:.6g}, max_var={b['max_var']:.6g})")
        if "test_bayesian" in all_results[best_model_id]:
            tb = all_results[best_model_id]["test_bayesian"]
            print(f"[Test Confidence] ≈{tb['confidence']:.6f} "
                  f"(mean_var={tb['predictive_variance_mean']:.6g}, "
                  f"max_var={tb['predictive_variance_max']:.6g})")

    # ----------- Also show Most Confident Model (by validation mean variance) ------------
    if confidence_ranking:
        confidence_ranking.sort(key=lambda x: (x[1], -x[3]))  # lowest mean_var; tie-break by higher Dice
        mid, mean_var, conf, bayes_dice = confidence_ranking[0]
        print(f"\nMost Confident Model (Validation Bayesian): {mid}")
        print(f"mean_var={mean_var:.6g}  confidence≈{conf:.6f}  Bayes Dice@t*={bayes_dice:.4f}")
