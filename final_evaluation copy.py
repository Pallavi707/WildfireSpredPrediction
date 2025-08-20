import os
import json
import torch
import numpy as np
from metrics1 import *
from models import *
from datasets import *
from leejunhyun_unet_models import U_Net
from milesial_unet_model import APAU_Net
from CellularAutomataPostprocessing import UNetWithPostCA, NeuralCA
from CellularAutomataAllfeaturespostporcessing import UNetWithPostCA19
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

MODELS = {
    "U_Net": U_Net,
    # "UNet1": UNet1,
    # "UNetWithPostCA": UNetWithPostCA,
    # "UNetWithPostCA19": UNetWithPostCA19,
    # "APAU_Net": APAU_Net,
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
    base = os.path.splitext(os.path.basename(weights_path))[0]  # e.g. model-U_Net-WBCE+DICE-bestF1Score-Rank-0
    ckpt_path = os.path.join(SAVE_MODEL_PATH, base.replace(".weights", "") + ".pt")
    # If the .pt has same stem as .weights (usual in your training script), this finds it:
    if not os.path.exists(ckpt_path):
        # direct replace extension
        ckpt_path = os.path.splitext(weights_path)[0] + ".pt"

    # 1) Try exact checkpoint
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")  # safe: .pt saved by you
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

            # Use original logits for: loss, auc_score, distance (they call sigmoid inside)
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
    all_results = {}
    best_model_id = None
    best_f1 = -1.0  # initialize with very low value

    for file in os.listdir(SAVE_MODEL_PATH):
        if not file.endswith(".weights"):
            continue

        model_file = os.path.join(SAVE_MODEL_PATH, file)
        print(f"\nEvaluating model file: {file}")

        # Extract model name and loss type from filename
        base = os.path.splitext(file)[0]  # Remove '.weights'
        parts = base.split("-")           # e.g., ['model','U_Net','WBCE+DICE','bestF1Score','Rank','0']

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
            all_results[model_id] = result

            print(f"Evaluation complete for: {model_id}")
            print(f"Threshold used: {t_star:.4f}")
            print(json.dumps(result["avg_metrics"], indent=2))

            # Track best model based on F1-score
            current_f1 = result["avg_metrics"]["f1"]
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_model_id = model_id

        except Exception as e:
            print(f"Error evaluating {file}: {e}")

    with open(RESULT_JSON, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nAll test results saved to {RESULT_JSON}")

    # ----------- Print Best Model -----------------
    if best_model_id:
        print(f"\nBest Performing Model: {best_model_id}")
        print(f"F1-score: {best_f1:.4f}")
        print(f"Metrics: {json.dumps(all_results[best_model_id]['avg_metrics'], indent=2)}")
