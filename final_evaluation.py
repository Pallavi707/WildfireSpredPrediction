import os
import json
import torch
import numpy as np
from metrics1 import *
from models import *
from datasets import *
from leejunhyun_unet_models import U_Net
from milesial_unet_model import APAU_Net

# Add seed setup here
import random
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

MODELS = {
    "U_Net": U_Net,
    # Add more models like APAU_Net, AttU_Net if needed
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

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
    pin_memory=True
)

# -------------- Evaluation ------------------
def evaluate_model(model, dataloader, loss_name):
    model.eval()

    per_image_dice = []
    per_image_iou = []
    per_image_f1 = []
    per_image_accuracy = []

    total_loss = total_iou = total_acc = total_f1 = total_auc = total_dice = total_prec = total_rec = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            outputs = model(images)

            total_loss += loss(labels, outputs, loss_name).item()
            total_iou += mean_iou(labels, outputs)
            total_acc += accuracy(labels, outputs)
            total_f1 += f1_score(labels, outputs)
            total_auc += auc_score(labels, outputs)
            total_dice += dice_score(labels, outputs)
            prec, rec = precision_recall(labels, outputs)
            total_prec += prec
            total_rec += rec

            for j in range(images.size(0)):
                per_image_dice.append(float(dice_score(labels[j], outputs[j])))
                per_image_iou.append(float(mean_iou(labels[j], outputs[j])))
                per_image_f1.append(float(f1_score(labels[j], outputs[j])))
                per_image_accuracy.append(float(accuracy(labels[j], outputs[j])))

    n_batches = len(dataloader)
    return {
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
        if file.endswith(".weights"):
            model_file = os.path.join(SAVE_MODEL_PATH, file)

            try:
                print(f"\nEvaluating model file: {file}")

                # Extract model name and loss type from filename
                base = os.path.splitext(file)[0]  # Remove '.weights'
                parts = base.split("-")

                if len(parts) < 4 or not parts[0].startswith("model"):
                    print(f"Skipping invalid filename: {file}")
                    continue

                model_name = parts[1]
                loss_name = "-".join(parts[2:-3])
                model_id = f"{model_name}-{loss_name}"

                if model_name not in MODELS:
                    print(f"Skipping: Unknown model '{model_name}' in file {file}")
                    continue

                model = MODELS[model_name](19, 1).to(DEVICE)
                model.load_state_dict(torch.load(model_file, map_location=DEVICE))

                result = evaluate_model(model, test_loader, loss_name)
                all_results[model_id] = result

                print(f"Evaluation complete for: {model_id}")
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
