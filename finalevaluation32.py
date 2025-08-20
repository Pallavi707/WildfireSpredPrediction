# final_evaluation.py — tile-aware thresholds, robust loss parsing, tolerant state_dict load
import os
import re
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from metrics1 import *
from models import *
from datasets import *
from leejunhyun_unet_models import U_Net
from milesial_unet_model import APAU_Net
from CellularAutomataAllfeaturespostporcessing import U2Net_CA
from CellularAutomataPostprocessing import UNet1_CA

# ----------------- Reproducibility ------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# ----------------- Config ------------------
SAVE_MODEL_PATH = "savedModels18"  # e.g., "savedModels18"
DATASET_PATH = "data18/next-day-wildfire-spread"
RESULT_JSON = os.path.join(SAVE_MODEL_PATH, "test_results.json")

# try to detect tile number (e.g., 18) from SAVE_MODEL_PATH to find *_18.json files
_tile_match = re.search(r"(\d+)$", SAVE_MODEL_PATH)
TILE_SUFFIX = _tile_match.group(1) if _tile_match else None

# base (non-suffixed) and tile-suffixed fallbacks
BEST_THRESHOLDS_PLAIN = os.path.join(SAVE_MODEL_PATH, "best_thresholds.json")
BEST_THRESHOLDS_TILED = os.path.join(SAVE_MODEL_PATH, f"best_thresholds_{TILE_SUFFIX}.json") if TILE_SUFFIX else None

BAYES_SUMMARY_PLAIN = os.path.join(SAVE_MODEL_PATH, "bayesian_eval_summary.json")
BAYES_SUMMARY_TILED = os.path.join(SAVE_MODEL_PATH, f"bayesian_eval_summary_{TILE_SUFFIX}.json") if TILE_SUFFIX else None

# --- Test-time MC config/paths ---
MC_PASSES = int(os.environ.get("MC_DROPOUT_PASSES", "30"))
DO_TEST_CONFIDENCE = True
TEST_BAYES_SUMMARY_JSON = os.path.join(SAVE_MODEL_PATH, "test_bayesian_eval_summary.json")

MODELS = {
    "U2Net_CA": U2Net_CA,
    # "U_Net": U_Net,
    # "APAU_Net": APAU_Net,
    # "UNet1_CA": UNet1_CA,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = torch.cuda.is_available()
BATCH_SIZE = 8

ALL_FEATURES = [
    'elevation','fws','population','pdsi','pr','sph','slope','PrevFireMask',
    'erc','NDVI','fpr','ftemp','th','EVI','vs','tmmx','fwd','aspect','tmmn'
]

TILE_SIZE = int(TILE_SUFFIX) if TILE_SUFFIX is not None else 64

print(f"Using TILE_SIZE: {TILE_SIZE} (from TILE_SUFFIX: {TILE_SUFFIX})")    


test_dataset = WildfireDataset(
    f"{DATASET_PATH}/test.data",
    f"{DATASET_PATH}/test.labels",
    features=list(range(len(ALL_FEATURES))),
    crop_size=TILE_SIZE
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=PIN_MEMORY
)

# -------------- Helpers ------------------
def logit(p: float) -> float:
    p = float(p); eps = 1e-6
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))

def _read_json(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def load_saved_threshold(model_name: str, loss_name: str, weights_path: str) -> float:
    """
    Priority:
      1) matching .pt checkpoint next to the .weights (reads 'best_threshold')
      2) best_thresholds_<tile>.json with key '<model>-<loss>-<tile>'
      3) best_thresholds_<tile>.json with key '<model>-<loss>'
      4) best_thresholds.json with either key
      5) default 0.5
    """
    

    print(f"Note: using default threshold 0.5 for {model_name}-{loss_name} (no saved threshold found).")
    return 0.3

# ---- Validation-time Bayesian summary ----------------------------------------
def load_bayesian_summary():
    # prefer tile-suffixed; fall back to plain
    db = _read_json(BAYES_SUMMARY_TILED) if BAYES_SUMMARY_TILED else None
    if isinstance(db, dict):
        return db
    db = _read_json(BAYES_SUMMARY_PLAIN)
    if isinstance(db, dict):
        return db
    print(f"[Confidence] {BAYES_SUMMARY_PLAIN} not found — confidence will be omitted.")
    return {}

def confidence_from_var(var_mean: float) -> float:
    return 1.0 / (1.0 + float(var_mean))

def extract_confidence(bayes_db: dict, model_name: str, loss_name: str, tile: str = None):
    # Try tile-suffixed key first, then non-tiled
    keys_to_try = []
    if tile:
        keys_to_try.append(f"{model_name}-{loss_name}-{tile}")
    keys_to_try.append(f"{model_name}-{loss_name}")

    entry = None
    for k in keys_to_try:
        if isinstance(bayes_db, dict) and k in bayes_db:
            entry = bayes_db[k]
            break
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

# ---- Test-time MC Dropout helpers ---------------------------------------
@torch.no_grad()
def _enable_mc_dropout(model: torch.nn.Module):
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
        for images, _labels in dataloader:
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

# -------------- Evaluation ------------------
def evaluate_model(model, dataloader, loss_name, threshold: float):
    """
    Evaluate using exactly the saved threshold t* by shifting logits:
    logits_shift = logits - logit(t*)
    """
    model.eval()
    t_logit = logit(threshold)

    per_image_dice, per_image_iou, per_image_f1, per_image_accuracy = [], [], [], []
    totals = dict(loss=0.0, iou=0.0, acc=0.0, f1=0.0, auc=0.0, dice=0.0, prec=0.0, rec=0.0)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits = model(images)

            # loss & AUC use unshifted logits
            totals["loss"] += loss(labels, logits, loss_name).item()
            totals["auc"]  += auc_score(labels, logits)

            # binarizing metrics use shifted logits
            logits_shift = logits - t_logit
            totals["iou"]  += mean_iou(labels, logits_shift)
            totals["acc"]  += accuracy(labels, logits_shift)
            totals["f1"]   += f1_score(labels, logits_shift)
            totals["dice"] += dice_score(labels, logits_shift)
            pr, rc = precision_recall(labels, logits_shift)
            totals["prec"] += pr
            totals["rec"]  += rc

            for j in range(images.size(0)):
                per_image_dice.append(float(dice_score(labels[j], logits_shift[j])))
                per_image_iou.append(float(mean_iou(labels[j], logits_shift[j])))
                per_image_f1.append(float(f1_score(labels[j], logits_shift[j])))
                per_image_accuracy.append(float(accuracy(labels[j], logits_shift[j])))

    n_batches = len(dataloader)
    return {
        "threshold_used": float(threshold),
        "avg_metrics": {
            "loss": float(totals["loss"] / n_batches),
            "iou": float(totals["iou"] / n_batches),
            "accuracy": float(totals["acc"] / n_batches),
            "f1": float(totals["f1"] / n_batches),
            "auc": float(totals["auc"] / n_batches),
            "dice": float(totals["dice"] / n_batches),
            "precision": float(totals["prec"] / n_batches),
            "recall": float(totals["rec"] / n_batches),
        },
        "per_image_metrics": {
            "dice": per_image_dice,
            "iou": per_image_iou,
            "f1": per_image_f1,
            "accuracy": per_image_accuracy
        }
    }

def _parse_filename(base_no_ext: str):
    """
    Parse model filename like:
      model-U2Net_CA-FOCAL+DICE-bestF1Score-18
    Robustly extracts: model_name, loss_name
    """
    parts = base_no_ext.split("-")
    if len(parts) < 3 or parts[0] != "model":
        return None, None

    model_name = parts[1]

    # find the index of the marker 'bestF1Score' (or last if missing)
    try:
        idx_best = parts.index("bestF1Score")
    except ValueError:
        idx_best = len(parts)

    # loss name is everything from parts[2] up to (but not including) idx_best
    if idx_best <= 2:
        loss_name = ""  # will be caught later
    else:
        loss_name = "-".join(parts[2:idx_best])

    return model_name, loss_name

# -------------- Main ------------------------
if __name__ == "__main__":
    bayes_db = load_bayesian_summary()

    all_results = {}
    best_model_id = None
    best_f1 = -1.0
    confidence_ranking = []  # (model_id, mean_var, confidence, bayes_val_dice)

    for file in os.listdir(SAVE_MODEL_PATH):
        if not file.endswith(".weights"):
            continue

        print('--------------------------------------------------------------------')
        print(f"\nEvaluating model file: {file}")
        model_file = os.path.join(SAVE_MODEL_PATH, file)
        base = os.path.splitext(file)[0]

        model_name, loss_name = _parse_filename(base)
        if not model_name:
            print(f"Skipping invalid filename: {file}")
            continue

        if model_name not in MODELS:
            print(f"Skipping: Unknown model '{model_name}' in file {file}")
            continue

        if not loss_name:
            print(f"⚠️ Could not parse loss name from '{file}'. Assuming 'FOCAL+DICE'.")
            loss_name = "FOCAL+DICE"

        # infer tile from filename for lookups (e.g., '-18')
        m_tile = re.search(r"-(\d+)$", base)
        tile = m_tile.group(1) if m_tile else None

        try:
            # Build model and load weights (tolerant)
            model = MODELS[model_name](19, 1).to(DEVICE)
            state = torch.load(model_file, map_location=DEVICE)

            try:
                load_msg = model.load_state_dict(state, strict=False)
                missing = getattr(load_msg, "missing_keys", [])
                unexpected = getattr(load_msg, "unexpected_keys", [])
                if missing:
                    print(f"⚠️ Missing keys while loading: {missing[:6]}{'…' if len(missing)>6 else ''}")
                if unexpected:
                    print(f"⚠️ Unexpected keys while loading: {unexpected[:6]}{'…' if len(unexpected)>6 else ''}")
            except Exception as e:
                print(f"Strict load failed ({e}), retrying strict=False…")
                load_msg = model.load_state_dict(state, strict=False)

            # Load exact threshold that was saved
            t_star = load_saved_threshold(model_name, loss_name, model_file)

            # Evaluate
            result = evaluate_model(model, test_loader, loss_name, threshold=t_star)

            # ---- Attach VALIDATION Bayesian confidence if available ----
            model_id = f"{model_name}-{loss_name}"  # use non-tiled id in results for stability
            conf_info = extract_confidence(bayes_db, model_name, loss_name, tile)
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

            # ---- Test-time confidence (MC on test set; reporting only) ----
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
                    test_bayes_db = _read_json(TEST_BAYES_SUMMARY_JSON) or {}
                    test_bayes_db[model_id] = result["test_bayesian"]
                    with open(TEST_BAYES_SUMMARY_JSON, "w") as f:
                        json.dump(test_bayes_db, f, indent=4)
                except Exception as e:
                    print(f"[Test Confidence] Failed to write {TEST_BAYES_SUMMARY_JSON}: {e}")

            all_results[model_id] = result

            print(f"Evaluation complete for: {model_id}")
            print(f"Threshold used (deterministic): {t_star:.4f}")
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

    # ----------- Print Best Model (by F1) -----------------
    if best_model_id:
        print(f"\nBest Performing Model (by F1): {best_model_id}")
        print(f"F1-score: {best_f1:.4f}")
        print(f"Metrics: {json.dumps(all_results[best_model_id]['avg_metrics'], indent=2)}")
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
