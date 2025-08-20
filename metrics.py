import torch
import torch.nn.functional as F

# -------------------- numeric stability -------------------- #
EPS = 1e-7
LOGIT_CLAMP = 40.0

def _safe_labels(x):
    # Replace NaN/Inf in labels with 0, keep in [0,1]
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)

def _safe_logits(x):
    # Replace NaN/Inf in logits and clamp to avoid BCE overflow
    return torch.nan_to_num(x, nan=0.0, posinf=LOGIT_CLAMP, neginf=-LOGIT_CLAMP).clamp(-LOGIT_CLAMP, LOGIT_CLAMP)

def _flatten(x):
    return x.reshape(-1)

# ============================ LOSSES ============================ #

def _make_mask_and_targets(gold_mask, pred_logits, ignore_index: int = -1):
    """
    gold_mask: [B,1,H,W] with {0,1} and possibly -1 to ignore
    pred_logits: [B,1,H,W] (raw logits)
    returns: mask (bool), targets (float in {0,1}), logits (sanitized)
    """
    gold_mask = _safe_labels(gold_mask)
    logits    = _safe_logits(pred_logits)

    # valid elements: finite and not equal to ignore_index
    # (labels were sanitized to [0,1], but in case dataset used -1, handle separately)
    if ignore_index is not None:
        mask = torch.isfinite(gold_mask) & (gold_mask != float(ignore_index))
    else:
        mask = torch.isfinite(gold_mask)

    targets = gold_mask.float().clamp(0.0, 1.0)

    # shape assertions
    if logits.shape != targets.shape:
        raise ValueError(f"[metrics] Shape mismatch: pred_logits {logits.shape} vs labels {targets.shape}")

    return mask, targets, logits

def sigmoid_focal_loss_with_logits(logits, targets, mask=None, alpha=0.85, gamma=2.0, reduction="mean"):
    logits  = _safe_logits(logits)
    targets = _safe_labels(targets)

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p   = torch.sigmoid(logits)
    pt  = torch.where(targets > 0.5, p, 1.0 - p)

    alpha_t = torch.where(
        targets > 0.5,
        torch.as_tensor(alpha, device=logits.device, dtype=logits.dtype),
        torch.as_tensor(1.0 - alpha, device=logits.device, dtype=logits.dtype),
    )
    focal = alpha_t * (1.0 - pt).clamp(0.0, 1.0).pow(gamma) * bce

    if mask is not None:
        focal = focal[mask]
        if focal.numel() == 0:
            # return 0 but keep graph
            return logits.new_zeros((), requires_grad=True)

    if reduction == "mean":
        return focal.mean()
    elif reduction == "sum":
        return focal.sum()
    return focal

def soft_dice_loss_logits(logits, targets, mask=None, smooth=1e-6):
    logits  = _safe_logits(logits)
    targets = _safe_labels(targets)

    probs = torch.sigmoid(logits)

    if mask is not None:
        probs   = probs[mask]
        targets = targets[mask]

    if probs.numel() == 0:
        return logits.new_zeros((), requires_grad=True)

    probs   = _flatten(probs)
    targets = _flatten(targets)

    intersection = (probs * targets).sum()
    union        = probs.sum() + targets.sum()
    dice = 1.0 - ((2.0 * intersection + smooth) / (union + smooth))
    return dice

def bce_with_logits_masked(logits, targets, mask=None, reduction="mean", pos_weight=None):
    logits  = _safe_logits(logits)
    targets = _safe_labels(targets)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    if mask is not None:
        loss = loss[mask]
        if loss.numel() == 0:
            return logits.new_zeros((), requires_grad=True)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss

def focal_dice_loss(gold_mask, pred_logits, alpha=0.85, gamma=2.0, dice_weight=2.0, ignore_index=-1):
    mask, targets, logits = _make_mask_and_targets(gold_mask, pred_logits, ignore_index=ignore_index)
    if not mask.any():
        return logits.new_zeros((), requires_grad=True)

    fl = sigmoid_focal_loss_with_logits(logits, targets, mask=mask, alpha=alpha, gamma=gamma, reduction="mean")
    dl = soft_dice_loss_logits(logits, targets, mask=mask, smooth=1e-6)
    loss = fl + dice_weight * dl

    if not torch.isfinite(loss):
        # one extra sanitize & clamp tighter
        logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        fl = sigmoid_focal_loss_with_logits(logits, targets, mask=mask, alpha=alpha, gamma=gamma, reduction="mean")
        dl = soft_dice_loss_logits(logits, targets, mask=mask, smooth=1e-6)
        loss = fl + dice_weight * dl
    return loss

def wbce_dice_loss(gold_mask, pred_logits, dice_weight=1.0, ignore_index=-1, pos_weight=None):
    mask, targets, logits = _make_mask_and_targets(gold_mask, pred_logits, ignore_index=ignore_index)
    if not mask.any():
        return logits.new_zeros((), requires_grad=True)
    bce = bce_with_logits_masked(logits, targets, mask=mask, reduction="mean", pos_weight=pos_weight)
    dl  = soft_dice_loss_logits(logits, targets, mask=mask, smooth=1e-6)
    return bce + dice_weight * dl

def focal_only_loss(gold_mask, pred_logits, alpha=0.85, gamma=2.0, ignore_index=-1):
    mask, targets, logits = _make_mask_and_targets(gold_mask, pred_logits, ignore_index=ignore_index)
    if not mask.any():
        return logits.new_zeros((), requires_grad=True)
    return sigmoid_focal_loss_with_logits(logits, targets, mask=mask, alpha=alpha, gamma=gamma, reduction="mean")

def loss(gold_mask, pred_logits, loss_name: str):
    """
    Dispatcher used by your train/validation loops.
    Expects raw logits for pred_logits.
    """
    name = loss_name.upper().strip()
    if name in ("FOCAL+DICE", "FOCAL_DICE", "FOCALDICE"):
        return focal_dice_loss(gold_mask, pred_logits, alpha=0.85, gamma=1.0, dice_weight=2.0, ignore_index=-1)
    elif name == "FOCAL":
        return focal_only_loss(gold_mask, pred_logits, alpha=0.85, gamma=2.0, ignore_index=-1)
    elif name in ("WBCE+DICE", "WBCE_DICE", "WBCEDICE"):
        return wbce_dice_loss(gold_mask, pred_logits, dice_weight=1.0, ignore_index=-1, pos_weight=None)
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")

# ============================ METRICS ============================ #

def _confusion(y_true, y_pred_bin):
    """
    Computes TP, FP, FN, TN for binary {0,1} tensors.
    Shapes: [B,1,H,W] or [1,H,W] or broadcastable. Returns float scalars on y_true.device.
    """
    y = _safe_labels(y_true).round()
    p = _safe_labels(y_pred_bin).round()

    y = y.to(dtype=torch.bool)
    p = p.to(dtype=torch.bool)

    tp = torch.sum(p & y).to(dtype=torch.float32)
    fp = torch.sum(p & ~y).to(dtype=torch.float32)
    fn = torch.sum(~p & y).to(dtype=torch.float32)
    tn = torch.sum(~p & ~y).to(dtype=torch.float32)

    return tp, fp, fn, tn

def precision_recall(y_true, y_pred_bin):
    tp, fp, fn, _ = _confusion(y_true, y_pred_bin)
    prec = tp / (tp + fp + EPS)
    rec  = tp / (tp + fn + EPS)
    return prec, rec

def accuracy(y_true, y_pred_bin):
    tp, fp, fn, tn = _confusion(y_true, y_pred_bin)
    acc = (tp + tn) / (tp + tn + fp + fn + EPS)
    return acc

def mean_iou(y_true, y_pred_bin):
    """
    Binary IoU (a.k.a. Jaccard index) with smoothing for empty unions.
    """
    y = _safe_labels(y_true).round()
    p = _safe_labels(y_pred_bin).round()

    inter = torch.sum((y == 1) & (p == 1)).to(dtype=torch.float32)
    union = torch.sum((y == 1) | (p == 1)).to(dtype=torch.float32)
    return (inter + EPS) / (union + EPS)

def dice_score(y_true, y_pred_bin):
    """
    Sørensen–Dice coefficient (F1 for sets).
    """
    y = _safe_labels(y_true).round()
    p = _safe_labels(y_pred_bin).round()

    inter = torch.sum((y == 1) & (p == 1)).to(dtype=torch.float32)
    s = torch.sum(y == 1).to(dtype=torch.float32) + torch.sum(p == 1).to(dtype=torch.float32)
    return (2.0 * inter + EPS) / (s + EPS)

def f1_score(y_true, y_pred_bin):
    prec, rec = precision_recall(y_true, y_pred_bin)
    return (2.0 * prec * rec) / (prec + rec + EPS)

def auc_score(y_true, y_pred):
    """
    ROC AUC.
    - If y_pred appears binary (<=2 unique values), return 0.5 (neutral) to avoid misuse.
    - If y_pred is probabilistic in [0,1], compute AUC via Mann–Whitney U (rank statistic).
    """
    y = _safe_labels(y_true).view(-1)
    p = _safe_labels(y_pred).view(-1)

    # Count positives/negatives
    pos = (y > 0.5)
    neg = ~pos
    n_pos = pos.sum()
    n_neg = neg.sum()

    if n_pos == 0 or n_neg == 0:
        return torch.as_tensor(0.5, device=y_true.device, dtype=torch.float32)

    # If predictions look binary, return neutral 0.5
    uniq = torch.unique(p)
    if uniq.numel() <= 2:
        return torch.as_tensor(0.5, device=y_true.device, dtype=torch.float32)

    # Rank-based AUC (equivalent to probability that a random positive > random negative)
    # Sort predictions and compute ranks
    sorted_vals, sorted_idx = torch.sort(p)
    ranks = torch.empty_like(sorted_idx, dtype=torch.float32)
    ranks[sorted_idx] = torch.arange(1, p.numel() + 1, device=p.device, dtype=torch.float32)

    # Sum of ranks for positive samples
    rank_pos = ranks[pos]
    sum_rank_pos = rank_pos.sum()

    auc = (sum_rank_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg + EPS)
    # clamp to [0,1]
    return auc.clamp(0.0, 1.0)
