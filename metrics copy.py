import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# ---------------------- Focal Loss ----------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

# ---------------------- Dice Loss ----------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

# ---------------------- Combined Loss ----------------------
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        return self.focal(inputs, targets) + self.dice(inputs, targets)

# ---------------------- Evaluation Functions ----------------------
def mean_iou(gold_mask, pred_mask):
    pred_mask_binary = (pred_mask > 0.).int()
    mask = gold_mask != -1
    gold_mask_masked = gold_mask[mask]
    pred_mask_masked = pred_mask_binary[mask]
    intersection = (gold_mask_masked * pred_mask_masked).sum()
    union = gold_mask_masked.sum() + pred_mask_masked.sum() - intersection
    gold_mask_masked = 1 - gold_mask_masked
    pred_mask_masked = 1 - pred_mask_masked
    intersection_0 = (gold_mask_masked * pred_mask_masked).sum()
    union_0 = gold_mask_masked.sum() + pred_mask_masked.sum() - intersection_0
    return ((intersection / union) + (intersection_0 / union_0)) / 2

def accuracy(gold_mask, pred_mask):
    pred_mask_binary = (pred_mask > 0.).int()
    mask = gold_mask != -1
    gold_mask_masked = gold_mask[mask]
    pred_mask_masked = pred_mask_binary[mask]
    accuracy_1 = (gold_mask_masked == pred_mask_masked).sum() / len(gold_mask_masked)
    gold_mask_masked = 1 - gold_mask_masked
    pred_mask_masked = 1 - pred_mask_masked
    accuracy_0 = (gold_mask_masked == pred_mask_masked).sum() / len(gold_mask_masked)
    return (accuracy_1 + accuracy_0) / 2

def distance(gold_mask, pred_mask):
    mask = gold_mask != -1
    gold_mask_masked = gold_mask[mask].float()
    pred_mask_masked = torch.sigmoid(pred_mask[mask].float())
    return torch.linalg.norm(gold_mask_masked - pred_mask_masked)

def f1_score(gold_mask, pred_mask):
    pred_mask_binary = (pred_mask > 0.).int()
    mask = gold_mask != -1
    gold_mask_masked = gold_mask[mask]
    pred_mask_masked = pred_mask_binary[mask]
    tp_1 = (gold_mask_masked * pred_mask_masked).sum()
    fp_1 = ((1 - gold_mask_masked) * pred_mask_masked).sum()
    fn_1 = (gold_mask_masked * (1 - pred_mask_masked)).sum()
    precision_1 = tp_1 / (tp_1 + fp_1 + 1e-6)
    recall_1 = tp_1 / (tp_1 + fn_1 + 1e-6)
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1 + 1e-6)
    gold_mask_masked_0 = 1 - gold_mask_masked
    pred_mask_masked_0 = 1 - pred_mask_masked
    tp_0 = (gold_mask_masked_0 * pred_mask_masked_0).sum()
    fp_0 = ((1 - gold_mask_masked_0) * pred_mask_masked_0).sum()
    fn_0 = (gold_mask_masked_0 * (1 - pred_mask_masked_0)).sum()
    precision_0 = tp_0 / (tp_0 + fp_0 + 1e-6)
    recall_0 = tp_0 / (tp_0 + fn_0 + 1e-6)
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0 + 1e-6)
    return (f1_1 + f1_0) / 2

def auc_score(gold_mask, pred_mask):
    mask = gold_mask != -1
    gold_mask_masked = gold_mask[mask].float().cpu().numpy()
    pred_mask_masked = torch.sigmoid(pred_mask[mask]).detach().cpu().numpy()
    if len(np.unique(gold_mask_masked)) == 1:
        return np.nan
    return roc_auc_score(gold_mask_masked, pred_mask_masked)

def precision_recall(gold_mask, pred_mask):
    pred_mask_binary = (pred_mask > 0.).int()
    mask = gold_mask != -1
    gold_mask_masked = gold_mask[mask]
    pred_mask_masked = pred_mask_binary[mask]
    tp_1 = (gold_mask_masked * pred_mask_masked).sum()
    fp_1 = ((1 - gold_mask_masked) * pred_mask_masked).sum()
    fn_1 = (gold_mask_masked * (1 - pred_mask_masked)).sum()
    precision_1 = tp_1 / (tp_1 + fp_1 + 1e-6)
    recall_1 = tp_1 / (tp_1 + fn_1 + 1e-6)
    gold_mask_masked_0 = 1 - gold_mask_masked
    pred_mask_masked_0 = 1 - pred_mask_masked
    tp_0 = (gold_mask_masked_0 * pred_mask_masked_0).sum()
    fp_0 = ((1 - gold_mask_masked_0) * pred_mask_masked_0).sum()
    fn_0 = (gold_mask_masked_0 * (1 - pred_mask_masked_0)).sum()
    precision_0 = tp_0 / (tp_0 + fp_0 + 1e-6)
    recall_0 = tp_0 / (tp_0 + fn_0 + 1e-6)
    return (precision_1 + precision_0) / 2, (recall_1 + recall_0) / 2

def dice_score(gold_mask, pred_mask, smooth=1e-6):
    pred_mask_binary = (pred_mask > 0.).int()
    mask = gold_mask != -1
    gold_mask_masked = gold_mask[mask]
    pred_mask_masked = pred_mask_binary[mask]
    intersection_1 = (gold_mask_masked * pred_mask_masked).sum()
    union_1 = gold_mask_masked.sum() + pred_mask_masked.sum()
    dice_1 = (2.0 * intersection_1 + smooth) / (union_1 + smooth)
    gold_mask_masked_0 = 1 - gold_mask_masked
    pred_mask_masked_0 = 1 - pred_mask_masked
    intersection_0 = (gold_mask_masked_0 * pred_mask_masked_0).sum()
    union_0 = gold_mask_masked_0.sum() + pred_mask_masked_0.sum()
    dice_0 = (2.0 * intersection_0 + smooth) / (union_0 + smooth)
    return (dice_1 + dice_0) / 2

def loss(inputs, targets):
    return FocalDiceLoss(alpha=0.25, gamma=2.0)(inputs, targets)