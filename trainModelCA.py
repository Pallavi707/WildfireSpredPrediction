import os
from datetime import datetime
import argparse
import torch
import torch.nn as nn
from collections import Counter
import numpy as np
from models import *
from datasets import *
import platform
from metrics import *
import pickle
from torch.utils.data import DataLoader
#from milesial_unet_model import UNet, UNetWithRefinerCA
from Unet_CA import *

TRAIN = 'train'
VAL = 'validation'
MASTER_RANK = 0
SAVE_INTERVAL = 1

DATASET_PATH = 'data/next-day-wildfire-spread'
SAVE_MODEL_PATH = 'savedModels'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--master', default='sardine', help='master node')
    parser.add_argument('-p', '--port', default='30437', help='master node')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    args = parser.parse_args()
    print(f'Initializing training on single GPU')
    train(0, args)

def create_data_loaders(rank, gpu, world_size, selected_features=None):
    batch_size = 64
    ALL_FEATURES = ['elevation', 'fws', 'population', 'pdsi', 'pr', 'sph', 'slope', 'PrevFireMask',
                    'erc', 'NDVI', 'fpr', 'ftemp', 'th', 'EVI', 'vs', 'tmmx', 'fwd',
                    'aspect', 'tmmn']

    if selected_features is not None:
        feature_indices = [ALL_FEATURES.index(feature) for feature in selected_features]
    else:
        feature_indices = list(range(len(ALL_FEATURES)))
        selected_features = ALL_FEATURES

    print(f"\nSelected features and their indices:\n{list(zip(selected_features, feature_indices))}")

    datasets = {
        TRAIN: RotatedWildfireDataset(f"{DATASET_PATH}/{TRAIN}.data", f"{DATASET_PATH}/{TRAIN}.labels",
                                      features=feature_indices, crop_size=64),
        VAL: WildfireDataset(f"{DATASET_PATH}/{VAL}.data", f"{DATASET_PATH}/{VAL}.labels",
                             features=feature_indices, crop_size=64)
    }

    dataLoaders = {
        TRAIN: DataLoader(datasets[TRAIN], batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True),
        VAL: DataLoader(datasets[VAL], batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    }

    return dataLoaders

def perform_validation(model, loader):
    model.eval()
    total_loss = total_iou = total_accuracy = total_f1 = 0
    total_auc = total_dice = total_precision = total_recall = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # ✅ Apply CA and then sigmoid
            final_out = model(images)
            outputs = torch.sigmoid(final_out)

            labels = torch.flatten(labels)
            outputs = torch.flatten(outputs)

            total_loss += loss(labels, outputs).item()
            total_iou += mean_iou(labels, outputs)
            total_accuracy += accuracy(labels, outputs)
            total_f1 += f1_score(labels, outputs)
            total_auc += auc_score(labels, outputs)
            total_dice += dice_score(labels, outputs)
            precision, recall = precision_recall(labels, outputs)
            total_precision += precision
            total_recall += recall

    size = len(loader)
    print(f"Validation - Loss: {total_loss/size:.4f}, IoU: {total_iou/size:.4f}, Accuracy: {total_accuracy/size:.4f}")
    print(f"F1 Score: {total_f1/size:.4f}, AUC: {total_auc/size:.4f}, Dice: {total_dice/size:.4f}")
    print(f"Precision: {total_precision/size:.4f}, Recall: {total_recall/size:.4f}")
    return total_loss/size, total_iou/size, total_accuracy/size, total_f1/size, total_auc/size, total_dice/size, total_precision/size, total_recall/size


def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dataLoaders = create_data_loaders(rank, gpu, args.gpus * args.nodes)
    print("\n🔍 Inspecting label distribution in the training set...")
    all_labels = []
    for _, labels in dataLoaders[TRAIN]:
        all_labels.append(labels.flatten().cpu().numpy())
    all_labels = np.concatenate(all_labels)
    print("Label Distribution:", Counter(all_labels))
    torch.manual_seed(0)

    model_names = ["UNetWithPostCA"]
    for model_name in model_names:
        print(f"\n===== Training {model_name} =====")

        if model_name == "UNetWithPostCA":
            model = UNetWithPostCA(in_channels=19, hidden_channels=64, out_channels=1, ca_steps=5)
        else:
            continue

        model = model.cuda(gpu)
        criterion = FocalDiceLoss(alpha=0.25, gamma=2.0).cuda(gpu)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.003, momentum=0.9)

        total_step = len(dataLoaders[TRAIN])
        best_epoch = 0
        best_f1_score = -float("inf")
        train_loss_history = []
        val_metrics_history = []

        for epoch in range(args.epochs):
            model.train()
            loss_train = 0

            for i, (images, labels) in enumerate(dataLoaders[TRAIN]):
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                final_out = model(images)   # ✅ updated
                labels = labels.view_as(final_out)  # reshape to (B, 1, H, W)

                loss = criterion(final_out, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_train += loss.item()

            train_loss_history.append(loss_train / len(dataLoaders[TRAIN]))
            print(f"\n--- Validation Results for {model_name} ---")
            metrics = perform_validation(model, dataLoaders[VAL])
            val_metrics_history.append(metrics)
            _, _, _, f1, _, _, _, _ = metrics

            if f1 > best_f1_score:
                best_f1_score = f1
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(SAVE_MODEL_PATH, f"{model_name}_bestF1.pth"))
                print("Model saved.")

        pickle.dump(train_loss_history, open(f"{SAVE_MODEL_PATH}/train_loss_{model_name}.pkl", "wb"))
        pickle.dump(val_metrics_history, open(f"{SAVE_MODEL_PATH}/val_metrics_{model_name}.pkl", "wb"))

        print(f"{model_name} training complete. Best epoch: {best_epoch}, Best F1: {best_f1_score:.4f}")

    print("\n====== Final Model Performance Comparison (Best F1 Score) ======")
    summary = {}
    for model_name in model_names:
        metrics_path = os.path.join(SAVE_MODEL_PATH, f"val_metrics_{model_name}.pkl")
        if not os.path.exists(metrics_path):
            print(f"Metrics for {model_name} not found. Skipping.")
            continue

        val_metrics_history = pickle.load(open(metrics_path, "rb"))
        best_epoch = max(range(len(val_metrics_history)), key=lambda i: val_metrics_history[i][3])
        best_metrics = val_metrics_history[best_epoch]
        summary[model_name] = {
            "Best Epoch": best_epoch,
            "Loss": best_metrics[0],
            "IoU": best_metrics[1],
            "Accuracy": best_metrics[2],
            "F1": best_metrics[3],
            "AUC": best_metrics[4],
            "Dice": best_metrics[5],
            "Precision": best_metrics[6],
            "Recall": best_metrics[7],
        }

    print(f"{'Model':<15} {'Epoch':<6} {'Loss':<8} {'IoU':<8} {'Acc':<8} {'F1':<8} {'AUC':<8} {'Dice':<8} {'Prec':<8} {'Recall':<8}")
    for model_name, metrics in summary.items():
        print(f"{model_name:<15} {metrics['Best Epoch']:<6} "
              f"{metrics['Loss']:<8.4f} {metrics['IoU']:<8.4f} {metrics['Accuracy']:<8.4f} "
              f"{metrics['F1']:<8.4f} {metrics['AUC']:<8.4f} {metrics['Dice']:<8.4f} "
              f"{metrics['Precision']:<8.4f} {metrics['Recall']:<8.4f}")

if __name__ == '__main__':
    main()
