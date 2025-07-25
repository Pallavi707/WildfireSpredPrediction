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
from milesial_unet_model import UNet, APAU_Net
from leejunhyun_unet_models import U_Net, R2U_Net, AttU_Net, R2AttU_Net, AttU_Net_S
import pickle
import random



# Import custom loss and evaluation functions


TRAIN = 'train'
VAL = 'validation'
MASTER_RANK = 0
SAVE_INTERVAL = 1

# Make sure to change these paths!
DATASET_PATH = 'data/next-day-wildfire-spread'
SAVE_MODEL_PATH = 'savedModels'

loss_functions = ['WBCE','WBCE + DICE','FOCAL', 'FOCAL+DICE']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--master', default='sardine',
                        help='master node')
    parser.add_argument('-p', '--port', default='30437',
                         help = 'master node')
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=10, type=int,
                        metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    print(f'initializing training on single GPU')
    

    for loss_name in loss_functions:
        print(f"\n====== Training with loss: {loss_name} ======")
        print("******************************************************************************")
        train(0, args, loss_name)
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
        print(f"Using selected features: {selected_features}")
        print(f"Feature indices being used: {feature_indices}")
    else:
        feature_indices = list(range(len(ALL_FEATURES)))
        selected_features = ALL_FEATURES
        print("Using all features by default.")

    print(f"\nSelected features and their indices:\n{list(zip(selected_features, feature_indices))}")

    datasets = {
        TRAIN: RotatedWildfireDataset(
            f"{DATASET_PATH}/{TRAIN}.data",
            f"{DATASET_PATH}/{TRAIN}.labels",
            features=feature_indices,
            crop_size=64
        ),
        VAL: WildfireDataset(
            f"{DATASET_PATH}/{VAL}.data",
            f"{DATASET_PATH}/{VAL}.labels",
            features=feature_indices,
            crop_size=64
        )
    }

    dataLoaders = {
        TRAIN: torch.utils.data.DataLoader(
            dataset=datasets[TRAIN],
            batch_size=batch_size,
            shuffle=True,  
            num_workers=0,
            pin_memory=True,
        ),
        VAL: torch.utils.data.DataLoader(
            dataset=datasets[VAL],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    }

    return dataLoaders

def perform_validation(model, loader, loss_name, return_per_image_metrics=False):
    model.eval()

    # Per-image metrics
    per_image_dice = []
    per_image_iou = []
    per_image_f1 = []

    # Total accumulators for average metrics
    total_loss = 0
    total_iou = 0
    total_accuracy = 0
    total_f1 = 0
    total_auc = 0
    total_dice = 0
    total_precision = 0
    total_recall = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            outputs = model(images)

            # Total (averaged) metrics
            total_loss += loss(labels, outputs, loss_name).item()
            total_iou += mean_iou(labels, outputs)
            total_accuracy += accuracy(labels, outputs)
            total_f1 += f1_score(labels, outputs)
            total_auc += auc_score(labels, outputs)
            total_dice += dice_score(labels, outputs)

            precision, recall = precision_recall(labels, outputs)
            total_precision += precision
            total_recall += recall

            # Per-image metrics
            for j in range(images.size(0)):
                label = labels[j]
                pred = outputs[j]
                per_image_dice.append(float(dice_score(label, pred)))
                per_image_iou.append(float(mean_iou(label, pred)))
                per_image_f1.append(float(f1_score(label, pred)))

    # Averages
    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader)
    avg_accuracy = total_accuracy / len(loader)
    avg_f1 = total_f1 / len(loader)
    avg_auc = total_auc / len(loader)
    avg_dice = total_dice / len(loader)
    avg_precision = total_precision / len(loader)
    avg_recall = total_recall / len(loader)

    print(f"Validation - Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, Accuracy: {avg_accuracy:.4f}")
    print(f"F1 Score: {avg_f1:.4f}, AUC: {avg_auc:.4f}, Dice: {avg_dice:.4f}")
    print(f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")

    # Return both when needed
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
                "dice": per_image_dice,
                "iou": per_image_iou,
                "f1": per_image_f1
            }
        }

    # Return averages only by default
    return avg_loss, avg_iou, avg_accuracy, avg_f1, avg_auc, avg_dice, avg_precision, avg_recall



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


    model = U_Net(19, 1).cuda() # since only using single gpu, no need for DistributedDataParallel
    #torch.cuda.set_device(gpu)
    #model.cuda(gpu)


    #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([5])).cuda(gpu)
  

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.003, momentum=0.9)

    start = datetime.now()
    print(f'TRAINING ON: {platform.node()}, Starting at: {datetime.now()}')

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

            # Forward pass
            outputs = model(images)


            # Not entirely sure if this flattening is required or not
            #labels = torch.flatten(labels)
            #outputs = torch.flatten(outputs)

            loss_value = loss(labels, outputs, loss_name) 
            #loss = torchvision.ops.sigmoid_focal_loss(outputs, labels, alpha=0.85, gamma=2, reduction="mean")

            loss_train += loss_value.item()


            # Backward and optimize
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            if i % 20 == 0:
                print('Epoch [{}/{}], Steps [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1,
                    args.epochs,
                    i,
                    total_step,
                    loss_value.item())
                )

        train_loss_history.append(loss_train / len(dataLoaders[TRAIN]))

        if validate:
            metrics = perform_validation(model, dataLoaders[VAL], loss_name)
            val_metrics_history.append(metrics)

            curr_avg_loss_val, _, _, curr_f1_score, _, _, _, _ = metrics

            if best_f1_score < curr_f1_score:
                print("Saving model...")
                best_epoch = epoch
                best_f1_score = curr_f1_score
                filename = f'model-{model.__class__.__name__}-{loss_name}-bestF1Score-Rank-{rank}.weights'
                torch.save(model.state_dict(), f'{SAVE_MODEL_PATH}/{filename}')
                print("Model has been saved!")
            else:
                print("Model is not being saved")

    pickle.dump(train_loss_history, open(f"{SAVE_MODEL_PATH}/train_loss_history.pkl", "wb"))
    pickle.dump(val_metrics_history, open(f"{SAVE_MODEL_PATH}/val_metrics_history.pkl", "wb"))

    if gpu == 0:
        end_time = datetime.now()
        elapsed = end_time - start
        elapsed_seconds = elapsed.total_seconds()

        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        seconds = int(elapsed_seconds % 60)

        print(f"Training completed in {hours} hours, {minutes} minutes, and {seconds} seconds.")
        print(f"Total time (seconds): {elapsed_seconds}")
        print(f"Start time: {start}")
        print(f"Endtime: {end_time}")
        print(f"Best epoch: {best_epoch + 1}")
        print(f"Best F1 score: {best_f1_score}")

       
        # Load best model for final validation
        model_path = f"{SAVE_MODEL_PATH}/model-{model.__class__.__name__}-{loss_name}-bestF1Score-Rank-{rank}.weights"
        model.load_state_dict(torch.load(model_path))

        # Re-evaluate with per-image metrics
        results = perform_validation(model, dataLoaders[VAL], loss_name, return_per_image_metrics=True)

        # Save per-image metrics to JSON
        import json
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

        print(f"Logged per-image + avg metrics to {json_file} for: {model_id}")
    print("Training complete!")


if __name__ == '__main__':
    main()
