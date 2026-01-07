import os
import json
import heapq
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import MobileNetV2
from util.trainUtil.AlbumentationsDataset import AlbumentationsDataset
from util.trainUtil.util import plot_metrics
from util.trainUtil.train_one_batch import train_one_batch, evaluate_testset
from util.trainUtil.util import load_config

CONFIG_PATH = "config/bird_MobileNetV2.json"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          device, epochs, save_dir, config):

    print(f"Total epochs: {config['epochs']}")
    print(f"Train batch size: {config['batch_size_train']}, Validation batch size: {config['batch_size_val']}")
    print(f"Input size: {config['input_size']}, Validation resize: {config['val_resize']}")
    print(f"Model save path: {config['save']}")
    print("Training begins!")
    print("----------------------------------------------------------------")
    train_losses, val_accuracies = [], []
    best_val_acc, patience_counter = 0.0, 0

    patience = config["early_stopping_patience"]
    val_interval = config["val_interval"]
    max_saved_models = config["max_saved_models"]
    saved_models = []

    df_train_log, df_val_log = pd.DataFrame(), pd.DataFrame()

    # Warm-up: initial log
    images, labels = next(iter(train_loader))
    log_train = train_one_batch(images, labels, device, model, criterion, optimizer, 0, 0)
    df_train_log = pd.concat(
        [df_train_log, pd.DataFrame([{"epoch": 0, "batch": 0, **log_train}])],
        ignore_index=True
    )

    log_val = evaluate_testset(val_loader, device, model, criterion, 0)
    df_val_log = pd.concat(
        [df_val_log, pd.DataFrame([{"epoch": 0, **log_val}])],
        ignore_index=True
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"),
                1):
            log_train = train_one_batch(images, labels, device, model, criterion, optimizer, epoch + 1, batch_idx)
            df_train_log = pd.concat([df_train_log, pd.DataFrame([log_train])], ignore_index=True)
            running_loss += log_train['train_loss']

        train_loss = running_loss / len(train_loader)
        if (epoch + 1) % val_interval == 0:
            train_losses.append(train_loss)

            # Validation phase
            model.eval()
            log_test = evaluate_testset(val_loader, device, model, criterion, epoch + 1)
            df_val_log = pd.concat([df_val_log, pd.DataFrame([log_test])], ignore_index=True)

            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc="Validation", colour="green"):
                    outputs = model(images.to(device))
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.to(device)).sum().item()

            val_acc = correct / total
            val_accuracies.append(val_acc)

            print(
                f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {train_loss:.6f}, Validation Accuracy: {val_acc:.6f}")

            # Save top-k models
            model_path = os.path.join(save_dir, "model",
                                      f"epoch{epoch + 1}_acc{val_acc:.6f}.pth")
            if len(saved_models) < max_saved_models:
                torch.save(model.state_dict(), model_path)
                heapq.heappush(saved_models, (val_acc, model_path))
            elif val_acc > saved_models[0][0]:
                _, old_model_path = heapq.heappop(saved_models)
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)
                torch.save(model.state_dict(), model_path)
                heapq.heappush(saved_models, (val_acc, model_path))

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc, patience_counter = val_acc, 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}.")
                    break

        scheduler.step()

        # Plot metrics every 10 epochs
        if (epoch + 1) % 10 == 0:
            plot_metrics(train_losses, val_accuracies, epoch, save_dir + 'epoch')

    df_train_log.to_csv(save_dir + 'train_log.csv', index=False)
    df_val_log.to_csv(save_dir + 'val_log.csv', index=False)


def main():
    # Load config
    config = load_config(CONFIG_PATH)
    for subdir in ['config', 'epoch', 'model']:
        os.makedirs(config["save"] + subdir, exist_ok=True)

    # Save config copy
    with open(os.path.join(config["save"] + 'config', os.path.basename(CONFIG_PATH)), "w") as f:
        json.dump(config, f, indent=4)

    # Data transforms
    train_transform = A.Compose([
        A.RandomResizedCrop(*config["input_size"]),
        A.HorizontalFlip(p=0.5),
        # A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.5),
        # A.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.2, p=0.5),
        # A.Lambda(image=lambda img, **kwargs: img ** (1 / 1.5), p=0.5),
        # A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
        # A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.5),
        # A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=0.5),
        A.Normalize(mean=config["normalize_mean"],
                    std=config["normalize_std"]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(*config["val_resize"]),
        A.CenterCrop(*config["input_size"]),
        A.Normalize(mean=config["normalize_mean"],
                    std=config["normalize_std"]),
        ToTensorV2(),
    ])

    # Dataset setup
    data_path = config["image_path"]
    train_dataset_raw = datasets.ImageFolder(root=os.path.join(data_path, "train"))
    val_dataset_raw = datasets.ImageFolder(root=os.path.join(data_path, "val"))
    print("The training information is as follows:")
    print(f"Training samples: {len(train_dataset_raw)}")
    print(f"Validation samples: {len(val_dataset_raw)}")
    print(f"Number of classes: {len(train_dataset_raw.classes)}")
    print(f"Class names: {train_dataset_raw.classes}")
    # Build dataset with augmentation
    train_paths, train_labels = [], []
    for root, _, files in os.walk(os.path.join(data_path, "train")):
        for file in files:
            train_paths.append(os.path.join(root, file))
            train_labels.append(train_dataset_raw.class_to_idx[os.path.basename(root)])

    val_paths, val_labels = [], []
    for root, _, files in os.walk(os.path.join(data_path, "val")):
        for file in files:
            val_paths.append(os.path.join(root, file))
            val_labels.append(val_dataset_raw.class_to_idx[os.path.basename(root)])

    train_dataset = AlbumentationsDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = AlbumentationsDataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size_train"],
                              shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size_val"],
                            shuffle=False, num_workers=config["num_workers"])

    # Model, optimizer, scheduler
    model = MobileNetV2(num_classes=config["num_classes"]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=config["learning_rate"],
                           weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=config["scheduler_step_size"],
                                          gamma=config["scheduler_gamma"])

    # Start training
    train(model, train_loader, val_loader, criterion, optimizer,
          scheduler, DEVICE, config["epochs"], config["save"], config)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
