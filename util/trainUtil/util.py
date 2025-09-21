import json
import os
from matplotlib import pyplot as plt


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def plot_metrics(train_losses, val_accuracies, epoch, result):
    plt.figure(figsize=(8, 6))
    epochs = [(i + 1) * 5 for i in range(len(val_accuracies))]
    plt.plot(epochs, train_losses, label="Train Loss", color='blue')
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", color='orange')

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"Training Loss & Validation Accuracy (Epoch {epoch + 1})")
    plt.legend()
    os.makedirs(result, exist_ok=True)

    min_loss, max_acc = min(train_losses), max(val_accuracies)
    min_loss_epoch = epochs[train_losses.index(min_loss)]
    max_acc_epoch = epochs[val_accuracies.index(max_acc)]

    plt.scatter(min_loss_epoch, min_loss, color='red', label='Min Loss')
    plt.text(min_loss_epoch, min_loss + 0.1, f'{min_loss:.6f}', fontsize=10, color='red', ha='center')
    plt.scatter(max_acc_epoch, max_acc, color='green', label='Max Accuracy')
    plt.text(max_acc_epoch, max_acc + 0.02, f'{max_acc:.6f}', fontsize=10, color='green', ha='center')

    plt.savefig(f"{result}/epoch_{epoch + 1}.png")
    plt.close()
