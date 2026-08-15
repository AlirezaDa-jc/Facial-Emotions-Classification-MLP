import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

import config
from dataset import get_dataloaders
from model import MLP
from utils import evaluate_metrics, set_seed, train_one_epoch, validate


def plot_curves(train_losses, val_losses, train_accs, val_accs, exp_name):
    """Plots train vs val loss and accuracy curves side by side and saves to disk."""
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    epochs_range = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs_range, train_losses, label="Train Loss")
    axes[0].plot(epochs_range, val_losses, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{exp_name} - Loss Curve")
    axes[0].legend()

    axes[1].plot(epochs_range, train_accs, label="Train Acc")
    axes[1].plot(epochs_range, val_accs, label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{exp_name} - Accuracy Curve")
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(config.PLOTS_DIR, f"{exp_name}_curves.png")
    plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_confusion_matrix(cm, class_names, exp_name):
    """Plots and saves the confusion matrix as a heatmap."""
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{exp_name} - Confusion Matrix")
    plt.tight_layout()
    save_path = os.path.join(config.PLOTS_DIR, f"{exp_name}_confusion_matrix.png")
    plt.savefig(save_path)
    plt.show()
    plt.close()


def run_experiment(config_name):
    """Runs one full training + evaluation cycle for a named config in config.CONFIGS."""
    if config_name not in config.CONFIGS:
        raise ValueError(
            f"Unknown config name: {config_name}. Available: {list(config.CONFIGS.keys())}"
        )

    cfg = config.CONFIGS[config_name]
    print(f"\n===== Running experiment: {config_name} =====")
    print(f"Config: {cfg}")
    print(f"Using device: {config.DEVICE}")

    # Seed everything before data split, model init, and training
    set_seed(config.SEED)

    # Data
    train_loader, val_loader, test_loader, classes = get_dataloaders(cfg["batch_size"])
    print(f"Dataset Loaded. Classes ({len(classes)}): {classes}")

    # Model, Loss, Optimizer
    model = MLP(
        input_size=config.INPUT_SIZE,
        hidden_sizes=cfg["hidden_sizes"],
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
        activation=cfg["activation"],
    ).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    # Training Loop
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    print("\nStarting Training...")
    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch + 1:02d}/{config.EPOCHS:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    # Final Test Evaluation (loss/acc)
    test_loss, test_acc = validate(model, test_loader, criterion, config.DEVICE)
    print(f"\nFinal Test Evaluation | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # Full metrics: precision, recall, F1, confusion matrix
    metrics = evaluate_metrics(model, test_loader, config.DEVICE, class_names=classes)
    print(
        f"Macro Precision: {metrics['precision_macro']:.4f} | "
        f"Macro Recall: {metrics['recall_macro']:.4f} | "
        f"Macro F1: {metrics['f1_macro']:.4f}"
    )

    # Plots
    plot_curves(train_losses, val_losses, train_accs, val_accs, config_name)
    plot_confusion_matrix(metrics["confusion_matrix"], classes, config_name)

    # Save results to disk (json) so nothing needs to be re-run for the report
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    result_record = {
        "config_name": config_name,
        "config": cfg,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "precision_per_class": metrics["precision_per_class"].tolist(),
        "recall_per_class": metrics["recall_per_class"].tolist(),
        "f1_per_class": metrics["f1_per_class"].tolist(),
        "support_per_class": metrics["support_per_class"].tolist(),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "class_names": classes,
    }
    result_path = os.path.join(config.RESULTS_DIR, f"{config_name}.json")
    with open(result_path, "w") as f:
        json.dump(result_record, f, indent=2)
    print(f"Saved results to {result_path}")

    return result_record


if __name__ == "__main__":
    for experiment_name in config.CONFIGS:
        run_experiment(experiment_name)
