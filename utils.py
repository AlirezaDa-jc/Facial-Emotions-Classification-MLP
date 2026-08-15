import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def set_seed(seed):
    """Seeds Python, NumPy, and PyTorch (CPU + CUDA) for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Makes CUDA convolution/matmul algorithms deterministic (small perf cost).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Executes single training pass over dataset."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(dataloader), correct / total


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Executes single validation/test pass over dataset."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate_metrics(model, dataloader, device, class_names=None):
    """
    Runs the model on a dataloader and computes precision, recall, F1-score
    (macro-averaged, plus per-class), accuracy, and the confusion matrix.

    Returns a dict with:
        - accuracy: float
        - precision_macro, recall_macro, f1_macro: float
        - precision_per_class, recall_per_class, f1_per_class: arrays (len = num_classes)
        - confusion_matrix: 2D array (num_classes x num_classes)
        - class_names: list of class labels (if provided)
    """
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu()

        all_preds.append(predictions)
        all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    accuracy = accuracy_score(all_labels, all_preds)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_per_class": precision_pc,
        "recall_per_class": recall_pc,
        "f1_per_class": f1_pc,
        "support_per_class": support_pc,
        "confusion_matrix": cm,
        "class_names": class_names,
    }
