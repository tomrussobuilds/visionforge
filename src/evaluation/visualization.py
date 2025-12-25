"""
Visualization Utilities Module

This module provides functions for generating visual reports of the model's 
performance, including training loss/accuracy curves, normalized confusion 
matrices, and sample prediction grids with true vs. predicted labels.
"""

# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
from typing import Sequence, List
from pathlib import Path
import logging

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# =========================================================================== #
#                                Internal Imports
# =========================================================================== #
from src.core import Config

# =========================================================================== #
#                                CONFIGURATION
# =========================================================================== #
# Set a professional style for all plots
try:
    plt.style.use('seaborn-v0_8-muted')
except OSError:
    plt.style.use('ggplot') # Fallback style

# Global logger instance
logger = logging.getLogger("medmnist_pipeline")

# =========================================================================== #
#                               VISUALIZATION FUNCTIONS
# =========================================================================== #

def show_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    classes: List[str],
    n: int = 12,
    save_path: Path | None = None,
    cfg: Config | None = None
) -> None:
    """
    Lazy-extracts a batch from the loader and displays model predictions.
    
    Highlights correct (green) vs. incorrect (red) predictions in a grid.
    Handles denormalization and C,H,W to H,W,C transposition automatically.

    Args:
        model (nn.Module): Trained model for inference.
        loader (DataLoader): Test or Validation loader to sample from.
        device (torch.device): Computation device.
        classes (List[str]): Names of the target classes.
        n (int): Number of images to show (max based on batch size).
        save_path (Path | None): Output destination for the figure.
        cfg (Config | None): Used for title metadata and normalization stats.
    """
    model.eval()

    # 1. Lazy Extraction of a single batch
    # next(iter(loader)) grabs the first available batch
    batch = next(iter(loader))
    images_tensor, labels_tensor = batch[0], batch[1]

    # 2. Inference for the batch
    with torch.no_grad():
        images_tensor = images_tensor.to(device)
        outputs = model(images_tensor)
        preds = outputs.argmax(dim=1).cpu().numpy()
    
    # Move images back to CPU and convert to NumPy
    images_batch = images_tensor.cpu().numpy()
    true_labels = labels_tensor.cpu().numpy().flatten()
    
    # 3. Grid Setup
    n = min(n, len(images_batch))
    rows = int(np.ceil(n / 4))
    cols = 4
    plt.figure(figsize=(12, 3 * rows))

    for i in range(n):
        img = images_batch[i].copy()

        # Denormalization Logic using Config stats
        if cfg and hasattr(cfg, 'dataset'):
            # Ensure mean and std match the image shape (C, 1, 1) for broadcasting
            mean = np.array(cfg.dataset.mean).reshape(-1, 1, 1)
            std = np.array(cfg.dataset.std).reshape(-1, 1, 1)
            
            img = (img * std) + mean
            img = np.clip(img, 0, 1)

        # Transpose for Matplotlib: (C, H, W) -> (H, W, C)
        if img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))

        plt.subplot(rows, cols, i + 1)
        
        # Support for grayscale or RGB
        if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1):
            plt.imshow(img.squeeze(), cmap='gray')
        else:
            plt.imshow(img)
            
        color = "green" if true_labels[i] == preds[i] else "red"
        plt.title(f"T:{classes[true_labels[i]]}\nP:{classes[preds[i]]}", 
                  color=color, fontsize=10)
        plt.axis("off")
    
    # Title and Layout
    model_title = cfg.model_name if cfg else "Model"
    plt.suptitle(f"Test Predictions — {model_title}", fontsize=16)
    plt.tight_layout()

    # Save logic
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        logger.info(f"Sample predictions grid saved → {save_path.name}")
    else:
        plt.show()
    
    plt.close()

def plot_training_curves(
        train_losses: Sequence[float],
        val_accuracies: Sequence[float],
        out_path: Path | None = None,
        save_npz: bool = True,
        cfg: Config | None = None,
        out_dir: Path | None = None
) -> None:
    """
    Plots the training loss and validation accuracy curves on a dual-axis plot
    and optionally saves the raw numerical data.

    Args:
        train_losses (Sequence[float]): List of training losses per epoch.
        val_accuracies (Sequence[float]): List of validation accuracies per epoch.
        out_path (Path): Path to save the generated plot.
        save_npz (bool): Whether to save raw data to a .npz file.
    """
    if out_dir and not out_path:
        out_path = Path(out_dir) / "training_curves.png"
    elif not out_path:
        raise ValueError("Either out_path or out_dir must be provided.")
    
    fig, ax1 = plt.subplots(figsize=(9, 6))

    # Plot Training Loss on the left axis (ax1)
    ax1.plot(train_losses, color='#e74c3c', lw=2, label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='#e74c3c', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#e74c3c')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Plot Validation Accuracy on the right axis (ax2)
    ax2 = ax1.twinx()
    ax2.plot(val_accuracies, color='#3498db', lw=2, label="Validation Accuracy")
    ax2.set_ylabel("Accuracy", color='#3498db', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#3498db')

    plt.title("Training Loss & Validation Accuracy", fontsize=14, pad=15)
    fig.tight_layout()
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info(f"Training curves saved → {out_path}")

    if save_npz:
        npz_path = out_path.with_suffix('.npz')
        np.savez(
            npz_path,
            train_losses=train_losses,
            val_accuracies=val_accuracies
        )
        logger.info(f"Raw training data saved → {npz_path}")

    plt.close()

def plot_confusion_matrix(
        all_labels: np.ndarray,
        all_preds: np.ndarray,
        classes: List[str],
        out_path: Path,
        cfg: Config | None = None
) -> None:
    """
    Generates and saves a normalized confusion matrix plot.

    Args:
        all_labels (np.ndarray): Array of true labels.
        all_preds (np.ndarray): Array of predicted labels.
        classes (List[str]): List of class names for labeling.
        out_path (Path): Path to save the generated plot.
        cfg (Config | None): Configuration object for title metadata.
    """
    # Calculate the normalized confusion matrix (rows sum to 1)
    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=np.arange(len(classes)),
        normalize='true',
    )

    cm = np.nan_to_num(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes,
    )

    fig, ax = plt.subplots(figsize=(11, 9))
    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False,
        values_format='.3f'
    )

    model_name = cfg.model_name if cfg else "Model"
    plt.title(f"Confusion Matrix – {model_name}", fontsize=14, pad=20)
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved → {out_path}")