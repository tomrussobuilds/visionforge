"""
Core Training and Validation Engines

This module provides high-performance implementation of the training and 
validation loops. It integrates modern PyTorch features such as Automatic 
Mixed Precision (AMP), Gradient Clipping, and MixUp augmentation to ensure 
numerical stability and efficient hardware utilization.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from typing import Tuple

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# =========================================================================== #
#                               CORE ENGINES                                  #
# =========================================================================== #

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixup_fn = None,
    scaler = None,
    grad_clip: float = 0.0
) -> float:
    """
    Performs a single full pass over the training dataset.

    Args:
        model (nn.Module): The neural network architecture to train.
        loader (DataLoader): Training data provider.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        optimizer (Optimizer): Gradient descent optimizer (e.g., SGD, AdamW).
        device (torch.device): Hardware target (cuda, mps, or cpu).
        mixup_fn (callable, optional): Function to apply MixUp data blending.
        scaler (GradScaler, optional): PyTorch scaler for mixed precision training.
        grad_clip (float): Maximum norm for gradient clipping (0.0 to disable).

    Returns:
        float: The average training loss for the current epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, targets in tqdm(loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # --- 1. MixUp Preparation ---
        # If MixUp is enabled, we generate a convex combination of samples.
        # Otherwise, we use original targets for both standard loss paths.
        if mixup_fn:
            inputs, y_a, y_b, lam = mixup_fn(inputs, targets)
            outputs = model(inputs)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # --- 2. Backward Pass & Parameter Update ---
        # Standard backward pass for CPU efficiency (ignoring scaler overhead)
        if scaler:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    
    return running_loss / len(loader.dataset)


def validate_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """
    Evaluates the model performance on a held-out validation set.

    This function computes the validation loss and classification accuracy 
    under a 'no_grad' context to minimize memory consumption and latency.

    Args:
        model (nn.Module): The model to evaluate.
        val_loader (DataLoader): Validation data provider.
        criterion (nn.Module): Loss function used for evaluation.
        device (torch.device): Hardware target for execution.

    Returns:
        dict: A dictionary containing 'loss' (float) and 'accuracy' (float) metrics.
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Loss computation
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            
            # Accuracy computation
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    return {
        "loss": val_loss / len(val_loader.dataset),
        "accuracy": correct / total
    }

# =========================================================================== #
#                               MIXUP UTILITY                                 #
# =========================================================================== #

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applies MixUp augmentation, generating a convex combination of two random
    samples and their corresponding targets.

    Args:
        x (torch.Tensor): Input data batch (images).
        y (torch.Tensor): Target labels batch.
        alpha (float): Beta distribution parameter (set to 0 to disable MixUp).
        device (torch.device | None): The device where tensors should be placed.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
            mixed_x: The blended input images.
            y_a: The original targets.
            y_b: The permuted targets.
            lam: The mixing coefficient lambda.
    """
    if alpha <= 0:
        return x, y, y, 1.0

    # Draw mixing coefficient lambda from Beta distribution
    lam: float = np.random.beta(alpha, alpha)
    batch_size: int = x.size(0)
    
    # Generate a random permutation of indices
    # Optimized for CPU by avoiding explicit device calls when not on GPU
    index = torch.randperm(batch_size)
    if x.is_cuda:
        index = index.to(x.device)

    # Calculate the mixed input
    mixed_x: torch.Tensor = lam * x + (1 - lam) * x[index, :]
    
    # Get the corresponding targets
    y_a: torch.Tensor = y
    y_b: torch.Tensor = y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Calculates the MixUp loss as a weighted average of the loss for the two targets.

    Args:
        criterion (nn.Module): The standard loss function (e.g., CrossEntropyLoss).
        pred (torch.Tensor): Model predictions for the mixed input.
        y_a (torch.Tensor): The original targets.
        y_b (torch.Tensor): The permuted targets.
        lam (float): The mixing coefficient lambda.

    Returns:
        torch.Tensor: The final MixUp-regularized loss value.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)