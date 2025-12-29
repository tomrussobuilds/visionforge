"""
Model Orchestration and Architecture Definition Module

This module provides a factory for deep learning architectures adapted for 
the MedMNIST ecosystem. It specializes in fine-tuning standard Torchvision 
models to handle low-resolution biomedical images (28x28 pixels).

Key Architectural Adaptations:
1. Spatial Preservation: Replaces the standard ResNet 7x7 (stride 2) entry 
   convolution with a 3x3 (stride 1) layer and removes initial pooling. This 
   prevents excessive information loss in the early stages of feature extraction.
2. Cross-Modal Weight Transfer: Implements bicubic interpolation of pre-trained 
   ImageNet weights and handles channel-depth conversion (e.g., RGB to Grayscale).
3. Dynamic Head Reconfiguration: Automatically adjusts the final linear 
   layers based on the target dataset's class cardinality.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import Config, LOGGER_NAME

# =========================================================================== #
#                               MODEL DEFINITION                              #
# =========================================================================== #

# Global logger instance
logger = logging.getLogger(LOGGER_NAME)

def build_resnet18_adapted(
        device: torch.device,
        num_classes: int,
        in_channels: int,
        cfg: Config
    ) -> nn.Module:
    """
    Loads a ResNet-18 model and adapts its structure for the MedMNIST 
    ecosystem (28x28 inputs) using the provided configuration manifest.

    The adaptation steps are:
    1. Selective Weight Loading: Dynamically determines whether to load 
       ImageNet-1K weights based on the hierarchical `cfg.model` state.
    2. Entry-Point Reconstruction: Replaces the original 7x7 `conv1` (stride 2) 
       with a 3x3 `conv1` (stride 1) to prevent downsampling of small inputs.
    3. Resolution Preservation: Removes the `maxpool` layer (via Identity) 
       to retain critical spatial features for 28x28 resolution.
    4. Weight Morphing: Performs bicubic interpolation of pre-trained weights 
       to align the 7x7 source kernels with the new 3x3 target geometry.
    5. Head Reconfiguration: Maps the final 512-feature vector to the 
       target dataset's class cardinality.

    Args:
        device (torch.device): The computation accelerator (CPU/CUDA/MPS).
        num_classes (int): Number of target output units.
        in_channels (int): Dimensionality of input tensors (1 for Gray, 3 for RGB).
        cfg (Config): The global experiment manifest for policy-aware adaptation.

    Returns:
        nn.Module: The spatially-preserved and weight-morphed ResNet-18 model.
    """
    # 1. Load ResNet-18 with policy-driven weight initialization
    # We use the ModelConfig sub-module to determine pre-training status
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if cfg.model.pretrained else None
    model = models.resnet18(weights=weights)
    
    # Store the original conv1 layer for weight transfer
    old_conv = model.conv1

    # 2. Define the new initial convolution layer (3x3, stride 1)
    # Standard ResNet-18 downsamples 28x28 input to 6x6 via conv1+maxpool.
    # Our adaptation preserves the full 28x28 resolution at the entry point.
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=64,
        kernel_size=3,  # Optimized for small-scale medical features
        stride=1,       # Prevents immediate spatial collapse
        padding=1,
        bias=False
    )

    # 3. Transfer weights from the old 7x7 layer to the new 3x3 layer
    # This maintains the feature extraction capabilities of ImageNet pre-training.
    if cfg.model.pretrained:
        with torch.no_grad():
            w = old_conv.weight
            # Interpolate the 7x7 weights to 3x3 using bicubic interpolation
            w = F.interpolate(w, size=(3,3), mode='bicubic', align_corners=True)

            if in_channels == 1:
                # RGB-to-Grayscale conversion: average weights across channel dimension
                w = w.mean(dim=1, keepdim=True)

            new_conv.weight.copy_(w)
    
    # Apply structural modifications to the backbone
    model.conv1 = new_conv
    
    # Remove MaxPool: Replacing with Identity retains spatial map dimensions
    model.maxpool = nn.Identity()
    
    # 4. Replace the final classification head
    # Maps the global average pooled features to the target labels
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Target device synchronization
    model = model.to(device)

    return model