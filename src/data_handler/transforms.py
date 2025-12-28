"""
Data Transformations Module

This module defines the image augmentation pipelines for training and 
the standard normalization for validation/testing. It also includes 
utilities for deterministic worker initialization. It supports both RGB
and Grayscale datasets dynamically.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from typing import Tuple

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
from torchvision.transforms import v2

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import Config, DatasetMetadata


# =========================================================================== #
#                             TRANSFORMATION PIPELINES                        #
# =========================================================================== #

def get_augmentations_description(cfg: Config) -> str:
    """
    Generates a descriptive string of the augmentations using values from Config.
    Used for logging and run traceability.
    """ 
    params = {
        "HFlip": cfg.augmentation.hflip,
        "Rotation": f"{cfg.augmentation.rotation_angle}°",
        "Jitter": cfg.augmentation.jitter_val,
        "ResizedCrop": f"{cfg.dataset.img_size} (0.9, 1.0)"
    }

    descr = [f"{k}({v})" for k, v in params.items()]
    
    if cfg.training.mixup_alpha > 0:
        descr.append(f"MixUp(α={cfg.training.mixup_alpha})")
    
    return ", ".join(descr)


def get_pipeline_transforms(
    cfg: Config,
    ds_meta: DatasetMetadata
) -> Tuple[v2.Compose, v2.Compose]:
    """
    Defines the image transformation pipelines for training and evaluation.

    This function dynamically constructs the Torchvision V2 augmentation 
    pipeline based on the dataset metadata (RGB vs Grayscale) and the 
    global experiment configuration. It ensures that 1-channel datasets 
    are promoted to 3-channel tensors to maintain compatibility with 
    architectures like ResNet-18.

    Args:
        cfg (Config): The validated global configuration object.
        ds_meta (DatasetMetadata): Metadata specific to the current MedMNIST dataset.

    Returns:
        Tuple[v2.Compose, v2.Compose]: A tuple containing (train_transform, val_transform).
    """
    
    # 1. Resolve Channel Logic
    # Identify if the dataset is native RGB or requires grayscale-to-RGB promotion.
    is_rgb = ds_meta.in_channels == 3
    
    # 2. Extract Normalization Stats from Registry (Pydantic models)
    # Ensuring consistency between dataset domain and pixel distributions.
    if ds_meta.in_channels == 1:
        mean = [ds_meta.mean[0]] * 3
        std = [ds_meta.std[0]] * 3
    else:
        mean = ds_meta.mean
        std = ds_meta.std

    def get_base_ops():
        """Foundational operations common to all pipelines."""
        ops = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
        # Promote 1-channel to 3-channel for architecture compatibility
        if not is_rgb:
            ops.append(v2.Grayscale(num_output_channels=3))
        return ops
        
    # --- TRAINING PIPELINE ---
    train_transform = v2.Compose([
        *get_base_ops(),
        v2.RandomHorizontalFlip(p=cfg.augmentation.hflip),
        v2.RandomRotation(cfg.augmentation.rotation_angle),
        v2.ColorJitter(
            brightness=cfg.augmentation.jitter_val,
            contrast=cfg.augmentation.jitter_val,
            saturation=cfg.augmentation.jitter_val if is_rgb else 0.0,
        ),
        v2.RandomResizedCrop(
            size=cfg.dataset.img_size,
            scale=(cfg.augmentation.min_scale, 1.0),
            antialias=True,
            interpolation=v2.InterpolationMode.BILINEAR,
        ),
        v2.Normalize(mean=mean, std=std),
    ])
    
    # --- VALIDATION/INFERENCE PIPELINE ---
    val_transform = v2.Compose([
        *get_base_ops(),
        v2.Normalize(mean=mean, std=std),
    ])
    
    return train_transform, val_transform