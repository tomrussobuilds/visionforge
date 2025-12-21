"""
PyTorch Dataset Definition Module

This module contains the custom Dataset class for MedMNIST, handling
the conversion from NumPy arrays to PyTorch tensors and applying 
image transformations for training and inference.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from typing import Tuple
from pathlib import Path

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# =========================================================================== #
#                                DATASET CLASS                                #
# =========================================================================== #

class MedMNISTDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """
    PyTorch Dataset for MedMNIST data. Handles NumPy to Tensor conversion
    and optional image transformations.
    """
    def __init__(self,
                 images: np.ndarray,
                 labels: np.ndarray,
                 path: Path,
                 transform: transforms.Compose | None = None):
        """
        Args:
            images (np.ndarray): Image data (N, H, W) or (N, H, W, C).
            labels (np.ndarray): Label data (N,).
            transform (transforms.Compose | None): Torchvision transformations.
        """
        # We keep images as uint8 if we use ToPILImage in transforms, 
        # or float32 if we process them directly.
        self.images = images
        self.labels = labels.astype(np.int64)
        self.path = path
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            # Our pipeline in transforms.py starts with ToPILImage(),
            # which expects a uint8 array of shape (H, W, C) or (H, W).
            img = self.transform(img)
        else:
            # Fallback: Manual conversion to Tensor (C, H, W)
            # 1. Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # 2. Add channel dimension if grayscale (H, W) -> (1, H, W)
            if img.ndim == 2:
                img = torch.from_numpy(img).unsqueeze(0)
            else:
                # (H, W, C) -> (C, H, W)
                img = torch.from_numpy(img).permute(2, 0, 1)

        return img, torch.tensor(label, dtype=torch.long)