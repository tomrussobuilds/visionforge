"""
Synthetic Data Handler for Testing.

Provides tiny synthetic MedMNIST datasets for unit tests without any downloads.
"""

# =========================================================================== #
#                                Standard Imports                               #
# =========================================================================== #
from dataclasses import dataclass

# =========================================================================== #
#                               Third-Party Imports                            #
# =========================================================================== #
import numpy as np


# =========================================================================== #
#                                DATA CLASSES                                   #
# =========================================================================== #
@dataclass
class SyntheticMedMNISTData:
    """Tiny synthetic dataset for testing (NO downloads)."""

    train_images: np.ndarray
    train_labels: np.ndarray
    val_images: np.ndarray
    val_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    path: str = "/synthetic"


# =========================================================================== #
#                                FACTORY FUNCTIONS                              #
# =========================================================================== #
def create_synthetic_dataset(num_classes=8, samples=100):
    """Create synthetic dataset (NO Zenodo)."""
    return SyntheticMedMNISTData(
        train_images=np.random.randint(0, 255, (samples, 28, 28, 3), dtype=np.uint8),
        train_labels=np.random.randint(0, num_classes, (samples, 1), dtype=np.uint8),
        val_images=np.random.randint(0, 255, (samples // 2, 28, 28, 3), dtype=np.uint8),
        val_labels=np.random.randint(0, num_classes, (samples // 2, 1), dtype=np.uint8),
        test_images=np.random.randint(0, 255, (samples // 2, 28, 28, 3), dtype=np.uint8),
        test_labels=np.random.randint(0, num_classes, (samples // 2, 1), dtype=np.uint8),
    )
