"""
Dataset Fetching and Loading Module

This module handles the physical retrieval of any MedMNIST dataset, including
robust download logic, MD5 verification, and loading into structured containers.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging
import time
from pathlib import Path
from dataclasses import dataclass

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import requests
import numpy as np

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from scripts.core import (
    md5_checksum, validate_npz_keys, 
    DatasetMetadata
)

# =========================================================================== #
#                                DATA CONTAINERS                              #
# =========================================================================== #

@dataclass(frozen=True)
class MedMNISTData:
    """A generic container for MedMNIST dataset splits stored as NumPy arrays."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    path: Path


# =========================================================================== #
#                                FETCHING LOGIC                               #
# =========================================================================== #
# Global logger instance
logger = logging.getLogger("medmnist_pipeline")

def ensure_dataset_npz(
        metadata: DatasetMetadata,
        retries: int = 5,
        delay: float = 5.0,
    ) -> Path:
    """
    Downloads a MedMNIST dataset NPZ file robustly with retries and MD5 validation.

    Args:
        metadata (DatasetMetadata): Metadata containing URL, MD5, name and target path.
        retries (int): Max number of download attempts.
        delay (float): Delay (seconds) between retries.

    Returns:
        Path: Path to the successfully validated .npz file.
    """
    target_npz = metadata.path

    def _is_valid(path: Path) -> bool:
        """Checks file existence, header (ZIP/NPZ), and MD5 checksum."""
        if not path.exists() or path.stat().st_size < 50_000:
            return False
        
        try:
            # Check for ZIP header (NPZ files are ZIP archives)
            with open(path, "rb") as f:
                if f.read(2) != b"PK":
                    return False
        except IOError:
            return False
            
        return md5_checksum(path) == metadata.md5_checksum
    
    # 1. Check if valid file already exists
    if _is_valid(target_npz):
        logger.info(f"Valid dataset '{metadata.name}' found at: {target_npz}")
        return target_npz

    # 2. Cleanup corrupted file
    if target_npz.exists():
        logger.warning(f"Corrupted dataset found, deleting: {target_npz}")
        target_npz.unlink()

    # 3. Download logic with streaming
    logger.info(f"Downloading {metadata.name} from {metadata.url}")
    target_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_npz.with_suffix(".tmp")
    
    for attempt in range(1, retries + 1):
        try:
            with requests.get(metadata.url, timeout=60, stream=True) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            if not _is_valid(tmp_path):
                raise ValueError("Downloaded file failed MD5 or header validation")

            tmp_path.replace(target_npz) # Atomic move
            logger.info(f"Successfully downloaded and verified: {metadata.name}")
            return target_npz

        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()

            if attempt == retries:
                logger.error(f"Failed to download {metadata.name} after {retries} attempts")
                raise RuntimeError(f"Could not download {metadata.name}") from e

            logger.warning(f"Attempt {attempt}/{retries} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)

    raise RuntimeError("Unexpected error in dataset download logic.")


def load_medmnist(metadata: DatasetMetadata) -> MedMNISTData:
    """
    Ensures the dataset is present and loads it from the NPZ file.
    """
    path = ensure_dataset_npz(metadata)

    logger.info(f"Loading {metadata.name} into memory...")

    with np.load(path) as data:
        validate_npz_keys(data)
        
        return MedMNISTData(
            X_train=np.array(data["train_images"]),
            X_val=np.array(data["val_images"]),
            X_test=np.array(data["test_images"]),
            y_train=data["train_labels"].ravel(),
            y_val=data["val_labels"].ravel(),
            y_test=data["test_labels"].ravel(),
            path=path
        )