"""
Input/Output & Persistence Utilities.

This module manages the pipeline's interaction with the filesystem, handling 
configuration serialization (YAML), model checkpoint restoration, and dataset 
integrity verification via MD5 checksums and schema validation.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging
import os
import yaml
import hashlib
from pathlib import Path
from typing import Any, Dict

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import numpy as np
import torch

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .paths import LOGGER_NAME

# =========================================================================== #
#                                  I/O Utilities                              #
# =========================================================================== #

def save_config_as_yaml(data: Any, yaml_path: Path) -> Path:
    """
    Saves a configuration object as a YAML file.
    
    Orchestrates the conversion from Pydantic/Dict to a sanitized format
    and ensures safe persistence to the filesystem.
    """
    logger = logging.getLogger(LOGGER_NAME)
    try:
        # 1. Extraction (Pydantic to Dict)
        if hasattr(data, "model_dump"):
            try:
                raw_dict = data.model_dump(mode='json')
            except Exception as e:
                logger.warning(f"JSON dump failed, using fallback serialization: {e}")
                raw_dict = data.model_dump()
        else:
            raw_dict = data

        # 2. Sanitization (Recursive cleanup)
        final_data = _sanitize_for_yaml(raw_dict)

        # 3. Persistence (Physical write)
        _persist_yaml_atomic(final_data, yaml_path)

        logger.info(f"Configuration frozen successfully at → {yaml_path.name}")
        return yaml_path

    except Exception as e:
        logger.error(f"Failed to save configuration YAML: {e}")
        raise

def _sanitize_for_yaml(obj: Any) -> Any:
    """
    Recursively converts non-serializable types (Path, etc.) into YAML-friendly formats.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_yaml(i) for i in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj

def _persist_yaml_atomic(data: Any, path: Path) -> None:
    """
    Handles the physical writing to disk with directory creation and OS buffer flushing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(
            data, f, 
            default_flow_style=False, 
            sort_keys=False, 
            indent=4, 
            allow_unicode=True
        )
        f.flush()
        os.fsync(f.fileno())

def load_config_from_yaml(yaml_path: Path) -> Dict[str, Any]:
    """
    Loads a configuration dictionary from a YAML file.

    Args:
        yaml_path (Path): Path to the source YAML file.

    Returns:
        Dict[str, Any]: The raw configuration dictionary.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML configuration file not found at: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def validate_npz_keys(data: np.lib.npyio.NpzFile) -> None:
    """
    Validates that the loaded NPZ dataset contains all required MedMNIST keys.

    Args:
        data (np.lib.npyio.NpzFile): The loaded NPZ file object.

    Raises:
        ValueError: If any required key (images/labels) is missing.
    """
    required_keys = {
        "train_images", "train_labels",
        "val_images", "val_labels",
        "test_images", "test_labels",
    }

    missing = required_keys - set(data.files)
    if missing:
        found = list(data.files)
        raise ValueError(
            f"NPZ archive is corrupted or invalid. Missing keys: {missing}"
            f" | Found keys: {found}"
        )


def md5_checksum(path: Path) -> str:
    """
    Calculates the MD5 checksum of a file using buffered reading.

    Args:
        path (Path): Path to the file to verify.

    Returns:
        str: The calculated hexadecimal MD5 hash.
    """
    hash_md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_model_weights(
        model: torch.nn.Module, 
        path: Path, 
        device: torch.device
) -> None:
    """
    Restores model state from a checkpoint using secure weight-only loading.
    
    Args:
        model (torch.nn.Module): The model instance to populate.
        path (Path): Filesystem path to the checkpoint file.
        device (torch.device): Target device for mapping the tensors.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {path}")
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)