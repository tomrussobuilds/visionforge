"""
Input/Output & Persistence Utilities.

This module manages the pipeline's interaction with the filesystem, handling
configuration serialization (YAML), model checkpoint restoration, and dataset
integrity verification via MD5 checksums and schema validation.
"""

# Exposed Interface
from .checkpoints import load_model_weights
from .data_io import md5_checksum, validate_npz_keys
from .serialization import load_config_from_yaml, save_config_as_yaml

# Exports
__all__ = [
    # Serialization
    "save_config_as_yaml",
    "load_config_from_yaml",
    # Checkpoints
    "load_model_weights",
    # Data Integrity
    "validate_npz_keys",
    "md5_checksum",
]
