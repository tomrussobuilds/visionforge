"""
Input/Output & Persistence Utilities.

This module manages the pipeline's interaction with the filesystem, handling 
configuration serialization (YAML), model checkpoint restoration, and dataset 
integrity verification via MD5 checksums and schema validation.
"""

# =========================================================================== #
#                                Exposed Interface                            #
# =========================================================================== #

# 1. Configuration & Serialization (from .serialization)
from .serialization import (
    save_config_as_yaml,
    load_config_from_yaml
)

# 2. Model Weight Management (from .checkpoints)
from .checkpoints import (
    load_model_weights
)

# 3. Data Integrity & Validation (from .data_io)
from .data_io import (
    validate_npz_keys,
    md5_checksum
)

# =========================================================================== #
#                                     Exports                                 #
# =========================================================================== #

__all__ = [
    # Serialization
    "save_config_as_yaml",
    "load_config_from_yaml",
    
    # Checkpoints
    "load_model_weights",
    
    # Data Integrity
    "validate_npz_keys",
    "md5_checksum"
]