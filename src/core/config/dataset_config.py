"""
Dataset Registry Orchestration & Metadata Resolution.

This module centralizes the logic for bridging static dataset metadata with
runtime execution requirements. It ensures that any vision dataset, regardless 
of its native format (Grayscale/RGB), is normalized and shaped to meet the 
input specifications of the selected model architecture.

Key Responsibilities:
    * Adaptive Normalization: Adjusts mean/std statistics based on channel logic.
    * Feature Promotion: Automates Grayscale-to-RGB conversion for ImageNet weights.
    * Resource Budgeting: Enforces sampling limits and class balancing.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import argparse
from typing import Optional
from pathlib import Path

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
from pydantic import BaseModel, Field, ConfigDict

# =========================================================================== #
#                               Internal Imports                              #
# =========================================================================== #
from .types import (
    ImageSize, ValidatedPath, PositiveInt
)
from ..metadata import DatasetMetadata
from ..paths import DATASET_DIR

# =========================================================================== #
#                             DATASET CONFIGURATION                           #
# =========================================================================== #

class DatasetConfig(BaseModel):
    """
    Validated manifest for a specific dataset execution context.
    
    This class acts as the bridge between static registry metadata and 
    runtime preferences. It resolves channel promotion and sampling 
    policies required for the training pipeline.
    """
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True
    )

    # --- Core Metadata (Injected) ---
    metadata: Optional[DatasetMetadata] = Field(
        default=None,
        exclude=True,
    )
    
    # --- User-Defined Parameters (Runtime) ---
    data_root: ValidatedPath = DATASET_DIR
    use_weighted_sampler: bool = True
    max_samples: Optional[PositiveInt] = Field(default=20000)
    img_size: ImageSize = Field(
        description="Target square resolution for the model input"
    )
    force_rgb: bool = Field(
        default=True,
        description="Convert grayscale to 3-channel to enable ImageNet weights"
    )

    # --- Computed Properties (Read-Only) ---

    @property
    def dataset_name(self) -> str:
        """The short identifier of the dataset (e.g., 'bloodmnist')."""
        return self.metadata.name
    
    @property
    def num_classes(self) -> int:
        """The total number of unique target classes defined in metadata."""
        return self.metadata.num_classes

    @property
    def in_channels(self) -> int:
        """The native number of channels in the source dataset (1 or 3)."""
        return self.metadata.in_channels
    
    @property
    def effective_in_channels(self) -> int:
        """The actual depth the model will receive (3 if forced/native RGB)."""
        return 3 if self.force_rgb else self.in_channels
    
    @property
    def mean(self) -> tuple[float, ...]:
        """Channel-wise mean, expanded if force_rgb is active on grayscale."""
        m = self.metadata.mean
        return (m[0],) * 3 if (self.force_rgb and self.in_channels == 1) else m
    
    @property
    def std(self) -> tuple[float, ...]:
        """Channel-wise std, expanded if force_rgb is active on grayscale."""
        s = self.metadata.std
        return (s[0],) * 3 if (self.force_rgb and self.in_channels == 1) else s

    @property
    def processing_mode(self) -> str:
        """Technical description of the channel resolution strategy."""
        if self.in_channels == 3:
            return "NATIVE-RGB"
        return "RGB-PROMOTED" if self.effective_in_channels == 3 else "NATIVE-GRAY"

    # --- Factory Methods ---

    @classmethod
    def from_args(cls, args: argparse.Namespace, metadata: DatasetMetadata) -> "DatasetConfig":
        """
        Encapsulates decision logic to build a DatasetConfig from CLI inputs.
        
        Resolves conflicts between CLI arguments (args) and dataset constraints 
        (metadata), specifically handling RGB promotion and sampling limits.
        """
        # 1. Resolve RGB logic: prioritizes CLI, fallbacks to auto-promotion if pretrained
        is_pretrained = getattr(args, "pretrained", True)
        force_rgb_cli = getattr(args, "force_rgb", None)
        
        resolved_force_rgb = (
            force_rgb_cli if force_rgb_cli is not None 
            else (metadata.in_channels == 1 and is_pretrained)
        )
            
        # 2. Resolve sampling constraints: handles 0/negative as None (no limit)
        cli_max = getattr(args, "max_samples", None)
        if cli_max is not None and cli_max <= 0:
            resolved_max = None
        else:
            resolved_max = cli_max or cls.model_fields['max_samples'].default

        resolved_img_size = getattr(args, "img_size", None) or metadata.native_resolution

        return cls(
            metadata=metadata,
            data_root=Path(getattr(args, "data_dir", DATASET_DIR)),
            max_samples=resolved_max,
            use_weighted_sampler=getattr(args, "use_weighted_sampler", True),
            force_rgb=resolved_force_rgb,
            img_size=resolved_img_size
        )