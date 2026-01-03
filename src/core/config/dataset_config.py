"""
Dataset Registry Resolver & Metadata Schema.

This module acts as the interface between the MedMNIST Metadata Registry 
and the training pipeline. It handles the dynamic resolution of dataset-specific 
constraints, such as channel counts, normalization constants (mean/std), 
and automated 3-channel expansion (RGB) for grayscale datasets when 
using pretrained models.

Key Architectural Features:
    * Metadata Binding: Resolves dataset names to official MedMNIST properties.
    * Channel Orchestration: Manages 'force_rgb' logic to ensure compatibility 
      between grayscale medical images and ImageNet-pretrained architectures.
    * Class Balance Logic: Configures Weighted Sampler settings based on 
      registry-provided class distributions.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import argparse
from typing import Optional

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
    Resolves dataset-specific constraints and normalization metadata.
    
    Acts as the interface between the MedMNIST Metadata Registry and the 
    training pipeline, handling automated 3-channel expansion for 
    pretrained models and class-balance sampling logic.
    """
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True
    )

    # --- Core Metadata (Injected) ---
    metadata: DatasetMetadata
    
    # --- User-Defined Parameters (Runtime) ---
    data_root: ValidatedPath = DATASET_DIR
    use_weighted_sampler: bool = True
    max_samples: Optional[PositiveInt] = Field(default=20000)
    img_size: ImageSize = Field(
        default=28,
        description="Target square resolution for the model input"
    )
    force_rgb: bool = Field(
        default=True,
        description="Convert grayscale to 3-channel to enable ImageNet weights"
    )

    # --- Computed Properties (Derived from Metadata) ---

    @property
    def dataset_name(self) -> str:
        """The short identifier of the dataset (e.g., 'bloodmnist')."""
        return self.metadata.name
    
    @property
    def num_classes(self) -> int:
        """The total number of unique target classes in the dataset."""
        return len(self.metadata.classes)

    @property
    def in_channels(self) -> int:
        """The native number of channels in the source dataset images (1 or 3)."""
        return self.metadata.in_channels
    
    @property
    def effective_in_channels(self) -> int:
        """The actual number of channels the model will receive after 'force_rgb' logic."""
        return 3 if self.force_rgb else self.in_channels
    
    @property
    def mean(self) -> tuple[float, ...]:
        """Channel-wise mean for normalization, expanded if force_rgb is active."""
        if self.force_rgb and self.in_channels == 1:
            return (self.metadata.mean[0],) * 3
        return self.metadata.mean
    
    @property
    def std(self) -> tuple[float, ...]:
        """Channel-wise std for normalization, expanded if force_rgb is active."""
        if self.force_rgb and self.in_channels == 1:
            return (self.metadata.std[0],) * 3
        return self.metadata.std
    
    @property
    def processing_mode(self) -> str:
        """
        Human-readable description of the channel processing strategy:
        - NATIVE-RGB: Dataset is already RGB.
        - RGB-PROMOTED: Grayscale dataset expanded to 3 channels.
        - NATIVE-GRAY: Dataset kept as single-channel.
        """
        if self.in_channels == 3:
            return "NATIVE-RGB"
        if self.effective_in_channels == 3:
            return "RGB-PROMOTED"
        return "NATIVE-GRAY"

    # --- Factory Methods ---
    @classmethod
    def from_args(cls, args: argparse.Namespace, metadata: DatasetMetadata) -> "DatasetConfig":
        """
        Factory method to resolve dataset configuration from CLI arguments and registry metadata.
        
        Args:
            args: Parsed command-line arguments.
            metadata: Static dataset properties from the registry.
        """
        # 1. Resolve RGB logic
        is_pretrained = getattr(args, "pretrained", True)
        force_rgb_arg = getattr(args, "force_rgb", None)
        should_force_rgb = (
            force_rgb_arg if force_rgb_arg is not None 
            else (metadata.in_channels == 1 and is_pretrained)
        )
            
        # 2. Resolve sampling constraints
        # Get value from CLI; if missing or 0, we decide the fallback
        cli_max = getattr(args, "max_samples", None)
        
        # If user explicitly passed 0 or -1, they want the FULL dataset
        if cli_max is not None and cli_max <= 0:
            final_max_samples = None
        # If user passed a specific value (e.g. 5000), use it
        elif cli_max is not None and cli_max > 0:
            final_max_samples = cli_max
        # Otherwise (if None or not provided), use the Class Default (20000)
        else:
            final_max_samples = cls.model_fields['max_samples'].default

        return cls(
            metadata=metadata,
            max_samples=final_max_samples,
            use_weighted_sampler=getattr(args, "use_weighted_sampler", True),
            force_rgb=should_force_rgb,
            img_size=getattr(args, "img_size", 28)
        )