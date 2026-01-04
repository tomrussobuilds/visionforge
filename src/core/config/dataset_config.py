"""
Dataset Registry Resolver & Metadata Schema.

This module acts as the interface between the MedMNIST Metadata Registry 
and the training pipeline. It handles the dynamic resolution of dataset-specific 
constraints, ensuring that domain-specific medical data aligns with deep 
learning architecture requirements.

Key Architectural Features:
    * Metadata Binding: Injects static MedMNIST properties (classes, distribution) 
      into a validated Pydantic manifest.
    * Channel Orchestration: Implements 'force_rgb' promotion logic, allowing 
      single-channel medical datasets (e.g., PneumoniaMNIST) to utilize 
      architectures pretrained on 3-channel natural images (ImageNet).
    * Statistical Expansion: Dynamically adapts normalization constants (mean/std) 
      based on the resolved channel strategy, preventing statistical mismatch 
      during the 'promotion' of grayscale images.
    * Sample Budgeting: Manages dataset truncation and weighted sampling 
      policies to handle class imbalance and computational constraints.

By centralizing these transformations, the resolver ensures that the data 
loading pipeline remains agnostic to the specific MedMNIST variant being processed.
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
        m = self.metadata.mean
        return (m[0],) * 3 if (self.force_rgb and self.in_channels == 1) else m
    
    @property
    def std(self) -> tuple[float, ...]:
        """Channel-wise std for normalization, expanded if force_rgb is active."""
        s = self.metadata.std
        return (s[0],) * 3 if (self.force_rgb and self.in_channels == 1) else s
    
    @property
    def processing_mode(self) -> str:
        """
        Human-readable description of the channel processing strategy.
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
        """
        # 1. Resolve RGB logic
        is_pretrained = getattr(args, "pretrained", True)
        force_rgb_arg = getattr(args, "force_rgb", None)
        should_force_rgb = (
            force_rgb_arg if force_rgb_arg is not None 
            else (metadata.in_channels == 1 and is_pretrained)
        )
            
        # 2. Resolve sampling constraints
        cli_max = getattr(args, "max_samples", None)
        if cli_max is not None and cli_max <= 0:
            final_max_samples = None
        elif cli_max is not None and cli_max > 0:
            final_max_samples = cli_max
        else:
            final_max_samples = cls.model_fields['max_samples'].default

        return cls(
            metadata=metadata,
            data_root=Path(getattr(args, "data_dir", DATASET_DIR)),
            max_samples=final_max_samples,
            use_weighted_sampler=getattr(args, "use_weighted_sampler", True),
            force_rgb=should_force_rgb,
            img_size=getattr(args, "img_size", 28)
        )