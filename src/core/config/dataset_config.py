"""
Dataset Registry Orchestration & Metadata Resolution.

Bridges static dataset metadata with runtime execution requirements. Normalizes 
datasets regardless of native format (Grayscale/RGB) to meet model architecture 
input specifications.

Key Responsibilities:
    * Adaptive normalization: Adjusts mean/std based on channel logic
    * Feature promotion: Automates grayscale-to-RGB for ImageNet weights
    * Resource budgeting: Enforces sampling limits and class balancing
    * Multi-resolution support: Resolves metadata by selected resolution
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
from .types import ImageSize, ValidatedPath, PositiveInt
from ..metadata import (
    DatasetMetadata, DatasetRegistryWrapper, DATASET_REGISTRY
)
from ..paths import DATASET_DIR

# =========================================================================== #
#                          Dataset Configuration                              #
# =========================================================================== #

class DatasetConfig(BaseModel):
    """
    Validated manifest for dataset execution context.
    
    Bridges static registry metadata with runtime preferences. Resolves 
    channel promotion and sampling policies for the training pipeline 
    with multi-resolution support.
    """
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True
    )

    name: Optional[str] = Field(
        default=None,
        description="Dataset identifier (e.g., 'bloodmnist')"
    )
    metadata: Optional[DatasetMetadata] = Field(
        default=None, exclude=True
    )

    # Runtime parameters
    data_root: ValidatedPath = DATASET_DIR
    use_weighted_sampler: bool = True
    max_samples: Optional[PositiveInt] = Field(default=20000)
    img_size: ImageSize = Field(
        description="Target square resolution for model input",
        default=28,
    )
    force_rgb: bool = Field(
        default=True,
        description="Convert grayscale to RGB for ImageNet weights"
    )
    resolution: Optional[int] = Field(
        default=28,
        description="Target dataset resolution (28 or 224)"
    )

    # --- Properties ---
    @property
    def dataset_name(self) -> str:
        """Dataset identifier (e.g., 'bloodmnist')."""
        if self.metadata is None or self.metadata.name is None:
            self.metadata = DatasetRegistryWrapper().get_dataset(
                list(DATASET_REGISTRY.keys())[0]
            )
        return self.metadata.name
    
    @property
    def num_classes(self) -> int:
        """Number of unique target classes."""
        return self.metadata.num_classes

    @property
    def in_channels(self) -> int:
        """Native dataset channels (1 or 3)."""
        return self.metadata.in_channels
    
    @property
    def effective_in_channels(self) -> int:
        """Actual channels model receives (3 if force_rgb enabled)."""
        return 3 if self.force_rgb \
            else self.in_channels
    
    @property
    def mean(self) -> tuple[float, ...]:
        """Channel-wise mean, expanded if force_rgb on grayscale."""
        m = self.metadata.mean
        return (m[0],) * 3 if self.force_rgb and self.in_channels == 1 \
            else m
    
    @property
    def std(self) -> tuple[float, ...]:
        """Channel-wise std, expanded if force_rgb on grayscale."""
        s = self.metadata.std
        return (s[0],) * 3 if self.force_rgb and self.in_channels == 1 \
            else s

    @property
    def processing_mode(self) -> str:
        """Channel resolution strategy description."""
        if self.in_channels == 3:
            return "NATIVE-RGB"
        return "RGB-PROMOTED" if self.effective_in_channels == 3 \
            else "NATIVE-GRAY"

    # --- Encapsulated Logic in Separate Methods ---
    @classmethod
    def resolve_rgb_promotion(cls, args, resolved_metadata) -> bool:
        """Resolve the RGB promotion logic."""
        is_pretrained = getattr(args, "pretrained", True)
        force_rgb_cli = getattr(args, "force_rgb", None)
        return force_rgb_cli if force_rgb_cli is not None \
            else (resolved_metadata.in_channels == 1 and is_pretrained)

    @classmethod
    def resolve_max_samples(cls, args) -> Optional[PositiveInt]:
        """Resolve max_samples logic."""
        cli_max = getattr(args, "max_samples", None)
        return None if (cli_max is not None and cli_max <= 0) \
            else (cli_max or cls.model_fields['max_samples'].default)

    @classmethod
    def resolve_img_size(cls, args, resolved_metadata) -> int:
        """Resolve image size from CLI args or metadata."""
        return getattr(args, "img_size", None) or resolved_metadata.native_resolution

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "DatasetConfig":
        """
        Factory method to create a DatasetConfig from CLI arguments or runtime config.

        Resolves conflicts between CLI args and dataset registry metadata,
        handling RGB promotion, sampling limits, image size, and multi-resolution support.

        Args:
            args: Parsed CLI arguments containing dataset selection and runtime preferences.

        Returns:
            Configured DatasetConfig instance with proper metadata.
        """
        # 1. Determine dataset name from CLI args or config
        dataset_name = getattr(args, "dataset", None) or getattr(args, "name", None)

        # If no dataset is specified, fall back to the first dataset in the registry
        if dataset_name is None:
            dataset_name = list(DATASET_REGISTRY.keys())[0]

        # 2. Determine resolution from CLI args (default is 28)
        resolution = getattr(args, "resolution", 28)

        # 3. Load dataset metadata from registry (ignores any old metadata)
        wrapper = DatasetRegistryWrapper(resolution=resolution)
        try:
            # Attempt to get dataset metadata using the provided dataset name
            resolved_metadata = wrapper.get_dataset(dataset_name)
        except KeyError:
            # Handle the case where the dataset is not found in the registry
            available_datasets = list(wrapper.registry.keys())
            raise KeyError(
                f"Dataset '{dataset_name}' not found. Available datasets: {available_datasets}"
            )

        # 4. Resolve RGB promotion, sampling limits, and image size using helper methods
        resolved_force_rgb = cls.resolve_rgb_promotion(args, resolved_metadata)
        resolved_max = cls.resolve_max_samples(args)
        resolved_img_size = cls.resolve_img_size(args, resolved_metadata)

        # 5. Construct and return DatasetConfig with all resolved parameters
        return cls(
            data_root=Path(getattr(args, "data_dir", DATASET_DIR)),
            metadata=resolved_metadata,
            max_samples=resolved_max,
            use_weighted_sampler=getattr(args, "use_weighted_sampler", True),
            force_rgb=resolved_force_rgb,
            img_size=resolved_img_size,
            resolution=resolution
        )
