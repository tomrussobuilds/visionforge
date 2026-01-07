"""
Dataset Metadata Base Definitions

This module defines the schema for dataset metadata using Pydantic 
to ensure immutability, type safety, and seamless integration 
with the global configuration engine.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from typing import List, Tuple
from pathlib import Path

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
from pydantic import BaseModel, Field, ConfigDict

# =========================================================================== #
#                                METADATA SCHEMA                              #
# =========================================================================== #

class DatasetMetadata(BaseModel):
    """
    Metadata container for a MedMNIST dataset.
    
    This structure ensures that all dataset-specific constants are grouped
    and immutable throughout the execution of the pipeline. It serves as
    the static definition that feeds into the dynamic DatasetConfig.
    """
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True
    )

    name: str = Field(
        ...,
        description="Short identifier (e.g., 'pathmnist')"
    )
    display_name: str = Field(
        ...,
        description="Full dataset name for reporting"
    )
    md5_checksum: str = Field(
        ...,
        description="MD5 hash for archive integrity"
    )
    url: str = Field(
        ...,
        description="Source URL for automated downloads"
    )
    path: Path = Field(
        ...,
        description="Relative or absolute path to the .npz archive"
    )
    
    classes: List[str] = Field(
        ...,
        description="List of class labels in index order"
    )
    in_channels: int = Field(
        ...,
        description="1 for Grayscale, 3 for RGB"
    )
    native_resolution: int = Field(
        default=28, 
        description="Native pixel resolution of the samples (e.g., 28 or 224)"
    )
    
    # Normalization parameters
    mean: Tuple[float, ...] = Field(
        ...,
        description="Channel-wise mean for normalization"
    )
    std: Tuple[float, ...] = Field(
        ...,
        description="Channel-wise std for normalization"
    )
    
    # Behavioral flags
    is_anatomical: bool = Field(
        default=True, 
        description="True if the dataset has a fixed anatomical orientation (e.g., ChestMNIST)"
    )
    is_texture_based: bool = Field(
        default=True,
        description="True if the dataset classification relies on texture patterns (e.g., PathMNIST)"
    )

    @property
    def normalization_info(self) -> str:
        """Returns a formatted string of mean/std for reporting purposes."""
        return f"Mean: {self.mean} | Std: {self.std}"
    
    @property
    def resolution_str(self) -> str:
        """Returns the formatted native resolution string (e.g., '28x28' or '224x224')."""
        return f"{self.native_resolution}x{self.native_resolution}"

    @property
    def num_classes(self) -> int:
        """Returns the total number of target classes."""
        return len(self.classes)
    
    def __repr__(self) -> str:
        return f"<DatasetMetadata: {self.display_name} ({self.resolution_str}, {self.num_classes} classes)>"