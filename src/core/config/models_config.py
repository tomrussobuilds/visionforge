"""
Model Architecture Configuration Module.

This module defines the schema and validation logic for the deep learning 
architectures used in the pipeline. It handles parameters related to 
model identification, weight initialization (pre-training), and 
structural adaptations like dropout for regularization.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import argparse

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
from pydantic import (
    BaseModel, ConfigDict, Field
)

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .types import DropoutRate
from ..metadata import DatasetMetadata

# =========================================================================== #
#                                MODEL CONFIGURATION                         #
# =========================================================================== #

class ModelConfig(BaseModel):
    """
    Configuration for Model Architecture and Weight Initialization.
    
    This sub-config manages the structural identity of the model and its
    initial state. It is initialized via metadata injection to ensure 
    perfect alignment between the dataset properties and the model's 
    input/output layers.
    """
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True
    )
    
    name: str = Field(
        default="resnet18",
        description="The unique identifier for the model architecture."
    )
    
    pretrained: bool = Field(
        default=True,
        description="Whether to initialize the model with pre-trained weights."
    )
    
    dropout: DropoutRate = Field(
        default=0.2,
        description="Dropout probability for the classification head."
    )

    # --- Structural Fields (Injected/Derived) ---
    in_channels: int = Field(
        ..., 
        description="Number of input channels the architecture expects."
    )
    num_classes: int = Field(
        ..., 
        description="Number of output logits for the final classification layer."
    )

    @classmethod
    def from_args(cls, args: argparse.Namespace, metadata: DatasetMetadata) -> "ModelConfig":
        """
        Factory method to create a ModelConfig from CLI arguments and metadata.
        
        Logic:
            - num_classes is derived directly from the dataset metadata.
            - in_channels is determined by the dataset's native channels, 
              accounting for automatic RGB promotion if using pretrained weights 
              on grayscale images.
        """
        is_pretrained = getattr(args, 'pretrained', True)
        
        # Centralized channel logic
        force_rgb_arg = getattr(args, 'force_rgb', None)
        should_force_rgb = force_rgb_arg if force_rgb_arg is not None else \
                           (metadata.in_channels == 1 and is_pretrained)

        return cls(
            name=getattr(args, 'model_name', "resnet18"),
            pretrained=is_pretrained,
            dropout=getattr(args, 'dropout', 0.2),
            num_classes=len(metadata.classes),
            in_channels=3 if should_force_rgb else metadata.in_channels
        )