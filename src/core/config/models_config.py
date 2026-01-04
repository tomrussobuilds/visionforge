"""
Model Architecture Configuration Module.

This module defines the declarative schema for deep learning architectures, 
managing the structural lifecycle from identification to structural adaptation. 
It acts as the geometric bridge between dataset properties and neural 
connectivity, ensuring input/output layer alignment.

Key Architectural Features:
    * Structural Identity: Manages model selection and weight initialization 
      policies (e.g., ImageNet pre-training vs. random initialization).
    * Geometric Alignment: Automatically calculates required 'in_channels' and 
      'num_classes' by reconciling dataset metadata with the 'force_rgb' 
      promotion logic.
    * Regularization Policy: Controls structural regularization parameters like 
      Dropout, ensuring they are validated against domain-specific ranges.
    * Metadata Injection: Leverages factory methods to perform 'Late Binding' 
      of architectural constraints at runtime.

By centralizing architectural definitions, the module prevents common mismatch 
errors (e.g., Logit-Class mismatch) before the model is instantiated in memory.
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
            - in_channels resolves the 'force_rgb' logic to promote grayscale 
              images to 3 channels if using pretrained ImageNet weights.
        """
        is_pretrained = getattr(args, 'pretrained', True)
        
        # Centralized channel promotion logic
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