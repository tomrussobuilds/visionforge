"""
Model Architecture Configuration Module.

This module defines the schema and validation logic for the deep learning 
architectures used in the pipeline. It handles parameters related to 
model identification, weight initialization (pre-training), and 
structural adaptations.
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
#                                MODEL CONFIGURATION                         #
# =========================================================================== #

class ModelConfig(BaseModel):
    """
    Configuration for Model Architecture and Weight Initialization.
    
    This sub-config manages the structural identity of the model and its
    initial state. By isolating these parameters, the pipeline can easily 
    scale to support diverse architectures (ResNet, ViT, Swin) while 
    maintaining a consistent interface for the Model Factory.
    """
    model_config = ConfigDict(
        extra="forbid",
        frozen=True
    )
    
    name: str = Field(
        default="resnet_18_adapted",
        description="The unique identifier for the model architecture."
    )
    
    pretrained: bool = Field(
        default=True,
        description="Whether to initialize the model with ImageNet weights."
    )

    # In future, additional architecture-specific fields can be added here:
    # dropout: float = Field(0.0, ge=0.0, le=1.0)
    # num_heads: Optional[int] = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ModelConfig":
        """
        Factory method to create a ModelConfig from CLI arguments.
        
        Args:
            args (argparse.Namespace): The parsed command-line arguments.

        Returns:
            ModelConfig: A validated model configuration instance.
        """
        return cls(
            name=getattr(args, 'model_name', "resnet_18_adapted"),
            pretrained=getattr(args, 'pretrained', True)
        )