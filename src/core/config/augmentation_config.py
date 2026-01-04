"""
Data Augmentation & Test-Time Augmentation (TTA) Schema.

This module defines the declarative schema for the stochastic transformation 
pipeline. It synchronizes geometric and photometric noise levels used 
during training with the intensity of perturbations applied during 
Test-Time Augmentation (TTA), ensuring a calibrated approach to model robustness.

Key Architectural Features:
    * Geometric Invariance: Manages parameters for horizontal flips and 
      rotational degrees, helping the model generalize across different 
      imaging orientations common in medical scans.
    * Photometric Consistency: Controls color jitter and scaling factors to 
      account for variations in image acquisition and lighting.
    * TTA Ensemble Strategy: Defines the intensity of pixel shifts, scaling, 
      and Gaussian blur for Test-Time Augmentation, allowing for more stable 
      and confident predictions through stochastic averaging.
    * Validation Guard: Leverages domain-specific types (e.g., RotationDegrees) 
      to ensure that augmentation intensities remain within physically 
      plausible ranges for medical diagnostic imaging.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import argparse

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
from pydantic import BaseModel, Field, ConfigDict

# =========================================================================== #
#                               Internal Imports                              #
# =========================================================================== #
from .types import (
    Probability, RotationDegrees, NonNegativeFloat
)

# =========================================================================== #
#                          AUGMENTATION CONFIGURATION                         #
# =========================================================================== #

class AugmentationConfig(BaseModel):
    """
    Configures the stochastic transformation pipeline for training and TTA.
    
    Standardizes parameters for geometric and photometric augmentations, 
    ensuring consistency between training-time noise and Test-Time 
    Augmentation (TTA) intensity.
    """
    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )
    
    # Training-time Augmentations
    hflip: Probability = Field(default=0.5)
    rotation_angle: RotationDegrees = Field(default=10)
    jitter_val: NonNegativeFloat = Field(default=0.2)
    min_scale: Probability = Field(default=0.9)

    # Test-time Augmentations (TTA)
    tta_translate: float = Field(
        default=2.0,
        description="Pixel shift (in pixels) for TTA perturbations."
    )
    tta_scale: float = Field(
        default=1.1,
        description="Scaling factor used to create TTA variants."
    )
    tta_blur_sigma: float = Field(
        default=0.4,
        description="Gaussian blur intensity for TTA variants."
    )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "AugmentationConfig":
        """
        Factory method to resolve augmentation policies from CLI arguments.
        Ensures a single point of entry for transformation hyperparameters.
        """
        params = {
            "hflip": getattr(args, 'hflip', 0.5),
            "rotation_angle": getattr(args, 'rotation_angle', 10),
            "jitter_val": getattr(args, 'jitter_val', 0.2),
            "min_scale": getattr(args, 'min_scale', 0.9),
            "tta_translate": getattr(args, 'tta_translate', 2.0),
            "tta_scale": getattr(args, 'tta_scale', 1.1),
            "tta_blur_sigma": getattr(args, 'tta_blur_sigma', 0.4)
        }
        return cls(**params)