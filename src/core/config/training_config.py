"""
Optimization & Regularization Configuration Schema.

This module defines the declarative schema for the training lifecycle, 
orchestrating the optimization landscape and regularization boundaries. 
It synchronizes learning rate dynamics (Cosine Annealing) with advanced 
data augmentation strategies (Mixup) and precision policies (AMP).

Key Architectural Features:
    * Optimization Dynamics: Manages learning rate schedules and momentum 
      parameters to navigate the loss landscape effectively.
    * Regularization Suite: Controls Label Smoothing and Mixup coefficients, 
      facilitating model generalization on medical imaging datasets.
    * Precision & Stability: Configures Automatic Mixed Precision (AMP) and 
      Gradient Clipping to optimize hardware throughput and training stability.
    * Reproducibility Guard: Integrates global seeding and strict 
      determinism flags to ensure bit-perfect experimental replication.

By enforcing strict boundary validation (e.g., probability ranges), the 
schema prevents unstable training states before the first batch is processed.
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
    PositiveInt, NonNegativeInt, NonNegativeFloat,
    Probability, SmoothingValue, LearningRate, GradNorm
)

# =========================================================================== #
#                             TRAINING CONFIGURATION                          #
# =========================================================================== #

class TrainingConfig(BaseModel):
    """
    Defines the optimization landscape and regularization strategies.
    
    This class ensures that all training-related hyperparameters are 
    validated against physical and domain-specific constraints.
    """
    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )
    
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    reproducible: bool = Field(
        default=False,
        description="If True, enables strict reproducibility mode"
    )
    batch_size: PositiveInt = Field(default=128)
    epochs: PositiveInt = Field(default=60)
    patience: NonNegativeInt = Field(default=15)
    learning_rate: LearningRate = Field(default=0.008)
    min_lr: LearningRate = Field(default=1e-6)
    momentum: Probability = Field(default=0.9)
    weight_decay: NonNegativeFloat = Field(default=5e-4)
    label_smoothing: SmoothingValue = 0.0
    
    mixup_alpha: NonNegativeFloat = Field(
        default=0.2,
        description="Mixup interpolation coefficient"
    )
    mixup_epochs: NonNegativeInt = Field(
        default=20,
        description="Number of epochs to apply mixup"
    )
    
    use_tta: bool = True
    cosine_fraction: Probability = Field(default=0.5)
    use_amp: bool = False
    
    grad_clip: Optional[GradNorm] = Field(
        default=1.0,
        description="Max norm for gradient clipping; None to disable"
    )

    scheduler_type: str = Field(
        default="cosine",
        description="Type of LR scheduler: 'cosine', 'plateau', 'step', or 'none'"
    )
    
    scheduler_patience: NonNegativeInt = Field(
        default=5, 
        description="Patience for ReduceLROnPlateau"
    )
    scheduler_factor: Probability = Field(
        default=0.1, 
        description="Reduction factor for Plateau/StepLR"
    )
    step_size: PositiveInt = Field(
        default=20, 
        description="Period of learning rate decay for StepLR"
    )
    criterion_type: str = Field(
        default="cross_entropy",
        description="Loss function: 'cross_entropy', 'focal', or 'bce_logit'"
    )
    weighted_loss: bool = Field(
        default=False,
        description="Whether to apply class-frequency weighting"
    )
    focal_gamma: NonNegativeFloat = Field(
        default=2.0,
        description="Focusing parameter for Focal Loss"
    )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        """
        Factory method to map CLI arguments to the TrainingConfig schema.
        Only overrides fields that are present in the args Namespace.
        """
        arg_map = {
            "seed": "seed",
            "reproducible": "reproducible",
            "batch_size": "batch_size",
            "learning_rate": "lr",
            "momentum": "momentum",
            "weight_decay": "weight_decay",
            "epochs": "epochs",
            "patience": "patience",
            "mixup_alpha": "mixup_alpha",
            "mixup_epochs": "mixup_epochs",
            "use_tta": "use_tta",
            "cosine_fraction": "cosine_fraction",
            "use_amp": "use_amp",
            "grad_clip": "grad_clip",
            "label_smoothing": "label_smoothing",
            "min_lr": "min_lr",
            "scheduler_type": "scheduler_type",
            "scheduler_patience": "scheduler_patience",
            "scheduler_factor": "scheduler_factor",
            "step_size": "step_size",
            "criterion_type": "criterion_type",
            "weighted_loss": "weighted_loss",
            "focal_gamma": "focal_gamma"
        }
        
        params = {}
        for config_key, arg_key in arg_map.items():
            val = getattr(args, arg_key, None)
            if val is not None:
                params[config_key] = val
        
        return cls(**params)