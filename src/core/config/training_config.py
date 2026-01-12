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
    PositiveInt, NonNegativeInt, NonNegativeFloat, WeightDecay,
    Probability, SmoothingValue, LearningRate, GradNorm, Momentum
)

# =========================================================================== #
#                             TRAINING CONFIGURATION                          #
# =========================================================================== #

class TrainingConfig(BaseModel):
    """
    Defines the optimization landscape and regularization strategies.
    
    This class ensures all training hyperparameters are validated and provides
    clear structure for reproducibility, optimization, regularization, and
    scheduler configuration.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )

    # ==================== Reproducibility ====================
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    reproducible: bool = Field(
        default=False,
        description="Enable strict reproducibility mode"
    )

    # ==================== Training Loop ====================
    batch_size: PositiveInt = Field(
        default=128,
        description="Number of samples per batch"
    )
    epochs: PositiveInt = Field(
        default=60,
        description="Maximum number of training epochs"
    )
    patience: NonNegativeInt = Field(
        default=15,
        description="Early stopping patience in epochs"
    )

    # ==================== Optimization ====================
    learning_rate: LearningRate = Field(
        default=0.008,
        description="Initial learning rate"
    )
    min_lr: LearningRate = Field(
        default=1e-6,
        description="Minimum learning rate"
    )
    momentum: Momentum = Field(
        default=0.9,
        description="SGD momentum factor"
    )
    weight_decay: WeightDecay = Field(
        default=5e-4,
        description="Weight decay (L2 regularization)"
    )
    grad_clip: Optional[GradNorm] = Field(
        default=1.0,
        description="Maximum gradient norm; None to disable"
    )

    # ==================== Regularization ====================
    label_smoothing: SmoothingValue = Field(
        default=0.0,
        description="Label smoothing factor for classification"
    )
    mixup_alpha: NonNegativeFloat = Field(
        default=0.2, 
        description="Mixup interpolation coefficient"
    )
    mixup_epochs: NonNegativeInt = Field(
        default=20,
        description="Number of epochs to apply Mixup"
    )
    use_tta: bool = Field(
        default=True,
        description="Enable Test Time Augmentation (TTA)"
    )

    # ==================== Scheduler ====================
    scheduler_type: str = Field(
        default="cosine",
        description="LR scheduler type: 'cosine', 'plateau', 'step', or 'none'"
    )
    cosine_fraction: Probability = Field(
        default=0.5,
        description="Fraction of total epochs for cosine annealing"
    )
    scheduler_patience: NonNegativeInt = Field(
        default=5,
        description="Patience for ReduceLROnPlateau"
    )
    scheduler_factor: Probability = Field(
        default=0.1,
        description="LR reduction factor for Plateau/StepLR"
    )
    step_size: PositiveInt = Field(
        default=20,
        description="Period of LR decay for StepLR"
    )
    use_amp: bool = Field(
        default=False,
        description="Enable Automatic Mixed Precision (AMP)"
    )

    # ==================== Loss ====================
    criterion_type: str = Field(
        default="cross_entropy",
        description="Loss function: 'cross_entropy', 'focal', or 'bce_logit'"
    )
    weighted_loss: bool = Field(
        default=False,
        description="Apply class-frequency weighting"
    )
    focal_gamma: NonNegativeFloat = Field(
        default=2.0,
        description="Focusing parameter for Focal Loss"
    )

    # ==================== Factory Method ====================
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        """
        Factory method to instantiate TrainingConfig from CLI arguments.

        Only overrides schema fields that exist in the args Namespace and are not None.
        This approach automatically adapts to new fields added to the schema.

        Args:
            args (argparse.Namespace): Parsed command-line arguments.

        Returns:
            TrainingConfig: Config instance with CLI-overridden values.
        """
        args_dict = vars(args)
        # Keep only args that match schema fields and are not None
        valid_fields = cls.model_fields.keys()
        params = {
            k: v for k, v in args_dict.items() if k in valid_fields and v is not None
        }

        return cls(**params)
