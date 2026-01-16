"""
Hyperparameter Search Space Definitions for Optuna.

Defines search distributions for each optimizable parameter, respecting
the type constraints defined in src/core/config/types.py.

Each search space is defined with:
    - Type-aware bounds (matching Pydantic validators)
    - Distribution strategy (log/uniform/categorical)
    - Domain expertise defaults (medical imaging best practices)
"""

# =========================================================================== #
#                         Standard Imports                                    #
# =========================================================================== #
from typing import Dict, Callable

# =========================================================================== #
#                         Third-Party Imports                                 #
# =========================================================================== #
import optuna

# =========================================================================== #
#                        SEARCH SPACE DEFINITIONS                             #
# =========================================================================== #

class SearchSpaceRegistry:
    """
    Centralized registry of hyperparameter search distributions.
    
    Each method returns a dict of {param_name: suggest_function} where
    suggest_function takes a Trial object and returns a sampled value.
    """
    
    @staticmethod
    def get_optimization_space() -> Dict[str, Callable]:
        """
        Core optimization hyperparameters (learning rate, weight decay, etc.).
        
        Returns:
            Dict mapping parameter names to sampling functions
        """
        return {
            "learning_rate": lambda trial: trial.suggest_float(
                "learning_rate", 
                1e-5, 1e-2,  # LearningRate bounds: gt=1e-8, lt=1.0
                log=True
            ),
            "weight_decay": lambda trial: trial.suggest_float(
                "weight_decay",
                1e-6, 1e-3,  # WeightDecay bounds: ge=0.0, le=0.2
                log=True
            ),
            "momentum": lambda trial: trial.suggest_float(
                "momentum",
                0.85, 0.95  # Momentum bounds: ge=0.0, lt=1.0
            ),
            "min_lr": lambda trial: trial.suggest_float(
                "min_lr",
                1e-7, 1e-5,
                log=True
            ),
        }
    
    @staticmethod
    def get_regularization_space() -> Dict[str, Callable]:
        """
        Regularization strategies (mixup, label smoothing, dropout).
        
        Returns:
            Dict of regularization parameter samplers
        """
        return {
            "mixup_alpha": lambda trial: trial.suggest_float(
                "mixup_alpha",
                0.0, 0.4  # Common range for medical imaging
            ),
            "label_smoothing": lambda trial: trial.suggest_float(
                "label_smoothing",
                0.0, 0.2  # SmoothingValue bounds: ge=0.0, le=0.3
            ),
            "dropout": lambda trial: trial.suggest_float(
                "dropout",
                0.1, 0.5  # DropoutRate bounds: ge=0.0, le=0.9
            ),
        }
    
    @staticmethod
    def get_batch_size_space(resolution: int = 28) -> Dict[str, Callable]:
        """
        Batch size as categorical (powers of 2 for GPU efficiency).
        
        RESOLUTION-AWARE: Smaller batches for high-res to prevent OOM.
        
        Args:
            resolution: Input image resolution (28 or 224)
        
        Returns:
            Dict with batch_size sampler
        """
        if resolution >= 224:
            # High-res (224×224): Conservative batches for 8GB VRAM
            batch_choices = [8, 12, 16]
        else:
            # Low-res (28×28): Can handle larger batches
            batch_choices = [16, 32, 48, 64]
        
        return {
            "batch_size": lambda trial: trial.suggest_categorical(
                "batch_size",
                batch_choices
            ),
        }
    
    @staticmethod
    def get_scheduler_space() -> Dict[str, Callable]:
        """
        Learning rate scheduler parameters.
        
        Returns:
            Dict of scheduler-related samplers
        """
        return {
            "cosine_fraction": lambda trial: trial.suggest_float(
                "cosine_fraction",
                0.3, 0.7  # Probability bounds: ge=0.0, le=1.0
            ),
            "scheduler_patience": lambda trial: trial.suggest_int(
                "scheduler_patience",
                3, 10  # NonNegativeInt
            ),
        }
    
    @staticmethod
    def get_augmentation_space() -> Dict[str, Callable]:
        """
        Data augmentation intensity parameters.
        
        Returns:
            Dict of augmentation samplers
        """
        return {
            "rotation_angle": lambda trial: trial.suggest_int(
                "rotation_angle",
                0, 15  # RotationDegrees: ge=0, le=360
            ),
            "jitter_val": lambda trial: trial.suggest_float(
                "jitter_val",
                0.0, 0.15
            ),
            "min_scale": lambda trial: trial.suggest_float(
                "min_scale",
                0.9, 1.0
            ),
        }
    
    @staticmethod
    def get_full_space(resolution: int = 28) -> Dict[str, Callable]:
        """
        Combined search space with all available parameters.
        
        Args:
            resolution: Input image resolution for batch size calculation
        
        Returns:
            Unified dict of all parameter samplers
        """
        full_space = {}
        full_space.update(SearchSpaceRegistry.get_optimization_space())
        full_space.update(SearchSpaceRegistry.get_regularization_space())
        full_space.update(SearchSpaceRegistry.get_batch_size_space(resolution))
        full_space.update(SearchSpaceRegistry.get_scheduler_space())
        full_space.update(SearchSpaceRegistry.get_augmentation_space())
        return full_space
    
    @staticmethod
    def get_quick_space(resolution: int = 28) -> Dict[str, Callable]:
        """
        Reduced search space for fast exploration (most impactful params).
        
        Focuses on:
            - Learning rate (most critical)
            - Weight decay
            - Batch size (resolution-aware)
            - Dropout
        
        Args:
            resolution: Input image resolution for batch size calculation
        
        Returns:
            Dict of high-impact parameter samplers
        """
        space = {}
        space.update(SearchSpaceRegistry.get_optimization_space())
        space.update({
            "batch_size": SearchSpaceRegistry.get_batch_size_space(resolution)["batch_size"],
            "dropout": SearchSpaceRegistry.get_regularization_space()["dropout"],
        })
        return space


# =========================================================================== #
#                          PRESET CONFIGURATIONS                              #
# =========================================================================== #

def get_search_space(preset: str = "quick", resolution: int = 28):
    """
    Factory function to retrieve a search space preset.
    
    Args:
        preset: Name of the preset ("quick", "full", "optimization_only", etc.)
        resolution: Input image resolution (affects batch_size choices)
        
    Returns:
        Dict of parameter samplers or FullSearchSpace instance
        
    Raises:
        ValueError: If preset name not recognized
    """
    # Non-batch-size-dependent presets
    simple_presets = {
        "optimization_only": SearchSpaceRegistry.get_optimization_space,
        "regularization_only": SearchSpaceRegistry.get_regularization_space,
    }
    
    if preset in simple_presets:
        return simple_presets[preset]()
    
    # Resolution-dependent presets
    if preset == "quick":
        return SearchSpaceRegistry.get_quick_space(resolution)
    elif preset == "full":
        return FullSearchSpace(resolution)
    else:
        raise ValueError(
            f"Unknown preset '{preset}'. Available: quick, full, "
            f"optimization_only, regularization_only"
        )


class FullSearchSpace:
    """
    Resolution-aware full search space with dynamic batch_size constraints.
    
    Prevents OOM errors by limiting batch sizes based on input resolution.
    """
    
    def __init__(self, resolution: int = 28):
        """
        Initialize search space with resolution context.
        
        Args:
            resolution: Input image resolution (28, 224, etc.)
        """
        self.resolution = resolution
    
    def sample_params(self, trial: optuna.Trial) -> dict:
        """
        Sample hyperparameters with resolution-aware constraints.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dict of sampled hyperparameters
        """
        # Determine max safe batch_size based on resolution
        if self.resolution >= 224:
            # 224×224: Max batch_size=16 even with AMP 
            batch_choices = [8, 12, 16]
        else:
            # 28×28: Can handle larger batches
            batch_choices = [16, 32, 48, 64]
        
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "momentum": trial.suggest_float("momentum", 0.85, 0.95),
            "min_lr": trial.suggest_float("min_lr", 1e-8, 1e-5, log=True),
            "mixup_alpha": trial.suggest_float("mixup_alpha", 0.0, 0.4),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "batch_size": trial.suggest_categorical("batch_size", batch_choices),
            "cosine_fraction": trial.suggest_float("cosine_fraction", 0.3, 0.7),
            "scheduler_patience": trial.suggest_int("scheduler_patience", 3, 10),
            "rotation_angle": trial.suggest_int("rotation_angle", 0, 15),
            "jitter_val": trial.suggest_float("jitter_val", 0.0, 0.15),
            "min_scale": trial.suggest_float("min_scale", 0.90, 1.0),
        }