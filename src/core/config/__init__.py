"""
Configuration Package Initialization.

This package exposes the core Experiment Manifest (Config), its constituent 
sub-schemas, and operational handlers. By centralizing exports here, the rest 
of the application can interact with the configuration engine and infrastructure 
orchestration through a unified interface.
"""

# =========================================================================== #
#                                 Main Engine                                 #
# =========================================================================== #
from .engine import Config

# =========================================================================== #
#                                Sub-Configurations                           #
# =========================================================================== #
from .system_config import SystemConfig
from .training_config import TrainingConfig
from .augmentation_config import AugmentationConfig
from .dataset_config import DatasetConfig
from .evaluation_config import EvaluationConfig
from .models_config import ModelConfig

# =========================================================================== #
#                             Operational Handlers                            #
# =========================================================================== #
from .infrastructure_config import InfrastructureManager

# =========================================================================== #
#                                    Types                                    #
# =========================================================================== #
from .types import ValidatedPath

# =========================================================================== #
#                                   EXPORTS                                   #
# =========================================================================== #
__all__ = [
    "Config",
    "SystemConfig",
    "TrainingConfig",
    "AugmentationConfig",
    "DatasetConfig",
    "EvaluationConfig",
    "ModelConfig",
    "InfrastructureManager",
    "ValidatedPath"
]