"""
Optuna objective components for the training pipeline.

This package provides the Optuna objective function and its supporting
components, structured around single-responsibility modules for
configuration building, metric extraction, and training execution.
"""

# =========================================================================== #
#                             Internal Imports                                #
# =========================================================================== #
from .objective import OptunaObjective
from .training_executor import TrialTrainingExecutor
from .metric_extractor import MetricExtractor
from .config_builder import TrialConfigBuilder

# =========================================================================== #
#                              Public API                                     #
# =========================================================================== #
__all__ = [
    "OptunaObjective",
    "TrialTrainingExecutor",
    "MetricExtractor",
    "TrialConfigBuilder",
]
