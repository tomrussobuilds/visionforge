"""
Optuna Hyperparameter Optimization Module.

Provides components for automated hyperparameter search:
    - SearchSpaceRegistry: Predefined search distributions
    - OptunaObjective: Objective function for optimization
    - OptunaOrchestrator: Study lifecycle management
    - run_optimization: Convenience function for complete workflow
"""

from .search_spaces import SearchSpaceRegistry, get_search_space
from .objective import OptunaObjective, PrunableTrainer
from .orchestrator import OptunaOrchestrator, run_optimization
from .early_stopping import StudyEarlyStoppingCallback, get_early_stopping_callback


__all__ = [
    "SearchSpaceRegistry",
    "get_search_space",
    "OptunaObjective",
    "PrunableTrainer",
    "OptunaOrchestrator",
    "run_optimization",
    "StudyEarlyStoppingCallback",
    "get_early_stopping_callback",
]