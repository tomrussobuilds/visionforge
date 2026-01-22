"""
Optuna Study Orchestrator Package.

High-level coordination for hyperparameter optimization studies.
Provides a modular architecture for study creation, execution,
visualization, and result export.

Main Components:
    - OptunaOrchestrator: Primary orchestration class
    - run_optimization: Convenience function for full pipeline

Usage:
    from orchard.optimization.orchestrator import OptunaOrchestrator, run_optimization

    study = run_optimization(cfg=config, device=device, paths=paths)
"""

from .exporters import (
    export_best_config,
    export_study_summary,
    export_top_trials,
)
from .orchestrator import (
    OptunaOrchestrator,
    run_optimization,
)

__all__ = [
    "OptunaOrchestrator",
    "run_optimization",
    "export_best_config",
    "export_study_summary",
    "export_top_trials",
]
