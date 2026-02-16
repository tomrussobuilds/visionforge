"""
Experiment Tracking Module.

Provides optional MLflow integration for experiment tracking, metric logging,
and artifact management. Designed as an opt-in feature controlled via config.

When MLflow is not installed, the module provides a no-op tracker that silently
skips all logging operations, ensuring zero overhead for users without the
tracking dependency.
"""

from .tracker import MLflowTracker, NoOpTracker, TrackerProtocol, create_tracker

__all__ = [
    "MLflowTracker",
    "NoOpTracker",
    "TrackerProtocol",
    "create_tracker",
]
