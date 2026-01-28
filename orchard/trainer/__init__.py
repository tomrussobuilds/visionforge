"""
Trainer Package Facade

This package exposes the central ModelTrainer class, the optimization factories,
and the low-level execution engines, providing a unified interface for the
training lifecycle.
"""

# Internal Imports
from .engine import mixup_data, train_one_epoch, validate_epoch
from .setup import get_criterion, get_optimizer, get_scheduler
from .trainer import ModelTrainer

# Exports
__all__ = [
    "ModelTrainer",
    "train_one_epoch",
    "validate_epoch",
    "mixup_data",
    "get_optimizer",
    "get_scheduler",
    "get_criterion",
]
