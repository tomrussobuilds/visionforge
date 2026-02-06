"""
Configuration Package Initialization.

Provides a unified, flat public API for configuration components while
avoiding eager imports of heavy or optional dependencies (e.g. torch).

Architecture:
    - Lazy Import Pattern (PEP 562): Uses __getattr__ for on-demand loading
    - Deferred Dependencies: torch and pydantic loaded only when needed
    - Flat API: All configs accessible from orchard.core.config namespace
    - Caching: Loaded modules cached in globals() for performance

Implementation:
    1. __all__: Public API contract listing all available configs
    2. _LAZY_IMPORTS: Mapping from config names to module paths
    3. __getattr__: Dynamic loader triggered on first access
    4. __dir__: IDE/introspection support for auto-completion

Example:
    >>> from orchard.core.config import Config, HardwareConfig
    >>> # torch is NOT imported yet (lazy loading)
    >>> cfg = Config.from_args(args)
    >>> # NOW torch is imported (triggered by Config access)
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Config",
    "HardwareConfig",
    "TelemetryConfig",
    "TrainingConfig",
    "AugmentationConfig",
    "DatasetConfig",
    "EvaluationConfig",
    "ModelConfig",
    "InfrastructureManager",
    "InfraManagerProtocol",
    "ValidatedPath",
    "OptunaConfig",
    "ExportConfig",
]

# LAZY IMPORTS MAPPING
_LAZY_IMPORTS: dict[str, str] = {
    "Config": "orchard.core.config.manifest",
    "HardwareConfig": "orchard.core.config.hardware_config",
    "TelemetryConfig": "orchard.core.config.telemetry_config",
    "TrainingConfig": "orchard.core.config.training_config",
    "AugmentationConfig": "orchard.core.config.augmentation_config",
    "DatasetConfig": "orchard.core.config.dataset_config",
    "EvaluationConfig": "orchard.core.config.evaluation_config",
    "ModelConfig": "orchard.core.config.models_config",
    "InfrastructureManager": "orchard.core.config.infrastructure_config",
    "InfraManagerProtocol": "orchard.core.config.infrastructure_config",
    "ValidatedPath": "orchard.core.config.types",
    "OptunaConfig": "orchard.core.config.optuna_config",
    "ExportConfig": "orchard.core.config.export_config",
}


# LAZY LOADER FUNCTION
def __getattr__(name: str) -> Any:
    """
    Lazily import configuration components on first access.

    Prevents importing heavy dependencies (e.g. torch) unless the
    corresponding configuration class is actually used.
    """
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_LAZY_IMPORTS[name])
    attr = getattr(module, name)

    # Cache on module for future access
    globals()[name] = attr
    return attr


# DIR SUPPORT
def __dir__() -> list[str]:
    """
    Support for dir() and IDE auto-completion.

    Returns:
        Sorted list of public configuration class names
    """
    return sorted(__all__)
