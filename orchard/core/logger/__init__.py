"""
Telemetry and Reporting Package.

This package centralizes experiment logging, environment reporting, and 
visual telemetry. It provides high-level utilities to initialize system-wide 
loggers and format experiment metadata for reproducibility.

Available Components:
    - Logger: Static utility for stream and file logging initialization.
    - Reporter: Metadata reporting engine for environment baseline status.
"""

# =========================================================================== #
#                                Logger & Reporter Exports                    #
# =========================================================================== #
from .logger import Logger
from .reporter import (
    Reporter,
    log_optimization_header,
    log_study_summary,
    log_best_config_export,
    LogStyle,
    log_trial_start
)

# =========================================================================== #
#                                __all__ Definition                           #
# =========================================================================== #
__all__ = [
    "Logger",
    "Reporter",
    "log_optimization_header",
    "log_study_summary",
    "log_best_config_export",
    "LogStyle",
    "log_trial_start"
]