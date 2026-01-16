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
from .reporter import Reporter

# =========================================================================== #
#                                __all__ Definition                           #
# =========================================================================== #
__all__ = ["Logger", "Reporter"]