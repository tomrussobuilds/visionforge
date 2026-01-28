"""
Logging Management Module

Handles centralized logging configuration with dynamic reconfiguration support.
Enables transition from console-only logging (bootstrap phase) to dual console+file
logging once experiment directories are provisioned by RootOrchestrator.

Key Features:
    - Singleton-like Behavior: Prevents duplicate logger configurations
    - Dynamic Reconfiguration: Switches from console-only to file-based logging
    - Rotating File Handler: Automatic log rotation with size limits
    - Thread-safe: Safe for concurrent access across modules
    - Timestamp-based Files: Unique log files per experiment session
"""

# Standard Imports
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Final, Optional

# Internal Imports
from ..paths import LOGGER_NAME


# LOGGER CLASS
class Logger:
    """
    Manages centralized logging configuration with singleton-like behavior.

    Provides a unified logging interface for the entire framework with support for
    dynamic reconfiguration. Initially bootstraps with console-only output, then
    transitions to dual console+file logging when experiment directories become available.

    The logger implements pseudo-singleton semantics via class-level tracking (_configured_names)
    to prevent duplicate handler registration while allowing intentional reconfiguration
    when log directories are provided.

    Lifecycle:
        1. Bootstrap Phase: Console-only logging (no log_dir specified)
        2. Orchestration Phase: RootOrchestrator calls setup() with log_dir
        3. Reconfiguration: Existing handlers removed, file handler added
        4. Audit Trail: Log file path stored in _active_log_file for reference

    Class Attributes:
        _configured_names (Dict[str, bool]): Tracks which logger names have been configured
        _active_log_file (Optional[Path]): Current active log file path for auditing

    Attributes:
        name (str): Logger identifier (typically LOGGER_NAME constant)
        log_dir (Optional[Path]): Directory for log file storage
        log_to_file (bool): Enable file logging (requires log_dir)
        level (int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_bytes (int): Maximum log file size before rotation (default: 5MB)
        backup_count (int): Number of rotated log files to retain (default: 5)
        logger (logging.Logger): Underlying Python logger instance

    Example:
        >>> # Bootstrap phase (console-only)
        >>> logger = Logger().get_logger()
        >>> logger.info("Framework initializing...")

        >>> # Orchestration phase (add file logging)
        >>> logger = Logger.setup(
        ...     name=LOGGER_NAME,
        ...     log_dir=Path("./outputs/run_123/logs"),
        ...     level="INFO"
        ... )
        >>> logger.info("Logging to file now")

        >>> # Retrieve log file path
        >>> log_path = Logger.get_log_file()
        >>> print(f"Logs saved to: {log_path}")

    Notes:
        - Reconfiguration is idempotent: calling setup() multiple times is safe
        - All handlers are properly closed before reconfiguration
        - Log files use UTC timestamps for consistency across time zones
        - RotatingFileHandler prevents disk space exhaustion
    """

    _configured_names: Final[Dict[str, bool]] = {}
    _active_log_file: Optional[Path] = None

    def __init__(
        self,
        name: str = LOGGER_NAME,
        log_dir: Optional[Path] = None,
        log_to_file: bool = True,
        level: int = logging.INFO,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 5,
    ):
        """
        Initializes the Logger with specified configuration.

        Args:
            name: Logger identifier (default: LOGGER_NAME constant)
            log_dir: Directory for log file storage (None = console-only)
            log_to_file: Enable file logging if log_dir provided (default: True)
            level: Logging level as integer constant (default: logging.INFO)
            max_bytes: Maximum log file size before rotation in bytes (default: 5MB)
            backup_count: Number of rotated backup files to retain (default: 5)
        """
        self.name = name
        self.log_dir = log_dir
        self.log_to_file = log_to_file and (log_dir is not None)
        self.level = level
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        self.logger = logging.getLogger(name)

        if name not in Logger._configured_names or log_dir is not None:
            self._setup_logger()
            Logger._configured_names[name] = True

    def _setup_logger(self) -> None:
        """
        Configures log handlers: Console always, File only if log_dir is provided.

        Removes existing handlers before configuration to prevent duplicate output
        during reconfiguration. Creates a console handler for stdout and optionally
        adds a rotating file handler when log_dir is specified.

        Handler Configuration:
            - Console: Always enabled, outputs to sys.stdout
            - File: Enabled only when log_dir is provided, uses RotatingFileHandler
            - Format: "YYYY-MM-DD HH:MM:SS - LEVEL - message"
            - Rotation: Automatic when max_bytes threshold exceeded
        """
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )

        self.logger.setLevel(self.level)
        self.logger.propagate = False

        # Clean up existing handlers to prevent duplicates during reconfiguration
        if self.logger.hasHandlers():
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)

        # 1. Console Handler (Standard Output)
        console_h = logging.StreamHandler(sys.stdout)
        console_h.setFormatter(formatter)
        self.logger.addHandler(console_h)

        # 2. Rotating File Handler (Activated when log_dir is known)
        if self.log_to_file and self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = self.log_dir / f"{self.name}_{timestamp}.log"

            file_h = RotatingFileHandler(
                filename, maxBytes=self.max_bytes, backupCount=self.backup_count, encoding="utf-8"
            )
            file_h.setFormatter(formatter)
            self.logger.addHandler(file_h)

            Logger._active_log_file = filename

    def get_logger(self) -> logging.Logger:
        """
        Returns the configured logging.Logger instance.

        Returns:
            The underlying Python logging.Logger instance with configured handlers
        """
        return self.logger

    @classmethod
    def get_log_file(cls) -> Optional[Path]:
        """
        Returns the current active log file path for auditing.

        Returns:
            Path to the active log file, or None if file logging is not enabled
        """
        return cls._active_log_file

    @classmethod
    def setup(
        cls, name: str, log_dir: Optional[Path] = None, level: str = "INFO", **kwargs
    ) -> logging.Logger:
        """
        Main entry point for configuring the logger, called by RootOrchestrator.

        Bridges semantic LogLevel strings (INFO, DEBUG, WARNING) to Python logging
        constants. Provides convenient string-based level specification while internally
        using numeric logging constants.

        Args:
            name: Logger identifier (typically LOGGER_NAME constant)
            log_dir: Directory for log file storage (None = console-only mode)
            level: Logging level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            **kwargs: Additional arguments passed to Logger constructor

        Returns:
            Configured logging.Logger instance ready for use

        Environment Variables:
            DEBUG: If set to "1", overrides level to DEBUG regardless of level parameter

        Example:
            >>> logger = Logger.setup(
            ...     name="VisionForge",
            ...     log_dir=Path("./outputs/run_123/logs"),
            ...     level="INFO"
            ... )
            >>> logger.info("Training started")
        """
        if os.getenv("DEBUG") == "1":
            numeric_level = logging.DEBUG
        else:
            numeric_level = getattr(logging, level.upper(), logging.INFO)

        return cls(name=name, log_dir=log_dir, level=numeric_level, **kwargs).get_logger()


# GLOBAL INSTANCE
# Initial bootstrap instance (Console-only).
# Level is set to INFO by default, overridden by setup() during orchestration.
logger: Final[logging.Logger] = Logger().get_logger()
