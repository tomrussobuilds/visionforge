"""
Infrastructure & Resource Lifecycle Management.

This module defines the `InfrastructureManager`, an operational handler responsible 
for enforcing environment stability. It mediates between the high-level 
orchestrator and low-level OS utilities, handling process sanitization, 
concurrency guarding (file locking), and hardware resource cleanup.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import os
import logging
from typing import Optional

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
from pydantic import BaseModel, ConfigDict

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from ..environment import (
    ensure_single_instance, 
    release_single_instance, 
    kill_duplicate_processes
)

# =========================================================================== #
#                             INFRASTRUCTURE MANAGER                          #
# =========================================================================== #

class InfrastructureManager(BaseModel):
    """
    Operational executor for environment safeguarding and resource management.
    
    The InfrastructureManager offloads system-level tasks from the configuration 
    schemas and the central orchestrator. It ensures that the execution 
    environment is "clean" before a run starts and "released" after it ends.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True
    )

    def prepare_environment(self, cfg, logger: Optional[logging.Logger] = None) -> None:
        """
        Coordinates the pre-execution sequence to ensure environment integrity.

        This includes:
            1. Terminating zombie or duplicate processes if permitted.
            2. Acquiring an advisory lock to prevent race conditions.

        Args:
            cfg (Config): The global configuration manifest.
            logger (Optional[logging.Logger]): Active logger for status reporting.
        """
        # 1. Process Sanitization
        if cfg.system.allow_process_kill:
            # Prevent accidental termination in shared HPC/Cluster environments
            is_shared = any(env in os.environ for env in ["SLURM_JOB_ID", "PBS_JOBID"])
            if not is_shared:
                kill_duplicate_processes(logger=logger)
            elif logger:
                logger.debug("Skipping process kill: Shared cluster environment detected.")

        # 2. Concurrency Guarding
        ensure_single_instance(
            lock_file=cfg.system.lock_file_path,
            logger=logger or logging.getLogger("Infrastructure")
        )

    def release_resources(self, cfg, logger: Optional[logging.Logger] = None) -> None:
        """
        Handles the graceful release of system and hardware resources.

        Designed to be called during the orchestrator's cleanup phase to ensure 
        that lock files are unlinked and compute caches are cleared.

        Args:
            cfg (Config): The global configuration manifest.
            logger (Optional[logging.Logger]): Active logger for status reporting.
        """
        # 1. Release Filesystem Lock
        try:
            release_single_instance(cfg.system.lock_file_path)
        
            msg = "System resource lock released successfully."
            if logger:
                logger.info(f" » {msg}")
            else:
                logging.debug(msg)
        except Exception as e:
            if logger:
                logger.warning(f"Failed to release lock file: {e}")

        # 2. Hardware Memory Cleanup
        self._flush_compute_cache()

    def _flush_compute_cache(self) -> None:
        """
        Clears volatile memory buffers for supported hardware backends.
        Prevents memory fragmentation across consecutive experimental runs.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()