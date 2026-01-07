"""
Infrastructure & Resource Lifecycle Management.

This module provides the operational bridge between the declarative configuration 
and the physical execution environment. It manages the 'clean-start' and 
'graceful-stop' sequences, ensuring that hardware resources are optimized 
and that concurrent experimental runs do not collide via filesystem-level locks.

Key Operational Tasks:
    * Process Sanitization: Guards against ghost processes and accidental 
      multi-process collisions in local environments.
    * Environment Locking: Implements a mutual exclusion (Mutex) strategy 
      to synchronize access to experimental outputs.
    * Resource De-allocation: Ensures GPU/MPS caches are flushed and temporary 
      system artifacts are purged upon exit.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import os
import logging
from typing import Optional, Any

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
    environment is "clean" before a run starts and "released" after it ends,
    preventing resource leakage.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True
    )

    def prepare_environment(self, cfg: Any, logger: Optional[logging.Logger] = None) -> None:
        """
        Coordinates the pre-execution sequence to ensure environment integrity.

        This includes:
            1. Terminating zombie or duplicate processes if permitted.
            2. Acquiring an advisory lock to prevent race conditions.

        Args:
            cfg: The global configuration manifest.
            logger (Optional[logging.Logger]): Active logger for status reporting.
        """
        # 1. Process Sanitization
        if cfg.system.allow_process_kill:
            # Prevent accidental termination in shared HPC/Cluster environments
            is_shared = any(env in os.environ for env in ["SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID"])
            if not is_shared:
                kill_duplicate_processes(logger=logger)
            elif logger:
                logger.debug(" » [SYS] Shared environment detected: skipping process kill.")

        # 2. Concurrency Guarding
        ensure_single_instance(
            lock_file=cfg.system.lock_file_path,
            logger=logger or logging.getLogger("Infrastructure")
        )

    def release_resources(self, cfg: Any, logger: Optional[logging.Logger] = None) -> None:
        """
        Handles the graceful release of system and hardware resources.

        Designed to be called during the orchestrator's cleanup phase to ensure 
        that lock files are unlinked and compute caches are cleared.

        Args:
            cfg: The global configuration manifest.
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
            try:
                torch.mps.empty_cache()
            except Exception as e:
                pass