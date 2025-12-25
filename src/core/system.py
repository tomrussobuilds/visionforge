"""
System and Hardware Utilities Module

This module provides low-level abstractions for hardware acceleration (CUDA/MPS),
environment-aware reproducibility (seeding, worker allocation), and OS-level 
process management (exclusive locking, duplicate termination).
"""
# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
import os
import fcntl
import sys
import random
import time
import logging
from pathlib import Path
from typing import Optional

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import numpy as np
import torch
import psutil

# =========================================================================== #
#                               SYSTEM UTILITIES
# =========================================================================== #

# Global variable to hold the lock file descriptor and prevent GC cleanup
_lock_fd: Optional[int] = None

def set_seed(seed: int) -> None:
    """
    Sets random seeds for reproducibility across NumPy, Python, and PyTorch.
    Ensures deterministic behavior even for CUDNN convolution algorithms.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Force deterministic algorithms (potentially slower but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_num_workers() -> int:
    """
    Determines optimal DataLoader workers based on reproducibility settings.
    Returns 0 (single-thread) if DOCKER_REPRODUCIBILITY_MODE is enabled.
    """
    is_repro = os.environ.get("DOCKER_REPRODUCIBILITY_MODE", "0").upper() in ("1", "TRUE")
    return 0 if is_repro else 4


def detect_best_device() -> str:
    """
    Detects the most performant hardware accelerator available.
    Order: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    
    # Check for Apple Silicon support
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def get_cuda_name() -> str:
    """Returns the human-readable GPU name or empty string if unavailable."""
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""


def load_model_weights(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    """Low-level utility to restore model state from a checkpoint."""
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True)
    )


def to_device_obj(device_str: str) -> torch.device:
    """Converts a device string ('cuda', 'cpu', 'mps') into a torch.device object."""
    return torch.device(device_str)


def kill_duplicate_processes(
        logger: logging.Logger,
        script_name: Optional[str] = None
    ) -> None:
    """
    Scans for and terminates other Python instances running the same script
    to prevent resource contention or log corruption.
    """
    if script_name is None:
        script_name = os.path.basename(sys.argv[0])
    
    current_pid = os.getpid()
    killed = 0
    python_names = {'python', 'python3', 'python.exe', 'python3.exe'}

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            pinfo = proc.info
            # Match python executables and skip our own process
            if pinfo['name'] not in python_names or pinfo['pid'] == current_pid:
                continue
            
            cmdline = pinfo['cmdline']
            # Check if the script name is in the command line arguments
            if cmdline and any(script_name in arg for arg in cmdline):
                proc.terminate()
                killed += 1
                logger.warning(f"Terminated duplicate process: PID {pinfo['pid']}")
        
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    if killed:
        logger.info(f"Cleaned up {killed} duplicate process(es). Cooling down...")
        time.sleep(1.5)


def ensure_single_instance(
        lock_file: Path,
        logger: logging.Logger
    ) -> None:
    """
    Uses flock (Unix) to ensure only one instance of the script runs.
    The lock is released only when the process exits.
    """
    global _lock_fd
    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        # Open for writing and keep fd alive globally
        f = open(lock_file, 'w')
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fd = f 
        logger.info("Exclusive system lock acquired.")
    except (IOError, BlockingIOError):
        logger.error("CRITICAL: Another instance is already running. Aborting to prevent conflicts.")
        sys.exit(1)