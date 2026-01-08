"""
Hardware Acceleration & Computing Environment.

This module provides high-level abstractions for hardware discovery (CUDA/MPS),
and compute resource optimization. It manages the detection of available 
accelerators and synchronizes PyTorch threading with system capabilities.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import os
import platform
import logging

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
import matplotlib

# =========================================================================== #
#                               System Utilities                              #
# =========================================================================== #

def configure_system_libraries() -> None:
    """
    Configures third-party libraries for headless environments.
    Sets Matplotlib to 'Agg' backend on Linux/Docker to avoid GUI issues.
    Also sets logging level for Matplotlib to WARNING to reduce verbosity.
    """
    is_linux = platform.system() == "Linux"
    is_docker = any([
        os.environ.get("IN_DOCKER") == "TRUE",
        os.path.exists("/.dockerenv")
    ])
    
    if is_linux or is_docker:
        matplotlib.use("Agg")  
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        
    if platform.system() == "Windows":
        logging.debug("Windows environment detected: fcntl locking is unavailable.")

# =========================================================================== #
#                              Hardware Utilities                             #
# =========================================================================== #

def get_num_workers() -> int:
    """
    Determines optimal DataLoader workers with a safe cap for RAM stability.

    Returns:
        int: Recommended number of subprocesses for data loading.
    """
    total_cores = os.cpu_count() or 2
    if total_cores <= 4:
        return 2
    return min(total_cores // 2, 8)

def apply_cpu_threads(num_workers: int) -> int:
    """
    Calculates and sets optimal compute threads to avoid resource contention.
    Synchronizes PyTorch intra-op parallelism with OMP/MKL environment variables.
    
    Args:
        num_workers (int): Number of active DataLoader workers.

    Returns:
        int: The number of threads applied to the system.
    """
    total_cores = os.cpu_count() or 1
    optimal_threads = max(2, total_cores - num_workers)
    
    torch.set_num_threads(optimal_threads)
    os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
    os.environ["MKL_NUM_THREADS"] = str(optimal_threads)

    return optimal_threads

def detect_best_device() -> str:
    """
    Detects the most performant hardware accelerator available (CUDA > MPS > CPU).
    
    Returns:
        str: The best available device string.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_cuda_name() -> str:
    """
    Returns the human-readable name of the primary GPU device.
    
    Returns:
        str: GPU model name or empty string if unavailable.
    """
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""

def get_vram_info(device_idx: int = 0) -> str:
    """
    Safely retrieves and formats VRAM availability for a specific CUDA device.
    
    Args:
        device_idx (int): The index of the GPU to query.
        
    Returns:
        str: Formatted string 'Free / Total GB' or error/status message.
    """
    if not torch.cuda.is_available():
        return "N/A"
    
    try:
        if device_idx >= torch.cuda.device_count():
            return "Invalid Device Index"
            
        free, total = torch.cuda.mem_get_info(device_idx)
        return f"{free / 1024**3:.2f} GB / {total / 1024**3:.2f} GB"
    except Exception as e:
        logging.debug(f"VRAM query failed: {e}")
        return "Unknown (Query failed)"

def to_device_obj(device_str: str) -> torch.device:
    """
    Converts a device string into a live torch.device object.
    
    Args:
        device_str (str): Target device ('cuda', 'cpu', 'mps').

    Returns:
        torch.device: The active computing device object.
    """
    return torch.device(device_str)
