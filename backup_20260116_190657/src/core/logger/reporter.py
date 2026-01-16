"""
Telemetry & Environment Reporting Engine.

This module provides the `Reporter` class, a specialized utility for formatting 
and emitting experiment metadata. It decouples the visual representation of 
the pipeline state from the core orchestration logic, ensuring a clean 
separation between process management and telemetry output.

The reporter handles:
    - Hardware capability visualization.
    - Dataset metadata resolution reporting.
    - Execution strategy and hyperparameter summaries.
    - Filesystem workspace mapping.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging
from typing import TYPE_CHECKING

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
from pydantic import BaseModel, ConfigDict
import torch

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from ..environment import (
    get_cuda_name, determine_tta_mode, get_vram_info
)

if TYPE_CHECKING:
    from core.config import Config
    from core.paths import RunPaths

# =========================================================================== #
#                             REPORTER DEFINITION                             #
# =========================================================================== #

class Reporter(BaseModel):
    """
    Centralized logging and reporting utility for experiment lifecycle events.
    
    This class transforms complex configuration states and hardware objects 
    into human-readable logs. It is designed to be called by the Orchestrator 
    during the initialization phase to provide a baseline report of the 
    execution environment.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def log_initial_status(
        self, 
        logger: logging.Logger, 
        cfg: "Config", 
        paths: "RunPaths", 
        device: "torch.device",
        applied_threads: int,
        num_workers: int
    ) -> None:
        """
        Logs the verified baseline environment configuration upon initialization.
        Uses formatted headers for visual consistency with the main pipeline.

        Args:
            logger: The active experiment logger instance.
            cfg: The validated global configuration manifest.
            paths: The dynamic path orchestrator for the current session.
            device: The resolved PyTorch compute device.
            applied_threads: The actual number of intra-op threads assigned.
            num_workers: The number of DataLoader workers.
        """
        header = (
            f"\n{'━' * 80}\n"
            f"{' ENVIRONMENT INITIALIZATION ':^80}\n"
            f"{'━' * 80}"
        )
        logger.info(header)
        
        self._log_hardware_section(logger, cfg, device, applied_threads, num_workers)
        logger.info("")
        
        self._log_dataset_section(logger, cfg)
        logger.info("")
        
        self._log_strategy_section(logger, cfg, device)
        logger.info("")
        
        logger.info(f"[FILESYSTEM]")
        logger.info(f"  » {'Run Root':<16}: {paths.root}")
        logger.info(f"{'━' * 80}\n")

    def _log_hardware_section(
        self, 
        logger: logging.Logger, 
        cfg: "Config", 
        device: "torch.device", 
        applied_threads: int,
        num_workers: int
    ) -> None:
        """
        Logs hardware-specific configuration, GPU metadata, and threading state.
        
        This method identifies the active compute device and provides transparency 
        on hardware utilization. It includes a fallback warning if the requested 
        accelerator (CUDA/MPS) is unavailable and the system reverted to CPU.
        """
        requested_device = cfg.hardware.device.lower()
        active_type = device.type
        
        logger.info(f"[HARDWARE]")
        logger.info(f"  » {'Active Device':<16}: {str(device).upper()}")
        
        if requested_device != "cpu" and active_type == "cpu":
            logger.warning(
                f"  [!] FALLBACK: Requested '{requested_device}' is unavailable. "
                f"Operating on CPU."
            )
        
        if active_type == 'cuda':
            logger.info(f"  » {'GPU Model':<16}: {get_cuda_name()}")
            logger.info(f"  » {'VRAM Available':<16}: {get_vram_info(device.index or 0)}")
        
        logger.info(f"  » {'DataLoader':<16}: {num_workers} workers")
        logger.info(f"  » {'Compute Fabric':<16}: {applied_threads} threads")
    
    def _log_dataset_section(self, logger: logging.Logger, cfg: "Config") -> None:
        """
        Logs dataset metadata, resolution, and anatomical characteristics.
        Leverages the DatasetMetadata schema for rich telemetry.
        """
        ds = cfg.dataset
        meta = ds.metadata
        
        logger.info(f"[DATASET]")
        logger.info(f"  » {'Name':<16}: {meta.display_name}")
        logger.info(f"  » {'Classes':<16}: {meta.num_classes} categories")
        logger.info(f"  » {'Resolution':<16}: {ds.img_size}px (Native: {meta.resolution_str})")
        logger.info(f"  » {'Channels':<16}: {meta.in_channels}")
        logger.info(f"  » {'Anatomical':<16}: {meta.is_anatomical}")
        logger.info(f"  » {'Texture-based':<16}: {meta.is_texture_based}")
    
    def _log_strategy_section(
        self,
        logger: logging.Logger,
        cfg: "Config",
        device: "torch.device"
    ) -> None:
        """Logs high-level training strategies, models, and hyperparameters."""
        train = cfg.training
        sys = cfg.hardware
        tta_status = determine_tta_mode(train.use_tta, device.type)
        
        repro_mode = "Bit-Perfect (Strict)" if sys.use_deterministic_algorithms else "Standard"
        seed_value = train.seed
        
        logger.info(f"[STRATEGY]")
        logger.info(f"  » {'Architecture':<16}: {cfg.model.name}")
        logger.info(f"  » {'Weights':<16}: {'Pretrained' if cfg.model.pretrained else 'Random'}")
        logger.info(f"  » {'Precision':<16}: {'AMP (Mixed)' if train.use_amp else 'FP32 (Full)'}")
        logger.info(f"  » {'TTA Mode':<16}: {tta_status}")
        
        logger.info(f"  » {'Repro. Mode':<16}: {repro_mode}")
        logger.info(f"  » {'Global Seed':<16}: {seed_value}")
        logger.info("")
        
        logger.info(f"[HYPERPARAMETERS]")
        logger.info(f"  » {'Epochs':<16}: {train.epochs}")
        logger.info(f"  » {'Batch Size':<16}: {train.batch_size}")
        
        lr = train.learning_rate
        lr_str = f"{lr:.2e}" if isinstance(lr, (float, int)) else str(lr)
        logger.info(f"  » {'Initial LR':<16}: {lr_str}")