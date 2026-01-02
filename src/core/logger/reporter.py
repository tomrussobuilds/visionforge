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

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from ..environment import (
    get_cuda_name, determine_tta_mode
)

if TYPE_CHECKING:
    from core.config import Config
    from core.paths import RunPaths
    import torch

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
        applied_threads: int
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
        """
        header = (
            f"\n{'━' * 80}\n"
            f"{' ENVIRONMENT INITIALIZATION ':^80}\n"
            f"{'━' * 80}"
        )
        logger.info(header)
        
        self._log_hardware_section(logger, cfg, device, applied_threads)
        logger.info("")
        
        self._log_dataset_section(logger, cfg)
        logger.info("")
        
        self._log_strategy_section(logger, cfg, device)
        logger.info("")
        
        logger.info(f"[FILESYSTEM]")
        logger.info(f"  » Run Root:     {paths.root}")
        logger.info(f"{'━' * 80}\n")

    def _log_hardware_section(
        self, 
        logger: logging.Logger, 
        cfg: "Config", 
        device: "torch.device", 
        applied_threads: int
    ) -> None:
        """
        Logs hardware-specific configuration, GPU metadata, and threading state.
        
        This method identifies the active compute device and provides transparency 
        on hardware utilization. It includes a fallback warning if the requested 
        accelerator (CUDA/MPS) is unavailable and the system reverted to CPU.

        Args:
            logger: The active experiment logger instance.
            cfg: The validated global configuration manifest.
            device: The resolved PyTorch compute device.
            applied_threads: The actual number of intra-op threads assigned.
        """
        requested_device = cfg.system.device.lower()
        active_type = device.type
        
        logger.info(f"[HARDWARE]")
        logger.info(f"  » Active Device:  {str(device).upper()}")
        
        if requested_device != "cpu" and active_type == "cpu":
            logger.warning(
                f"  [!] FALLBACK: Requested '{requested_device}' is unavailable. "
                f"Operating on CPU."
            )
        
        if active_type == 'cuda':
            gpu_name = get_cuda_name()
            if gpu_name: 
                logger.info(f"  » GPU Model:      {gpu_name}")
        
        num_workers = getattr(cfg.training, "num_workers", 0)
        logger.info(f"  » DataLoader:     {num_workers} workers")
        logger.info(f"  » Compute Fabric: {applied_threads} threads")

    def _log_dataset_section(self, logger: logging.Logger, cfg: "Config") -> None:
        """
        Logs dataset metadata, resolution, and anatomical characteristics.

        Args:
            logger: The active experiment logger instance.
            cfg: The validated global configuration manifest.
        """
        ds = cfg.dataset
        logger.info(f"[DATASET]")
        logger.info(f"  » Name:         {ds.dataset_name}")
        logger.info(f"  » Resolution:   {ds.img_size}px")
        logger.info(f"  » Mode:         {ds.processing_mode}")
        logger.info(f"  » Anatomical:   {ds.is_anatomical}")
        logger.info(f"  » Texture:      {ds.is_texture_based}")

    def _log_strategy_section(
        self,
        logger: logging.Logger,
        cfg: "Config",
        device: "torch.device"
    ) -> None:
        """
        Logs high-level training strategies, models, and hyperparameters.

        Args:
            logger: The active experiment logger instance.
            cfg: The validated global configuration manifest.
            device: The resolved PyTorch compute device.
        """
        train = cfg.training
        tta_status = determine_tta_mode(train.use_tta, device.type)
        
        repro_status = "ENABLED (Bit-Perfect)" if getattr(train, "reproducible", False) else "DISABLED (Standard)"
        
        logger.info(f"[STRATEGY]")
        logger.info(f"  » Model:        {cfg.model.name}")
        logger.info(f"  » Pretrained:   {cfg.model.pretrained}")
        logger.info(f"  » TTA Mode:     {tta_status}")
        logger.info(f"  » Reproduce:    {repro_status}")
        
        logger.info(f"[HYPERPARAMETERS]")
        logger.info(f"  » Epochs:       {train.epochs}")
        logger.info(f"  » Batch Size:   {train.batch_size}")
        logger.info(f"  » Learn Rate:   {train.learning_rate:.2e}")