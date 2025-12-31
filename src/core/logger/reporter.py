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
    get_cuda_name, apply_cpu_threads, determine_tta_mode
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
        device: "torch.device"
    ) -> None:
        """
        Logs the verified baseline environment configuration upon initialization.
        Uses formatted headers for visual consistency with the main pipeline.

        Args:
            logger: The active experiment logger.
            cfg: The validated global configuration manifest.
            paths: The dynamic path orchestrator for the current session.
            device: The resolved PyTorch compute device.
        """
        header = (
            f"\n{'━' * 80}\n"
            f"{' ENVIRONMENT INITIALIZATION ':^80}\n"
            f"{'━' * 80}"
        )
        logger.info(header)
        
        self._log_hardware_section(logger, cfg, device)
        logger.info("")
        
        self._log_dataset_section(logger, cfg)
        logger.info("")
        
        self._log_strategy_section(logger, cfg, device)
        
        logger.info(f"[FILESYSTEM]")
        logger.info(f"  » Run Root:     {paths.root}")

    def _log_hardware_section(self, logger, cfg, device):
        """Logs hardware-specific configuration and fallback warnings."""
        req_dev = cfg.system.device
        
        logger.info(f"[HARDWARE]")
        logger.info(f"  » Device:       {str(device).upper()}")
        
        if req_dev != "cpu" and device.type == "cpu":
            logger.warning(f"  [!] FALLBACK: Requested {req_dev} is unavailable.")
        
        if device.type == 'cuda':
            gpu_name = get_cuda_name()
            if gpu_name: 
                logger.info(f"  » GPU:          {gpu_name}")
        
        elif device.type == 'cpu':
            opt_threads = apply_cpu_threads(cfg.num_workers)
            logger.info(f"  » Workers:      {cfg.num_workers}")
            logger.info(f"  » CPU Threads:  {opt_threads}")

    def _log_dataset_section(self, logger, cfg):
        """Logs dataset metadata and channel processing modes."""
        ds = cfg.dataset

        logger.info(f"[DATASET]")
        logger.info(f"  » Name:         {ds.dataset_name}")
        logger.info(f"  » Resolution:   {ds.img_size}px")
        logger.info(f"  » Mode:         {ds.processing_mode}")
        logger.info(f"  » Anatomical:   {ds.is_anatomical}")
        logger.info(f"  » Texture:      {ds.is_texture_based}")

    def _log_strategy_section(self, logger, cfg, device):
        """Logs high-level training and augmentation strategies."""
        train = cfg.training
        tta_status = determine_tta_mode(train.use_tta, device.type)
        
        logger.info(f"[STRATEGY]")
        logger.info(f"  » Model:        {cfg.model.name}")
        logger.info(f"  » Pretrained:   {cfg.model.pretrained}")
        logger.info(f"  » TTA Mode:     {tta_status}")
        
        logger.info(f"[HYPERPARAMETERS]")
        logger.info(f"  » Epochs:       {train.epochs}")
        logger.info(f"  » Batch Size:   {train.batch_size}")
        logger.info(f"  » Learn Rate:   {train.learning_rate:.2e}")