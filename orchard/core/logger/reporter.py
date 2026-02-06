"""
Environment Reporter.

Provides formatted logging for experiment initialization and environment configuration.
Transforms complex configuration states and hardware objects into human-readable logs.
"""

import logging
from typing import TYPE_CHECKING

import torch
from pydantic import BaseModel, ConfigDict

from ..environment import determine_tta_mode, get_cuda_name, get_vram_info
from .styles import LogStyle

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Config
    from ..paths import RunPaths


class Reporter(BaseModel):
    """
    Centralized logging and reporting utility for experiment lifecycle events.

    Transforms complex configuration states and hardware objects into
    human-readable logs. Called by Orchestrator during initialization.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def log_initial_status(
        self,
        logger_instance: logging.Logger,
        cfg: "Config",
        paths: "RunPaths",
        device: "torch.device",
        applied_threads: int,
        num_workers: int,
    ) -> None:
        """
        Logs verified baseline environment configuration upon initialization.

        Args:
            logger_instance: Active experiment logger
            cfg: Validated global configuration manifest
            paths: Dynamic path orchestrator for current session
            device: Resolved PyTorch compute device
            applied_threads: Number of intra-op threads assigned
            num_workers: Number of DataLoader workers
        """
        # Newline + Header Block
        logger_instance.info("")
        logger_instance.info(LogStyle.HEAVY)
        logger_instance.info(f"{'ENVIRONMENT INITIALIZATION':^80}")
        logger_instance.info(LogStyle.HEAVY)

        # Hardware Section
        self._log_hardware_section(logger_instance, cfg, device, applied_threads, num_workers)
        logger_instance.info("")

        # Dataset Section
        self._log_dataset_section(logger_instance, cfg)
        logger_instance.info("")

        # Strategy Section
        self._log_strategy_section(logger_instance, cfg, device)
        logger_instance.info("")

        # Hyperparameters Section
        logger_instance.info("[HYPERPARAMETERS]")
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Epochs':<18}: {cfg.training.epochs}"
        )
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Batch Size':<18}: {cfg.training.batch_size}"
        )
        lr = cfg.training.learning_rate
        lr_str = f"{lr:.2e}" if isinstance(lr, (float, int)) else str(lr)
        logger_instance.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {'Initial LR':<18}: {lr_str}")
        logger_instance.info("")

        # Filesystem Section
        logger_instance.info("[FILESYSTEM]")
        logger_instance.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {'Run Root':<18}: {paths.root}")

        # Closing separator
        logger_instance.info(LogStyle.HEAVY)
        logger_instance.info("")

    def _log_hardware_section(
        self,
        logger_instance: logging.Logger,
        cfg: "Config",
        device: "torch.device",
        applied_threads: int,
        num_workers: int,
    ) -> None:
        """Logs hardware-specific configuration and GPU metadata."""
        requested_device = cfg.hardware.device.lower()
        active_type = device.type

        logger_instance.info("[HARDWARE]")
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Active Device':<18}: {str(device).upper()}"
        )

        if requested_device != "cpu" and active_type == "cpu":
            logger_instance.warning(
                f"{LogStyle.INDENT}{LogStyle.WARNING} "
                f"FALLBACK: Requested '{requested_device}' unavailable, using CPU"
            )

        if active_type == "cuda":
            logger_instance.info(
                f"{LogStyle.INDENT}{LogStyle.ARROW} {'GPU Model':<18}: {get_cuda_name()}"
            )
            logger_instance.info(
                f"{LogStyle.INDENT}{LogStyle.ARROW} "
                f"{'VRAM Available':<18}: {get_vram_info(device.index or 0)}"
            )

        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'DataLoader':<18}: {num_workers} workers"
        )
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Compute Threads':<18}: {applied_threads} threads"
        )

    def _log_dataset_section(self, logger_instance: logging.Logger, cfg: "Config") -> None:
        """Logs dataset metadata and characteristics."""
        ds = cfg.dataset
        meta = ds.metadata

        logger_instance.info("[DATASET]")
        logger_instance.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {'Name':<18}: {meta.display_name}")
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Classes':<18}: {meta.num_classes} categories"
        )
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} "
            f"{'Resolution':<18}: {ds.img_size}px (Native: {meta.resolution_str})"
        )
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Channels':<18}: {meta.in_channels}"
        )
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Anatomical':<18}: {meta.is_anatomical}"
        )
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Texture-based':<18}: {meta.is_texture_based}"
        )

    def _log_strategy_section(
        self, logger_instance: logging.Logger, cfg: "Config", device: "torch.device"
    ) -> None:
        """Logs high-level training strategies and models."""
        train = cfg.training
        sys = cfg.hardware
        tta_status = determine_tta_mode(train.use_tta, device.type)

        repro_mode = "Strict" if sys.use_deterministic_algorithms else "Standard"

        logger_instance.info("[STRATEGY]")
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Architecture':<18}: {cfg.model.name}"
        )
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} "
            f"{'Weights':<18}: {'Pretrained' if cfg.model.pretrained else 'Random'}"
        )

        # Add weight variant if present (for ViT)
        if cfg.model.weight_variant:
            logger_instance.info(
                f"{LogStyle.INDENT}{LogStyle.ARROW} "
                f"{'Weight Variant':<18}: {cfg.model.weight_variant}"
            )

        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} "
            f"{'Precision':<18}: {'AMP (Mixed)' if train.use_amp else 'FP32'}"
        )
        logger_instance.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {'TTA Mode':<18}: {tta_status}")
        logger_instance.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {'Repro. Mode':<18}: {repro_mode}")
        logger_instance.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {'Global Seed':<18}: {train.seed}")
