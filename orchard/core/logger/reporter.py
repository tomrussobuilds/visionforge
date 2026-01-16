"""
Telemetry & Environment Reporting Engine.

Provides formatted logging utilities for RootOrchestrator and OptunaOrchestrator.
Centralizes all heavy logging logic for consistent, readable output.

The reporter handles:
    - Hardware capability visualization
    - Dataset metadata resolution reporting
    - Execution strategy and hyperparameter summaries
    - Optimization trial formatting
    - Study completion summaries
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging
from typing import TYPE_CHECKING, Any, Dict

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
from ..paths import RunPaths, LOGGER_NAME

if TYPE_CHECKING:
    from ..config import Config
    import optuna

logger = logging.getLogger(LOGGER_NAME)


# =========================================================================== #
#                        EXPERIMENT INITIALIZATION LOGGING                    #
# =========================================================================== #

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
        num_workers: int
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
        header = (
            f"\n{'━' * 80}\n"
            f"{' ENVIRONMENT INITIALIZATION ':^80}\n"
            f"{'━' * 80}"
        )
        logger_instance.info(header)
        
        self._log_hardware_section(logger_instance, cfg, device, applied_threads, num_workers)
        logger_instance.info("")
        
        self._log_dataset_section(logger_instance, cfg)
        logger_instance.info("")
        
        self._log_strategy_section(logger_instance, cfg, device)
        logger_instance.info("")
        
        logger_instance.info(f"[HYPERPARAMETERS]")
        logger_instance.info(f"  » {'Epochs':<16}: {cfg.training.epochs}")
        logger_instance.info(f"  » {'Batch Size':<16}: {cfg.training.batch_size}")
        lr = cfg.training.learning_rate
        lr_str = f"{lr:.2e}" if isinstance(lr, (float, int)) else str(lr)
        logger_instance.info(f"  » {'Initial LR':<16}: {lr_str}")
        logger_instance.info("")
        
        logger_instance.info(f"[FILESYSTEM]")
        logger_instance.info(f"  » {'Run Root':<16}: {paths.root}")
        logger_instance.info(f"{'━' * 80}\n")

    def _log_hardware_section(
        self, 
        logger_instance: logging.Logger, 
        cfg: "Config", 
        device: "torch.device", 
        applied_threads: int,
        num_workers: int
    ) -> None:
        """Logs hardware-specific configuration and GPU metadata."""
        requested_device = cfg.hardware.device.lower()
        active_type = device.type
        
        logger_instance.info(f"[HARDWARE]")
        logger_instance.info(f"  » {'Active Device':<16}: {str(device).upper()}")
        
        if requested_device != "cpu" and active_type == "cpu":
            logger_instance.warning(
                f"  [!] FALLBACK: Requested '{requested_device}' is unavailable. "
                f"Operating on CPU."
            )
        
        if active_type == 'cuda':
            logger_instance.info(f"  » {'GPU Model':<16}: {get_cuda_name()}")
            logger_instance.info(f"  » {'VRAM Available':<16}: {get_vram_info(device.index or 0)}")
        
        logger_instance.info(f"  » {'DataLoader':<16}: {num_workers} workers")
        logger_instance.info(f"  » {'Compute Fabric':<16}: {applied_threads} threads")
    
    def _log_dataset_section(self, logger_instance: logging.Logger, cfg: "Config") -> None:
        """Logs dataset metadata and characteristics."""
        ds = cfg.dataset
        meta = ds.metadata
        
        logger_instance.info(f"[DATASET]")
        logger_instance.info(f"  » {'Name':<16}: {meta.display_name}")
        logger_instance.info(f"  » {'Classes':<16}: {meta.num_classes} categories")
        logger_instance.info(f"  » {'Resolution':<16}: {ds.img_size}px (Native: {meta.resolution_str})")
        logger_instance.info(f"  » {'Channels':<16}: {meta.in_channels}")
        logger_instance.info(f"  » {'Anatomical':<16}: {meta.is_anatomical}")
        logger_instance.info(f"  » {'Texture-based':<16}: {meta.is_texture_based}")
    
    def _log_strategy_section(
        self,
        logger_instance: logging.Logger,
        cfg: "Config",
        device: "torch.device"
    ) -> None:
        """Logs high-level training strategies and models."""
        train = cfg.training
        sys = cfg.hardware
        tta_status = determine_tta_mode(train.use_tta, device.type)
        
        repro_mode = "Strict" if sys.use_deterministic_algorithms else "Standard"
        
        logger_instance.info(f"[STRATEGY]")
        logger_instance.info(f"  » {'Architecture':<16}: {cfg.model.name}")
        logger_instance.info(f"  » {'Weights':<16}: {'Pretrained' if cfg.model.pretrained else 'Random'}")
        logger_instance.info(f"  » {'Precision':<16}: {'AMP (Mixed)' if train.use_amp else 'FP32'}")
        logger_instance.info(f"  » {'TTA Mode':<16}: {tta_status}")
        logger_instance.info(f"  » {'Repro. Mode':<16}: {repro_mode}")
        logger_instance.info(f"  » {'Global Seed':<16}: {train.seed}")


# =========================================================================== #
#                        OPTIMIZATION LOGGING                                 #
# =========================================================================== #

def log_optimization_header(cfg: "Config") -> None:
    """
    Log Optuna optimization session header.
    
    Args:
        cfg: Configuration with optuna settings
    """
    logger.info("")
    logger.info("#" * 84)
    logger.info("                   OPTUNA HYPERPARAMETER OPTIMIZATION")
    logger.info("#" * 84)
    logger.info(f"  Dataset: {cfg.dataset.dataset_name}")
    logger.info(f"  Model: {cfg.model.name}")
    logger.info(f"  Search Space: {cfg.optuna.search_space_preset}")
    logger.info(f"  Trials: {cfg.optuna.n_trials}")
    logger.info(f"  Epochs per Trial: {cfg.optuna.epochs}")
    logger.info(f"  Metric: {cfg.optuna.metric_name}")
    logger.info(f"  Device: {cfg.hardware.device}")
    logger.info(f"  Pruning: {'Enabled' if cfg.optuna.enable_pruning else 'Disabled'}")
    
    if cfg.optuna.enable_early_stopping:
        threshold = cfg.optuna.early_stopping_threshold or "auto"
        logger.info(f"  Early Stopping: Enabled (threshold={threshold}, patience={cfg.optuna.early_stopping_patience})")
    
    logger.info("#" * 84)


def log_trial_start(trial_number: int, params: Dict[str, Any]) -> None:
    """
    Log trial start with formatted parameters (grouped by category).
    
    Args:
        trial_number: Trial index
        params: Sampled hyperparameters
    """
    logger.info(f"Trial {trial_number} hyperparameters:")
    
    categories = [
        ("Optimization", ["learning_rate", "weight_decay", "momentum", "min_lr"]),
        ("Regularization", ["mixup_alpha", "label_smoothing", "dropout"]),
        ("Scheduling", ["cosine_fraction", "scheduler_patience", "batch_size"]),
        ("Augmentation", ["rotation_angle", "jitter_val", "min_scale"])
    ]
    
    for category_name, param_list in categories:
        category_params = {k: v for k, v in params.items() if k in param_list}
        if category_params:
            logger.info(f"  [{category_name}]")
            for key, value in category_params.items():
                if isinstance(value, float):
                    if value < 0.001:
                        logger.info(f"    • {key:<20} : {value:.2e}")
                    else:
                        logger.info(f"    • {key:<20} : {value:.4f}")
                else:
                    logger.info(f"    • {key:<20} : {value}")


def log_study_summary(study: "optuna.Study", metric_name: str) -> None:
    """
    Log optimization study completion summary.
    
    Args:
        study: Completed Optuna study
        metric_name: Name of optimization metric
    """
    import optuna
    
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    
    logger.info("")
    logger.info("=" * 84)
    logger.info("                         OPTIMIZATION COMPLETE")
    logger.info("=" * 84)
    logger.info(f"  Total Trials      : {len(study.trials)}")
    logger.info(f"  Completed         : {len(completed)}")
    logger.info(f"  Pruned            : {len(pruned)}")
    if failed:
        logger.info(f"  Failed            : {len(failed)}")
    logger.info("")
    
    if completed:
        logger.info(f"  Best {metric_name:<13} : {study.best_value:.6f}")
        logger.info(f"  Best Trial        : {study.best_trial.number}")
        logger.info("=" * 84)
        logger.info("")
        
        logger.info("Best Hyperparameters:")
        log_trial_start(study.best_trial.number, study.best_params)
    else:
        logger.warning("  No trials completed successfully")
        logger.info("=" * 84)
    
    logger.info("")


def log_best_config_export(config_path: Any) -> None:
    """
    Log best configuration export information.
    
    Args:
        config_path: Path to exported YAML config
    """
    logger.info("")
    logger.info("#" * 84)
    logger.info("                         BEST CONFIGURATION EXPORTED")
    logger.info("#" * 84)
    logger.info(f"  Configuration saved to: {config_path}")
    logger.info("")
    logger.info("  To train with optimized hyperparameters:")
    logger.info(f"    python main.py --config {config_path}")
    logger.info("")
    logger.info("  To visualize optimization results:")
    logger.info(f"    firefox {config_path.parent.parent}/figures/param_importances.html")
    logger.info("#" * 84)
    logger.info("")