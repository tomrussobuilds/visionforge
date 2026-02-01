"""
OptunaOrchestrator Core Implementation.

Coordinates hyperparameter optimization lifecycle through Optuna integration.
Manages study creation, trial execution, and artifact generation while
delegating specialized tasks to focused submodules (builders, exporters, visualizers).

Architecture:
    - High-level coordination: OptunaOrchestrator manages study lifecycle
    - Delegation pattern: Specialized modules handle samplers, pruners, callbacks
    - Config integration: Seamless override of base Config per trial
    - Artifact generation: Automated visualization and results export

Key Components:
    OptunaOrchestrator: Study lifecycle manager
    run_optimization: Convenience function for full pipeline execution

Typical Usage:
    >>> from orchard.optimization.orchestrator import run_optimization
    >>> study = run_optimization(cfg=config, device=device, paths=paths)
    >>> print(f"Best trial: {study.best_trial.number}")
"""

import logging

import optuna

from orchard.core import (
    LOGGER_NAME,
    Config,
    LogStyle,
    RunPaths,
    log_optimization_header,
    log_study_summary,
)

from ..objective.objective import OptunaObjective
from ..search_spaces import get_search_space
from .builders import build_callbacks, build_pruner, build_sampler
from .exporters import export_best_config, export_study_summary, export_top_trials
from .utils import has_completed_trials
from .visualizers import generate_visualizations

logger = logging.getLogger(LOGGER_NAME)


# STUDY ORCHESTRATOR
class OptunaOrchestrator:
    """
    High-level manager for Optuna hyperparameter optimization studies.

    Coordinates the complete optimization lifecycle: study creation, trial execution,
    and post-processing artifact generation. Integrates with VisionForge's Config
    and RunPaths infrastructure, delegating specialized tasks (sampler/pruner building,
    visualization, export) to focused submodules.

    This orchestrator serves as the entry point for hyperparameter tuning, wrapping
    Optuna's API with VisionForge-specific configuration and output management.

    Attributes:
        cfg (Config): Template configuration that will be overridden per trial
        device (torch.device): Hardware target for training (CPU/CUDA/MPS)
        paths (RunPaths): Output directory structure for artifacts and results

    Example:
        >>> orchestrator = OptunaOrchestrator(cfg=config, device=device, paths=paths)
        >>> study = orchestrator.optimize()
        >>> print(f"Best AUC: {study.best_value:.3f}")
        >>> # Artifacts saved to paths.figures/ and paths.exports/
    """

    def __init__(self, cfg: Config, device, paths: RunPaths):
        """Initialize orchestrator.

        Args:
            cfg: Base Config to override per trial
            device: PyTorch device
            paths: Root directory for outputs
        """
        self.cfg = cfg
        self.device = device
        self.paths = paths

        logger.info(f"OptunaOrchestrator initialized. Output: {paths.root}")

    def create_study(self) -> optuna.Study:
        """Create or load Optuna study with configured sampler and pruner.

        Returns:
            Configured Optuna study instance
        """
        # Pass cfg to builders
        sampler = build_sampler(self.cfg.optuna.sampler_type, self.cfg)
        pruner = build_pruner(self.cfg.optuna.enable_pruning, self.cfg.optuna.pruner_type, self.cfg)
        storage_url = self.cfg.optuna.get_storage_url(self.paths)

        logger.info(LogStyle.DOUBLE)
        logger.info(f"Starting Optuna optimization: {self.cfg.optuna.n_trials} trials")
        logger.info(LogStyle.DOUBLE)

        study = optuna.create_study(
            study_name=self.cfg.optuna.study_name,
            direction=self.cfg.optuna.direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage_url,
            load_if_exists=self.cfg.optuna.load_if_exists,
        )

        return study

    def optimize(self) -> optuna.Study:
        """Execute hyperparameter optimization.

        Returns:
            Completed study with trial results
        """
        study = self.create_study()
        search_space = get_search_space(
            self.cfg.optuna.search_space_preset,
            resolution=self.cfg.dataset.resolution,
            include_models=self.cfg.optuna.enable_model_search,
        )

        objective = OptunaObjective(
            cfg=self.cfg,
            search_space=search_space,
            device=self.device,
        )

        # Configure logging and callbacks
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        log_optimization_header(self.cfg)

        callbacks = build_callbacks(self.cfg)

        try:
            study.optimize(
                objective,
                n_trials=self.cfg.optuna.n_trials,
                timeout=self.cfg.optuna.timeout,
                n_jobs=self.cfg.optuna.n_jobs,
                show_progress_bar=self.cfg.optuna.show_progress_bar,
                callbacks=callbacks,
            )
        except KeyboardInterrupt:
            logger.warning("Optimization interrupted by user. Saving partial results...")

        self._post_optimization_processing(study)

        return study

    def _post_optimization_processing(self, study: optuna.Study) -> None:
        """Execute all post-optimization tasks.

        Args:
            study: Completed Optuna study
        """
        log_study_summary(study, self.cfg.optuna.metric_name)

        if not has_completed_trials(study):
            logger.warning(
                "No completed trials. Skipping visualizations, best config, "
                "and detailed exports."
            )
            export_study_summary(study, self.paths, self.cfg.optuna.metric_name)
            return

        if self.cfg.optuna.save_plots:
            generate_visualizations(study, self.paths.figures)

        if self.cfg.optuna.save_best_config:
            export_best_config(study, self.cfg, self.paths)

        export_study_summary(study, self.paths, self.cfg.optuna.metric_name)
        export_top_trials(study, self.paths, self.cfg.optuna.metric_name)


def run_optimization(cfg: Config, device, paths: RunPaths) -> optuna.Study:
    """
    Convenience function to run complete optimization pipeline.

    Args:
        cfg: Global configuration with optuna section
        device: PyTorch device for training
        paths: RunPaths instance for output management

    Returns:
        Completed Optuna study with trial results

    Example:
        >>> study = run_optimization(cfg=config, device="cuda", paths=paths)
        >>> print(f"Best AUC: {study.best_value:.3f}")
    """
    orchestrator = OptunaOrchestrator(cfg=cfg, device=device, paths=paths)
    return orchestrator.optimize()
