"""
Optuna Study Orchestrator.

High-level coordinator for hyperparameter optimization studies.
Manages study creation, execution, visualization, and result export.

Key Responsibilities:
    * Study lifecycle management (create, load, optimize)
    * Sampler and pruner configuration
    * Progress tracking and logging
    * Visualization generation (importance plots, optimization history)
    * Best trial export to YAML config
"""
# =========================================================================== #
#                         Standard Imports                                    #
# =========================================================================== #
import logging

# =========================================================================== #
#                         Third-Party Imports                                 #
# =========================================================================== #
import optuna
from optuna.samplers import (
     TPESampler, CmaEsSampler, RandomSampler, GridSampler
)
from optuna.pruners import (
     MedianPruner, PercentilePruner, HyperbandPruner, NopPruner
)
# =========================================================================== #
#                         Internal Imports                                    #
# =========================================================================== #
from orchard.core import Config, LOGGER_NAME, save_config_as_yaml, RunPaths
from .search_spaces import get_search_space
from .objective import OptunaObjective
from .early_stopping import get_early_stopping_callback

logger = logging.getLogger(LOGGER_NAME)


# =========================================================================== #
#                         STUDY ORCHESTRATOR                                  #
# =========================================================================== #

class OptunaOrchestrator:
    """
    High-level manager for Optuna optimization studies.
    
    Coordinates study creation, execution, and artifact generation
    while integrating with existing Config and logging infrastructure.
    
    Attributes:
        cfg: Template configuration for trials
        device: Computation device
        paths: RunPaths instance
    """
    
    def __init__(self, cfg: Config, device, paths: RunPaths):
        """
        Initialize orchestrator.
        
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
        """
        Create or load Optuna study with configured sampler and pruner.
        
        Returns:
            Configured Optuna study instance
        """
        # Configure sampler
        sampler = self._get_sampler()
        
        # Configure pruner
        pruner = self._get_pruner()
        
        # Configure storage
        storage_url = self.cfg.optuna.get_storage_url(self.paths)
        
        logger.info(
            f"Creating study '{self.cfg.optuna.study_name}' with:\n"
            f"  Sampler: {self.cfg.optuna.sampler_type}\n"
            f"  Pruner: {self.cfg.optuna.pruner_type}\n"
            f"  Storage: {storage_url or 'in-memory'}"
        )
        
        study = optuna.create_study(
            study_name=self.cfg.optuna.study_name,
            direction=self.cfg.optuna.direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage_url,
            load_if_exists=self.cfg.optuna.load_if_exists
        )
        
        return study
    
    def optimize(self) -> optuna.Study:
        """
        Execute hyperparameter optimization.
        
        Returns:
            Completed study with trial results
        """
        # Create study
        study = self.create_study()
        
        # Load search space
        search_space = get_search_space(
            self.cfg.optuna.search_space_preset,
            resolution=self.cfg.dataset.resolution
        )
        
        # Create objective function
        objective = OptunaObjective(
            cfg=self.cfg,
            search_space=search_space,
            device=self.device,
            metric_name=self.cfg.optuna.metric_name,
            enable_pruning=self.cfg.optuna.enable_pruning,
            warmup_epochs=self.cfg.optuna.pruning_warmup_epochs
        )
        
        # Run optimization
        logger.info(
            f"\n{'=' * 80}\n"
            f"Starting Optuna optimization: {self.cfg.optuna.n_trials} trials\n"
            f"{'=' * 80}"
        )
        
        early_stop_callback = get_early_stopping_callback(
            metric_name=self.cfg.optuna.metric_name,
            direction=self.cfg.optuna.direction,
            threshold=self.cfg.optuna.early_stopping_threshold,
            patience=self.cfg.optuna.early_stopping_patience,
            enabled=self.cfg.optuna.enable_early_stopping
        )

        callbacks = [early_stop_callback] if early_stop_callback else []

        try:
            study.optimize(
                objective,
                n_trials=self.cfg.optuna.n_trials,
                timeout=self.cfg.optuna.timeout,
                n_jobs=self.cfg.optuna.n_jobs,
                show_progress_bar=self.cfg.optuna.show_progress_bar,
                callbacks=callbacks
            )
        except KeyboardInterrupt:
            logger.warning("Optimization interrupted by user. Saving partial results...")
        
        # Post-optimization processing
        self._log_study_summary(study)
        
        if self.cfg.optuna.save_plots:
            self._generate_visualizations(study)
        
        if self.cfg.optuna.save_best_config:
            self._export_best_config(study)
        
        return study
    
    def _get_sampler(self) -> optuna.samplers.BaseSampler:
        """
        Create sampler based on configuration.
        
        Returns:
            Configured Optuna sampler
        """
        sampler_map = {
            "tpe": TPESampler,
            "cmaes": CmaEsSampler,
            "random": RandomSampler,
            "grid": GridSampler,
        }
        
        sampler_cls = sampler_map.get(self.cfg.optuna.sampler_type)
        if sampler_cls is None:
            raise ValueError(
                f"Unknown sampler: {self.cfg.optuna.sampler_type}"
            )
        
        return sampler_cls()
    
    def _get_pruner(self) -> optuna.pruners.BasePruner:
        """
        Create pruner based on configuration.
        
        Returns:
            Configured Optuna pruner
        """
        if not self.cfg.optuna.enable_pruning:
            return NopPruner()
        
        pruner_map = {
            "median": MedianPruner,
            "percentile": lambda: PercentilePruner(percentile=25.0),
            "hyperband": HyperbandPruner,
            "none": NopPruner,
        }
        
        pruner_factory = pruner_map.get(self.cfg.optuna.pruner_type)
        if pruner_factory is None:
            raise ValueError(
                f"Unknown pruner: {self.cfg.optuna.pruner_type}"
            )
        
        return pruner_factory()
    
    def _log_study_summary(self, study: optuna.Study) -> None:
        """
        Log optimization results summary.
        
        Args:
            study: Completed Optuna study
        """
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        
        logger.info(
            f"\n{'=' * 80}\n"
            f"Optimization Complete\n"
            f"{'=' * 80}\n"
            f"  Total trials: {len(study.trials)}\n"
            f"  Completed: {len(completed)}\n"
            f"  Pruned: {len(pruned)}\n"
        )
        
        if completed:
            logger.info(
                f"  Best {self.cfg.optuna.metric_name}: {study.best_value:.4f}\n"
                f"  Best trial: {study.best_trial.number}\n"
                f"{'=' * 80}"
            )
            
            logger.info("\nBest Hyperparameters:")
            for param, value in study.best_params.items():
                logger.info(f"  {param}: {value}")
        else:
            logger.warning("No trials completed successfully")
    
    def _generate_visualizations(self, study: optuna.Study) -> None:
        """
        Generate and save Optuna visualization plots.
        
        Args:
            study: Completed Optuna study
        """
        # Skip if no completed trials
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            logger.warning("No completed trials. Skipping visualizations.")
            return
        
        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_slice,
                plot_parallel_coordinate
            )
            
            plots = {
                "optimization_history": plot_optimization_history,
                "param_importances": plot_param_importances,
                "slice": plot_slice,
                "parallel_coordinate": plot_parallel_coordinate,
            }
            
            for plot_name, plot_fn in plots.items():
                try:
                    fig = plot_fn(study)
                    output_path = self.paths.figures / f"{plot_name}.html"
                    fig.write_html(str(output_path))
                    logger.info(f"Saved {plot_name} to {output_path}")
                except Exception as e:
                    logger.warning(f"Failed to generate {plot_name}: {e}")
        
        except ImportError:
            logger.warning(
                "plotly not installed. Skipping visualization generation. "
                "Install with: pip install plotly"
            )
    
    def _export_best_config(self, study: optuna.Study) -> None:
        """
        Export best trial configuration as YAML.
        
        Creates a new Config with best hyperparameters and saves it
        for easy reproduction of optimal results.
        
        Args:
            study: Completed Optuna study
        """
        # Skip if no completed trials
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            logger.warning("No completed trials. Cannot export best config.")
            return
        
        # Build config dict with best parameters
        config_dict = self.cfg.model_dump()
        
        for param_name, value in study.best_params.items():
            # Map to nested config structure (same logic as objective)
            if param_name in ["learning_rate", "weight_decay", "momentum", "min_lr",
                              "mixup_alpha", "label_smoothing", "batch_size",
                              "cosine_fraction", "scheduler_patience"]:
                config_dict["training"][param_name] = value
            elif param_name == "dropout":
                config_dict["model"][param_name] = value
            elif param_name in ["rotation_angle", "jitter_val", "min_scale"]:
                config_dict["augmentation"][param_name] = value
        
        # Restore normal epochs for final training (not Optuna short epochs)
        config_dict["training"]["epochs"] = self.cfg.training.epochs
        
        # Create and validate new config
        best_config = Config(**config_dict)
        
        # Save to YAML
        output_path = self.paths.reports / "best_config.yaml"
        save_config_as_yaml(best_config, output_path)
        
        logger.info(f"Best configuration saved to {output_path}")
        logger.info(f"To use: python main.py --config {output_path}")


# =========================================================================== #
#                          CONVENIENCE FUNCTIONS                              #
# =========================================================================== #

def run_optimization(cfg: Config, device, paths: RunPaths) -> optuna.Study:
    """
    Convenience function to run complete optimization pipeline.
    
    Args:
        cfg: Base configuration template
        device: PyTorch device
        paths: RunPaths instance
        
    Returns:
        Completed Optuna study
    """
    orchestrator = OptunaOrchestrator(
        cfg=cfg,
        device=device,
        paths=paths
    )
    
    return orchestrator.optimize()