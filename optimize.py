"""
Hyperparameter Optimization Entry Point for VisionForge.

Orchestrates Optuna-driven hyperparameter search with complete lifecycle
management: study creation, trial execution, pruning, visualization,
and best configuration export.

Workflow:
    1. Configuration parsing (CLI/YAML)
    2. Environment initialization via RootOrchestrator
    3. Optuna study creation with sampler/pruner
    4. Trial execution with early stopping and pruning
    5. Visualization generation (plots, reports)
    6. Best configuration export for final training

Usage:
    # Optimize with YAML recipe (recommended)
    python optimize.py --config recipes/optuna_resnet_18_adapted.yaml
    
    # Quick search with custom parameters
    python optimize.py --dataset pathmnist --n_trials 20 --epochs 10
    
    # Resume interrupted study
    python optimize.py --config recipes/optuna_vit_tiny.yaml --load_if_exists true

Key Features:
    - TPE/CMA-ES/Random/Grid sampler support
    - Median/Percentile/Hyperband pruning
    - Early stopping at target thresholds
    - Architecture and weight variant search
    - SQLite persistence for resumability
    - Interactive HTML visualizations
"""

# =========================================================================== #
#                            INTERNAL IMPORTS                                 #
# =========================================================================== #
from orchard.core import (
    RootOrchestrator, Config, parse_args, LogStyle
)
from orchard.optimization import run_optimization

# =========================================================================== #
#                           MAIN EXECUTION                                    #
# =========================================================================== #

def main() -> None:
    """
    Main orchestrator for hyperparameter optimization execution.
    
    Coordinates the complete optimization lifecycle from configuration parsing
    to final result reporting. Utilizes RootOrchestrator context manager for
    resource safety and automatic cleanup.
    
    Workflow:
        1. Parse CLI arguments (YAML config or direct flags)
        2. Build unified Config with Pydantic validation
        3. Initialize orchestrator (device, filesystem, logging)
        4. Execute Optuna study with trial pruning
        5. Generate importance plots and export best config
        6. Save study summary and top trials
        7. Report final statistics and next steps
    
    Raises:
        KeyboardInterrupt: User interrupted optimization (graceful cleanup)
        Exception: Any fatal error during optimization (logged and re-raised)
    """
    # Parse CLI arguments (supports both YAML and direct flags)
    args = parse_args()
    
    # Build configuration (triggers Pydantic validation)
    cfg = Config.from_args(args)
    
    # Use orchestrator context manager for resource safety
    # Guarantees cleanup even if optimization crashes
    with RootOrchestrator(cfg) as orchestrator:

        # Access synchronized services provided by orchestrator
        paths  = orchestrator.paths
        logger = orchestrator.run_logger
        device = orchestrator.get_device()
                
        try:
            # ================================================================ #
            #                  HYPERPARAMETER OPTIMIZATION                     #
            # ================================================================ #
            
            # Execute Optuna study with trial pruning and early stopping
            study = run_optimization(
                cfg    = cfg,
                device = device,
                paths  = paths
            )
            
            # ================================================================ #
            #                     OPTIMIZATION SUMMARY                         #
            # ================================================================ #
            completed = [t for t in study.trials if t.state.name == 'COMPLETE']
            pruned    = [t for t in study.trials if t.state.name == 'PRUNED']
            failed    = [t for t in study.trials if t.state.name == 'FAIL']
            
            logger.info(f"\n{LogStyle.DOUBLE}")
            logger.info(f"{'OPTIMIZATION EXECUTION SUMMARY':^80}")
            logger.info(LogStyle.DOUBLE)
            logger.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Dataset        : {cfg.dataset.dataset_name}")
            logger.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Search Space   : {cfg.optuna.search_space_preset}")
            logger.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Total Trials   : {len(study.trials)}")
            logger.info(f"{LogStyle.INDENT}{LogStyle.SUCCESS} Completed      : {len(completed)}")
            logger.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Pruned         : {len(pruned)}")
            if failed:
                logger.info(f"{LogStyle.INDENT}{LogStyle.WARNING} Failed         : {len(failed)}")
            
            # Only show best trial if there are completed trials
            if completed:
                try:
                    logger.info(f"{LogStyle.INDENT}{LogStyle.SUCCESS} Best {cfg.optuna.metric_name.upper():<9} : {study.best_value:.6f}")
                    logger.info(f"{LogStyle.INDENT}{LogStyle.SUCCESS} Best Trial     : {study.best_trial.number}")
                except ValueError:
                    # Edge case: completed trials exist but best_trial lookup fails
                    logger.warning(f"{LogStyle.INDENT}{LogStyle.WARNING} Best trial lookup failed (check study integrity)")
            else:
                logger.warning(f"{LogStyle.INDENT}{LogStyle.WARNING} No trials completed")
            
            logger.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Device         : {orchestrator.get_device()}")
            logger.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Artifacts      : {paths.root}")
            logger.info(f"{LogStyle.DOUBLE}\n")

        except KeyboardInterrupt:
            logger.warning(f"\n{LogStyle.WARNING} Interrupted by user. Optimization stopped gracefully.")
            # No re-raise - graceful exit

        except Exception as e:
            logger.error(f"\n{LogStyle.WARNING} Pipeline crashed: {e}", exc_info=True)
            raise

        finally:
            # Context manager handles automatic cleanup
            if 'paths' in locals() and paths:
                logger.info(f"Pipeline shutdown complete. Run directory: {paths.root}")

# =========================================================================== #
#                           ENTRY POINT                                       #
# =========================================================================== #

if __name__ == "__main__":
    main()