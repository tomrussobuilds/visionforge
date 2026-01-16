"""
Environment Orchestration & Lifecycle Management.

Provides RootOrchestrator, the central authority for initializing and managing 
experiment execution state. Synchronizes hardware (CUDA/CPU), filesystem (RunPaths), 
and telemetry (Logging) into a unified, reproducible context.

Key Responsibilities:
    - Deterministic seeding: Global RNG state locking for standard and 
      bit-perfect reproducibility
    - Resource guarding: Single-instance locking preventing race conditions 
      via InfrastructureManager
    - Path atomicity: Dynamic experiment workspace generation and validation
    - Hardware abstraction: Device-specific optimizations including compute 
      thread synchronization and DataLoader worker scaling
    - Lifecycle safety: Context Manager pattern guaranteeing resource cleanup 
      and state persistence during failures
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging
from typing import Optional, TYPE_CHECKING

# =========================================================================== #
#                             Third-Party Imports                             #
# =========================================================================== #
import torch

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .environment import (
    set_seed, to_device_obj, configure_system_libraries, apply_cpu_threads
)
from .config.infrastructure_config import InfrastructureManager
from .io import save_config_as_yaml
from .logger import Logger, Reporter
from .paths import RunPaths, setup_static_directories, LOGGER_NAME

if TYPE_CHECKING:
    from .config.engine import Config


# =========================================================================== #
#                            Root Orchestrator                                #
# =========================================================================== #

class RootOrchestrator:
    """
    High-level lifecycle controller for experiment environment.
    
    Central state machine coordinating transition from static configuration 
    (Pydantic-validated) to live execution state. Synchronizes hardware discovery, 
    filesystem provisioning, and telemetry initialization.

    Context Manager pattern guarantees system-level resources (kernel locks, 
    telemetry handlers) are safely acquired before execution and released 
    upon termination, ensuring atomicity during failures.

    Attributes:
        cfg: Validated global configuration manifest (SSOT)
        infra: Handler for OS-level resource guarding
        reporter: Engine for environment telemetry
        paths: Session-specific directory orchestrator
        run_logger: Active logger instance for session
        repro_mode: Bit-perfect determinism flag
        num_workers: Resolved DataLoader workers from policy
    """
    
    def __init__(self, cfg: "Config", log_initializer=Logger.setup) -> None:
        """
        Initializes orchestrator and binds execution policy.

        Args:
            cfg: Validated global configuration (SSOT)
            log_initializer: Strategy function for logging handlers
                (accepts name, log_dir, level)
        """
        self.cfg = cfg
        self.infra = InfrastructureManager()
        self.reporter = Reporter()
        self._log_initializer = log_initializer
        
        self.paths: Optional[RunPaths] = None
        self.run_logger: Optional[logging.Logger] = None
        self._device_cache: Optional[torch.device] = None
        
        # Policy extraction from SSOT
        self.repro_mode = self.cfg.hardware.use_deterministic_algorithms
        self.num_workers = self.cfg.hardware.effective_num_workers
    
    def __enter__(self) -> "RootOrchestrator":
        """
        Context Manager entry - triggers core service initialization.

        Returns:
            Initialized RootOrchestrator ready for pipeline execution
        """
        try:
            self.initialize_core_services()
            return self
        except Exception as e:
            self.cleanup()
            raise e

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Context Manager exit - ensures resource release and lock cleanup.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception instance if raised
            exc_tb: Exception traceback if raised

        Returns:
            False to allow exception propagation
        """
        self.cleanup()
        return False
     
    # --- Private Lifecycle Phases ---

    def _phase_1_determinism(self) -> None:
        """Enforces global RNG seeding and algorithmic determinism."""
        set_seed(self.cfg.training.seed, strict=self.repro_mode)

    def _phase_2_hardware_optimization(self) -> int:
        """
        Configures compute thread affinity and accelerator libraries.
        
        Returns:
            Number of CPU threads applied to runtime
        """
        applied_threads = apply_cpu_threads(self.num_workers)
        configure_system_libraries()
        return applied_threads
    
    def _phase_3_filesystem_provisioning(self) -> None:
        """
        Constructs experiment workspace via RunPaths.
        Anchors relative paths to validated PROJECT_ROOT.
        """
        setup_static_directories()
        self.paths = RunPaths.create(
            dataset_slug=self.cfg.dataset.dataset_name,
            model_name=self.cfg.model.name,
            training_cfg=self.cfg.dump_serialized(),
            base_dir=self.cfg.telemetry.output_dir
        )

    def _phase_4_logging_initialization(self) -> None:
        """
        Bridges static Logger to session-specific filesystem.
        Reconfigures handlers for file-based persistence in run directory.
        """
        self.run_logger = self._log_initializer(
            name=LOGGER_NAME,
            log_dir=self.paths.logs,
            level=self.cfg.telemetry.log_level
        )

    def _phase_5_config_persistence(self) -> None:
        """
        Mirrors hydrated configuration to experiment root.
        Saves portable YAML manifest for session auditability.
        """
        save_config_as_yaml(
            data=self.cfg,
            yaml_path=self.paths.get_config_path()
        )

    def _phase_6_infrastructure_guarding(self) -> None:
        """
        Secures system-level resource locks via InfrastructureManager.
        Prevents concurrent execution conflicts and manages cleanup.
        """
        self.infra.prepare_environment(self.cfg, logger=self.run_logger)

    def _phase_7_environment_reporting(self, applied_threads: int) -> None:
        """
        Emits baseline environment report to active logging streams.
        Summarizes hardware, dataset metadata, and execution policies.
        """
        self.reporter.log_initial_status(
            logger=self.run_logger,
            cfg=self.cfg,
            paths=self.paths,
            device=self.get_device(),
            applied_threads=applied_threads,
            num_workers=self.num_workers
        )

    # --- Public Interface ---

    def initialize_core_services(self) -> RunPaths:
        """
        Executes linear sequence of environment initialization phases.

        Synchronizes global state through 7 phases, progressing from 
        deterministic seeding to full environment reporting.

        Returns:
            Verified and provisioned directory structure
        """
        self._phase_1_determinism()
        applied_threads = self._phase_2_hardware_optimization()
        self._phase_3_filesystem_provisioning()
        self._phase_4_logging_initialization()
        self._phase_5_config_persistence()
        self._phase_6_infrastructure_guarding()
        self._phase_7_environment_reporting(applied_threads)
        
        return self.paths

    def cleanup(self) -> None:
        """
        Releases system resources and removes execution lock file.
        
        Guarantees clean state for subsequent runs by unlinking 
        InfrastructureManager guards and closing logging handlers.
        """
        try:
            self.infra.release_resources(self.cfg, logger=self.run_logger)
        except Exception as e:
            err_msg = f"Failed to release system lock: {e}"
            if self.run_logger:
                for handler in self.run_logger.handlers[:]:
                    handler.close()
                    self.run_logger.removeHandler(handler)
            else:
                logging.error(err_msg)

    def get_device(self) -> torch.device:
        """
        Resolves and caches optimal computation device (CUDA/CPU/MPS).
        
        Returns:
            PyTorch device object for model execution
        """
        if self._device_cache is None:
            self._device_cache = to_device_obj(device_str=self.cfg.hardware.device)
        return self._device_cache