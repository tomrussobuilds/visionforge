"""
Core Environment Orchestrator

This module centralizes the initialization of the system environment, 
coupling static configuration with runtime safety and logging services 
to provide a consistent state for the experiment.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging
from pathlib import Path

# =========================================================================== #
#                             Third-Party Imports                             #
# =========================================================================== #
import torch

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .system import (
    set_seed, ensure_single_instance, get_cuda_name,
    to_device_obj, load_model_weights, configure_system_libraries,
    release_single_instance, apply_cpu_threads, determine_tta_mode
)
from .io import save_config_as_yaml
from .logger import Logger
from .paths import RunPaths, setup_static_directories, LOGGER_NAME

# =========================================================================== #
#                              Root Orchestrator RootOrchestrator             #
# =========================================================================== #

class RootOrchestrator:
    """
    High-level lifecycle controller for the experiment environment.
    
    The RootOrchestrator serves as the Single Source of Truth (SSOT) for the 
    transition between static configuration (Pydantic models) and the live 
    execution state. It enforces system-level constraints, manages directory 
    atomicity, and synchronizes telemetry (logging) across the pipeline.

    Responsibilities:
        - Enforcement of execution exclusivity (Kernel-level locking).
        - Management of run-specific filesystem hierarchies.
        - Global RNG synchronization for bit-perfect reproducibility.
        - Hardware abstraction and stateful weight restoration.
    """
    
    def __init__(self, cfg):
        """
        Initializes the orchestrator with the experiment configuration.

        Args:
            cfg (Config): The validated global configuration manifest.
        """
        self.cfg = cfg
        self.paths = None
        self.run_logger = None
        self._device_cache = None
    
    def __enter__(self) -> "RootOrchestrator":
        """
        Context Manager entry point. 
        Automatically triggers the core service initialization sequence.

        Returns:
            RootOrchestrator: The initialized instance ready for pipeline execution.
        """
        self.initialize_core_services()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Context Manager exit point.
        Ensures that system resources are released and lock files are unlinked
        regardless of whether the pipeline succeeded or raised an exception.

        Args:
            exc_type: The type of the exception raised (if any).
            exc_val: The instance of the exception raised (if any).
            exc_tb: The traceback of the exception raised (if any).

        Returns:
            bool: Always False to allow exception propagation to the caller.
        """
        self.cleanup()
        return False

    def initialize_core_services(self) -> RunPaths:
        """
        Triggers the deterministic sequence of core service initializations.

        This method synchronizes the environment following a strict order of 
        operations:
            1. System Libraries: Configures matplotlib for headless operation.
            2. RNG Seeding: Locks global state for reproducibility.
            3. Static Layout: Ensures baseline project structure.
            4. Path Mapping: Generates unique session-specific workspace.
            5. Telemetry: Hot-swaps logger to file-persistent handlers.
            6. Guarding: Acquires system locks and purges zombie processes.
            7. Persistence: Saves the validated configuration for reference.

        Returns:
            RunPaths: The verified path orchestrator for the current session.
        """
        # 1. Configure system libraries for the current environment
        configure_system_libraries()

        # 2. Reproducibility setup: Lock global random state
        set_seed(self.cfg.training.seed)

        # 3. Static environment setup: Prepare global folder structure
        setup_static_directories()

        # 4. Dynamic path initialization: Create run-specific folder
        self.paths = RunPaths(
            dataset_slug=self.cfg.dataset.dataset_name,
            model_name=self.cfg.model_name,
            base_dir=self.cfg.system.output_dir
        )

        # 5. Logger initialization: Start file and console logging
        Logger.setup(
            name=LOGGER_NAME,
            log_dir=self.paths.logs
        )
        self.run_logger = logging.getLogger(LOGGER_NAME)

        # 6. Environment initialization and safety: Lock instance and clean zombies
        self.cfg.system.manage_environment()

        ensure_single_instance(
            lock_file=self.cfg.system.lock_file_path,
            logger=self.run_logger
        )
        
        # 7. Metadata preservation: Save validated config as SSOT reference
        save_config_as_yaml(
            data=self.cfg.model_dump(mode='json'),
            yaml_path=self.paths.get_config_path()
        )
        
        self._log_initial_status()
        
        return self.paths

    def cleanup(self) -> None:
        """
        Releases system resources and removes the lock file.
        To be called in the 'finally' block of main.py.
        """
        # Release the system lock to allow future instances
        try:
            release_single_instance(self.cfg.system.lock_file_path)
            if self.run_logger:
                self.run_logger.info("System lock released cleanly.")
            else:
                logging.info("System lock released cleanly.")
        except Exception as e:
            logging.error(f"Error releasing system lock: {e}")


    def get_device(self) -> torch.device:
        """
        Resolves and caches the optimal computation device based on configuration.
        
        Returns:
            torch.device: The PyTorch device object for model execution.
        """
        if self._device_cache is None:
            self._device_cache = to_device_obj(
                device_str=self.cfg.system.device
            )
        return self._device_cache

    def load_weights(self, model: torch.nn.Module, path: Path) -> None:
        """
        Coordinates atomic weight loading and logs the event.

        Args:
            model (torch.nn.Module): The model instance to populate.
            path (Path): Filesystem path to the checkpoint file.
        """
        device = self.get_device()
        load_model_weights(model, path, device)
        self.run_logger.info(
            f"Checkpoint weights successfully restored from: {path.name}"
        )


    def _log_initial_status(self) -> None:
        """
        Logs the verified baseline environment configuration upon initialization.
        Organizzato per sezioni: Hardware, Dataset, Training.
        """
        device_obj = self.get_device()
        device_str_requested = self.cfg.system.device
        
        self.run_logger.info(f"--- Environment Status Report ---")
        
        # 1. Hardware & System
        self.run_logger.info(f"Execution Device: {str(device_obj).upper()}")
        if device_str_requested != "cpu" and device_obj.type == "cpu":
            self.run_logger.warning(f"HARDWARE FALLBACK: Requested {device_str_requested} is unavailable.")
        
        if device_obj.type == 'cuda':
            gpu_name = get_cuda_name()
            if gpu_name: self.run_logger.info(f"GPU Model: {gpu_name}")
        elif device_obj.type == 'cpu':
            opt_threads = apply_cpu_threads(self.cfg.num_workers)
            self.run_logger.info(f"CPU Threads: {opt_threads} (Workers: {self.cfg.num_workers})")

        # 2. Dataset & Domain - High Fidelity Logic
        if self.cfg.dataset.effective_in_channels == 3:
            mode_str = "NATIVE-RGB" if self.cfg.dataset.in_channels == 3 else "RGB-PROMOTED"
        else:
            mode_str = "NATIVE-GRAY"

        self.run_logger.info(f"Dataset: {self.cfg.dataset.dataset_name} ({self.cfg.dataset.img_size}px)")
        self.run_logger.info(f"Data Mode: {mode_str}")
        self.run_logger.info(f"Anatomical Constraints: {self.cfg.dataset.is_anatomical}")
        self.run_logger.info(f"Texture-Based Logic: {self.cfg.dataset.is_texture_based}")

        # 3. Training Strategy
        tta_status = determine_tta_mode(self.cfg.training.use_tta, device_obj.type)
        self.run_logger.info(f"TTA Status: {tta_status}")
        self.run_logger.info(
            f"Hyperparameters: Epochs={self.cfg.training.epochs}, "
            f"Batch={self.cfg.training.batch_size}, LR={self.cfg.training.learning_rate:.4f}"
        )
        
        self.run_logger.info(f"Run Directory: {self.paths.root}")
        self.run_logger.info(f"---------------------------------")