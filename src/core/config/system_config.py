"""
System & Infrastructure Configuration Schema.

This module defines the declarative schema for hardware abstraction and 
filesystem orchestration. It acts as the bridge between raw environment 
settings (CLI/YAML) and the physical resources of the machine, providing the 
Single Source of Truth (SSOT) for device selection and path integrity.

Key Functionalities:
    * Hardware Negotiation: Resolves 'auto' device requests into concrete 
      accelerators (CUDA/MPS) based on runtime availability and self-corrects 
      if requested resources are missing.
    * Path Sanitization: Enforces absolute path resolution for datasets and 
      experiment roots, delegating physical directory creation to the 
      RootOrchestrator to ensure side-effect-free validation.
    * Execution Policy: Calculates dynamic parameters like 'effective_num_workers' 
      and 'deterministic_mode' by reconciling hardware capabilities with 
      reproducibility constraints.
    * Metadata Blueprint: Provides the necessary parameters (lock files, slugs) 
      for the InfrastructureManager to perform environment sanitization and 
      mutual exclusion guarding.

By centralizing these definitions, the engine ensures that the experiment 
state is defined in a predictable, hardware-aligned, and immutable manifest.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import argparse
import tempfile
from pathlib import Path

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
from pydantic import (
    BaseModel, Field, field_validator, ConfigDict
)

# =========================================================================== #
#                               Internal Imports                              #
# =========================================================================== #
from .types import (
    ValidatedPath, ProjectSlug, LogFrequency, LogLevel
)
from ..environment import (
    detect_best_device, get_num_workers
)
from ..paths import DATASET_DIR, OUTPUTS_ROOT

# =========================================================================== #
#                             SYSTEM CONFIGURATION                            #
# =================================================================指標
# =========================================================================== #

class SystemConfig(BaseModel):
    """
    Declarative manifest for infrastructure and hardware abstraction.
    
    This class serves as the SSOT (Single Source of Truth) for the pipeline's 
    physical requirements. It validates that requested compute devices are 
    accessible and defines the filesystem boundaries for the experiment.
    """
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        json_encoders={
            Path: lambda v: str(v)
        }
    )

    # Hardware Configuration
    device: str = Field(
        default_factory=detect_best_device,
        description="Computing device (cpu, cuda, mps or auto)."
    )
    log_level: LogLevel = "INFO"

    # Filesystem Strategy
    data_dir: ValidatedPath = Field(default=DATASET_DIR)
    output_dir: ValidatedPath = Field(default=OUTPUTS_ROOT)

    # Execution Policy
    save_model: bool = True
    log_interval: LogFrequency = Field(default=10)
    project_name: ProjectSlug = "vision_experiment"
    allow_process_kill: bool = Field(
        default=True,
        description="Permission flag for InfrastructureManager to terminate duplicate processes."
    )

    # Internal state for policy resolution
    _reproducible_mode: bool = False

    @property
    def lock_file_path(self) -> Path:
        """
        Dynamically generates the cross-platform lock file location.
        Used by InfrastructureManager to ensure mutual exclusion during execution.
        """
        safe_name = self.project_name.replace("/", "_") 
        return Path(tempfile.gettempdir()) / f"{safe_name}.lock"
    
    @property
    def support_amp(self) -> bool:
        """Determines if the current validated device supports Automatic Mixed Precision (AMP)."""
        return self.device.lower().startswith("cuda") or \
                self.device.lower().startswith("mps")
    
    @property
    def effective_num_workers(self) -> int:
        """
        Calculates the optimal number of DataLoader workers. 
        Returns 0 if reproducibility is required to avoid non-deterministic 
        multiprocessing behavior, otherwise returns the system-detected maximum.
        """
        if self._reproducible_mode:
            return 0
        return get_num_workers()
    
    @property
    def use_deterministic_algorithms(self) -> bool:
        """Flag indicating whether PyTorch should enforce bit-perfect deterministic algorithms."""
        return self._reproducible_mode
        
    @field_validator("device")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        """
        SSOT Validation: Ensures the requested device actually exists on this system.
        If the requested accelerator (cuda/mps) is unavailable, it self-corrects to 'cpu'.
        """
        if v == "auto":
            return detect_best_device()
        
        requested = v.lower()
        if "cuda" in requested and not torch.cuda.is_available():
            return "cpu"
        if "mps" in requested and not torch.backends.mps.is_available():
            return "cpu"
        return requested

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SystemConfig":
        """
        Factory method to map CLI arguments to the SystemConfig schema.
        
        Initializes environmental policies such as reproducibility based on 
        the provided namespace.
        """
        instance = cls(
            device=getattr(args, 'device', "auto"),
            data_dir=Path(getattr(args, 'data_dir', DATASET_DIR)),
            output_dir=Path(getattr(args, 'output_dir', OUTPUTS_ROOT)),
            save_model=getattr(args, 'save_model', True),
            log_interval=getattr(args, 'log_interval', 10),
            project_name=getattr(args, 'project_name', "vision_experiment")
        )
        
        # Set the internal policy flag based on CLI 'reproducible' argument
        object.__setattr__(instance, '_reproducible_mode', getattr(args, 'reproducible', False))
        
        return instance