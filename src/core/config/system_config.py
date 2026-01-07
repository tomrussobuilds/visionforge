"""
Hardware & Filesystem Manifest.

This module defines the declarative schema for hardware abstraction and 
filesystem orchestration. It acts as the bridge between raw environment 
settings (CLI/YAML) and the physical resources of the machine, providing the 
Single Source of Truth (SSOT) for device selection and path integrity.

Key Functionalities:
    * Hardware Negotiation: Resolves 'auto' device requests into concrete 
      accelerators (CUDA/MPS) based on runtime availability.
    * Path Sanitization: Enforces absolute path resolution for datasets and 
      experiment roots, ensuring side-effect-free validation.
    * Execution Policy: Calculates dynamic parameters like 'effective_num_workers' 
      and 'deterministic_mode' by reconciling hardware capabilities.
    * Resource Identity: Provides the necessary parameters (lock files, slugs) 
      for process synchronization and environment mutual exclusion.

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
from ..paths import PROJECT_ROOT

# =========================================================================== #
#                             SYSTEM CONFIGURATION                            #
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
    data_dir: ValidatedPath = Field(default="./dataset")
    output_dir: ValidatedPath = Field(default="./outputs")

    # Execution Policy
    save_model: bool = True
    log_interval: LogFrequency = Field(default=10)
    project_name: ProjectSlug = "vision_experiment"
    allow_process_kill: bool = Field(
        default=True,
        description="Permission flag to terminate duplicate processes for environment cleanup."
    )

    # Internal state for policy resolution
    _reproducible_mode: bool = False

    def to_portable_dict(self) -> dict:
        """
        Converts the configuration instance into a portable dictionary.
        
        This method reconciles absolute system paths back to project-relative 
        paths (e.g., converting '/home/user/project/dataset' to './dataset'). 
        It ensures that exported YAML/JSON manifests are environment-agnostic 
        and do not leak local filesystem structures into logs or repositories.
        
        Returns:
            dict: A sanitized dictionary representation of the system config.
        """
        data = self.model_dump()
        
        path_fields = ["data_dir", "output_dir"]
        
        for field in path_fields:
            full_path = Path(data[field])
            if full_path.is_relative_to(PROJECT_ROOT):
                relative_path = full_path.relative_to(PROJECT_ROOT)
                data[field] = f"./{relative_path}"
            else:
                data[field] = str(full_path)

        return data
    
    @property
    def lock_file_path(self) -> Path:
        """
        Dynamically generates a cross-platform lock file location.
        
        This path is used for environment sanitization and to prevent 
        resource contention during concurrent experiment execution.
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
    
    @field_validator("data_dir", "output_dir", mode="before")
    @classmethod
    def resolve_relative_paths(cls, v):
        """
        Ensures paths are always anchored to the PROJECT_ROOT.
        If 'v' is already absolute, it's kept as is (allowing external mounts).
        """
        path = Path(v)
        if not path.is_absolute():
            return (PROJECT_ROOT / path).resolve()
        return path.resolve()
        
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
        
        Dynamically extracts parameters from the command-line interface by 
        inspecting available schema fields, ensuring CLI inputs override 
        defaults without manual key mapping.
        """
        # Get all field names defined in the Pydantic model
        schema_fields = cls.model_fields.keys()
        
        # Filter namespace for existing fields that are not None
        params = {
            k: getattr(args, k) 
            for k in schema_fields 
            if hasattr(args, k) and getattr(args, k) is not None
        }

        instance = cls(**params)
        
        # Internal state injection for private policies
        repro_flag = getattr(args, 'reproducible', False)
        object.__setattr__(instance, '_reproducible_mode', repro_flag)
        
        return instance