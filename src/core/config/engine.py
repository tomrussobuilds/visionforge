"""
Vision Pipeline Configuration & Orchestration Engine.

This module acts as the declarative core of the pipeline, defining the 
hierarchical schema and validation logic required to drive the Orchestrator. 
It leverages Pydantic to transform raw inputs (CLI, YAML) into a structured, 
type-safe manifest that synchronizes hardware state with experiment logic.

Key Architectural Features:
    * Hierarchical Aggregation: Unifies specialized sub-configs (System, Dataset, 
      Model, Training, Evaluation, Augmentation) into a single immutable object.
    * Cross-Domain Validation: Implements complex logic checks (e.g., AMP vs. 
      Device, LR bounds, Mixup scheduling) that span multiple sub-modules.
    * Metadata-Driven Injection: Centralizes the resolution of registered dataset 
      specifications, ensuring architectural synchronization across the entire stack.
    * Factory Polymorphism: Provides dual entry points for instantiation via 
      structured YAML files or dynamic CLI arguments.

By enforcing strict validation during the initialization phase, the engine 
guarantees that the RootOrchestrator operates within a logically sound 
and reproducible execution context.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import argparse
from pathlib import Path
from typing import Any, Dict

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
from pydantic import (
    BaseModel, ConfigDict, Field, model_validator
)

# =========================================================================== #
#                               Internal Imports                              #
# =========================================================================== #
from .system_config import SystemConfig
from .training_config import TrainingConfig
from .augmentation_config import AugmentationConfig
from .dataset_config import DatasetConfig
from .evaluation_config import EvaluationConfig
from .models_config import ModelConfig

# TODO: Rename 'metadata' to 'registry' in the future to complete de-branding
from ..metadata import DATASET_REGISTRY
from ..io import load_config_from_yaml
from ..paths import PROJECT_ROOT

# =========================================================================== #
#                                MAIN CONFIGURATION                          #
# =========================================================================== #

class Config(BaseModel):
    """
    Main Experiment Manifest and Orchestration Schema.
    
    Aggregates specialized sub-configurations into a single validated object.
    It provides the blueprint for the RootOrchestrator to execute experiments.
    """
    model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            frozen=True
    )
    
    system: SystemConfig = Field(default_factory=SystemConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    def dump_portable(self) -> Dict[str, Any]:
        """
        Serializes the entire configuration with environment-agnostic paths.
        
        Converts absolute filesystem paths (system and dataset) into relative 
        anchors relative to the PROJECT_ROOT.
        
        Returns:
            Dict[str, Any]: A sanitized dictionary safe for cross-platform sharing.
        """
        full_data = self.model_dump()

        # Sanitize system paths
        full_data["system"] = self.system.to_portable_dict()    

        # Sanitize dataset root if applicable
        dataset_section = full_data.get("dataset", {})
        data_root = dataset_section.get("data_root")
        
        if data_root:
            dr_path = Path(data_root)
            if dr_path.is_relative_to(PROJECT_ROOT):
                relative_dr = dr_path.relative_to(PROJECT_ROOT)
                full_data["dataset"]["data_root"] = f"./{relative_dr}"
            
        return full_data    

    def dump_serialized(self) -> Dict[str, Any]:
        """
        Converts the config into a JSON-compatible dictionary.
        Essential for saving 'config.yaml' without serialization errors.
        """
        return self.model_dump(mode="json")
    
    @model_validator(mode="after")
    def validate_logic(self) -> "Config":
        """
        Cross-field logic validation after instantiation.
        
        This validator ensures hardware, model, and dataset parameters are 
        logically aligned. It supports 'Late Binding' by skipping checks 
        that depend on DatasetMetadata if they haven't been injected yet.
        """
        # Skip validation if metadata hasn't been injected (prevents AttributeError during YAML load)
        if self.dataset.metadata is None:
            return self
        
        # 1. Training Logic
        if self.training.mixup_epochs > self.training.epochs:
            raise ValueError(
                f"mixup_epochs ({self.training.mixup_epochs}) cannot exceed "
                f"total epochs ({self.training.epochs})."
            )
            
        # 2. Hardware vs Feature alignment
        if self.system.device == "cpu" and self.training.use_amp:
            raise ValueError("AMP cannot be enabled when using CPU device.")
            
        # 3. Model vs Dataset consistency (Late Bound validation)
        # These properties (in_channels) are resolved dynamically from metadata
        if self.model.pretrained and self.dataset.in_channels != 3:
            raise ValueError(
                f"Pretrained {self.model.name} requires 3-channel input (RGB). "
                f"Current dataset provides {self.dataset.in_channels} channels. "
                "Set 'force_rgb: true' in dataset config to fix this."
            )
            
        # 4. Optimizer bounds
        if self.training.min_lr >= self.training.learning_rate:
            raise ValueError(
                f"min_lr ({self.training.min_lr}) must be less than "
                f"initial learning_rate ({self.training.learning_rate})."
            )
            
        return self

    @property
    def run_slug(self) -> str:
        """Unique identifier for the experiment folder based on setup."""
        return f"{self.dataset.dataset_name}_{self.model.name}"
    
    @property
    def num_workers(self) -> int:
        """Proxies the effective number of workers from system policy."""
        return self.system.effective_num_workers
    
    @classmethod
    def _resolve_dataset_metadata(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """
        Internal helper to identify and fetch dataset specs from the registry.
        
        Args:
            args (argparse.Namespace): Parsed command line arguments.
            
        Returns:
            Dict[str, Any]: Metadata dictionary for the requested dataset.
        """
        ds_name_raw = getattr(args, 'dataset', None)
        if not ds_name_raw:
            raise ValueError(
                "Dataset name must be provided via --dataset or a config file."
            )
            
        ds_name = ds_name_raw.lower()
        if ds_name not in DATASET_REGISTRY:
             raise ValueError(f"Dataset '{ds_name_raw}' not found in registry.")
        
        return DATASET_REGISTRY[ds_name]
    
    @classmethod
    def from_yaml(cls, yaml_path: Path, metadata: Dict[str, Any]) -> "Config":
        """
        Factory method to create a fully 'hydrated' Config instance from a YAML file.
        
        Args:
            yaml_path: Path to the configuration file.
            metadata: The resolved dataset metadata to inject.
        """
        raw_data = load_config_from_yaml(yaml_path)
        
        # 1. Primary instantiation (metadata is None)
        cfg = cls(**raw_data)
        
        # 2. Metadata Injection (Hydration)
        object.__setattr__(cfg.dataset, 'metadata', metadata)
        
        # 3. Final Validation
        return cls.model_validate(cfg)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """
        Unified entry point for CLI-based instantiation.
        
        Leverages _resolve_dataset_metadata to ensure the dataset identity 
        is validated against the registry before any configuration is built.
        """
        # 1. Resolve metadata once (The Source of Truth)
        ds_meta = cls._resolve_dataset_metadata(args)

        # 2. Routing logic
        if getattr(args, 'config', None):
            # If a YAML is provided, hydrate it with the resolved metadata
            return cls.from_yaml(Path(args.config), metadata=ds_meta)
        
        # 3. Manual Assembly (CLI-only flow)
        return cls(
            system=SystemConfig.from_args(args),
            training=TrainingConfig.from_args(args),
            augmentation=AugmentationConfig.from_args(args),
            dataset=DatasetConfig.from_args(args, metadata=ds_meta),
            model=ModelConfig.from_args(args),
            evaluation=EvaluationConfig.from_args(args)
        )