"""
Vision Pipeline Configuration & Orchestration Engine.

Declarative core defining the hierarchical schema and validation logic 
for the pipeline. Transforms raw inputs (CLI, YAML) into a structured, 
type-safe manifest synchronizing hardware state with experiment logic.

Key Features:
    * Hierarchical aggregation: Unifies specialized sub-configs (Hardware, 
      Dataset, Model, Training, Evaluation, Augmentation) into single immutable object
    * Cross-domain validation: Complex logic checks (AMP vs Device, LR bounds, 
      Mixup scheduling) spanning multiple sub-modules
    * Metadata-driven injection: Centralizes dataset specification resolution, 
      ensuring architectural synchronization
    * Factory polymorphism: Dual entry points via YAML files or CLI arguments

Strict validation during initialization guarantees logically sound and 
reproducible execution context for the RootOrchestrator.
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
from pydantic import BaseModel, ConfigDict, Field, model_validator

# =========================================================================== #
#                               Internal Imports                              #
# =========================================================================== #
from .hardware_config import HardwareConfig
from .telemetry_config import TelemetryConfig
from .training_config import TrainingConfig
from .augmentation_config import AugmentationConfig
from .dataset_config import DatasetConfig
from .evaluation_config import EvaluationConfig
from .models_config import ModelConfig
from ..metadata.wrapper import DatasetRegistryWrapper
from ..io import load_config_from_yaml
from ..paths import PROJECT_ROOT


# =========================================================================== #
#                            Main Configuration                               #
# =========================================================================== #

class Config(BaseModel):
    """
    Main experiment manifest aggregating specialized sub-configurations.
    
    Provides the validated blueprint for RootOrchestrator to execute experiments.
    """
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        frozen=True
    )
    
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    @model_validator(mode="after")
    def validate_logic(self) -> "Config":
        """
        Cross-field logic validation after instantiation.
        
        Validates:
        - ResNet adapted requires 28x28 resolution
        - Mixup epochs within total epochs
        - AMP compatibility with device
        - Pretrained models require 3-channel input
        - Learning rate bounds
        """
        # 1. Model-specific resolution constraints
        if "resnet_18_adapted" in self.model.name.lower() and self.dataset.resolution != 28:
            raise ValueError(
                f"resnet_18_adapted requires resolution=28, got {self.dataset.resolution}"
            )

        # Skip remaining validations if metadata not injected
        if self.dataset.metadata is None:
            return self

        # 2. Training logic
        if self.training.mixup_epochs > self.training.epochs:
            raise ValueError(
                f"mixup_epochs ({self.training.mixup_epochs}) exceeds "
                f"total epochs ({self.training.epochs})"
            )

        # 3. Hardware-feature alignment
        if self.hardware.device == "cpu" and self.training.use_amp:
            raise ValueError("AMP requires GPU (CUDA/MPS), cannot use with CPU")

        # 4. Model-dataset consistency
        if self.model.pretrained and self.dataset.effective_in_channels != 3:
            raise ValueError(
                f"Pretrained {self.model.name} requires RGB (3 channels), "
                f"but dataset will provide {self.dataset.effective_in_channels} channels. "
                f"Set 'force_rgb: true' in dataset config or disable pretraining"
            )

        # 5. Optimizer bounds
        if self.training.min_lr >= self.training.learning_rate:
            raise ValueError(
                f"min_lr ({self.training.min_lr}) must be less than "
                f"learning_rate ({self.training.learning_rate})"
            )

        return self

    @property
    def run_slug(self) -> str:
        """Unique experiment folder identifier."""
        return f"{self.dataset.dataset_name}_{self.model.name}"
    
    @property
    def num_workers(self) -> int:
        """Effective DataLoader workers from hardware policy."""
        return self.hardware.effective_num_workers

    def dump_portable(self) -> Dict[str, Any]:
        """
        Serializes config with environment-agnostic paths.
        
        Converts absolute paths to relative anchors from PROJECT_ROOT.
        """
        full_data = self.model_dump()
        full_data["hardware"] = self.hardware.model_dump()
        full_data["telemetry"] = self.telemetry.to_portable_dict()

        # Sanitize dataset root path
        dataset_section = full_data.get("dataset", {})
        data_root = dataset_section.get("data_root")
        
        if data_root:
            dr_path = Path(data_root)
            if dr_path.is_relative_to(PROJECT_ROOT):
                relative_dr = dr_path.relative_to(PROJECT_ROOT)
                full_data["dataset"]["data_root"] = f"./{relative_dr}"
            
        return full_data

    def dump_serialized(self) -> Dict[str, Any]:
        """Converts config to JSON-compatible dict for YAML serialization."""
        return self.model_dump(mode="json")

    @classmethod
    def _resolve_dataset_metadata(cls, args: argparse.Namespace) -> dict:
        """
        Fetches dataset specs from registry.
        
        Args:
            args: Parsed CLI arguments
            
        Returns:
            Dataset metadata dictionary
            
        Raises:
            ValueError: If dataset not found or name missing
        """
        ds_name_raw = getattr(args, 'dataset', None)
        if not ds_name_raw:
            raise ValueError("Dataset name required via --dataset or config file")

        resolution = getattr(args, 'resolution', None)
        ds_name = ds_name_raw.lower()
        
        wrapper = DatasetRegistryWrapper(resolution=resolution)
        if ds_name not in wrapper.registry:
            raise ValueError(
                f"Dataset '{ds_name_raw}' not found in registry for resolution {resolution}"
            )

        return wrapper.get_dataset(ds_name)

    @classmethod
    def _hydrate_yaml(cls, yaml_path: Path, metadata: dict) -> "Config":
        """
        Loads YAML config and injects dataset metadata.
        
        Args:
            yaml_path: Path to config YAML
            metadata: Dataset metadata to inject
            
        Returns:
            Validated Config instance
        """
        raw_data = load_config_from_yaml(yaml_path)
        cfg = cls(**raw_data)
        
        # Inject metadata (frozen model workaround)
        object.__setattr__(cfg.dataset, 'metadata', metadata)
        
        return cls.model_validate(cfg)

    @classmethod
    def from_yaml(cls, yaml_path: Path, metadata: dict) -> "Config":
        """
        Factory from YAML file with metadata injection.
        
        Args:
            yaml_path: Path to config YAML
            metadata: Dataset metadata
            
        Returns:
            Hydrated Config instance
        """
        return cls._hydrate_yaml(yaml_path, metadata)

    @classmethod
    def _build_from_yaml_or_args(
        cls,
        args: argparse.Namespace,
        ds_meta: dict
    ) -> "Config":
        """
        Constructs Config from YAML or CLI arguments.
        
        Prioritizes YAML if provided, otherwise builds from CLI args.
        
        Args:
            args: Parsed argparse namespace
            ds_meta: Dataset metadata
            
        Returns:
            Configured instance
        """
        if getattr(args, "config", None):
            yaml_path = Path(args.config)
            raw_data = load_config_from_yaml(yaml_path)
            
            yaml_dataset_name = raw_data.get("dataset", {}).get("name")
            if yaml_dataset_name:
                resolution = raw_data.get("dataset", {}).get("resolution", 28)
                wrapper = DatasetRegistryWrapper(resolution=resolution)
                ds_meta = wrapper.get_dataset(yaml_dataset_name)
            
            return cls.from_yaml(yaml_path, metadata=ds_meta)
        
        return cls(
            hardware=HardwareConfig.from_args(args),
            telemetry=TelemetryConfig.from_args(args),
            training=TrainingConfig.from_args(args),
            augmentation=AugmentationConfig.from_args(args),
            dataset=DatasetConfig.from_args(args, metadata=ds_meta),
            model=ModelConfig.from_args(args),
            evaluation=EvaluationConfig.from_args(args),
        )
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """
        Factory from CLI arguments.
        
        Args:
            args: Parsed argparse namespace
            
        Returns:
            Configured instance with resolved metadata
        """
        ds_meta = cls._resolve_dataset_metadata(args)
        return cls._build_from_yaml_or_args(args, ds_meta)