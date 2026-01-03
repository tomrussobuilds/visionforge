"""
MedMNIST Configuration & Orchestration Engine.

This module acts as the declarative core of the pipeline, defining the 
hierarchical schema and validation logic required to drive the Orchestrator. 
It leverages Pydantic to transform raw inputs (CLI, YAML) into a structured, 
type-safe manifest that synchronizes hardware state with experiment logic.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import os
import argparse
from pathlib import Path

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
from pydantic import (
    BaseModel, ConfigDict, Field, field_validator, model_validator
)

# =========================================================================== #
#                               Internal Imports                              #
# =========================================================================== #
from .types import NonNegativeInt
from .system_config import SystemConfig
from .training_config import TrainingConfig
from .augmentation_config import AugmentationConfig
from .dataset_config import DatasetConfig
from .evaluation_config import EvaluationConfig
from .models_config import ModelConfig

from ..metadata import DATASET_REGISTRY
from ..environment import get_num_workers
from ..io import load_config_from_yaml

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

    num_workers: NonNegativeInt = Field(default_factory=get_num_workers)

    @model_validator(mode="after")
    def validate_logic(self) -> "Config":
        """
        Cross-field logic validation after instantiation.
        Ensures hardware, model, and dataset parameters are logically aligned.
        """
        # 1. Training Logic
        if self.training.mixup_epochs > self.training.epochs:
            raise ValueError(
                f"mixup_epochs ({self.training.mixup_epochs}) cannot exceed "
                f"total epochs ({self.training.epochs})."
            )
            
        # 2. Hardware vs Feature alignment
        if self.system.device == "cpu" and self.training.use_amp:
            raise ValueError("AMP cannot be enabled when using CPU device.")
            
        # 3. Model vs Dataset consistency (Sfrutta le nuove properties)
        if self.model.pretrained and self.dataset.in_channels == 1 and not self.dataset.force_rgb:
            raise ValueError(
                "Pretrained models require 3-channel input. "
                "Set force_rgb=True in dataset config to enable RGB promotion."
            )
            
        # 4. Optimizer bounds
        if self.training.min_lr >= self.training.learning_rate:
            raise ValueError(
                f"min_lr ({self.training.min_lr}) must be less than "
                f"initial learning_rate ({self.training.learning_rate})."
            )
        return self

    @field_validator("num_workers")
    @classmethod
    def check_cpu_count(cls, v: int) -> int:
        """Limits worker count to the physical capabilities of the host."""
        cpu_count = os.cpu_count() or 1
        return min(v, cpu_count)
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        """Factory method to create a validated Config instance from a YAML file."""
        raw_data = load_config_from_yaml(yaml_path)
        return cls(**raw_data)        
            
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """
        Factory method providing a unified entry point for CLI-based instantiation.
        
        Centralizes the resolution of dataset metadata and injects it into 
        downstream configurations to ensure architectural synchronization.
        """
        if hasattr(args, 'config') and args.config:
            return cls.from_yaml(Path(args.config))

        # --- CENTRAL METADATA RESOLUTION ---
        ds_name = getattr(args, 'dataset', "BloodMNIST")
        if ds_name.lower() not in DATASET_REGISTRY:
             raise ValueError(f"Dataset '{ds_name}' not found in DATASET_REGISTRY.")
        
        ds_meta = DATASET_REGISTRY[ds_name.lower()]
        ds_config = DatasetConfig.from_args(args, metadata=ds_meta)

        # --- INJECTION-BASED SUB-CONFIGURATIONS ---
        config_data = {
            "system": SystemConfig.from_args(args),
            "training": TrainingConfig.from_args(args),
            "augmentation": AugmentationConfig.from_args(args),
            "dataset": ds_config,
            "model": ModelConfig.from_args(args, metadata=ds_meta),
            "evaluation": EvaluationConfig.from_args(args)
        }

        if hasattr(args, 'num_workers') and args.num_workers is not None:
            config_data["num_workers"] = args.num_workers

        return cls(**config_data)