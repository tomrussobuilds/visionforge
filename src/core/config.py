"""
    Main Configuration Manifest for the MedMNIST Pipeline.
    
    This root container aggregates specialized sub-configurations (System, Training, 
    Augmentation, Dataset, Evaluation) into a unified, immutable schema. It acts 
    as the central validator for cross-module logic, ensuring that hardware 
    capabilities (AMP, Device) align with training strategies (MixUp, Pretraining).

    The class provides robust factory methods (`from_args`, `from_yaml`) to 
    seamlessly bridge external inputs into a type-safe, validated environment.

    Attributes:
        system (SystemConfig): Infrastructure and hardware settings.
        training (TrainingConfig): Optimization and regularization hyperparameters.
        augmentation (AugmentationConfig): Data transformation and TTA parameters.
        dataset (DatasetConfig): Metadata and constraints for the specific MedMNIST task.
        evaluation (EvaluationConfig): Visualization and reporting preferences.
        num_workers (int): Validated CPU worker count for data loading.
        model_name (str): Architecture identifier.
        pretrained (bool): Flag to enable/disable ImageNet weight transfer.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import os
import argparse
from pathlib import Path
import tempfile
from typing import Annotated, Optional

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
from pydantic import (
    BaseModel, Field, ConfigDict, field_validator, model_validator, AfterValidator
    )

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .system import detect_best_device, get_num_workers, kill_duplicate_processes
from .paths import DATASET_DIR, OUTPUTS_ROOT
from .io import load_config_from_yaml
from .metadata import DATASET_REGISTRY

# =========================================================================== #
#                                TYPE ALIASES                                 #
# =========================================================================== #

def _ensure_dir(v: Path) -> Path:
    "Ensure paths are absolute and create folders if missing."
    v.mkdir(parents=True, exist_ok=True)
    return v.resolve()

ValidatedPath = Annotated[Path, AfterValidator(_ensure_dir)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
SmoothingValue = Annotated[float, Field(ge=0.0, le=0.3)]
LearningRate = Annotated[float, Field(gt=1e-7, lt=1.0)]
Percentage = Annotated[float, Field(gt=0.0, le=1.0)]
Degrees = Annotated[int, Field(ge=0, le=180)]

# =========================================================================== #
#                                SUB-CONFIGURATIONS                           #
# =========================================================================== #

class SystemConfig(BaseModel):
    """Sub-configuration for system paths and hardware settings."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )
    device: str = Field(
        default_factory=detect_best_device,
        description="Computing device (cpu, cuda, mps or auto)."
    )
    data_dir: ValidatedPath = Field(default=DATASET_DIR)
    output_dir: ValidatedPath = Field(default=OUTPUTS_ROOT)
    save_model: bool = True
    log_interval: PositiveInt = Field(default=10)
    project_name: str = "medmnist_experiment"
    allow_process_kill: bool = Field(
        default=True,
        description="Enable automatic termination of duplicate processes."
    )

    @property
    def lock_file_path(self) -> Path:
        """Dynamically generates a cross-platform lock file path."""
        return Path(tempfile.gettempdir()) / f"{self.project_name}.lock"

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
    
    def manage_environment(self) -> None:
        """"
        Handles environment setup tasks such as killing duplicates if enabled.
        Safeguards against termination in shared cluster environments.
        """
        if not self.allow_process_kill:
            return
        is_shared = any(env in os.environ for env in ["SLURM_JOB_ID", "PBS_JOBID"])
        if is_shared:
            return
        kill_duplicate_processes()
        

class TrainingConfig(BaseModel):
    """Sub-configuration for core training hyperparameters."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )
    
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    batch_size: PositiveInt = Field(default=128)
    epochs: PositiveInt = Field(default=60)
    patience: NonNegativeInt = Field(default=15)
    learning_rate: PositiveFloat = Field(default=0.008)
    min_lr: LearningRate = Field(default=1e-6)
    momentum: Probability = Field(default=0.9)
    weight_decay: NonNegativeFloat = Field(default=5e-4)
    label_smoothing: SmoothingValue = 0.0
    mixup_alpha: NonNegativeFloat = Field(
        default=0.002,
        description="Mixup interpolation coefficient"
        )
    mixup_epochs: NonNegativeInt = Field(
        default=30,
        description="Number of epochs to apply mixup")
    use_tta: bool = True
    cosine_fraction: Probability = Field(default=0.5)
    use_amp: bool = False
    grad_clip: Optional[PositiveFloat] = Field(
        default=1.0,
        description="Max norm for gradient clipping; None to disable"
    )


class AugmentationConfig(BaseModel):
    """Sub-configuration for data augmentation parameters."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )
    
    hflip: Probability = Field(default=0.5)
    rotation_angle: Annotated[int, Field(default=10, ge=0, le=180)]
    jitter_val: NonNegativeFloat = Field(default=0.2)
    min_scale: Probability = Field(default=0.9)

    tta_translate: float = Field(
        default=2.0,
        description="Pixel shift for TTA"
    )
    tta_scale: float = Field(
        default=1.1,
        description="Scale factor for TTA"
    )
    tta_blur_sigma: float = Field(
        default=0.4,
        description="Gaussian blur sigma for TTA"
    )


class DatasetConfig(BaseModel):
    """Sub-configuration for dataset-specific metadata and sampling."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )
    
    dataset_name: str = "BloodMNIST"
    max_samples: Optional[PositiveInt] = Field(default=20000)
    use_weighted_sampler: bool = True
    in_channels: Annotated[PositiveInt, Field(ge=1, le=3)] = 3
    num_classes: PositiveInt = Field(
        default=8,
        description="Number of target classes in the dataset"
    )
    img_size: PositiveInt = Field(
        default=28,
        description="Target square resolution for the model input"
    )
    force_rgb: bool = Field(
        default=True,
        description="Convert grayscale to 3-channel to enable ImageNet weights"
    )
    mean: tuple[float, ...] = Field(
        default=(0.5, 0.5, 0.5),
        description="Channel-wise mean for normalization",
        min_length=1,
        max_length=3
    )
    std: tuple[float, ...] = Field(
        default=(0.5, 0.5, 0.5),
        description="Channel-wise std for normalization",
        min_length=1,
        max_length=3
    )
    normalization_info: str = "N/A"
    is_anatomical: bool = True
    is_texture_based: bool = True

    @property
    def effective_in_channels(self) -> int:
        """Returns the actual number of channels the model will see"""
        return 3 if self.force_rgb else self.in_channels
    

class EvaluationConfig(BaseModel):
    """Sub-configuration for model evaluation and reporting."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )
    
    n_samples: PositiveInt = Field(default=12)
    fig_dpi: PositiveInt = Field(default=200)
    img_size: tuple[int, int] = (10, 10)
    cmap_confusion: str = "Blues"
    plot_style: str = "seaborn-v0_8-muted"
    report_format: str = "xlsx"

    @field_validator("report_format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        supported = ["xlsx", "csv", "json"]
        if v.lower() not in supported:
            raise ValueError(f"Format {v} not supported. Use {supported}")
        return v.lower()
    
    save_confusion_matrix: bool = True
    save_predictions_grid: bool = True
    grid_cols: PositiveInt = Field(default=4)
    fig_size_predictions: tuple[int, int] = (12, 8)


# =========================================================================== #
#                                MAIN CONFIGURATION                          #
# =========================================================================== #

class Config(BaseModel):
    """
    Main Configuration Manifest for the MedMNIST Pipeline.
    
    This root container aggregates specialized sub-configurations (System, Training, 
    Augmentation, Dataset, Evaluation) into a unified, immutable schema. It acts 
    as the central validator for cross-module logic, ensuring that hardware 
    capabilities (AMP, Device) align with training strategies (MixUp, Pretraining).

    The class provides robust factory methods (`from_args`, `from_yaml`) to 
    seamlessly bridge external inputs into a type-safe, validated environment.

    Attributes:
        system (SystemConfig): Infrastructure and hardware settings.
        training (TrainingConfig): Optimization and regularization hyperparameters.
        augmentation (AugmentationConfig): Data transformation and TTA parameters.
        dataset (DatasetConfig): Metadata and constraints for the specific MedMNIST task.
        evaluation (EvaluationConfig): Visualization and reporting preferences.
        num_workers (int): Validated CPU worker count for data loading.
        model_name (str): Architecture identifier.
        pretrained (bool): Flag to enable/disable ImageNet weight transfer.
    """
    model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            frozen=True
    )
    
    # Nested configurations - Explicit access required (e.g., cfg.training.seed)
    system: SystemConfig = Field(default_factory=SystemConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    
    num_workers: NonNegativeInt = Field(default_factory=get_num_workers)
    model_name: str = "ResNet-18 Adapted"
    pretrained: bool = True


    @model_validator(mode="after")
    def validate_logic(self) -> "Config":
        """Cross-field logic validation after instantiation."""
        if self.training.mixup_epochs > self.training.epochs:
            raise ValueError(
                f"mixup_epochs ({self.training.mixup_epochs}) cannot exceed "
                f"epochs ({self.training.epochs})."
            )
        is_cpu = self.system.device == "cpu"
        if is_cpu and self.training.use_amp:
            raise ValueError("AMP cannot be enabled when using CPU device.")
        if self.pretrained and self.dataset.in_channels == 1 and not self.dataset.force_rgb:
            raise ValueError(
                "Pretrained models require 3-channel input. "
                "Set force_rgb=True in dataset config."
            )
        if self.training.min_lr >= self.training.learning_rate:
            raise ValueError(
                f"min_lr ({self.training.min_lr}) must be less than "
                f"learning_rate ({self.training.learning_rate})."
            )
        return self

    @field_validator("num_workers")
    @classmethod
    def check_cpu_count(cls, v: int) -> int:
        cpu_count = os.cpu_count() or 1
        return min(v, cpu_count)
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        """
        Factory method to create a validated Config instance from a YAML file.
        """
        raw_data = load_config_from_yaml(yaml_path)
        return cls(**raw_data)        
            
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """
        Factory method to create a validated Config instance from a CLI namespace.
        
        This method orchestrates the transformation of raw argparse parameters into 
        the hierarchical Pydantic schema, applying conditional logic for hardware 
        compatibility and dataset metadata resolution.
        """
        
        # 1. Short-circuit: If a --config YAML is provided, load direttaly from it
        if hasattr(args, 'config') and args.config:
            return cls.from_yaml(Path(args.config))

        # --- ENCAPSULATION ---

        def resolve_dataset_metadata():
            """Retrieve static metadata for the dataset from the central registry."""
            key = args.dataset.lower()
            if key not in DATASET_REGISTRY:
                raise ValueError(f"Dataset '{args.dataset}' not supported in DATASET_REGISTRY.")
            return DATASET_REGISTRY[key]

        def build_training_subconfig():
            """Map training parameters ensuring defaults are present."""
            return TrainingConfig(
                seed=getattr(args, 'seed', 42),
                batch_size=getattr(args, 'batch_size', 128),
                learning_rate=getattr(args, 'lr', 0.008),
                momentum=getattr(args, 'momentum', 0.9),
                weight_decay=getattr(args, 'weight_decay', 5e-4),
                epochs=getattr(args, 'epochs', 60),
                patience=getattr(args, 'patience', 15),
                mixup_alpha=getattr(args, 'mixup_alpha', 0.2),
                mixup_epochs=getattr(args, 'mixup_epochs', 20),
                use_tta=getattr(args, 'use_tta', True),
                cosine_fraction=getattr(args, 'cosine_fraction', 0.5),
                use_amp=getattr(args, 'use_amp', False),
                grad_clip=getattr(args, 'grad_clip', 1.0),
                label_smoothing=getattr(args, 'label_smoothing', 0.0),
                min_lr=getattr(args, 'min_lr', 1e-6)
            )

        def build_augmentation_subconfig():
            """Map augmentation parameters ensuring defaults are present."""
            return AugmentationConfig(
                hflip=getattr(args, 'hflip', 0.5),
                rotation_angle=getattr(args, 'rotation_angle', 10),
                jitter_val=getattr(args, 'jitter_val', 0.2),
                min_scale=getattr(args, 'min_scale', 0.9),
                tta_translate=getattr(args, 'tta_translate', 2.0),
                tta_scale=getattr(args, 'tta_scale', 1.1),
                tta_blur_sigma=getattr(args, 'tta_blur_sigma', 0.4)
            )

        # --- LOGIC EXECUTION ---

        ds_meta = resolve_dataset_metadata()
        
        # Determine RGB logic: User override or automatic for pretrained grayscale
        force_rgb_arg = getattr(args, 'force_rgb', None)
        should_force_rgb = force_rgb_arg if force_rgb_arg is not None else \
                          (ds_meta.in_channels == 1 and getattr(args, 'pretrained', False))
        
        # Determine final max_samples value
        final_max_samples = args.max_samples if (getattr(args, 'max_samples', 0) > 0) else None

        # --- CONFIGURATION BUILDING ---

        return cls(
            model_name=getattr(args, 'model_name', "ResNet-18 Adapted"),
            pretrained=getattr(args, 'pretrained', True),
            num_workers=getattr(args, 'num_workers', 4),
            system=SystemConfig(
                device=getattr(args, 'device', "auto"),
                data_dir=Path(getattr(args, 'data_dir', "./data")),
                output_dir=Path(getattr(args, 'output_dir', "./outputs")),
                save_model=getattr(args, 'save_model', True),
                log_interval=getattr(args, 'log_interval', 10),
                project_name=getattr(args, 'project_name', "medmnist_experiment")
            ),
            training=build_training_subconfig(),
            augmentation=build_augmentation_subconfig(),
            dataset=DatasetConfig(
                dataset_name=ds_meta.name,
                max_samples=final_max_samples,
                use_weighted_sampler=getattr(args, 'use_weighted_sampler', True),
                in_channels=ds_meta.in_channels,
                num_classes=len(ds_meta.classes),
                mean=ds_meta.mean,
                std=ds_meta.std,
                normalization_info=f"Mean={ds_meta.mean}, Std={ds_meta.std}",
                is_anatomical=ds_meta.is_anatomical,
                is_texture_based=ds_meta.is_texture_based,
                force_rgb=should_force_rgb,
                img_size=getattr(args, 'img_size', 28)
            ),
            evaluation=EvaluationConfig(
                n_samples=getattr(args, 'n_samples', 12),
                fig_dpi=getattr(args, 'fig_dpi', 200),
                plot_style=getattr(args, 'plot_style', "seaborn-v0_8-muted"),
                report_format=getattr(args, 'report_format', "xlsx")
            )
        )