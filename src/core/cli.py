"""
Argument Parsing Module

This module handles the command-line interface (CLI) for the training pipeline.
It bridges terminal inputs with the hierarchical Pydantic configuration.
"""

# =========================================================================== #
#                               Standard Imports                              #
# =========================================================================== #
import argparse

# =========================================================================== #
#                               Internal Imports                              #
# =========================================================================== #
from .config.system_config import SystemConfig
from .config.training_config import TrainingConfig
from .config.evaluation_config import EvaluationConfig
from .config.augmentation_config import AugmentationConfig

# =========================================================================== #
#                                ARGUMENT PARSING                             #
# =========================================================================== #

def parse_args() -> argparse.Namespace:
    """
    Configure and analyze command line arguments for the training script.
    """
    from .metadata import DATASET_REGISTRY

    parser = argparse.ArgumentParser(
        description="MedMNIST training pipeline based on adapted ResNet-18.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Instantiate specific configs that HAVE default values for all fields
    sys_def   = SystemConfig()
    train_def = TrainingConfig()
    eval_def  = EvaluationConfig()
    aug_def   = AugmentationConfig()
    # ModelConfig and DatasetConfig are NOT instantiated here because they 
    # require mandatory data-dependent fields (channels, classes, metadata).

    # Group: Global Strategy
    strat_group = parser.add_argument_group("Global Strategy")

    strat_group.add_argument(
        '--config', 
        type=str, 
        default=None, 
        help="Path to a YAML configuration file. If provided, all other CLI arguments are ignored."
    )
    strat_group.add_argument(
        '--project_name',
        type=str,
        default=sys_def.project_name,
        help="Logical name for the experiment suite (used for logging and locks)."
    )
    strat_group.add_argument(
        '--reproducible',
        action='store_true',
        dest='reproducible',
        help="Enforces strict determinism: deterministic algorithms and num_workers=0."
    )

    # Group: System & Hardware
    sys_group = parser.add_argument_group("System & Hardware")
    
    sys_group.add_argument(
        '--device',
        type=str,
        default=sys_def.device,
        help="Computing device (cpu, cuda, mps)."
    )
    sys_group.add_argument(
        '--num_workers',
        type=int,
        dest='num_workers',
        default=None,
        help="Number of subprocesses for data loading."
    )

    # Group: Paths & Logging
    path_group = parser.add_argument_group("Paths & Logging")

    path_group.add_argument(
        '--data_dir',
        type=str,
        default=str(sys_def.data_dir),
        help="Path to directory containing raw MedMNIST .npz files."
    )
    path_group.add_argument(
        '--output_dir',
        type=str,
        default=str(sys_def.output_dir),
        help="Base directory for experiment outputs and runs."
    )
    path_group.add_argument(
        '--log_interval',
        type=int,
        default=sys_def.log_interval,
        help="How many batches to wait before logging training status."
    )
    path_group.add_argument(
        '--no_save',
        action='store_false',
        dest='save_model',
        default=sys_def.save_model,
        help="Disable saving the best model checkpoint."
    )
    path_group.add_argument(
        '--resume',
        type=str,
        default=None,
        help="Path to a .pth checkpoint to resume training or run evaluation."
    )
    # Group: Training Hyperparameters
    train_group = parser.add_argument_group("Training Hyperparameters")
    
    train_group.add_argument(
        '--epochs',
        type=int,
        default=train_def.epochs
    )
    train_group.add_argument(
        '--batch_size',
        type=int,
        default=train_def.batch_size
    )
    train_group.add_argument(
        '--lr', '--learning_rate',
        type=float,
        default=train_def.learning_rate
    )
    train_group.add_argument(
        '--seed',
        type=int,
        default=train_def.seed
    )
    train_group.add_argument(
        '--patience',
        type=int,
        default=train_def.patience
    )
    train_group.add_argument(
        '--momentum',
        type=float,
        default=train_def.momentum
    )
    train_group.add_argument(
        '--weight_decay',
        type=float,
        default=train_def.weight_decay
    )
    train_group.add_argument(
        '--cosine_fraction',
        type=float,
        default=train_def.cosine_fraction,
        help="Fraction of total epochs to apply cosine annealing before switching to ReduceLROnPlateau."
    )
    train_group.add_argument(
        '--use_amp',
        action='store_true',
        default=train_def.use_amp,
        help="Enable Automatic Mixed Precision (FP16) during training."
    )
    train_group.add_argument(
        '--no_amp',
        action='store_false',
        dest='use_amp',
        help="Disable Automatic Mixed Precision."
    )
    train_group.add_argument(
        '--grad_clip',
        type=float,
        default=train_def.grad_clip,
        help="Maximum norm for gradient clipping (set to 0 to disable)."
    )
    train_group.add_argument(
        '--label_smoothing',
        type=float,
        default=0.0,
        help="Label smoothing factor (0.0 to 1.0)."
    )

    # Group: Regularization & Augmentation
    aug_group = parser.add_argument_group("Regularization & Augmentation")
    
    aug_group.add_argument(
        '--mixup_alpha',
        type=float,
        default=train_def.mixup_alpha
    )
    aug_group.add_argument(
        '--mixup_epochs',
        type=int,
        default=train_def.mixup_epochs,
        help="Number of epochs to apply MixUp augmentation."
    )
    aug_group.add_argument(
        '--no_tta',
        action='store_false',
        dest='use_tta',
        default=train_def.use_tta,
        help="Disable TTA during final evaluation."
    )
    # Collegamento corretto ai default di AugmentationConfig
    aug_group.add_argument(
        '--hflip',
        type=float,
        default=aug_def.hflip
    )
    aug_group.add_argument(
        '--rotation_angle',
        type=int,
        default=aug_def.rotation_angle
    )
    aug_group.add_argument(
        '--jitter_val',
        type=float,
        default=aug_def.jitter_val
    )

    # Group: Dataset Selection and Configuration
    dataset_group = parser.add_argument_group("Dataset Configuration")

    dataset_group.add_argument(
        '--dataset',
        type=str,
        default="bloodmnist",
        choices=DATASET_REGISTRY.keys(),
        help="Target MedMNIST dataset."
    )
    dataset_group.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help="Max training samples (Use 0 or -1 for full dataset)."
    )
    dataset_group.add_argument(
        '--balanced',
        action='store_true',
        dest='use_weighted_sampler',
        default=False,
        help="Use WeightedRandomSampler to handle class imbalance."
    )
    dataset_group.add_argument(
        '--force_rgb',
        action='store_true',
        dest='force_rgb',
        default=None, 
        help="Force conversion of grayscale to RGB."
    )
    dataset_group.add_argument(
        '--no_force_rgb',
        action='store_false',
        dest='force_rgb',
        help="Disable grayscale to RGB conversion."
    )
    dataset_group.add_argument(
        '--is_anatomical',
        type=lambda x: (str(x).lower() == 'true'),
        default=None,
        help="Override anatomical orientation flag. If None, uses Registry default."
    )
    dataset_group.add_argument(
        '--is_texture_based',
        type=lambda x: (str(x).lower() == 'true'),
        default=None,
        help="Override texture-based flag. If None, uses Registry default."
    )

    # Group: Model Selection
    model_group = parser.add_argument_group("Model Configuration")

    model_group.add_argument(
        '--model_name',
        type=str,
        default="resnet_18_adapted",
        help="Architecture identifier."
    )
    model_group.add_argument(
        '--pretrained',
        action='store_true',
        default=True,
        help="Load ImageNet weights for the backbone (default: True)."
    )
    model_group.add_argument(
        '--no_pretrained',
        action='store_false',
        dest='pretrained',
        help="Initialize model with random weights."
    )

    # Group: Evaluation & Reporting
    eval_group = parser.add_argument_group("Evaluation & Reporting")

    eval_group.add_argument(
        '--n_samples',
        type=int,
        default=eval_def.n_samples,
        help="Number of images to display in the prediction grid."
    )
    eval_group.add_argument(
        '--fig_dpi',
        type=int,
        default=eval_def.fig_dpi,
        help="Resolution (DPI) for saved plots."
    )
    eval_group.add_argument(
        '--report_format',
        type=str,
        default=eval_def.report_format,
        choices=["xlsx", "csv", "json"],
        help="Format for the final experiment summary."
    )
    eval_group.add_argument(
        '--plot_style',
        type=str,
        default=eval_def.plot_style,
        help="Matplotlib style for visualizations (e. g., 'ggplot', 'bmh')."
    )

    return parser.parse_args()