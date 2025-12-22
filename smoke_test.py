"""
Smoke Test Module for MedMNIST Pipeline

This script performs a rapid, end-to-end execution of the training and 
evaluation pipeline. It uses a minimal subset of data and a single epoch 
to verify:
1. Model initialization and forward/backward passes.
2. Checkpoint saving and loading.
3. Visualization utility compatibility (specifically training curves and matrices).
4. Reporting and directory structure integrity.
"""

# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
import logging
from dataclasses import replace

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import torch

# =========================================================================== #
#                                Internal Imports
# =========================================================================== #
from scripts.core import (
    Config, Logger, set_seed, DATASET_REGISTRY, RunPaths, 
    setup_static_directories
)
from scripts.data_handler import (
    load_medmnist, get_dataloaders, get_augmentations_transforms
)
from scripts.models import get_model
from scripts.trainer import ModelTrainer
from scripts.evaluation import run_final_evaluation

# =========================================================================== #
#                               SMOKE TEST EXECUTION
# =========================================================================== #

def run_smoke_test() -> None:
    """
    Orchestrates a lightweight version of the main pipeline to ensure 
    code stability and prevent regression bugs.
    """
    print("\n" + "="*60)
    print("INITIALIZING SMOKE TEST".center(60))
    print("="*60 + "\n")

    # 1. Minimal Configuration Setup
    dataset_key = "bloodmnist"
    ds_meta = DATASET_REGISTRY[dataset_key]

    cfg = Config(
        model_name="ResNet-18 Adapted",
        dataset_name=ds_meta.name,
        seed=42,
        batch_size=4,
        epochs=1,
        patience=1,
        use_tta=True,
        normalization_info="Smoke Test Normalization",
    )
    
    set_seed(cfg.seed)

    # 2. Environment Initialization
    setup_static_directories()
    
    # Prefix the run ID to distinguish smoke tests from real experiments
    paths = RunPaths(f"SMOKE_TEST_{cfg.model_name}", cfg.dataset_name)
    
    Logger.setup(
        name=paths.project_id,
        log_dir=paths.logs
    )
    run_logger = logging.getLogger(paths.project_id)
    device = torch.device("cpu") # Force CPU for universal smoke test compatibility
    
    run_logger.info("Starting Smoke Test: Environment verified.")

    # 3. Data Loading and Mocking
    # Load the dataset and truncate it using replace() for frozen dataclass support
    data = load_medmnist(ds_meta)
    
    run_logger.info("Mocking data subsets for rapid testing...")
    data = replace(
        data,
        X_train=data.X_train[:16],
        y_train=data.y_train[:16],
        X_val=data.X_val[:8],
        y_val=data.y_val[:8],
        X_test=data.X_test[:8],
        y_test=data.y_test[:8]
    )

    # Re-initialize dataloaders with the mocked data
    train_loader, val_loader, test_loader = get_dataloaders(data, cfg)

    # 4. Model Factory Check
    model = get_model(device=device, cfg=cfg)
    run_logger.info(f"Model {cfg.model_name} instantiated on {device}.")

    # 5. Training Loop Execution
    run_logger.info("Executing training epoch...")
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=cfg,
        output_dir=paths.models
    )
    
    best_path, train_losses, val_accuracies = trainer.train()

    # 6. Final Evaluation & Visualization Verification
    run_logger.info("Running final evaluation and reporting...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    
    aug_info = get_augmentations_transforms(cfg)

    macro_f1, test_acc = run_final_evaluation(
        model=model,
        test_loader=test_loader,
        test_images=data.X_test,
        test_labels=data.y_test,
        class_names=ds_meta.classes,
        train_losses=train_losses,
        val_accuracies=val_accuracies,
        device=device,
        paths=paths,
        cfg=cfg,
        use_tta=cfg.use_tta,
        aug_info=aug_info
    )

    run_logger.info(f"SMOKE TEST PASSED: Acc {test_acc:.4f} | F1 {macro_f1:.4f}")
    print(f"\nSmoke test completed. Check outputs in: {paths.root}\n")


# =========================================================================== #
#                               ENTRY POINT
# =========================================================================== #

if __name__ == "__main__":
    try:
        run_smoke_test()
    except Exception as e:
        # Using a generic logger check to handle potential setup failures
        logging.error(f"SMOKE TEST FAILED: {str(e)}", exc_info=True)
        raise