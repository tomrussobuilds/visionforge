"""
Unit tests for the refactored OptunaObjective.

These tests validate the behavior of the Optuna objective and its
supporting components using dependency injection, enabling high
coverage through isolated and deterministic unit tests.
"""

# =========================================================================== #
#                              Standard Imports                               #
# =========================================================================== #
from unittest.mock import MagicMock, patch

# =========================================================================== #
#                             Third-Party Imports                             #
# =========================================================================== #
import pytest
import torch

# =========================================================================== #
#                             Internal Imports                                #
# =========================================================================== #
from orchard.optimization.objective import (
    MetricExtractor,
    OptunaObjective,
    TrialConfigBuilder,
    TrialTrainingExecutor,
)

# =========================================================================== #
#                    TRIAL CONFIG BUILDER TESTS                               #
# =========================================================================== #


@pytest.mark.unit
def test_config_builder_preserves_metadata():
    """Test TrialConfigBuilder preserves dataset metadata."""
    mock_cfg = MagicMock()
    mock_cfg.model_dump.return_value = {
        "dataset": {"resolution": 28},
        "training": {},
        "model": {},
        "augmentation": {},
    }
    mock_cfg.dataset.resolution = 28
    mock_cfg.dataset._ensure_metadata = MagicMock()
    mock_cfg.dataset._ensure_metadata.name = "bloodmnist"
    mock_cfg.optuna.epochs = 20

    builder = TrialConfigBuilder(mock_cfg)

    # Verify metadata is cached
    assert builder.base_metadata == mock_cfg.dataset._ensure_metadata
    assert builder.optuna_epochs == 20


@pytest.mark.unit
def test_config_builder_applies_param_overrides():
    """Test TrialConfigBuilder applies parameter overrides correctly."""
    mock_cfg = MagicMock()
    config_dict = {
        "dataset": {"resolution": 28, "metadata": None},
        "training": {"learning_rate": 0.001, "epochs": 60},
        "model": {"dropout": 0.0},
        "augmentation": {"rotation_angle": 0},
    }
    mock_cfg.model_dump.return_value = config_dict.copy()
    mock_cfg.dataset.resolution = 28
    mock_cfg.dataset._ensure_metadata = MagicMock()
    mock_cfg.optuna.epochs = 20

    builder = TrialConfigBuilder(mock_cfg)

    trial_params = {
        "learning_rate": 0.0001,
        "dropout": 0.3,
        "rotation_angle": 15,
    }

    # Test _apply_param_overrides directly
    test_dict = config_dict.copy()
    test_dict["dataset"]["metadata"] = builder.base_metadata
    test_dict["training"]["epochs"] = builder.optuna_epochs
    builder._apply_param_overrides(test_dict, trial_params)

    # Verify overrides
    assert test_dict["training"]["learning_rate"] == 0.0001
    assert test_dict["model"]["dropout"] == 0.3
    assert test_dict["augmentation"]["rotation_angle"] == 15
    assert test_dict["training"]["epochs"] == 20


# =========================================================================== #
#                    METRIC EXTRACTOR TESTS                                   #
# =========================================================================== #


@pytest.mark.unit
def test_metric_extractor_extracts_correct_metric():
    """Test MetricExtractor extracts specified metric."""
    extractor = MetricExtractor(metric_name="auc")

    val_metrics = {"loss": 0.5, "accuracy": 0.85, "auc": 0.92}

    result = extractor.extract(val_metrics)

    assert result == 0.92


@pytest.mark.unit
def test_metric_extractor_raises_on_missing_metric():
    """Test MetricExtractor raises KeyError for missing metric."""
    extractor = MetricExtractor(metric_name="f1")

    val_metrics = {"loss": 0.5, "accuracy": 0.85, "auc": 0.92}

    with pytest.raises(KeyError, match="f1"):
        extractor.extract(val_metrics)


@pytest.mark.unit
def test_metric_extractor_tracks_best():
    """Test MetricExtractor tracks best metric."""
    extractor = MetricExtractor(metric_name="auc")

    best1 = extractor.update_best(0.80)
    assert best1 == 0.80
    assert extractor.best_metric == 0.80

    best2 = extractor.update_best(0.90)
    assert best2 == 0.90
    assert extractor.best_metric == 0.90

    best3 = extractor.update_best(0.85)
    assert best3 == 0.90
    assert extractor.best_metric == 0.90


# =========================================================================== #
#                    TRAINING EXECUTOR TESTS                                  #
# =========================================================================== #


@pytest.mark.unit
def test_training_executor_should_prune_warmup():
    """Test TrialTrainingExecutor respects warmup period."""
    mock_trial = MagicMock()
    mock_trial.should_prune.return_value = True

    mock_cfg = MagicMock()
    mock_cfg.training.use_amp = False
    mock_cfg.training.epochs = 30
    mock_cfg.training.scheduler_type = "cosine"
    mock_cfg.optuna.enable_pruning = True
    mock_cfg.optuna.pruning_warmup_epochs = 10

    executor = TrialTrainingExecutor(
        model=MagicMock(),
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=MagicMock(),
        scheduler=MagicMock(),
        criterion=MagicMock(),
        cfg=mock_cfg,
        device=torch.device("cpu"),
        metric_extractor=MetricExtractor("auc"),
    )

    # Before warmup - should NOT prune
    assert executor._should_prune(mock_trial, epoch=5) is False

    # After warmup - should prune
    assert executor._should_prune(mock_trial, epoch=15) is True


@pytest.mark.unit
def test_training_executor_disabled_pruning():
    """Test TrialTrainingExecutor with pruning disabled."""
    mock_trial = MagicMock()
    mock_trial.should_prune.return_value = True

    mock_cfg = MagicMock()
    mock_cfg.training.use_amp = False
    mock_cfg.training.epochs = 30
    mock_cfg.training.grad_clip = 0.0
    mock_cfg.training.scheduler_type = "cosine"
    mock_cfg.training.mixup_epochs = 0
    mock_cfg.optuna.enable_pruning = False
    mock_cfg.optuna.pruning_warmup_epochs = 10

    assert mock_cfg.optuna.enable_pruning is False

    executor = TrialTrainingExecutor(
        model=MagicMock(),
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=MagicMock(),
        scheduler=MagicMock(),
        criterion=MagicMock(),
        cfg=mock_cfg,
        device=torch.device("cpu"),
        metric_extractor=MetricExtractor("auc"),
    )

    # These should ALL be False
    result1 = executor._should_prune(mock_trial, epoch=5)
    result2 = executor._should_prune(mock_trial, epoch=15)

    assert result1 is False, f"Expected False but got {result1}"
    assert result2 is False, f"Expected False but got {result2}"


@pytest.mark.unit
def test_training_executor_validate_epoch_error_handling():
    """Test TrialTrainingExecutor handles validation errors."""
    mock_cfg = MagicMock()
    mock_cfg.training.use_amp = False
    mock_cfg.training.epochs = 30
    mock_cfg.optuna.enable_pruning = True
    mock_cfg.optuna.pruning_warmup_epochs = 5

    executor = TrialTrainingExecutor(
        model=MagicMock(),
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=MagicMock(),
        scheduler=MagicMock(),
        criterion=MagicMock(),
        cfg=mock_cfg,
        device=torch.device("cpu"),
        metric_extractor=MetricExtractor("auc"),
    )

    with patch(
        "orchard.trainer.validate_epoch",
        side_effect=RuntimeError("Validation failed"),
    ):
        result = executor._validate_epoch()

    # Should return default metrics
    assert result == {"loss": 999.0, "accuracy": 0.0, "auc": 0.0}


# =========================================================================== #
#                    OPTUNA OBJECTIVE TESTS                                   #
# =========================================================================== #


@pytest.mark.unit
def test_optuna_objective_init_with_defaults():
    """Test OptunaObjective initializes with dependency injection."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 20
    mock_cfg.optuna.metric_name = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()
    mock_cfg.dataset._ensure_metadata.name = "bloodmnist"

    search_space = {"learning_rate": MagicMock()}
    device = torch.device("cpu")

    # Use dependency injection
    mock_dataset = MagicMock()
    mock_dataset.path = "/fake/path"
    mock_dataset_loader = MagicMock(return_value=mock_dataset)

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=device,
        dataset_loader=mock_dataset_loader,
    )

    # Verify injected loader was called
    mock_dataset_loader.assert_called_once_with(mock_cfg.dataset._ensure_metadata)
    assert objective.medmnist_data == mock_dataset
    assert objective._dataset_loader == mock_dataset_loader


@pytest.mark.unit
def test_optuna_objective_uses_injected_dependencies():
    """Test OptunaObjective uses all injected dependencies."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 20
    mock_cfg.optuna.metric_name = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space = {}
    device = torch.device("cpu")

    # Mock all injectable dependencies
    mock_dataset_loader = MagicMock(return_value=MagicMock())
    mock_dataloader_factory = MagicMock()
    mock_model_factory = MagicMock()

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=device,
        dataset_loader=mock_dataset_loader,
        dataloader_factory=mock_dataloader_factory,
        model_factory=mock_model_factory,
    )

    # Verify all injected dependencies are stored
    assert objective._dataset_loader == mock_dataset_loader
    assert objective._dataloader_factory == mock_dataloader_factory
    assert objective._model_factory == mock_model_factory

    # Verify dataset loader was called during __init__
    mock_dataset_loader.assert_called_once_with(mock_cfg.dataset._ensure_metadata)


@pytest.mark.unit
def test_optuna_objective_sample_params_dict():
    """Test OptunaObjective samples params from dict search space."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 20
    mock_cfg.optuna.metric_name = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()
    mock_cfg.training.momentum = 0.9
    mock_cfg.training.mixup_alpha = 0.0
    mock_cfg.training.cosine_fraction = 1.0
    mock_suggest = MagicMock(return_value=0.001)
    search_space = {"learning_rate": mock_suggest}

    # Use injected loader to avoid patching
    mock_loader = MagicMock(return_value=MagicMock())

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=torch.device("cpu"),
        dataset_loader=mock_loader,
    )

    mock_trial = MagicMock()
    params = objective._sample_params(mock_trial)

    # Verify suggest function was called
    mock_suggest.assert_called_once_with(mock_trial)
    assert params == {"learning_rate": 0.001}


@pytest.mark.unit
def test_optuna_objective_sample_params_object():
    """Test OptunaObjective samples params from object with sample_params."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 20
    mock_cfg.optuna.metric_name = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    # Mock search space with sample_params method
    mock_search_space = MagicMock()
    mock_search_space.sample_params.return_value = {"lr": 0.01, "dropout": 0.3}

    # Mock dataset loader to avoid calling real load_medmnist
    mock_dataset_loader = MagicMock(return_value=MagicMock())

    # Inject both mocks
    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=mock_search_space,
        device=torch.device("cpu"),
        dataset_loader=mock_dataset_loader,
    )

    mock_trial = MagicMock()
    params = objective._sample_params(mock_trial)

    mock_search_space.sample_params.assert_called_once_with(mock_trial)
    assert params == {"lr": 0.01, "dropout": 0.3}

    mock_dataset_loader.assert_called_once_with(mock_cfg.dataset._ensure_metadata)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
