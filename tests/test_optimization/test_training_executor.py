"""
Comprehensive Test Suite for TrialTrainingExecutor.

Tests cover initialization, Optuna integration (reporting/pruning),
scheduler stepping, and error handling during validation.
"""

# =========================================================================== #
#                         Standard Imports                                    #
# =========================================================================== #
from unittest.mock import MagicMock, patch

# =========================================================================== #
#                         Third-Party Imports                                 #
# =========================================================================== #
import optuna
import pytest
import torch
import torch.nn as nn

# =========================================================================== #
#                         Internal Imports                                    #
# =========================================================================== #
from orchard.optimization import MetricExtractor, TrialTrainingExecutor

# =========================================================================== #
#                    FIXTURES                                                 #
# =========================================================================== #


@pytest.fixture
def mock_cfg():
    """Mock Config specific for Optuna trials."""
    cfg = MagicMock()
    cfg.training.epochs = 5
    cfg.training.use_amp = False
    cfg.training.grad_clip = 1.0
    cfg.training.scheduler_type = "step"

    cfg.optuna.enable_pruning = True
    cfg.optuna.pruning_warmup_epochs = 2
    return cfg


@pytest.fixture
def mock_trial():
    """Mock Optuna trial."""
    trial = MagicMock(spec=optuna.Trial)
    trial.number = 42
    trial.should_prune.return_value = False
    return trial


@pytest.fixture
def executor(mock_cfg):
    """TrialTrainingExecutor instance with mocked components."""
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    metric_extractor = MetricExtractor(metric_name="auc")

    return TrialTrainingExecutor(
        model=model,
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=nn.CrossEntropyLoss(),
        cfg=mock_cfg,
        device=torch.device("cpu"),
        metric_extractor=metric_extractor,
    )


# =========================================================================== #
#                    TESTS: Initialization                                    #
# =========================================================================== #


@pytest.mark.unit
def test_executor_init(executor):
    """Test correctly mapping config to executor attributes."""
    assert executor.epochs == 5
    assert executor.enable_pruning is True
    assert executor.warmup_epochs == 2
    assert executor.scaler is None


# =========================================================================== #
#                    TESTS: Optuna Integration                                #
# =========================================================================== #


@pytest.mark.unit
def test_should_prune_respects_warmup(executor, mock_trial):
    """Ensure pruning is never triggered before warmup_epochs."""
    executor.enable_pruning = True
    executor.warmup_epochs = 3
    mock_trial.should_prune.return_value = True

    assert executor._should_prune(mock_trial, epoch=1) is False
    assert executor._should_prune(mock_trial, epoch=3) is True


@pytest.mark.unit
def test_should_prune_respects_flag(executor, mock_trial):
    """Ensure pruning is disabled if enable_pruning is False."""
    executor.enable_pruning = False
    executor.warmup_epochs = 0
    mock_trial.should_prune.return_value = True

    assert executor._should_prune(mock_trial, epoch=1) is False


# =========================================================================== #
#                    TESTS: Scheduler Logic                                   #
# =========================================================================== #


@pytest.mark.unit
def test_step_scheduler_plateau(executor):
    """Test plateau scheduler receives val_loss."""
    executor.cfg.training.scheduler_type = "plateau"
    executor.scheduler = MagicMock()

    executor._step_scheduler(val_loss=0.5)
    executor.scheduler.step.assert_called_once_with(0.5)


@pytest.mark.unit
def test_step_scheduler_standard(executor):
    """Test standard scheduler (StepLR) is called without arguments."""
    executor.cfg.training.scheduler_type = "step"
    executor.scheduler = MagicMock()

    executor._step_scheduler(val_loss=0.5)
    executor.scheduler.step.assert_called_once_with()


@pytest.mark.unit
def test_return_if_scheduler_is_none(executor):
    """Ensures the function exits early when the scheduler is not initialized."""
    executor.scheduler = None
    # This should not raise any AttributeError when calling internal step logic
    result = executor._step_scheduler(val_loss=0.5)
    assert result is None


# =========================================================================== #
#                    TESTS: Full Execution Loop                               #
# =========================================================================== #


@pytest.mark.integration
@patch("orchard.optimization.objective.TrialTrainingExecutor._train_epoch")
@patch("orchard.optimization.objective.TrialTrainingExecutor._validate_epoch")
def test_execute_full_loop(mock_val, mock_train, executor, mock_trial):
    """Test a complete successful execution of the trial."""
    mock_train.return_value = 0.4
    mock_val.return_value = {"loss": 0.3, "accuracy": 0.8, "auc": 0.85}

    with patch.object(executor.scheduler, "step"):
        executor.epochs = 2
        best_metric = executor.execute(mock_trial)

    assert best_metric == 0.85
    assert mock_trial.report.call_count == 2
    mock_trial.report.assert_any_call(0.85, 1)


@pytest.mark.integration
@patch("orchard.optimization.objective.TrialTrainingExecutor._train_epoch")
@patch("orchard.optimization.objective.TrialTrainingExecutor._validate_epoch")
def test_execute_pruning_raises(mock_val, mock_train, executor, mock_trial):
    """Test that TrialPruned is raised and execution stops."""
    mock_train.return_value = 0.4
    mock_val.return_value = {"loss": 0.3, "auc": 0.5}

    executor.warmup_epochs = 1
    executor.enable_pruning = True
    mock_trial.should_prune.return_value = True

    with pytest.raises(optuna.TrialPruned):
        executor.execute(mock_trial)

    assert mock_train.call_count == 1


# =========================================================================== #
#                    TESTS: Error Handling                                    #
# =========================================================================== #


@pytest.mark.unit
@patch("orchard.optimization.objective.TrialTrainingExecutor._train_epoch")
@patch("orchard.optimization.objective.TrialTrainingExecutor._validate_epoch")
@patch("orchard.core.logger")
def test_execute_invalid_validation_handling(
    mock_logger, mock_val, mock_train, executor, mock_trial
):
    """Test that the executor catches None/non-dict validation results."""
    # Case 1: Validation returns None
    mock_train.return_value = 0.5
    mock_val.return_value = None
    executor.epochs = 1

    assert executor.execute(mock_trial) == 0.0

    # Case 2: Validation returns wrong type
    mock_val.return_value = "not_a_dict"
    assert executor.execute(mock_trial) == 0.0


def calculate_discount(price, discount_rate):
    """
    Calculates the final price after applying a discount rate.

    Args:
        price (float): The original price of the item. Must be non-negative.
        discount_rate (float): The discount percentage (0 to 1).

    Returns:
        float: The discounted price.

    Raises:
        ValueError: If price is negative or discount_rate is outside the [0, 1] range.
    """
    if price < 0:
        raise ValueError("Price cannot be negative.")

    if not (0 <= discount_rate <= 1):
        raise ValueError("Discount rate must be between 0 and 1.")

    return price * (1 - discount_rate)


# --- Unit Tests ---


@pytest.mark.unit
@pytest.mark.parametrize(
    "price, rate, expected",
    [
        (100.0, 0.2, 80.0),  # Standard case
        (50.0, 0.0, 50.0),  # No discount
        (200.0, 1.0, 0.0),  # 100% discount
        (0.0, 0.5, 0.0),  # Zero price
    ],
)
def test_calculate_discount_success(price, rate, expected):
    """Tests valid inputs to ensure correct mathematical output."""
    assert calculate_discount(price, rate) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "price, rate",
    [
        (-10, 0.1),  # Negative price
        (100, -0.1),  # Negative discount
        (100, 1.1),  # Discount > 100%
    ],
)
def test_calculate_discount_errors(price, rate):
    """Tests invalid inputs to ensure appropriate exceptions are raised."""
    with pytest.raises(ValueError):
        calculate_discount(price, rate)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
