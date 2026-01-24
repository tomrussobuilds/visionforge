"""
Unit tests for Optuna factory functions (samplers, pruners, callbacks).

Tests the builder functions that construct Optuna components from configuration
strings, ensuring proper error handling and component instantiation.
"""

# Third-Party Imports
import pytest
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner, PercentilePruner
from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler

# Internal Imports
from orchard.optimization.orchestrator.builders import build_callbacks, build_pruner, build_sampler


@pytest.mark.unit
def test_build_sampler_tpe(mock_config):
    """Test building TPE sampler."""
    mock_config.optuna.sampler_type = "tpe"
    sampler = build_sampler("tpe", mock_config)
    assert isinstance(sampler, TPESampler)


@pytest.mark.unit
def test_build_sampler_random(mock_config):
    """Test building random sampler."""
    mock_config.optuna.sampler_type = "random"
    sampler = build_sampler("random", mock_config)
    assert isinstance(sampler, RandomSampler)


@pytest.mark.unit
def test_build_sampler_cmaes(mock_config):
    """Test building CMA-ES sampler."""
    mock_config.optuna.sampler_type = "cmaes"
    sampler = build_sampler("cmaes", mock_config)
    assert isinstance(sampler, CmaEsSampler)


@pytest.mark.unit
def test_build_sampler_invalid(mock_config):
    """Test building sampler with invalid type raises ValueError."""
    mock_config.optuna.sampler_type = "invalid_sampler"

    with pytest.raises(ValueError) as exc_info:
        build_sampler("invalid_sampler", mock_config)

    assert "Unknown sampler: invalid_sampler" in str(exc_info.value)
    assert "Valid options:" in str(exc_info.value)


@pytest.mark.unit
def test_build_pruner_disabled(mock_config):
    """Test that pruner returns NopPruner when pruning is disabled."""
    mock_config.optuna.enable_pruning = False
    mock_config.optuna.pruner_type = "median"

    pruner = build_pruner(False, "median", mock_config)
    assert isinstance(pruner, NopPruner)


@pytest.mark.unit
def test_build_pruner_median(mock_config):
    """Test building median pruner."""
    mock_config.optuna.enable_pruning = True
    mock_config.optuna.pruner_type = "median"

    pruner = build_pruner(True, "median", mock_config)
    assert isinstance(pruner, MedianPruner)


@pytest.mark.unit
def test_build_pruner_percentile(mock_config):
    """Test building percentile pruner."""
    mock_config.optuna.enable_pruning = True
    mock_config.optuna.pruner_type = "percentile"

    pruner = build_pruner(True, "percentile", mock_config)
    assert isinstance(pruner, PercentilePruner)


@pytest.mark.unit
def test_build_pruner_hyperband(mock_config):
    """Test building hyperband pruner."""
    mock_config.optuna.enable_pruning = True
    mock_config.optuna.pruner_type = "hyperband"

    pruner = build_pruner(True, "hyperband", mock_config)
    assert isinstance(pruner, HyperbandPruner)


@pytest.mark.unit
def test_build_pruner_invalid(mock_config):
    """Test building pruner with invalid type raises ValueError."""
    mock_config.optuna.enable_pruning = True
    mock_config.optuna.pruner_type = "invalid_pruner"

    with pytest.raises(ValueError) as exc_info:
        build_pruner(True, "invalid_pruner", mock_config)

    assert "Unknown pruner: invalid_pruner" in str(exc_info.value)
    assert "Valid options:" in str(exc_info.value)


@pytest.mark.unit
def test_build_callbacks_with_early_stopping(mock_config):
    """Test building callbacks when early stopping is enabled."""
    mock_config.optuna.enable_early_stopping = True
    mock_config.optuna.early_stopping_threshold = 0.95
    mock_config.optuna.early_stopping_patience = 5
    mock_config.optuna.direction = "maximize"
    mock_config.optuna.metric_name = "val_acc"

    callbacks = build_callbacks(mock_config)

    assert len(callbacks) == 1
    assert callbacks[0] is not None


@pytest.mark.unit
def test_build_callbacks_without_early_stopping(mock_config):
    """Test building callbacks when early stopping is disabled."""
    mock_config.optuna.enable_early_stopping = False

    callbacks = build_callbacks(mock_config)

    assert len(callbacks) == 0


# FIXTURE
@pytest.fixture
def mock_config():
    """Provide a mock configuration object for testing."""
    from unittest.mock import MagicMock

    config = MagicMock()
    optuna_mock = MagicMock()
    optuna_mock.sampler_type = "tpe"
    optuna_mock.enable_pruning = True
    optuna_mock.pruner_type = "median"
    optuna_mock.enable_early_stopping = False
    optuna_mock.early_stopping_threshold = 0.95
    optuna_mock.early_stopping_patience = 5
    optuna_mock.direction = "maximize"
    optuna_mock.metric_name = "val_acc"

    # Attach optuna mock to config
    config.optuna = optuna_mock

    return config
