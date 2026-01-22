"""
Test Suite for Exporting Configurations and Study Information.

This module contains test cases for verifying the export functionality of various
configurations such as best config, study summary, and top trials. It ensures
correct behavior of configuration serialization and export to formats such as YAML, JSON, and Excel.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import json
from unittest.mock import MagicMock

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import optuna
import pandas as pd
import pytest
from pydantic import ValidationError

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from orchard.core import Config, RunPaths
from orchard.optimization import export_best_config, export_study_summary, export_top_trials

# =========================================================================== #
#                                Fixtures                                     #
# =========================================================================== #


@pytest.fixture
def study():
    """Fixture for creating a mock Optuna Study object."""
    study = MagicMock(spec=optuna.Study)
    trial_mock = MagicMock(spec=optuna.trial.FrozenTrial)
    trial_mock.state = optuna.trial.TrialState.COMPLETE
    trial_mock.params = {"param1": 0.1, "param2": 0.2}
    trial_mock.value = 0.8
    trial_mock.number = 1
    trial_mock.datetime_start = None
    trial_mock.datetime_complete = None

    study.best_trial = trial_mock
    study.best_params = trial_mock.params
    study.trials = [trial_mock]  # Add the mock trial to the study
    study.study_name = "test_study"
    study.direction = optuna.study.StudyDirection.MAXIMIZE
    return study


@pytest.fixture
def paths(tmpdir):
    """Fixture for creating a mock RunPaths object."""
    paths = MagicMock(spec=RunPaths)
    paths.reports = tmpdir.mkdir("reports")
    return paths


@pytest.fixture
def config():
    """Fixture for creating a valid Config object with ModelConfig and TrainingConfig."""
    model_config = {
        "name": "resnet_18_adapted",
        "pretrained": True,
        "dropout": 0.2,
        "weight_variant": None,
    }

    training_config = {"epochs": 10, "mixup_epochs": 5}

    return Config(model=model_config, training=training_config)


@pytest.mark.unit
def test_export_study_summary(study, paths):
    """Test export of study summary to JSON."""
    export_study_summary(study, paths, metric_name="accuracy")

    output_path = paths.reports / "study_summary.json"
    assert output_path.exists()

    with open(output_path, "r") as f:
        summary = json.load(f)
        assert "study_name" in summary
        assert "direction" in summary
        assert "trials" in summary
        assert len(summary["trials"]) == 1
        assert "best_trial" in summary


@pytest.mark.unit
def test_export_top_trials(study, paths):
    """Test export of top trials to Excel."""
    export_top_trials(study, paths, metric_name="accuracy", top_k=1)

    output_path = paths.reports / "top_10_trials.xlsx"
    assert output_path.exists()

    df = pd.read_excel(output_path)
    assert "Rank" in df.columns
    assert "Trial" in df.columns
    assert "ACCURACY" in df.columns  # Updated for case sensitivity


@pytest.mark.unit
def test_export_best_config_invalid_config(study, paths):
    """Test handling of invalid config during export."""

    # Test invalid model: Pass a string instead of a ModelConfig
    invalid_model_config = "invalid_model"  # Invalid config

    # Test invalid epochs: Set epochs to a negative number
    invalid_training_config = {"epochs": -10}  # Negative epoch value

    with pytest.raises(ValidationError):
        # Trying to create a Config instance with invalid model and epochs
        invalid_config = Config(model=invalid_model_config, training=invalid_training_config)
        export_best_config(study, invalid_config, paths)

    output_path = paths.reports / "best_config.yaml"
    assert not output_path.exists()


@pytest.mark.unit
def test_export_study_summary_no_completed_trials(study, paths):
    """Test export when no completed trials exist."""
    study.trials = []
    export_study_summary(study, paths, metric_name="accuracy")

    output_path = paths.reports / "study_summary.json"
    assert output_path.exists()

    with open(output_path, "r") as f:
        summary = json.load(f)
        assert "n_completed" in summary
        assert summary["n_completed"] == 0
        assert "trials" in summary
        assert len(summary["trials"]) == 0


@pytest.mark.unit
def test_export_top_trials_no_completed_trials(study, paths):
    """Test export when no completed trials exist."""
    study.trials = []
    export_top_trials(study, paths, metric_name="accuracy", top_k=1)

    output_path = paths.reports / "top_10_trials.xlsx"
    assert not output_path.exists()
