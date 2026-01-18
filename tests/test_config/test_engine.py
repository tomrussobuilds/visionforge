"""
Test Suite for Config Engine.

Tests main Config class integration, cross-validation,
YAML hydration, and from_args factory.
"""
# =========================================================================== #
#                         Standard Imports                                    #
# =========================================================================== #
import argparse

# =========================================================================== #
#                         Third-Party Imports                                 #
# =========================================================================== #
import pytest
from pydantic import ValidationError

# =========================================================================== #
#                         Internal Imports                                    #
# =========================================================================== #
from orchard.core.config import Config

# =========================================================================== #
#                    CONFIG: BASIC CONSTRUCTION                               #
# =========================================================================== #

@pytest.mark.unit
def test_config_defaults():
    """Test Config with all default sub-configs."""
    config = Config()
    
    # Sub-configs should be instantiated
    assert config.hardware is not None
    assert config.training is not None
    assert config.dataset is not None
    assert config.model is not None

@pytest.mark.unit
def test_config_from_args_basic(basic_args):
    """Test Config.from_args() with basic arguments."""
    config = Config.from_args(basic_args)
    
    assert config.dataset.dataset_name == "bloodmnist"
    assert config.model.name == "resnet_18_adapted"
    assert config.training.epochs == 60


# =========================================================================== #
#                    CONFIG: CROSS-VALIDATION                                 #
# =========================================================================== #

@pytest.mark.unit
def test_resnet_18_requires_resolution_28():
    """Test resnet_18_adapted validation enforces resolution=28."""
    args = argparse.Namespace(
        dataset="bloodmnist",
        model_name="resnet_18_adapted",
        resolution=224, # Wrong
        pretrained=True
    )
    
    with pytest.raises(ValidationError, match="resnet_18_adapted requires resolution=28"):
        Config.from_args(args)

@pytest.mark.unit
def test_amp_requires_gpu():
    """Test AMP validation rejects CPU + AMP."""
    args = argparse.Namespace(
        dataset="bloodmnist",
        device="cpu",
        use_amp=True,  # Invalid with CPU
        pretrained=True
    )
    
    with pytest.raises(ValidationError, match="AMP requires GPU"):
        Config.from_args(args)

@pytest.mark.unit
def test_pretrained_requires_rgb():
    """Test pretrained model validation enforces RGB channels."""
    args = argparse.Namespace(
        dataset="organcmnist",  # Grayscale
        model_name="resnet_18_adapted",
        pretrained=True,
        force_rgb=False,  # This will cause validation error
        resolution=28
    )
    
    with pytest.raises(ValidationError, match="Pretrained.*requires RGB"):
        Config.from_args(args)

@pytest.mark.unit
def test_min_lr_less_than_lr_validation():
    """Test min_lr < learning_rate validation."""
    args = argparse.Namespace(
        dataset="bloodmnist",
        learning_rate=0.001,
        min_lr=0.01,  # Greater than LR!
        pretrained=True
    )
    
    with pytest.raises(ValidationError, match="min_lr.*must be"):
        Config.from_args(args)


# =========================================================================== #
#                    CONFIG: YAML HYDRATION                                   #
# =========================================================================== #

@pytest.mark.integration
def test_from_yaml_loads_correctly(temp_yaml_config, mock_metadata_28):
    """Test Config.from_yaml() loads YAML correctly."""
    config = Config.from_yaml(temp_yaml_config, metadata=mock_metadata_28)
    
    assert config.dataset.dataset_name == "bloodmnist"
    assert config.model.name == "resnet_18_adapted"
    assert config.training.epochs == 60
    assert config.training.batch_size == 128

@pytest.mark.integration
def test_yaml_optuna_section_loaded(temp_yaml_config, mock_metadata_28):
    """Test YAML with optuna section loads OptunaConfig."""
    config = Config.from_yaml(temp_yaml_config, metadata=mock_metadata_28)
    
    # Optuna section should be loaded
    assert config.optuna is not None
    assert config.optuna.study_name == "yaml_test_study"
    assert config.optuna.n_trials == 20

@pytest.mark.integration
def test_yaml_precedence_over_args(temp_yaml_config, mock_metadata_28):
    """Test YAML values override CLI arguments."""
    args = argparse.Namespace(
        config=str(temp_yaml_config),
        epochs=999,  # Should be ignored
        batch_size=999,  # Should be ignored
        dataset="bloodmnist",
        pretrained=True
    )
    
    config = Config.from_args(args)
    
    # YAML values should take precedence
    assert config.training.epochs == 60  # From YAML
    assert config.training.batch_size == 128  # From YAML


# =========================================================================== #
#                    CONFIG: SERIALIZATION                                    #
# =========================================================================== #

@pytest.mark.unit
def test_dump_portable_converts_paths():
    """Test dump_portable() makes paths relative."""
    config = Config()
    
    portable = config.dump_portable()
    
    # Paths should be relative or portable
    assert "dataset" in portable
    assert "telemetry" in portable

@pytest.mark.unit
def test_dump_serialized_json_compatible():
    """Test dump_serialized() produces JSON-compatible dict."""
    config = Config()
    
    serialized = config.dump_serialized()
    
    # Should be dict with all sub-configs
    assert isinstance(serialized, dict)
    assert "hardware" in serialized
    assert "training" in serialized


# =========================================================================== #
#                    CONFIG: PROPERTIES                                       #
# =========================================================================== #

@pytest.mark.unit
def test_run_slug_property():
    """Test run_slug combines dataset and model names."""
    config = Config()
    
    slug = config.run_slug
    
    assert "bloodmnist" in slug
    assert config.model.name in slug

@pytest.mark.unit
def test_num_workers_property():
    """Test num_workers delegates to hardware config."""
    config = Config()
    
    workers = config.num_workers

    assert workers >= 0
    assert workers == config.hardware.effective_num_workers


# =========================================================================== #
#                    CONFIG: EDGE CASES                                       #
# =========================================================================== #

@pytest.mark.unit
def test_frozen_immutability():
    """Test Config is frozen (immutable)."""
    config = Config()
    
    with pytest.raises(ValidationError):
        config.training = None

@pytest.mark.integration
def test_invalid_yaml_raises_error(temp_invalid_yaml):
    """Test invalid YAML raises validation error."""
    # This YAML has min_lr > learning_rate
    with pytest.raises(ValidationError):
        args = argparse.Namespace(
            config=str(temp_invalid_yaml),
            dataset="bloodmnist",
            pretrained=True
        )
        Config.from_args(args)