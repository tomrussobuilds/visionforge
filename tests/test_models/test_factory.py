"""
Smoke Tests for Models Factory Module.

Quick coverage tests to validate factory pattern and model instantiation.
These are minimal tests to boost coverage from 0% to ~20%.
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from orchard.models.factory import get_model


# FACTORY: BASIC INSTANTIATION
@pytest.mark.unit
def test_get_model_returns_nn_module():
    """Test get_model returns a torch.nn.Module instance."""
    device = torch.device("cpu")
    mock_cfg = MagicMock()
    mock_cfg.architecture.name = "mini_cnn"
    mock_cfg.architecture.dropout = 0.0
    mock_cfg.dataset.effective_in_channels = 3
    mock_cfg.dataset.num_classes = 10
    mock_cfg.dataset.img_size = 28

    model = get_model(device=device, cfg=mock_cfg)

    assert isinstance(model, nn.Module)


@pytest.mark.unit
def test_get_model_deploys_to_device():
    """Test get_model deploys model to specified device."""
    device = torch.device("cpu")
    mock_cfg = MagicMock()
    mock_cfg.architecture.name = "mini_cnn"
    mock_cfg.architecture.dropout = 0.0
    mock_cfg.dataset.effective_in_channels = 3
    mock_cfg.dataset.num_classes = 8
    mock_cfg.dataset.img_size = 28

    model = get_model(device=device, cfg=mock_cfg)

    assert next(model.parameters()).device.type == device.type


@pytest.mark.unit
def test_get_model_invalid_architecture():
    """Test get_model raises ValueError for unknown architecture."""
    device = torch.device("cpu")
    mock_cfg = MagicMock()
    mock_cfg.architecture.name = "invalid_model_xyz"
    mock_cfg.dataset.effective_in_channels = 3
    mock_cfg.dataset.num_classes = 10
    mock_cfg.dataset.img_size = 28

    with pytest.raises(ValueError, match="not registered"):
        get_model(device=device, cfg=mock_cfg)


@pytest.mark.unit
def test_get_model_case_insensitive():
    """Test get_model handles case-insensitive model names."""
    device = torch.device("cpu")
    mock_cfg = MagicMock()
    mock_cfg.architecture.name = "MINI_CNN"
    mock_cfg.architecture.dropout = 0.0
    mock_cfg.dataset.effective_in_channels = 3
    mock_cfg.dataset.num_classes = 10
    mock_cfg.dataset.img_size = 28

    model = get_model(device=device, cfg=mock_cfg)

    assert isinstance(model, nn.Module)


# FACTORY: REGISTRY VALIDATION
@pytest.mark.unit
@pytest.mark.parametrize(
    "model_name",
    ["mini_cnn", "resnet_18_adapted", "efficientnet_b0", "vit_tiny"],
)
def test_get_model_all_registered_models(model_name):
    """Test get_model can instantiate all registered models."""
    device = torch.device("cpu")
    mock_cfg = MagicMock()
    mock_cfg.architecture.name = model_name
    mock_cfg.architecture.dropout = 0.0
    mock_cfg.dataset.effective_in_channels = 3
    mock_cfg.dataset.num_classes = 10
    mock_cfg.dataset.img_size = 224 if model_name == "vit_tiny" else 28
    mock_cfg.architecture.pretrained = False
    mock_cfg.architecture.weight_variant = None

    model = get_model(device=device, cfg=mock_cfg)

    assert isinstance(model, nn.Module)


@pytest.mark.unit
def test_get_model_verbose_false_suppresses_logs(caplog):
    """Test verbose=False suppresses factory and builder logs."""
    device = torch.device("cpu")
    mock_cfg = MagicMock()
    mock_cfg.architecture.name = "mini_cnn"
    mock_cfg.architecture.dropout = 0.0
    mock_cfg.dataset.effective_in_channels = 3
    mock_cfg.dataset.num_classes = 10
    mock_cfg.dataset.img_size = 28

    with caplog.at_level("INFO"):
        model = get_model(device=device, cfg=mock_cfg, verbose=False)

    assert isinstance(model, nn.Module)
    assert "Initializing Architecture" not in caplog.text
    assert "Model deployed" not in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
