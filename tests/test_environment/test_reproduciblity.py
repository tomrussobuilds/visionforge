"""
Test Suite for Reproducibility Utilities.

Covers deterministic seeding, reproducibility mode detection,
and DataLoader worker initialization logic.
"""

# Standard Imports
import os
import random
from unittest.mock import MagicMock, patch

# Third-Party Imports
import numpy as np
import pytest
import torch

# Internal Imports
from orchard.core import is_repro_mode_requested, set_seed, worker_init_fn


# TESTS: MODE DETECTION
@pytest.mark.unit
def test_is_repro_mode_requested_cli_flag():
    """CLI flag alone enables reproducibility mode."""
    assert is_repro_mode_requested(cli_flag=True) is True


@pytest.mark.unit
def test_is_repro_mode_requested_env_var(monkeypatch):
    """Environment variable enables reproducibility mode."""
    monkeypatch.setenv("DOCKER_REPRODUCIBILITY_MODE", "TRUE")
    assert is_repro_mode_requested(cli_flag=False) is True


@pytest.mark.unit
def test_is_repro_mode_requested_disabled(monkeypatch):
    """No flags -> reproducibility disabled."""
    monkeypatch.delenv("DOCKER_REPRODUCIBILITY_MODE", raising=False)
    assert is_repro_mode_requested(cli_flag=False) is False


# TESTS: set_seed
@pytest.mark.unit
def test_set_seed_reproducibility_cpu():
    """set_seed enforces deterministic CPU behavior."""
    set_seed(123)

    a1 = random.random()
    b1 = np.random.rand()
    c1 = torch.rand(1)

    set_seed(123)

    a2 = random.random()
    b2 = np.random.rand()
    c2 = torch.rand(1)

    assert a1 == a2
    assert b1 == b2
    assert torch.equal(c1, c2)


@pytest.mark.unit
def test_set_seed_sets_python_hashseed():
    """PYTHONHASHSEED is correctly set."""
    set_seed(999)
    assert os.environ["PYTHONHASHSEED"] == "999"


@pytest.mark.unit
def test_set_seed_strict_mode_with_cuda_available():
    """Strict mode enables deterministic PyTorch behavior when CUDA is available."""
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.use_deterministic_algorithms") as mock_deterministic:
            set_seed(42, strict=True)
            mock_deterministic.assert_called_once_with(True)

            assert torch.backends.cudnn.deterministic is True
            assert torch.backends.cudnn.benchmark is False


@pytest.mark.unit
def test_set_seed_non_strict_mode_with_cuda_available():
    """Non-strict mode sets cudnn flags when CUDA is available."""
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.use_deterministic_algorithms") as mock_deterministic:
            set_seed(42, strict=False)
            mock_deterministic.assert_not_called()

            assert torch.backends.cudnn.deterministic is True
            assert torch.backends.cudnn.benchmark is False


@pytest.mark.unit
def test_set_seed_without_cuda():
    """set_seed works correctly when CUDA is not available."""
    with patch("torch.cuda.is_available", return_value=False):
        set_seed(42, strict=True)
        set_seed(42, strict=False)


@pytest.mark.unit
def test_set_seed_cuda_branches_coverage():
    """Ensures all CUDA-related branches are executed in tests."""
    with patch("torch.cuda.is_available", return_value=True):
        mock_cudnn = MagicMock()
        with patch("torch.backends.cudnn", mock_cudnn):
            set_seed(42, strict=True)
            assert mock_cudnn.deterministic is True
            assert mock_cudnn.benchmark is False

    with patch("torch.cuda.is_available", return_value=False):
        set_seed(42, strict=True)
        set_seed(42, strict=False)


# TESTS: worker_init_fn
@pytest.mark.unit
def test_worker_init_fn_no_worker_info(monkeypatch):
    """worker_init_fn is a no-op outside DataLoader workers."""
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: None)

    worker_init_fn(worker_id=0)


@pytest.mark.unit
def test_worker_init_fn_sets_deterministic_state(monkeypatch):
    """worker_init_fn initializes RNGs deterministically per worker."""

    class DummyWorkerInfo:
        seed = 1000

    monkeypatch.setattr(
        torch.utils.data,
        "get_worker_info",
        lambda: DummyWorkerInfo(),
    )

    worker_init_fn(worker_id=1)

    a1 = random.random()
    b1 = np.random.rand()
    c1 = torch.rand(1)

    worker_init_fn(worker_id=1)

    a2 = random.random()
    b2 = np.random.rand()
    c2 = torch.rand(1)

    assert a1 == a2
    assert b1 == b2
    assert torch.equal(c1, c2)


# INTEGRATION TESTS
@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_set_seed_strict_mode_real_cuda():
    """Integration test: strict mode with real CUDA hardware."""
    set_seed(42, strict=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_set_seed_non_strict_mode_real_cuda():
    """Integration test: non-strict mode with real CUDA hardware."""
    set_seed(42, strict=False)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
