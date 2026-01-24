"""
Minimal Test Suite for Training Engine.

Quick tests to cover core training/validation functions and eliminate codecov warnings.
"""

# Standard Imports
from unittest.mock import MagicMock, patch

# Third-Party Imports
import pytest
import torch
import torch.nn as nn

# Internal Imports
from orchard.trainer.engine import mixup_data, train_one_epoch, validate_epoch


# FIXTURES
@pytest.fixture
def simple_model():
    """Simple 2-layer network for testing."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 10),
    )


@pytest.fixture
def simple_loader():
    """Mock dataloader with 2 batches."""
    batch1 = (torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,)))
    batch2 = (torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,)))
    loader = MagicMock()
    loader.__iter__ = MagicMock(return_value=iter([batch1, batch2]))
    loader.__len__ = MagicMock(return_value=2)
    return loader


@pytest.fixture
def criterion():
    """CrossEntropy loss."""
    return nn.CrossEntropyLoss()


@pytest.fixture
def optimizer(simple_model):
    """SGD optimizer."""
    return torch.optim.SGD(simple_model.parameters(), lr=0.01)


# TESTS: train_one_epoch
@pytest.mark.unit
def test_train_one_epoch_basic(simple_model, simple_loader, criterion, optimizer):
    """Test train_one_epoch completes without errors."""
    device = torch.device("cpu")
    loss = train_one_epoch(
        model=simple_model,
        loader=simple_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epoch=1,
        total_epochs=10,
        use_tqdm=False,  # Disable for testing
    )
    assert isinstance(loss, float)
    assert loss > 0


@pytest.mark.unit
def test_train_one_epoch_with_tqdm(simple_model, simple_loader, criterion, optimizer):
    """Test train_one_epoch with tqdm enabled."""
    device = torch.device("cpu")
    loss = train_one_epoch(
        model=simple_model,
        loader=simple_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epoch=1,
        total_epochs=10,
        use_tqdm=True,  # Enable tqdm
    )
    assert isinstance(loss, float)
    assert loss > 0


@pytest.mark.unit
def test_train_one_epoch_with_grad_clip(simple_model, simple_loader, criterion, optimizer):
    """Test train_one_epoch with gradient clipping."""
    device = torch.device("cpu")
    loss = train_one_epoch(
        model=simple_model,
        loader=simple_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        grad_clip=1.0,
        use_tqdm=False,
    )
    assert isinstance(loss, float)


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_train_one_epoch_with_scaler(simple_model, simple_loader, criterion, optimizer):
    """Test train_one_epoch with AMP scaler."""
    device = torch.device("cpu")
    scaler = torch.amp.GradScaler()

    loss = train_one_epoch(
        model=simple_model,
        loader=simple_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scaler=scaler,
        use_tqdm=False,
    )
    assert isinstance(loss, float)


@pytest.mark.unit
def test_train_one_epoch_scaler_grad_clip_branch():
    """Test scaler + grad_clip branch explicitly."""
    from unittest.mock import MagicMock, patch

    # Create all mocks
    mock_model = MagicMock()
    mock_model.train = MagicMock()
    mock_model.parameters = MagicMock(return_value=[torch.randn(10, 10, requires_grad=True)])
    mock_model.return_value = torch.randn(4, 10)

    mock_loader = MagicMock()
    batch = (torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,)))
    mock_loader.__iter__ = MagicMock(return_value=iter([batch]))

    mock_criterion = MagicMock(return_value=torch.tensor(0.5))
    mock_optimizer = MagicMock()

    mock_scaler = MagicMock()
    mock_scaler.scale = MagicMock(return_value=MagicMock(backward=MagicMock()))
    mock_scaler.unscale_ = MagicMock()  # ← Track this
    mock_scaler.step = MagicMock()
    mock_scaler.update = MagicMock()

    device = torch.device("cpu")

    # Patch clip_grad_norm_
    with patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
        loss = train_one_epoch(
            model=mock_model,
            loader=mock_loader,
            criterion=mock_criterion,
            optimizer=mock_optimizer,
            device=device,
            scaler=mock_scaler,
            grad_clip=1.0,
            use_tqdm=False,
        )

        # Verify the scaler.unscale_ was called before clipping
        mock_scaler.unscale_.assert_called_once_with(mock_optimizer)

        # Verify clip_grad_norm_ was called
        mock_clip.assert_called_once()


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_train_one_epoch_with_scaler_and_grad_clip(
    simple_model, simple_loader, criterion, optimizer
):
    """Test train_one_epoch with AMP scaler AND gradient clipping."""
    device = torch.device("cpu")
    scaler = torch.amp.GradScaler()

    loss = train_one_epoch(
        model=simple_model,
        loader=simple_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scaler=scaler,
        grad_clip=1.0,
        use_tqdm=False,
    )
    assert isinstance(loss, float)


@pytest.mark.unit
def test_train_one_epoch_with_mixup(simple_model, simple_loader, criterion, optimizer):
    """Test train_one_epoch with MixUp."""
    device = torch.device("cpu")

    def mock_mixup(x, y):
        return x, y, y, 0.5

    loss = train_one_epoch(
        model=simple_model,
        loader=simple_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        mixup_fn=mock_mixup,
        use_tqdm=False,
    )
    assert isinstance(loss, float)


@pytest.mark.unit
def test_train_one_epoch_updates_tqdm_postfix(simple_model, simple_loader, criterion, optimizer):
    """Test that tqdm postfix is updated with loss."""
    device = torch.device("cpu")

    with patch("orchard.trainer.engine.tqdm") as mock_tqdm:
        mock_iterator = MagicMock()
        mock_tqdm.return_value = mock_iterator

        # Make the iterator behave like the loader
        batch1 = (torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,)))
        batch2 = (torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,)))
        mock_iterator.__iter__ = MagicMock(return_value=iter([batch1, batch2]))

        loss = train_one_epoch(
            model=simple_model,
            loader=simple_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_tqdm=True,  # Enable tqdm
        )

        # Check that set_postfix was called
        assert mock_iterator.set_postfix.called


# TESTS: VALIDATE EPOCH
@pytest.mark.unit
def test_validate_epoch_basic(simple_model, simple_loader, criterion):
    """Test validate_epoch returns correct metrics."""
    device = torch.device("cpu")
    metrics = validate_epoch(
        model=simple_model,
        val_loader=simple_loader,
        criterion=criterion,
        device=device,
    )

    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "auc" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


@pytest.mark.unit
def test_validate_epoch_binary_classification(simple_model, criterion):
    """Test validate_epoch with binary classification."""
    device = torch.device("cpu")

    # Modify model to output 2 classes (binary)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 2),
    )

    # Create loader with binary labels
    batch = (torch.randn(8, 1, 28, 28), torch.randint(0, 2, (8,)))
    loader = MagicMock()
    loader.__iter__ = MagicMock(return_value=iter([batch]))

    metrics = validate_epoch(
        model=model,
        val_loader=loader,
        criterion=criterion,
        device=device,
    )

    assert isinstance(metrics, dict)
    assert "auc" in metrics


@pytest.mark.unit
def test_validate_epoch_auc_error_handling(simple_model, criterion):
    """Test validate_epoch handles AUC calculation errors."""
    device = torch.device("cpu")

    # Create loader with constant predictions (causes AUC error)
    batch = (torch.randn(4, 1, 28, 28), torch.zeros(4, dtype=torch.long))
    loader = MagicMock()
    loader.__iter__ = MagicMock(return_value=iter([batch]))

    metrics = validate_epoch(
        model=simple_model,
        val_loader=loader,
        criterion=criterion,
        device=device,
    )

    # Should not crash, AUC should be 0.0
    assert metrics["auc"] >= 0.0


# TESTS: MIXUP DATA
@pytest.mark.unit
def test_mixup_data_basic():
    """Test mixup_data creates proper blends."""
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)

    assert mixed_x.shape == x.shape
    assert y_a.shape == y.shape
    assert y_b.shape == y.shape
    assert 0.0 <= lam <= 1.0


@pytest.mark.unit
def test_mixup_data_disabled():
    """Test mixup_data with alpha=0 returns original data."""
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.0)

    assert torch.equal(mixed_x, x)
    assert torch.equal(y_a, y)
    assert lam == 1.0


@pytest.mark.unit
def test_mixup_data_cuda_aware():
    """Test mixup_data handles CUDA tensors if available."""
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()

    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)

    assert mixed_x.device == x.device
    assert y_a.device == y.device


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
