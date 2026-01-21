"""
Tests suite for RootOrchestrator.
Tests all 7 phases, __enter__, __exit__, and edge cases.
Achieves high coverage through dependency injection and mocking.
"""

# =========================================================================== #
#                         Standard Imports                                    #
# =========================================================================== #
from unittest.mock import MagicMock, patch

# =========================================================================== #
#                         Third-Party Imports                                 #
# =========================================================================== #
import pytest
import torch

# =========================================================================== #
#                         Internal Imports                                    #
# =========================================================================== #
from orchard.core import LOGGER_NAME, RootOrchestrator, RunPaths

# =========================================================================== #
#                    ORCHESTRATOR: INITIALIZATION                             #
# =========================================================================== #


@pytest.mark.unit
def test_orchestrator_init_with_defaults():
    """Test RootOrchestrator initializes with default dependencies."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)

    assert orch.cfg == mock_cfg
    assert orch.repro_mode is False
    assert orch.num_workers == 4
    assert orch.paths is None
    assert orch.run_logger is None


@pytest.mark.unit
def test_orchestrator_init_extracts_policies():
    """Test RootOrchestrator extracts policies from config."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = True
    mock_cfg.hardware.effective_num_workers = 8

    orch = RootOrchestrator(cfg=mock_cfg)

    assert orch.repro_mode is True
    assert orch.num_workers == 8


@pytest.mark.unit
def test_init_lazy_attributes_and_policy_extraction():
    """Test __init__ sets lazy attributes and extracts policy flags."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = True
    mock_cfg.hardware.effective_num_workers = 5

    orch = RootOrchestrator(cfg=mock_cfg)

    # Lazy attributes should be None initially
    assert orch.paths is None
    assert orch.run_logger is None
    assert orch._device_cache is None

    # Policy extraction
    assert orch.repro_mode is True
    assert orch.num_workers == 5


# =========================================================================== #
#                    CONTEXT MANAGER: __ENTER__                               #
# =========================================================================== #


@pytest.mark.unit
def test_context_manager_enter():
    """Test __enter__ calls initialize_core_services."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)
    orch.initialize_core_services = MagicMock(return_value=MagicMock())

    result = orch.__enter__()

    orch.initialize_core_services.assert_called_once()
    assert result == orch


@pytest.mark.unit
def test_context_manager_enter_exception_cleanup():
    """Test __enter__ calls cleanup on exception."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)
    orch.initialize_core_services = MagicMock(side_effect=RuntimeError("Init failed"))
    orch.cleanup = MagicMock()

    with pytest.raises(RuntimeError, match="Init failed"):
        orch.__enter__()

    orch.cleanup.assert_called_once()


# =========================================================================== #
#                    CONTEXT MANAGER: __EXIT__                                #
# =========================================================================== #


@pytest.mark.unit
def test_context_manager_exit_calls_cleanup():
    """Test __exit__ calls cleanup."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)
    orch.cleanup = MagicMock()

    result = orch.__exit__(None, None, None)

    orch.cleanup.assert_called_once()
    assert result is False  # Exception propagation


@pytest.mark.unit
def test_context_manager_exit_propagates_exception():
    """Test __exit__ returns False to propagate exceptions."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)
    orch.cleanup = MagicMock()

    # Simulate exception in with block
    result = orch.__exit__(ValueError, ValueError("test"), None)

    assert result is False  # Allows exception to propagate


# =========================================================================== #
#                    GET DEVICE                                               #
# =========================================================================== #


@pytest.mark.unit
def test_get_device_returns_cpu():
    """Test get_device returns CPU device."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.device = "cpu"
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)

    device = orch.get_device()

    assert device.type == "cpu"


@pytest.mark.unit
def test_get_device_caches_result():
    """Test get_device caches device object."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.device = "cpu"
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)

    # First call
    device1 = orch.get_device()
    # Second call
    device2 = orch.get_device()

    # Should be the same object (cached)
    assert device1 is device2


@pytest.mark.unit
def test_get_device_calls_resolver_when_cache_none():
    """Test get_device calls device resolver when _device_cache is None."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.device = "cpu"
    mock_cfg.hardware.effective_num_workers = 1
    mock_resolver = MagicMock(return_value=torch.device("cpu"))

    orch = RootOrchestrator(cfg=mock_cfg, device_resolver=mock_resolver)
    orch._device_cache = None

    device = orch.get_device()

    mock_resolver.assert_called_once_with(device_str="cpu")
    assert device.type == "cpu"


# =========================================================================== #
#                    CLEANUP                                                  #
# =========================================================================== #


@pytest.mark.unit
def test_cleanup_handles_infra_release_exception(monkeypatch):
    """Test cleanup handles exceptions raised by infra.release_resources."""
    mock_cfg = MagicMock()
    mock_infra = MagicMock()
    mock_infra.release_resources.side_effect = RuntimeError("release failed")
    mock_logger = MagicMock()
    mock_handler = MagicMock()
    mock_logger.handlers = [mock_handler]

    orch = RootOrchestrator(cfg=mock_cfg, infra_manager=mock_infra)
    orch.run_logger = mock_logger

    orch.cleanup()

    mock_handler.close.assert_called_once()
    mock_logger.removeHandler.assert_called_once_with(mock_handler)


@pytest.mark.unit
def test_cleanup_release_resources_fails_no_logger(caplog):
    """Test cleanup logs error if release_resources fails and no logger."""
    import logging

    mock_cfg = MagicMock()
    mock_infra = MagicMock()
    mock_infra.release_resources.side_effect = RuntimeError("fail")

    orch = RootOrchestrator(cfg=mock_cfg)
    orch.infra = mock_infra
    orch.run_logger = None

    with caplog.at_level(logging.ERROR):
        orch.cleanup()

    assert any("fail" in rec.message for rec in caplog.records)


# =========================================================================== #
#                    ORCHESTRATOR: PHASES 1-7                                 #
# =========================================================================== #


@pytest.mark.unit
def test_phase_1_determinism_always_calls_seed_setter():
    """Test _phase_1_determinism always calls seed setter."""
    mock_cfg = MagicMock()
    mock_cfg.training.seed = 123
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_seed_setter = MagicMock()
    orch = RootOrchestrator(cfg=mock_cfg, seed_setter=mock_seed_setter)

    orch._phase_1_determinism()
    mock_seed_setter.assert_called_once_with(123, strict=False)


@pytest.mark.unit
def test_phase_2_hardware_optimization_applies_threads_and_system():
    mock_cfg = MagicMock()
    mock_cfg.hardware.effective_num_workers = 4
    mock_thread_applier = MagicMock(return_value=7)
    mock_system_configurator = MagicMock()

    orch = RootOrchestrator(
        cfg=mock_cfg,
        thread_applier=mock_thread_applier,
        system_configurator=mock_system_configurator,
    )
    threads = orch._phase_2_hardware_optimization()
    assert threads == 7
    mock_thread_applier.assert_called_once_with(4)
    mock_system_configurator.assert_called_once()


@pytest.mark.unit
def test_phase_3_filesystem_provisioning_calls_static_setup_and_runpaths():
    import orchard.core.orchestrator as orch_module

    orig_create = orch_module.RunPaths.create
    orch_module.RunPaths.create = MagicMock(return_value="runpaths_obj")

    mock_cfg = MagicMock()
    mock_cfg.dataset.dataset_name = "ds"
    mock_cfg.model.name = "model"
    mock_cfg.telemetry.output_dir = "/tmp/out"
    mock_cfg.dump_serialized = MagicMock(return_value={"some": "data"})
    mock_static_setup = MagicMock()

    orch = RootOrchestrator(cfg=mock_cfg, static_dir_setup=mock_static_setup)
    orch._phase_3_filesystem_provisioning()

    mock_static_setup.assert_called_once()
    orch_module.RunPaths.create.assert_called_once_with(
        dataset_slug="ds",
        model_name="model",
        training_cfg={"some": "data"},
        base_dir="/tmp/out",
    )
    assert orch.paths == "runpaths_obj"
    orch_module.RunPaths.create = orig_create


@pytest.mark.unit
def test_phase_4_logging_initialization_sets_logger():
    mock_cfg = MagicMock()
    mock_cfg.telemetry.log_level = "INFO"
    orch = RootOrchestrator(cfg=mock_cfg)
    orch.paths = MagicMock()
    orch.paths.logs = "/tmp/logs"
    mock_log_initializer = MagicMock(return_value="logger_obj")
    orch._log_initializer = mock_log_initializer

    orch._phase_4_logging_initialization()

    mock_log_initializer.assert_called_once_with(
        name=LOGGER_NAME,
        log_dir="/tmp/logs",
        level="INFO",
    )
    assert orch.run_logger == "logger_obj"


@pytest.mark.unit
def test_phase_5_config_persistence_saves_config():
    mock_cfg = MagicMock()
    orch = RootOrchestrator(cfg=mock_cfg)
    orch.paths = MagicMock()
    orch.paths.get_config_path = MagicMock(return_value="/tmp/config.yaml")
    mock_saver = MagicMock()
    orch._config_saver = mock_saver

    orch._phase_5_config_persistence()
    mock_saver.assert_called_once_with(data=mock_cfg, yaml_path="/tmp/config.yaml")


@pytest.mark.unit
def test_phase_6_infra_prepare_raises_warning_no_logger(caplog):
    """Test _phase_6_infrastructure_guarding logs warning to logging if no logger."""
    import logging

    mock_cfg = MagicMock()
    mock_infra = MagicMock()
    mock_infra.prepare_environment.side_effect = RuntimeError("fail")

    orch = RootOrchestrator(cfg=mock_cfg)
    orch.infra = mock_infra
    orch.run_logger = None

    with caplog.at_level(logging.WARNING):
        orch._phase_6_infrastructure_guarding()

    assert any("fail" in rec.message for rec in caplog.records)


@pytest.mark.unit
def test_phase_7_device_resolver_fails_fallback_to_cpu(caplog):
    """Test _phase_7_environment_reporting falls back to CPU if device resolver fails."""
    import logging

    mock_cfg = MagicMock()
    mock_cfg.hardware.effective_num_workers = 2
    mock_reporter = MagicMock()

    def failing_resolver(device_str):
        raise RuntimeError("device fail")

    orch = RootOrchestrator(cfg=mock_cfg, reporter=mock_reporter, device_resolver=failing_resolver)
    orch._device_cache = None
    orch.run_logger = None
    orch.paths = MagicMock()

    with caplog.at_level(logging.WARNING):
        orch._phase_7_environment_reporting(applied_threads=1)

    assert orch._device_cache.type == "cpu"
    assert any("fallback to CPU" in rec.message for rec in caplog.records)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
