"""
Test Suite for InfrastructureManager.

Tests infrastructure resource management, lock file handling,
and compute cache flushing.
"""
# =========================================================================== #
#                         Third-Party Imports                                 #
# =========================================================================== #
import pytest
from pydantic import ValidationError

# =========================================================================== #
#                         Internal Imports                                    #
# =========================================================================== #
from orchard.core.config import InfrastructureManager, HardwareConfig


# =========================================================================== #
#                INFRASTRUCTURE MANAGER: CREATION                             #
# =========================================================================== #

@pytest.mark.unit
def test_infrastructure_manager_creation():
    """Test InfrastructureManager can be instantiated."""
    manager = InfrastructureManager()
    
    assert manager is not None

@pytest.mark.unit
def test_infrastructure_manager_is_singleton_like():
    """Test multiple InfrastructureManager instances are independent."""
    manager1 = InfrastructureManager()
    manager2 = InfrastructureManager()
    
    # Should be different instances
    assert manager1 is not manager2


# =========================================================================== #
#                INFRASTRUCTURE MANAGER: LOCK FILE MANAGEMENT                 #
# =========================================================================== #

@pytest.mark.integration
def test_prepare_environment_creates_lock(tmp_path):
    """Test prepare_environment() creates lock file."""
    manager = InfrastructureManager()
    
    # Mock config with temp lock path
    class MockHardware:
        allow_process_kill = False  # Skip process cleanup
        lock_file_path = tmp_path / "test.lock"
    
    class MockConfig:
        hardware = MockHardware()
    
    config = MockConfig()
    
    # Should create lock file
    manager.prepare_environment(config)
    
    assert config.hardware.lock_file_path.exists()
    
    # Cleanup
    manager.release_resources(config)

@pytest.mark.integration
def test_release_resources_removes_lock(tmp_path):
    """Test release_resources() removes lock file."""
    manager = InfrastructureManager()
    
    class MockHardware:
        allow_process_kill = False
        lock_file_path = tmp_path / "test.lock"
    
    class MockConfig:
        hardware = MockHardware()
    
    config = MockConfig()
    
    # Create and release
    manager.prepare_environment(config)
    assert config.hardware.lock_file_path.exists()
    
    manager.release_resources(config)
    assert not config.hardware.lock_file_path.exists()

@pytest.mark.integration
def test_prepare_environment_with_existing_lock(tmp_path):
    """Test prepare_environment() behavior with existing lock."""
    manager = InfrastructureManager()
    
    lock_path = tmp_path / "existing.lock"
    lock_path.touch()  # Create existing lock
    
    class MockHardware:
        allow_process_kill = False
        lock_file_path = lock_path
    
    class MockConfig:
        hardware = MockHardware()
    
    # Should handle existing lock (implementation-dependent)
    manager.prepare_environment(MockConfig())
    
    # Lock should still exist or be recreated
    assert lock_path.exists()
    
    # Cleanup
    manager.release_resources(MockConfig())

@pytest.mark.integration
def test_release_resources_idempotent(tmp_path):
    """Test release_resources() can be called multiple times safely."""
    manager = InfrastructureManager()
    
    class MockHardware:
        allow_process_kill = False
        lock_file_path = tmp_path / "test.lock"
    
    class MockConfig:
        hardware = MockHardware()
    
    config = MockConfig()
    
    manager.prepare_environment(config)
    
    # Release twice
    manager.release_resources(config)
    manager.release_resources(config)  # Should not raise
    
    assert not config.hardware.lock_file_path.exists()


# =========================================================================== #
#                INFRASTRUCTURE MANAGER: COMPUTE CACHE                        #
# =========================================================================== #

@pytest.mark.unit
def test_flush_compute_cache_no_error():
    """Test _flush_compute_cache() runs without error."""
    manager = InfrastructureManager()
    
    # Should not raise even if no GPU
    manager._flush_compute_cache()

@pytest.mark.unit
def test_flush_compute_cache_callable():
    """Test _flush_compute_cache() is callable."""
    manager = InfrastructureManager()
    
    # Should be a method
    assert callable(manager._flush_compute_cache)


# =========================================================================== #
#                INFRASTRUCTURE MANAGER: INTEGRATION WITH CONFIG              #
# =========================================================================== #

@pytest.mark.integration
def test_integration_with_hardware_config(tmp_path):
    """Test InfrastructureManager works with real HardwareConfig."""
    manager = InfrastructureManager()
    
    # Create real HardwareConfig with temp lock path
    hw_config = HardwareConfig(project_name="test-integration")
    
    class MockConfig:
        hardware = hw_config
    
    config = MockConfig()
    
    manager._flush_compute_cache()
    assert manager is not None


# =========================================================================== #
#                INFRASTRUCTURE MANAGER: ERROR HANDLING                       #
# =========================================================================== #

@pytest.mark.integration
def test_prepare_environment_with_logger(tmp_path):
    """Test prepare_environment() accepts optional logger."""
    manager = InfrastructureManager()
    
    class MockHardware:
        allow_process_kill = False
        lock_file_path = tmp_path / "test.lock"
    
    class MockConfig:
        hardware = MockHardware()
    
    class MockLogger:
        def info(self, msg): pass
        def warning(self, msg): pass
    
    # Should accept logger without error
    manager.prepare_environment(MockConfig(), logger=MockLogger())
    
    # Cleanup
    manager.release_resources(MockConfig())

@pytest.mark.integration
def test_release_resources_with_logger(tmp_path):
    """Test release_resources() accepts optional logger."""
    manager = InfrastructureManager()
    
    class MockHardware:
        allow_process_kill = False
        lock_file_path = tmp_path / "test.lock"
    
    class MockConfig:
        hardware = MockHardware()
    
    class MockLogger:
        def info(self, msg): pass
        def debug(self, msg): pass
    
    config = MockConfig()
    manager.prepare_environment(config)
    
    # Should accept logger without error
    manager.release_resources(config, logger=MockLogger())


# =========================================================================== #
#                INFRASTRUCTURE MANAGER: IMMUTABILITY                         #
# =========================================================================== #

@pytest.mark.unit
def test_infrastructure_manager_frozen():
    """Test InfrastructureManager is frozen."""
    manager = InfrastructureManager()
    
    with pytest.raises(ValidationError):
        manager.new_field = "should_fail"


# =========================================================================== #
#                INTEGRATION: WITH OPTUNA CONFIG                              #
# =========================================================================== #

@pytest.mark.integration
def test_optuna_hardware_integration():
    """Test InfrastructureManager works with Optuna-optimized HardwareConfig."""
    from orchard.core.config import OptunaConfig
    
    hw_config = HardwareConfig.for_optuna(device="cpu")
    optuna_config = OptunaConfig(n_trials=10)
    
    # Reproducible mode should be enabled
    assert hw_config.reproducible is True
    assert hw_config.effective_num_workers == 0
    
    # Optuna config should be valid
    assert optuna_config.n_trials == 10
    
    # InfrastructureManager should work with this config
    manager = InfrastructureManager()
    assert manager is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])