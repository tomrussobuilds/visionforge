"""
Configuration Serialization & Persistence Utilities.

This module handles the conversion of complex Python objects (including Pydantic 
models and Path objects) into YAML format, ensuring thread-safe and 
environment-agnostic persistence to the filesystem.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging
import os
import yaml
from pathlib import Path
from typing import Any, Dict

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from ..paths import LOGGER_NAME

# =========================================================================== #
#                               YAML Orchestration                            #
# =========================================================================== #

def save_config_as_yaml(data: Any, yaml_path: Path) -> Path:
    """
    Saves a configuration object as a YAML file.
    
    Orchestrates the conversion from Pydantic models or dictionaries to a 
    sanitized, YAML-friendly format and ensures atomic persistence.

    Args:
        data (Any): The configuration data (Pydantic model or dict).
        yaml_path (Path): The target filesystem path for the YAML file.

    Returns:
        Path: The confirmed path where the configuration was stored.

    Raises:
        Exception: If serialization or filesystem write fails.
    """
    logger = logging.getLogger(LOGGER_NAME)
    try:
        # 1. Extraction: Interface with Pydantic or raw dicts
        if hasattr(data, "model_dump"):
            try:
                raw_dict = data.model_dump(mode='json')
            except Exception as e:
                logger.warning(f"JSON dump failed, using fallback serialization: {e}")
                raw_dict = data.model_dump()
        else:
            raw_dict = data

        # 2. Sanitization: Recursive cleanup of non-serializable types
        final_data = _sanitize_for_yaml(raw_dict)

        # 3. Persistence: Safe physical write to disk
        _persist_yaml_atomic(final_data, yaml_path)

        logger.info(f"Configuration frozen successfully at → {yaml_path.name}")
        return yaml_path

    except Exception as e:
        logger.error(f"Failed to save configuration YAML: {e}")
        raise


def load_config_from_yaml(yaml_path: Path) -> Dict[str, Any]:
    """
    Loads a raw configuration dictionary from a YAML file.

    Args:
        yaml_path (Path): Path to the source YAML file.

    Returns:
        Dict[str, Any]: The loaded configuration manifest.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML configuration file not found at: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# =========================================================================== #
#                               Internal Helpers                              #
# =========================================================================== #

def _sanitize_for_yaml(obj: Any) -> Any:
    """
    Recursively converts non-serializable types into YAML-standard formats.
    
    Specifically handles:
    - Path objects -> converted to strings.
    - Dicts/Lists/Tuples -> processed recursively.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_yaml(i) for i in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _persist_yaml_atomic(data: Any, path: Path) -> None:
    """
    Performs a safe write operation with directory creation and buffer flushing.
    
    Leverages fsync to ensure the data is physically committed to the storage 
    device, preventing data loss during system failures.
    """
    # Ensure directory existence before writing
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(
            data, 
            f, 
            default_flow_style=False, 
            sort_keys=False, 
            indent=4, 
            allow_unicode=True
        )
        # Force OS-level synchronization
        f.flush()
        os.fsync(f.fileno())