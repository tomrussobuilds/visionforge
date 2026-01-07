"""
Models Factory Module.

This module implements the Factory Pattern to decouple model instantiation 
from the main execution logic. It synchronizes architectural intent with 
geometric constraints derived from dataset metadata at runtime.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
import torch.nn as nn

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import Config, LOGGER_NAME
from .resnet_18_adapted import build_resnet18_adapted

# =========================================================================== #
#                                MODEL FACTORY LOGIC                          #
# =========================================================================== #

# Global logger instance
logger = logging.getLogger(LOGGER_NAME)

def get_model(
    device: torch.device,
    cfg: Config
) -> nn.Module:
    """
    Factory function to instantiate and prepare the model.

    Resolves structural parameters (input channels and output classes) 
    by accessing the computed properties of DatasetConfig. This ensures 
    the architecture is perfectly adapted to the 'effective' geometry 
    of the data (e.g., handling RGB promotion) before deployment.

    Args:
        device (torch.device): The hardware device (CPU/CUDA) to host the model.
        cfg (Config): The global configuration object, already hydrated with 
            metadata and resolved channel logic.

    Returns:
        nn.Module: The instantiated and hardware-assigned PyTorch model.

    Raises:
        ValueError: If the requested model_name is not registered in the factory.
    """
    
    # 1. Resolve architectural geometry from the SSOT (Single Source of Truth)
    # We no longer call cfg.model.get_structural_params() because the 
    # DatasetConfig already knows the final 'effective' channels.
    in_channels = cfg.dataset.effective_in_channels
    num_classes = cfg.dataset.num_classes
    
    model_name_lower = cfg.model.name.lower()

    logger.info(
        f"Initializing Architecture: {cfg.model.name} | "
        f"Input: {cfg.dataset.img_size}x{cfg.dataset.img_size}x{in_channels} | "
        f"Output: {num_classes} classes"
    )

    # 2. Routing logic (Factory Pattern)
    if "resnet_18_adapted" in model_name_lower:
        # We pass the resolved geometry directly to the builder
        model = build_resnet18_adapted(
            device=device, 
            cfg=cfg,
            in_channels=in_channels,
            num_classes=num_classes
        )
    else:
        error_msg = f"Model architecture '{cfg.model.name}' is not recognized by the Factory."
        logger.error(f" [!] {error_msg}")
        raise ValueError(error_msg)
    
    # 3. Finalize placement and telemetry
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(
        f"Model deployed to {str(device).upper()} | "
        f"Total Parameters: {total_params:,}"
    )
    
    return model