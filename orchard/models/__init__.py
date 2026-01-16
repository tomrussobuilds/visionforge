"""
Models Factory Package

This package implements the Factory Pattern to decouple model instantiation 
from the main execution logic. It routes requests to specific architecture 
definitions and ensures models are correctly adapted to the dataset geometry 
(channels and classes) resolved at runtime.
"""

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .factory import get_model

# =========================================================================== #
#                                PACKAGE INTERFACE                            #
# =========================================================================== #

__all__ = [
    "get_model"
]