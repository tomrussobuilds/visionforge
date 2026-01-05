"""
Evaluation Reporting & Visualization Schema.

This module governs the post-training diagnostic phase, defining the 
aesthetics and structural requirements for performance telemetry. It 
synchronizes visual artifact generation (Confusion Matrices, Prediction Grids) 
with quantitative data persistence (Excel/JSON/CSV).

Key Architectural Features:
    * Aesthetic Standardization: Enforces consistent DPI, colormaps, and plot 
      styles across experiments to facilitate side-by-side comparative analysis.
    * Diagnostic Layouts: Configures the geometry of prediction grids, allowing 
      for high-fidelity inspection of model "blind spots" (misclassifications).
    * Tabular Export Policy: Validates and manages the serialization format 
      for final metrics, ensuring compatibility with downstream data 
      visualization tools.
    * Resource Efficiency: Controls inference batch sizes to optimize memory 
      usage during the evaluation of large test sets.

By centralizing reporting parameters, the schema guarantees that every 
experiment produces a standardized, publication-quality diagnostic suite.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import argparse

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
from pydantic import BaseModel, Field, ConfigDict, field_validator

# =========================================================================== #
#                               Internal Imports                              #
# =========================================================================== #
from .types import (
    PositiveInt, BatchSize
)

# =========================================================================== #
#                           EVALUATION CONFIGURATION                          #
# =========================================================================== #

class EvaluationConfig(BaseModel):
    """
    Controls the visual reporting and performance metric persistence.
    
    Sets the aesthetics for confusion matrices, prediction grids, and 
    defines the export format for quantitative tabular reports.
    """
    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )
    
    # Inference Strategy
    batch_size: BatchSize = Field(
        default=64,
        description="Batch size used during inference/evaluation"
    )
    
    # Visualization Aesthetics
    n_samples: PositiveInt = Field(
        default=12,
        description="Number of images to show in the prediction grid"
    )
    fig_dpi: PositiveInt = Field(default=200)
    cmap_confusion: str = "Blues"
    plot_style: str = "seaborn-v0_8-muted"
    
    # Export Settings
    report_format: str = "xlsx"
    save_confusion_matrix: bool = True
    save_predictions_grid: bool = True
    
    # Grid Geometry
    grid_cols: PositiveInt = Field(default=4)
    fig_size_predictions: tuple[int, int] = (12, 8)

    @field_validator("report_format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Ensure the output format is compatible with downstream consumers."""
        supported = ["xlsx"]
        if v.lower() not in supported:
            return "xlsx"
        return v.lower()

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EvaluationConfig":
        """Map evaluation and reporting preferences from CLI arguments."""
        return cls(
            batch_size=getattr(args, 'eval_batch_size', 64),
            n_samples=getattr(args, 'eval_samples', 12),
            fig_dpi=getattr(args, 'fig_dpi', 200),
            plot_style=getattr(args, 'plot_style', "seaborn-v0_8-muted"),
            report_format=getattr(args, 'report_format', "xlsx"),
            save_confusion_matrix=not getattr(args, 'no_confusion', False)
        )