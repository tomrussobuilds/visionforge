"""
Evaluation and Reporting Package

This package coordinates model inference, performance visualization, 
and structured experiment reporting using a memory-efficient approach.
"""

# =========================================================================== #
#                                Inference Engine                             #
# =========================================================================== #
from .engine import evaluate_model

# =========================================================================== #
#                                Visualizations                               #
# =========================================================================== #
from .visualization import (
    plot_confusion_matrix,
    plot_training_curves,
    show_predictions
)

# =========================================================================== #
#                                Structured Reporting                         #
# =========================================================================== #
from .reporting import (
    TrainingReport, 
    create_structured_report
)

# =========================================================================== #
#                                Evaluation Pipeline                          #
# =========================================================================== #
from .pipeline import run_final_evaluation