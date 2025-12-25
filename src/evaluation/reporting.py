"""
Reporting & Experiment Summarization Module

This module orchestrates the generation of human-readable artifacts following 
the completion of a training pipeline. It specializes in transforming raw 
experiment metrics and configuration states into structured, professionally 
formatted Excel summaries.

Core Components:
    * TrainingReport: A frozen data container that aggregates metadata, 
      hyperparameters, and final performance metrics.
    * Excel Export: Logic for generating stylized spreadsheets with 
      automated column sizing and conditional formatting.
"""

# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence
import logging

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import pandas as pd

# =========================================================================== #
#                                Internal Imports
# =========================================================================== #
from src.core import Config


# =========================================================================== #
#                               EXCEL REPORTS
# =========================================================================== #

logger = logging.getLogger("medmnist_pipeline")

@dataclass(frozen=True)
class TrainingReport:
    """
    Structured data container for summarizing a complete training experiment.
    
    Provides a vertical DataFrame representation optimized for readability 
    in spreadsheet software.
    """
    timestamp: str
    model: str
    dataset: str
    best_val_accuracy: float
    test_accuracy: float
    test_macro_f1: float
    epochs_trained: int
    learning_rate: float
    batch_size: int
    augmentations: str
    normalization: str
    model_path: str
    log_path: str
    seed: int

    def to_vertical_df(self) -> pd.DataFrame:
        """Converts the report dataclass into a vertical pandas DataFrame."""
        data = asdict(self)
        return pd.DataFrame(list(data.items()), columns=["Parameter", "Value"]) 

    def save(self, path: Path) -> None:
        """
        Saves the report DataFrame to an Excel file with professional formatting.
        
        Applies specific column widths, header styles, and text wrapping to 
        ensure long paths and augmentation strings are legible.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_vertical_df()

        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Detailed Report', index=False)

            workbook = writer.book
            worksheet = writer.sheets['Detailed Report']

            # Professional Styling
            header_format = workbook.add_format({
                'bold': True, 'bg_color': '#D7E4BC', 'border': 1, 'align': 'center'
            })
            base_format = workbook.add_format({
                'border': 1, 'align': 'left', 'valign': 'vcenter'
            })
            wrap_format = workbook.add_format({
                'border': 1, 'text_wrap': True, 'valign': 'top', 'font_size': 10
            })

            worksheet.set_column('A:A', 25, base_format)
            worksheet.set_column('B:B', 60, wrap_format)

            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)         

        logger.info(f"Summary Excel report saved to → {path}")


def create_structured_report(
    val_accuracies: Sequence[float],
    macro_f1: float,
    test_acc: float,
    train_losses: Sequence[float],
    best_path: Path,
    log_path: Path,
    cfg: Config,
    aug_info: str | None = None,
) -> TrainingReport:
    """
    Constructs a TrainingReport object using final metrics and configuration.
    
    Dynamically extracts augmentation parameters if not explicitly provided.
    """
    if aug_info is None:
        # Dynamic extraction from Pydantic config
        aug_dict = cfg.augmentation.model_dump()
        aug_info = ", ".join([f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in aug_dict.items()])

    return TrainingReport(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model=cfg.model_name,
        dataset=cfg.dataset.dataset_name,
        best_val_accuracy=max(val_accuracies) if val_accuracies else 0.0,
        test_accuracy=test_acc,
        test_macro_f1=macro_f1,
        epochs_trained=len(train_losses),
        learning_rate=cfg.training.learning_rate,
        batch_size=cfg.training.batch_size,
        augmentations=aug_info,
        normalization=cfg.dataset.normalization_info,
        model_path=str(best_path),
        log_path=str(log_path),
        seed=cfg.training.seed,
    )