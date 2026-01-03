"""
Dynamic Run Orchestration and Experiment Directory Management.

This module provides the RunPaths class, which automates the creation of 
unique, hashed directory structures for each experiment. It ensures 
that model weights, logs, and diagnostic figures are organized, 
uniquely identified via short-hashes, and protected from overwrites.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import re
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
from pydantic import BaseModel, ConfigDict

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .constants import OUTPUTS_ROOT

# =========================================================================== #
#                                RUN MANAGEMENT                               #
# =========================================================================== #

class RunPaths(BaseModel):
    """
    Manages experiment-specific directories using an atomic hashing strategy.
    
    Instead of long timestamps, it uses a combination of DATE + SLUGS + HASH 
    to create clean, unique, and immutable directory structures.
    
    Example structure:
        outputs/20260103_pathmnist_resnet18_f3a1/
        ├── figures/
        ├── models/
        ├── reports/
        └── logs/
    """
    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True
    )

    # Core Identifiers
    run_id: str
    dataset_slug: str
    model_slug: str
    
    # Physical Paths
    root: Path
    figures: Path
    models: Path
    reports: Path
    logs: Path

    @classmethod
    def create(
        cls, 
        dataset_slug: str, 
        model_name: str, 
        cfg_dict: Dict[str, Any], 
        base_dir: Optional[Path] = None
    ) -> "RunPaths":
        """
        Factory method to initialize a unique run environment.
        
        Generates a 4-character hash based on key hyperparameters to ensure 
        atomicity even if multiple runs start on the same day.

        Args:
            dataset_slug (str): Identifier for the dataset (e.g., 'pathmnist').
            model_name (str): Human-readable model name.
            cfg_dict (Dict): Dictionary of hyperparameters used for hashing.
            base_dir (Path, optional): Custom base directory for outputs.

        Returns:
            RunPaths: A frozen instance with all paths pre-calculated and created.
        """
        # 1. Clean Slugs
        ds_slug = dataset_slug.lower()
        m_slug = re.sub(r'[^a-zA-Z0-9]', '', model_name.lower())
        
        # 2. Atomic Hashing (Prevents collisions with different params)
        # Hash based on model, dataset, and core training parameters
        hash_input = f"{ds_slug}_{m_slug}_{cfg_dict.get('lr')}_{cfg_dict.get('batch_size')}"
        run_hash = hashlib.blake2b(hash_input.encode(), digest_size=2).hexdigest()
        
        # 3. Clean Run ID: YYYYMMDD_dataset_model_hash
        date_str = time.strftime("%Y%m%d")
        run_id = f"{date_str}_{ds_slug}_{m_slug}_{run_hash}"
        
        # 4. Path Resolution
        base = base_dir or OUTPUTS_ROOT
        root_path = base / run_id
        
        # Instantiate (Pydantic handles the frozen assignment)
        instance = cls(
            run_id=run_id,
            dataset_slug=ds_slug,
            model_slug=m_slug,
            root=root_path,
            figures=root_path / "figures",
            models=root_path / "models",
            reports=root_path / "reports",
            logs=root_path / "logs"
        )
        
        # Physical creation of the tree
        instance._setup_run_directories()
        return instance

    # ======================================================================= #
    #                            Dynamic Properties                           #
    # ======================================================================= #

    @property
    def best_model_path(self) -> Path:
        """Standardized path for the top-performing model checkpoint."""
        return self.models / f"best_{self.model_slug}.pth"
    
    @property
    def final_report_path(self) -> Path:
        """Destination path for the comprehensive experiment summary."""
        return self.reports / "training_summary.csv"

    # ======================================================================= #
    #                              Public Methods                             #
    # ======================================================================= #

    def get_fig_path(self, filename: str) -> Path:
        """Generates an absolute path for a visualization artifact."""
        return self.figures / filename

    def get_config_path(self) -> Path:
        """Returns the path where the run configuration is archived."""
        return self.reports / "config.yaml"

    # ======================================================================= #
    #                             Internal Helpers                            #
    # ======================================================================= #

    def _setup_run_directories(self) -> None:
        """Creates the physical directory tree for the current run."""
        paths_to_create = [self.figures, self.models, self.reports, self.logs]
        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"RunPaths(run_id='{self.run_id}', root={self.root})"