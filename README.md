# MedMNIST Classification with Adapted ResNet-18

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![TorchVision](https://img.shields.io/badge/TorchVision-0.15%2B-red)



**97.19% Test Accuracy • 0.9693 Macro F1 • Single pretrained ResNet-18 • 28×28 images**

This repository provides a highly reproducible training pipeline for the BloodMNIST (from MedMNIST v2) using an adapted pretrained ResNet-18 architecture. The goal is to demonstrate solid performance using a minimal configuration that adheres to modern PyTorch best practices.

The results reflect the latest successful training run (early stopping at epoch 44).

### Confusion Matrix
<img src="docs/media/confusion_matrix_ResNet-18 Adapted.png" width="400">

### Training Curves
<img src="docs/media/training_curves.png" width="400">

### Sample Predictions
<img src="docs/media/sample_predictions_ResNet-18 Adapted.png" width="400">

### Final Results (60 epochs, seed 42)
| Metric                  | Value     |
|-------------------------|-----------|
| Best Validation Accuracy| **96.96%** |
| Test Accuracy (with TTA)| **97.19%** |
| Test Macro F1 (with TTA)| **0.9693** |

→ Confusion matrix, training curves, sample predictions and Excel report are automatically saved.

---

### Research Goals & Roadmap

This project started as a benchmark to evaluate the efficiency of a single, adapted ResNet-18 on the BloodMNIST dataset (28×28). Despite the rise of Vision Transformers (ViTs) and massive ensembles, this repository aims to demonstrate that a lightweight, classic architecture—when paired with modern training techniques—remains extremely competitive for medical imaging tasks.
The Scaling Strategy:

  - Phase 1 (Current): Optimization on 28×28 BloodMNIST using architectural adaptations (initial kernel resize, MaxPool removal).

  - Phase 2 (Near Term): Scaling the pipeline across the entire MedMNIST v2 suite (DermaMNIST, OrganMNIST, etc.) to validate the robustness of the hyperparameters.

  - Phase 3 (Long Term): Extending support for high-resolution medical formats (224×224) and multi-modal integration, transitioning the repo from a specific classifier to a versatile MedMNIST benchmarking framework.

### Key Features & Design Choices (Post-Refactoring)

**Modularity and Structure** 

Fully decoupled logic using specialized sub-packages (`core`, `data_handler`, `models`, `trainer`, `evaluation`).

**Robust Pathing**

Implemented dynamic `PROJECT_ROOT` detection in `constants.py` to ensure all outputs (models, logs, figures) are correctly saved relative to the project root, regardless of where the script is executed (local or Docker container).

**Accuracy vs. Reproducibility Balance** **The pipeline prioritizes fully deterministic reproducibility.** 

While running in "Fast Mode" (`num_workers > 0`) is faster, the "Strict Reproducibility" mode (`num_workers=0`) guarantees bit-per-bit identical results at the expense of a longer training time. This trade-off is managed automatically via `DOCKER_REPRODUCIBILITY_MODE` environment variable.

**Automated Reporting**

Generates high-resolution plots and comprehensive Excel reports (`.xlsx`) for every run.

**Registry-based Metadata**

Metadata for different MedMNIST variants is centralized in a `DATASET_REGISTRY` using `NamedTuples`. This ensures type safety, immutability, and makes adding supoprt for new datasets (like DermaMNIST) a matter of a single dictionary entry.

**Atomic Run Isolation**

Every execution is treated as a unique experiment. The `RunPaths` manager ensures that logs, checkpoints, and reports are isolated in timestamped directories, preventing accidental overwrites and maintaing a clean historical record of experiments.

**Regularization: MixUp**

To improve generalization on the 28x28 manifold, the pipeline implements MixUp during training. New samples are synthesized as a convex combination of two random samples from the mini-batch:

$$
\tilde{x} = \lambda x_i + (1 - \lambda) x_j
$$
$$
\tilde{y} = \lambda y_i + (1 - \lambda) y_j
$$

Where $\lambda \in [0, 1]$ is drawn from a $Beta(\alpha, \alpha)$ distribution. This process encourages the model to behave linearly in-between training samples, reducing overconfident predictions and improving robustness.

---


### ResNet-18 adapted for 28×28
- Initial 7 x7 convolution replaced with 3 x 3.
- Initial `MaxPool` removed to preserve 28 x 28 feature maps.
- ImageNet pretrained weights transferred via bicubic upsampling of the first convolutional layer.
- Reproduciblity & Robustness: – Full Reproducibility guaranteed (fixed seeds for PyTorch, NumPy, Python).
- `worker_init_fn` implemented to ensure determinism even when using multiple data loading workers (`num_workers > 0`).

---

### Defensive Utilities

A few tiny helpers included in `system.py` and `config.py` were added after real debugging incidents to ensure robust, unattended execution:

* **Dynamic `num_workers`:** The `Config` class automatically adjusts `num_workers` (0 vs 4) based on the `DOCKER_REPRODUCIBILITY_MODE` environment variable, balancing speed and determinism.
* **Process Management (`kill_duplicate_processes()`):** Stops accidental multi-launches that consume excessive CPU/RAM.
* **Safe Data I/O (`ensure_mnist_npz()`):** Robust dataset download with retries, MD5 check, and atomic write ensures pipeline reliability.
* **Robust Pathing:** The `get_project_root()` utility ensures all outputs (models, logs, figures) are saved correctly relative to the project root, regardless of the execution environment (host or Docker).

---

### Project Structure

```bash
med_mnist/
├── main.py                   # Global entry point
├── Dockerfile                # Image definition
├── .dockerignore             # Build optimization
├── .gitignore                # Repository filtering
├── requirements.txt          # Python dependencies
├── docs/
│   └── media/                # Stable README assets (tracked)
├── tools/                    # Maintenance (RAM checks, etc.)
├── legacy/                   # Archived code
├── scripts/                  # Modular package
│   ├── core/                 # Config & Constants
│   ├── data_handler/         # Loading & Augmentation
│   ├── models/               # Model Factory
│   ├── trainer/              # Training Loop & MixUp
│   └── evaluation/           # Engine & Reporting
└── outputs/                  # Results (ignored by Git)
    └── YYYYMMDD_HHMMSS/      # Timestamped run folder
```

### 1. Requirements

Install dependencies easily with pip:

```bash
pip install -r requirements.txt
```

Install dependencies easily with pip, or check the full list here:

[📦 See Full Requirements](requirements.txt)


### 2. Usage (Local Execution - Recommended for Speed)

Run the script from the project root. It will default to the fast mode (`num_workers=4`).

```bash
git clone https://github.com/tomrussobuilds/bloodmnist.git
cd bloodmnist
python main.py
```
Note: The entry point script is now `main.py`.

The script will automatically:

- Download BloodMNIST if missing
- Train for max 60 epochs with early stopping (`patience=15`)
- Save the best model → `outputs/YYYYMMDD_HHMMSS_bloodmnist_resnet18/models/best_model.pth`
- Generate figures, confusion matrix, Excel report → `figures/` and `reports/`

### 3. Docker Execution (Recommended for Portability & Reproducibility)

The pipeline is containerized using the included `Dockerfile`.

| Mode | Command | `num_workers` | Guarantees |
| :--- | :--- | :---: | :--- |
| **Fast Mode (Default)** | `docker run bloodmnist_image` | 4 | Best performance on CPU, **Not fully deterministic** |
| **Strict Reproducibility** | `-e DOCKER_REPRODUCIBILITY_MODE=TRUE` | 0 | **100% Deterministic** (bit-per-bit), but slower |

Run in Strict Reproducibility Mode (Recommended for testing):

```bash
docker run -it --rm -e DOCKER_REPRODUCIBILITY_MODE=TRUE bloodmnist_image
```

### Command Line Arguments (argparse)
You can fully configure training from the command line (via `main.py`).

| Arg | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| --epochs | int | 60 | Maximum number of training epochs. |
| --batch_size | int | 128 | Batch size for data loaders. |
| --lr | float | 0.008 | Initial learning rate for the SGD optimizer. |
| --seed | int | 42 | Random seed for reproducibility (influences PyTorch, NumPy, Python). |
| --mixup_alpha | float | 0.002 | $\alpha$ parameter for MixUp regularization. Set to 0 to disable MixUp. |
| --patience | int | 15 | Early stopping patience (epochs without validation improvement). |
| --momentum | float | 0.9 | Momentum factor for the SGD optimizer. |
| --weight_decay | float | 5e-4 | Weight decay (L2 penalty) for the optimizer |
| --no_tta | flag | (disabled) | Flag to disable Test-Time Augmentation (TTA) during final evaluation. |
| --hflip | float | 0.5 | Probability of Horizontal Flip augmentation. |
| --rotation_angle | int | 10 | Max rotation angle for random rotations. |
| --jitter_val | float | 0.2 | Strength of Color Jitter (brightness/contrast). |

Examples:

Run without TTA

```bash
python main.py --no_tta
```
Train for 100 epochs

```bash
python main.py --epochs 100
```
Lower LR for finer tuning

```bash
python main.py --lr 0.001
```
Disable MixUp

```bash
python main.py --mixup_alpha 0
```
Custom batch size & seed

```bash
python main.py --batch_size 64 --seed 123
```

### Reproducibility

The entire pipeline is deterministic (seed 42). Run the script twice will yield the same validation curve and the same final accuracy.

### Citation

If you use this repository in academic work or derivative projects:

```bibtex
@misc{bloodmnist_resnet18,
  title  = {BloodMNIST Classification with Adapted ResNet-18},
  author = {Tommaso Russo},
  year   = {2025},
  url    = {https://github.com/tomrussobuilds/bloodmnist}
}
```

### Conclusion & Future Work

This project is a starting point. While the adapted ResNet-18 has shown great results, the real goal is to turn this repository into a sandbox for medical imaging experimentation.

What’s next? I plan to keep testing:

  - More Architectures: Moving beyond ResNet to see how different backbones (EfficientNets, ConvNeXts, etc.) handle the MedMNIST manifold.

  - New Datasets: Scaling the same pipeline to other MedMNIST categories.

  - Better Tuning: Exploring different augmentation and regularization strategies.

Let's see how far we can push the accuracy on these datasets. If you have ideas or want to test a specific model, feel free to contribute!
