# 🔬 VisionForge: Type-Safe Deep Learning Framework

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-e92063?logo=pydantic&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-3.0%2B-00ADD8?logo=optuna&logoColor=white)

![Architecture](https://img.shields.io/badge/Architecture-Decoupled-blueviolet)
![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)
![License](https://img.shields.io/badge/license-MIT-green)

![Status](https://img.shields.io/badge/status-Active-success)
![Issues](https://img.shields.io/github/issues/tomrussobuilds/visionforge)

---

## 📌 Table of Contents

- [🎯 Overview](#-overview)
- [🚀 Quick Start](#-quick-start)
- [✨ Core Features](#-core-features)
- [🏗 System Architecture](#-system-architecture)
- [📊 Experiment Management](#-experiment-management)
- [🧩 Dependency Graph](#-dependency-graph)
- [🔬 Technical Deep Dive](#-technical-deep-dive)
- [📁 Project Structure](#-project-structure)
- [💻 Usage Patterns](#-usage-patterns)
- [🎯 Hyperparameter Optimization](#-hyperparameter-optimization)
- [✅ Environment Verification](#-environment-verification)
- [🐳 Containerized Deployment](#-containerized-deployment)
- [📊 Configuration Reference](#-configuration-reference)
- [🔄 Extending to New Datasets](#-extending-to-new-datasets)
- [📚 Citation](#-citation)
- [🗺 Development Roadmap](#-development-roadmap)

---

## 🎯 Overview

**VisionForge** is a research-grade PyTorch training framework engineered for reproducible, scalable computer vision experiments. Originally designed for medical imaging (MedMNIST v2), it has evolved into a domain-agnostic platform supporting multi-resolution architectures (28×28 to 224×224+), automated hyperparameter optimization, and cluster-safe execution.

**Key Differentiators:**
- **Type-Safe Configuration Engine**: Pydantic V2-based declarative manifests eliminate runtime errors
- **Zero-Conflict Execution**: Kernel-level file locking (`fcntl`) prevents concurrent runs from corrupting shared resources
- **Intelligent Hyperparameter Search**: Optuna integration with TPE sampling and Median Pruning
- **Hardware-Agnostic**: Auto-detection and optimization for CPU/CUDA/MPS backends
- **Audit-Grade Traceability**: BLAKE2b-hashed run directories with full YAML snapshots


**Representative Benchmarks** (RTX 5070, OrganCMNIST):

| Task | Architecture | Resolution | Trials | Time | Notes |
|------|-------------|-----------|--------|------|-------|
| **Training** | ResNet-18 | 28×28 | - | ~5 min (60 epochs) | BloodMNIST, GPU |
| **Training** | ResNet-18 | 28×28 | - | ~2.5h (60 epochs) | BloodMNIST, CPU (16 cores) |
| **Optimization** | EfficientNet-B0 | 224×224 | 3 | ~1.5h | **Early stopped** at AUC≥0.9999 |
| **Single Trial** | EfficientNet-B0 | 224×224 | 1 | ~29 min (15 epochs) | Batch size 8 |

>[!Note]
>Optimization times vary significantly with early stopping.
>The framework automatically stops when target performance is reached 
>(e.g., 3 trials vs planned 20 when AUC≥0.9999).

---

## 🚀 Quick Start

```bash
# 1. Clone and setup environment
git clone https://github.com/tomrussobuilds/visionforge.git
cd visionforge
pip install -r requirements.txt

# 2. Run a quick verification (1-epoch sanity check)
python -m tools.smoke_test

# 3. Train with optimized recipe (ResNet-18, 28×28)
python main.py --config recipes/config_resnet_18_adapted.yaml

# 4. Launch hyperparameter optimization (50 trials, ~15 min on GPU)
python optimize.py --config recipes/optuna_resnet_18_adapted.yaml

# 5. View optimization results
firefox outputs/*/figures/param_importances.html

# 6. Deploy best configuration
python main.py --config outputs/*/reports/best_config.yaml
```

---

## ✨ Core Features

### 🔒 Enterprise-Grade Execution Safety

**Tiered Configuration Engine (SSOT)**  
Built on Pydantic V2, the configuration system acts as a **Single Source of Truth**, transforming raw inputs (CLI/YAML) into an immutable, type-safe execution blueprint:

- **Late-Binding Metadata Injection**: Dataset specifications (normalization constants, class mappings) are resolved from a centralized registry at instantiation time
- **Cross-Domain Validation**: Post-construction logic guards prevent unstable states (e.g., enforcing RGB input for pretrained weights, validating AMP compatibility)
- **Path Portability**: Automatic serialization converts absolute paths to environment-agnostic anchors for cross-platform reproducibility

**Infrastructure Guard Layer**  
An independent `InfrastructureManager` bridges declarative configs with physical hardware:

- **Mutual Exclusion via `flock`**: Kernel-level advisory locking ensures only one training instance per workspace (prevents VRAM race conditions)
- **Process Sanitization**: `psutil` wrapper identifies and terminates ghost Python processes
- **HPC-Aware Safety**: Auto-detects cluster schedulers (SLURM/PBS/LSF) and suspends aggressive process cleanup to preserve multi-user stability

**Deterministic Run Isolation**  
Every execution generates a unique workspace using:
```
outputs/YYYYMMDD_DS_MODEL_HASH6/
```
Where `HASH6` is a BLAKE2b cryptographic digest (3-byte, deterministic) computed from the training configuration. Even minor hyperparameter variations produce isolated directories, preventing resource overlap and ensuring auditability.

### 🔬 Reproducibility Architecture

**Dual-Layer Reproducibility Strategy:**
1. **Standard Mode**: Global seeding (Seed 42) with performance-optimized algorithms
2. **Strict Mode**: Bit-perfect reproducibility via:
   - `torch.use_deterministic_algorithms(True)`
   - `worker_init_fn` for multi-process RNG synchronization
   - Auto-scaling to `num_workers=0` when determinism is critical

**Data Integrity Validation:**
- MD5 checksum verification for dataset downloads
- `validate_npz_keys` structural integrity checks before memory allocation

### ⚡ Performance Optimization

**Hybrid RAM Management:**
- **Small Datasets** (<50K samples): Full RAM caching for maximum throughput
- **Large Datasets** (>100K samples): Indexed slicing to prevent OOM errors

**Dynamic Path Anchoring:**
- "Search-up" logic locates project root via markers (`.git`, `README.md`)
- Ensures absolute path stability regardless of invocation directory

**Graceful Logger Reconfiguration:**
- Initial logs route to `STDOUT` for immediate feedback
- Hot-swap to timestamped file handler post-initialization without trace loss

### 🎯 Intelligent Hyperparameter Search

**Optuna Integration Features:**
- **TPE Sampling**: Tree-structured Parzen Estimator for efficient search space exploration
- **Median Pruning**: Early stopping of underperforming trials (30-50% time savings)
- **Persistent Studies**: SQLite storage enables resume-from-checkpoint
- **Type-Safe Constraints**: All search spaces respect Pydantic validation bounds
- **Auto-Visualization**: Parameter importance plots, optimization history, parallel coordinates

---

## 🏗 System Architecture

The framework implements **Separation of Concerns (SoC)** with five core layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                      RootOrchestrator                           │
│              (Lifecycle Manager & Context)                      │
│                                                                 │
│  Responsibilities:                                              │
│  • Phase 1-7 initialization sequence                            │
│  • Resource acquisition & cleanup (Context Manager)             │
│  • Device resolution & caching                                  │
└────────────┬─────────────────────────┬──────────────────────────┘
             │                         │
             │ uses                    │ uses
             │                         │
    ┌────────▼──────────┐     ┌────────▼───────────────┐
    │                   │     │                        │
    │  Config Engine    │     │  InfrastructureManager │
    │  (Pydantic V2)    │     │  (flock/psutil)        │
    │                   │     │                        │
    │  • Type safety    │     │  • Process cleanup     │
    │  • Validation     │     │  • Kernel locks        │
    │  • Metadata       │     │  • HPC detection       │
    │    injection      │     │                        │
    └───────────────────┘     └────────────────────────┘
             │
             │ provides config to
             │
    ┌────────▼───────────────────────────────────────────────┐
    │                                                        │
    │              Execution Pipeline                        │
    │                                                        │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
    │  │   Data   │  │  Model   │  │ Trainer  │              │
    │  │ Handler  │→ │ Factory  │→ │  Engine  │              │
    │  └──────────┘  └──────────┘  └────┬─────┘              │
    │                                   │                    │
    │                             ┌─────▼──────┐             │
    │                             │ Evaluation │             │
    │                             │  Pipeline  │             │
    │                             └────────────┘             │
    │                                                        │
    └────────────────────────────┬───────────────────────────┘
                                 │
                                 │ alternative path
                                 │
                        ┌────────▼──────────────┐
                        │  Optimization Engine  │
                        │      (Optuna)         │
                        │                       │
                        │  • Study management   │
                        │  • Trial execution    │
                        │  • Pruning logic      │
                        │  • Visualization      │
                        └───────────────────────┘
```

**Key Design Principles:**

1. Orchestrator owns both Config and InfrastructureManager
2. Config is the SSOT - all modules receive it as dependency
3. InfrastructureManager is stateless utility for OS-level operations
4. Execution pipeline is linear: Data → Model → Training → Eval
5. Optimization wraps the entire pipeline for each trial

---

## 📊 Experiment Management

Every run generates a complete artifact suite for total traceability:

**Artifact Structure:**
```
outputs/20260116_organcmnist_optuna_a3f7c2/
├── figures/
│   ├── param_importances.html      # Interactive importance plot
│   ├── optimization_history.html   # Trial progression
│   └── parallel_coordinates.html   # Hyperparameter relationships
├── reports/
│   ├── best_config.yaml            # Optimized configuration
│   ├── study_summary.json          # All trials metadata
│   └── top_10_trials.xlsx          # Best configurations
└── database/
    └── study.db                    # SQLite storage for resumption
```

> [!IMPORTANT]
> ### 📂 [View Sample Artifacts](./docs/artifacts)
> Explore Excel reports, YAML configs, and diagnostic plots from real experiments.

---

## 🧩 Dependency Graph

<p align="center">
<img src="docs/framework_map.svg?v=4" width="900" alt="System Dependency Graph">
</p>

> *Generated via `pydeps`. Highlights the centralized Config hub and modular architecture.*

<details>
<summary>🛠️ Regenerate Dependency Graph</summary>

```bash
pydeps orchard \
    --cluster \
    --max-bacon=0 \
    --max-module-depth=4 \
    --only orchard \
    --noshow \
    -T svg \
    -o docs/framework_map.svg
```

**Requirements:** `pydeps` + Graphviz (`sudo apt install graphviz` or `brew install graphviz`)

</details>

---

## 🔬 Technical Deep Dive

### Architecture Adaptation

Standard ResNet-18 is optimized for 224×224 ImageNet inputs. Direct application to 28×28 domains causes catastrophic information loss. Our adaptation strategy:

| Layer | Standard ResNet-18 | VisionForge Adapted | Rationale |
|-------|-------------------|---------------------|-----------|
| **Input Conv** | 7×7, stride=2, pad=3 | **3×3, stride=1, pad=1** | Preserve spatial resolution |
| **Max Pooling** | 3×3, stride=2 | **Identity (bypassed)** | Prevent 75% feature loss |
| **Stage 1 Input** | 56×56 (from 224) | **28×28 (from 28)** | Native resolution entry |

**Key Modifications:**
1. **Stem Redesign**: Replacing large-receptive-field convolution avoids immediate downsampling
2. **Pooling Removal**: MaxPool bypass maintains full spatial fidelity into residual stages
3. **Bicubic Weight Transfer**: Pretrained 7×7 weights are spatially interpolated to 3×3 geometry

### Mathematical Weight Transfer

To retain ImageNet-learned feature detectors, we apply bicubic interpolation:

**Source Tensor:**
$$W_{\text{src}} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times 7 \times 7}$$

**Transformation:**
$$W_{\text{dest}} = \mathcal{I}_{\text{bicubic}}(W_{\text{src}}, \text{size}=(3, 3))$$

**Result:** A 3×3 kernel preserving edge-detection patterns optimized for compact receptive fields, enabling faster convergence than random initialization.

### Training Regularization

**MixUp Augmentation** synthesizes training samples via convex combinations:

$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j \quad \text{where} \quad \lambda \sim \text{Beta}(\alpha, \alpha)$$

$$\tilde{y} = \lambda y_i + (1 - \lambda) y_j$$

This prevents overfitting on small-scale textures and improves generalization.

---

## 📁 Project Structure

```
visionforge/
├── main.py                         # Training entry point
├── optimize.py                     # Hyperparameter search entry point
├── Dockerfile                      # Multi-stage reproducible build
├── requirements.txt                # Pinned dependencies
├── recipes/                        # YAML configuration presets
│   ├── config_resnet_18_adapted.yaml
│   ├── config_efficientnet_b0.yaml
│   ├── optuna_resnet_18_adapted.yaml
│   └── optuna_efficientnet_b0.yaml
├── tools/                          # Diagnostic utilities
│   ├── smoke_test.py               # 1-epoch E2E verification
│   ├── health_check.py             # Dataset integrity validation
│   └── unit_test.py                # Unit test suite (WIP)
├── orchard/                        # Core framework package
│   ├── core/                       # Framework nucleus
│   │   ├── config/                 # Pydantic schemas
│   │   ├── environment/            # Hardware abstraction
│   │   ├── io/                     # Serialization utilities
│   │   ├── logger/                 # Telemetry system
│   │   ├── metadata/               # Dataset registry
│   │   ├── paths/                  # Path management
│   │   ├── cli.py                  # Argument parser
│   │   └── orchestrator.py         # Lifecycle coordinator
│   ├── data_handler/               # Loading strategies
│   ├── models/                     # Architecture factory
│   ├── trainer/                    # Training loop
│   ├── evaluation/                 # Metrics and visualization
│   └── optimization/               # Optuna integration
│       ├── objective.py            # Trial execution logic
│       ├── orchestrator.py         # Study management
│       └── search_spaces.py        # Hyperparameter distributions
└── outputs/                        # Isolated run workspaces
```

---

## 💻 Usage Patterns

### Configuration-Driven Execution

**Recommended Method:** YAML recipes ensure full reproducibility and version control.

```bash
# Verify environment
python -m tools.smoke_test

# Train with preset (28×28 ResNet-18)
python main.py --config recipes/config_resnet_18_adapted.yaml

# Train with preset (224×224 EfficientNet-B0)
python main.py --config recipes/config_efficientnet_b0.yaml
```

### CLI Overrides

For rapid experimentation (not recommended for production):

```bash
# Quick test on different dataset
python main.py --dataset dermamnist --epochs 10 --batch_size 64

# Custom learning rate schedule
python main.py --lr 0.001 --min_lr 1e-7 --epochs 100

# Disable augmentations
python main.py --mixup_alpha 0 --no_tta
```

> [!WARNING]
> **Configuration Precedence Order:**
> 1. **YAML file** (highest priority - if `--config` is provided)
> 2. **CLI arguments** (only used when no `--config` specified)
> 3. **Defaults** (from Pydantic field definitions)
>
> **When `--config` is provided, YAML values override CLI arguments.** This prevents configuration drift but means CLI flags are ignored. For reproducible research, always use YAML recipes.

---

## 🎯 Hyperparameter Optimization

### Quick Start

```bash
# Install Optuna (if not already present)
pip install optuna plotly kaleido

# Run optimization with preset (28×28, 50 trials, ~2-3h)
python optimize.py --config recipes/optuna_resnet_18_adapted.yaml

# Run optimization with preset (224×224, 20 trials, ~1h)
python optimize.py --config recipes/optuna_efficientnet_b0.yaml

# Custom search (20 trials, 10 epochs each)
python optimize.py --dataset pathmnist \
    --n_trials 20 \
    --epochs 10 \
    --search_space_preset quick

# Resume interrupted study
python optimize.py --config recipes/optuna_resnet_18_adapted.yaml \
    --load_if_exists true
```
---

### Search Space Coverage

**Full Space** (13 parameters):
- **Optimization**: `learning_rate`, `weight_decay`, `momentum`, `min_lr`
- **Regularization**: `mixup_alpha`, `label_smoothing`, `dropout`
- **Scheduling**: `cosine_fraction`, `scheduler_patience`
- **Augmentation**: `rotation_angle`, `jitter_val`, `min_scale`
- **Batch Size**: Resolution-aware categorical choices
  - 28×28: [16, 32, 48, 64]
  - 224×224: [8, 12, 16] (OOM-safe for 8GB VRAM)

**Quick Space** (4 parameters):
- `learning_rate`, `weight_decay`, `batch_size`, `dropout`

### Optimization Workflow

```bash
# Phase 1: Rapid exploration (20 trials, 10 epochs, ~30 min)
python optimize.py --config recipes/optuna_resnet_18_adapted.yaml \
    --n_trials 20 --epochs 10

# Phase 2: Comprehensive search (50 trials, 15 epochs, ~2-3h)
python optimize.py --config recipes/optuna_resnet_18_adapted.yaml

# Phase 3: Final training with best config (60 epochs)
python main.py --config outputs/*/reports/best_config.yaml --epochs 60
```

### Artifacts Generated

```
outputs/20250116_bloodmnist_optuna_a3f7c2/
├── figures/
│   ├── param_importances.html      # Interactive importance plot
│   ├── optimization_history.html   # Trial progression
│   └── parallel_coordinates.html   # Hyperparameter relationships
├── reports/
│   ├── best_config.yaml            # Optimized configuration
│   ├── study_summary.json          # All trials metadata
│   └── top_10_trials.xlsx          # Best configurations
└── database/
    └── study.db                    # SQLite storage for resumption
```

### Customization

Edit search spaces in `orchard/optimization/search_spaces.py`:

```python
class CustomSearchSpace:
    @staticmethod
    def get_optimization_space() -> Dict[str, Callable]:
        return {
            "learning_rate": lambda trial: trial.suggest_float(
                "learning_rate", 1e-4, 1e-2, log=True
            ),
            "weight_decay": lambda trial: trial.suggest_float(
                "weight_decay", 1e-5, 1e-3, log=True
            ),
        }
```

---

## ✅ Environment Verification

**Smoke Test** (1-epoch sanity check):
```bash
python -m tools.smoke_test
```

**Output:** Validates full pipeline in <30 seconds:
- Dataset loading and preprocessing
- Model instantiation and weight transfer
- Training loop execution
- Evaluation metrics computation
- Excel/PNG artifact generation

**Health Check** (dataset integrity):
```bash
python -m tools.health_check --dataset bloodmnist
```

**Output:** Verifies:
- MD5 checksum matching
- NPZ key structure (`train_images`, `train_labels`, `val_images`, etc.)
- Sample count validation

---

## 🐳 Containerized Deployment

### Build Image

```bash
docker build -t visionforge:latest .
```

### Execution Modes

**Standard Mode** (Performance Optimized):
```bash
docker run -it --rm \
  --gpus all \
  -u $(id -u):$(id -g) \
  -e TORCH_HOME=/tmp/torch_cache \
  -e MPLCONFIGDIR=/tmp/matplotlib_cache \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/outputs:/app/outputs \
  visionforge:latest \
  --config recipes/config_resnet_18_adapted.yaml
```

**Strict Reproducibility Mode** (Bit-Perfect Determinism):
```bash
docker run -it --rm \
  --gpus all \
  -u $(id -u):$(id -g) \
  -e IN_DOCKER=TRUE \
  -e DOCKER_REPRODUCIBILITY_MODE=TRUE \
  -e TORCH_HOME=/tmp/torch_cache \
  -e MPLCONFIGDIR=/tmp/matplotlib_cache \
  -e PYTHONHASHSEED=42 \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/outputs:/app/outputs \
  visionforge:latest \
  --config recipes/config_resnet_18_adapted.yaml \
  --reproducible
```

> [!NOTE]
> - `TORCH_HOME` and `MPLCONFIGDIR` prevent permission errors in containerized environments
> - `CUBLAS_WORKSPACE_CONFIG` is required for CUDA determinism
> - `--gpus all` requires NVIDIA Container Toolkit

---

## 📊 Configuration Reference

### Core Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `epochs` | int | 60 | [1, 1000] | Training epochs |
| `batch_size` | int | 128 | [1, 2048] | Samples per batch |
| `learning_rate` | float | 0.008 | (1e-8, 1.0) | Initial SGD learning rate |
| `min_lr` | float | 1e-6 | (0, lr) | Minimum LR for scheduler |
| `weight_decay` | float | 5e-4 | [0, 0.2] | L2 regularization |
| `momentum` | float | 0.9 | [0, 1) | SGD momentum |
| `mixup_alpha` | float | 0.002 | [0, 1] | MixUp strength (0=disabled) |
| `label_smoothing` | float | 0.0 | [0, 0.3] | Label smoothing factor |
| `dropout` | float | 0.0 | [0, 0.9] | Dropout probability |
| `seed` | int | 42 | - | Global random seed |
| `reproducible` | bool | False | - | Enable strict determinism |

### Augmentation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hflip` | float | 0.5 | Horizontal flip probability |
| `rotation_angle` | int | 10 | Max rotation degrees |
| `jitter_val` | float | 0.2 | ColorJitter intensity |
| `min_scale` | float | 0.95 | Minimum RandomResizedCrop scale |
| `no_tta` | bool | False | Disable test-time augmentation |

### Model Parameters

| Parameter | Type | Default | Options |
|-----------|------|---------|---------|
| `model_name` | str | "resnet_18_adapted" | `resnet_18_adapted`, `efficientnet_b0` |
| `pretrained` | bool | True | Use ImageNet weights |
| `force_rgb` | bool | True | Convert grayscale to 3-channel |
| `resolution` | int | 28 | [28, 224] |

### Dataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | str | "bloodmnist" | MedMNIST identifier |
| `data_root` | Path | `./dataset` | Dataset directory |
| `max_samples` | int | None | Cap training samples (debugging) |
| `use_weighted_sampler` | bool | True | Balance class distribution |

---

## 🔄 Extending to New Datasets

The framework is designed for zero-code dataset integration via the registry system:

### 1. Add Dataset Metadata

Edit `orchard/core/metadata/medmnist_v2_28x28.py`:

```python
DATASET_REGISTRY = {
    "custom_dataset": DatasetMetadata(
        name="custom_dataset",
        num_classes=10,
        in_channels=3,
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
        native_resolution=28,
        class_names=["class0", "class1", ...],
        url="https://example.com/dataset.npz",
        md5="abc123..."
    ),
}
```

### 2. Train Immediately

```bash
python main.py --dataset custom_dataset --epochs 30
```

No code changes required—the configuration engine automatically resolves metadata.

---

## 📚 Citation

```bibtex
@software{visionforge2025,
  author = {Tommaso Russo},
  title  = {VisionForge: Type-Safe Deep Learning Framework},
  year   = {2025},
  url    = {https://github.com/tomrussobuilds/visionforge},
  note   = {PyTorch framework with Pydantic configuration and Optuna optimization}
}
```

---

## 🗺 Development Roadmap

### ✅ Phase 1: Foundation (Completed)
- Architecture adaptation (3×3 stem, MaxPool removal)
- Pydantic-based configuration engine
- Infrastructure safety (flock, process management)

### ✅ Phase 2: Automation (Completed)
- YAML-driven execution model
- Optuna hyperparameter optimization
- Multi-resolution support (28×28, 224×224)

### 🔄 Phase 3: Modern Architectures (Current)
- Vision Transformer (ViT) integration
- EfficientNet-V2 support
- ConvNeXt architecture addition
- Attention mechanism analysis

### 🔮 Phase 4: Domain Transcendence (Planned)
- Abstract dataset registry for non-medical domains
- Multi-modal hooks (detection, segmentation)
- Distributed training support (DDP, FSDP)
- TorchScript/ONNX export pipeline
- Benchmark suite for architecture comparison

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please open an issue for discussion before submitting PRs.

## 📧 Contact

For questions or collaboration: [GitHub Issues](https://github.com/tomrussobuilds/visionforge/issues)

---

<p align="center">
<strong>Built with ❤️ for reproducible research</strong>
</p>