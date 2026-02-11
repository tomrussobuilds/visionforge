← [Back to Main README](../../README.md)

# Docker Training Guide

## 🐳 Containerized Deployment

### Prerequisites

Add your user to the `docker` group to avoid using `sudo` with every command:
```bash
sudo usermod -aG docker $USER
```
Log out and back in (or run `newgrp docker`) for the change to take effect.

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
  -e DOCKER_REPRODUCIBILITY_MODE=TRUE \
  -e PYTHONHASHSEED=42 \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/outputs:/app/outputs \
  visionforge:latest \
  --config recipes/config_resnet_18_adapted.yaml
```

> [!NOTE]
> - `TORCH_HOME`, `MPLCONFIGDIR`, `IN_DOCKER`, and `CUBLAS_WORKSPACE_CONFIG` are pre-set in the image and do not need to be passed at runtime
> - `PYTHONHASHSEED` must be set at container startup (before the Python interpreter loads) to guarantee hash determinism — the image default is `0`, override with `-e` if a specific seed is needed
> - `--gpus all` requires NVIDIA Container Toolkit

---
