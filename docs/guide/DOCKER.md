← [Back to Main README](../../README.md)

# Docker Training Guide

## 🐳 Containerized Deployment

### Prerequisites

**Docker permissions** — add your user to the `docker` group to avoid `sudo`:
```bash
sudo usermod -aG docker $USER
```
Log out and back in (or run `newgrp docker`) for the change to take effect.

**NVIDIA Container Toolkit** — required for GPU passthrough (`--gpus all`):
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Build Image

```bash
# Option 1: convenience script
bash docker/build.sh

# Option 2: manual build (from repo root)
docker build -t visionforge:latest -f docker/Dockerfile .
```

### Execution Modes

**Standard Mode** (Performance Optimized):
```bash
docker run -it --rm \
  --gpus all \
  --shm-size=8g \
  -u $(id -u):$(id -g) \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/outputs:/app/outputs \
  visionforge:latest \
  --config recipes/config_resnet_18.yaml
```

**Strict Reproducibility Mode** (Bit-Perfect Determinism):
```bash
docker run -it --rm \
  --gpus all \
  --shm-size=8g \
  -u $(id -u):$(id -g) \
  -e DOCKER_REPRODUCIBILITY_MODE=TRUE \
  -e PYTHONHASHSEED=42 \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/outputs:/app/outputs \
  visionforge:latest \
  --config recipes/config_resnet_18.yaml
```

> [!NOTE]
> - `TORCH_HOME`, `MPLCONFIGDIR`, `IN_DOCKER`, and `CUBLAS_WORKSPACE_CONFIG` are pre-set in the image and do not need to be passed at runtime
> - `PYTHONHASHSEED` must be set at container startup (before the Python interpreter loads) to guarantee hash determinism — the image default is `0`, override with `-e` if a specific seed is needed
> - `--gpus all` requires NVIDIA Container Toolkit

---
