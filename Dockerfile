# =====================================================
# CUDA-enabled image (works on GPU, falls back to CPU)
# =====================================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Reproducibility defaults (overridable at runtime with -e)
ENV PYTHONHASHSEED=0
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# Headless rendering (no display server in container)
ENV IN_DOCKER=TRUE
ENV MPLCONFIGDIR=/tmp/matplotlib_cache
ENV TORCH_HOME=/tmp/torch_cache

# System dependencies (rarely changes — cached as base layer)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip separately (avoids invalidating requirements layer)
RUN pip3 install --upgrade pip

# Set working directory
WORKDIR /app

# Install Python dependencies (rebuilds only when requirements.txt changes)
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy project files (changes most frequently — last layer)
COPY . .

# Default command
ENTRYPOINT ["python3", "main.py"]

CMD []
