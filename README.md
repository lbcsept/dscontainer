# Multi-Architecture Docker Environment

Python 3.12, PyTorch with CUDA 12.x, and HuggingFace Transformers in Docker.

## Quick Start

### Build & Run NVIDIA GPU (ARM64):

```bash
docker build --build-arg arch=linux/arm64 -f Dockerfile.arm64 -t torch-nvidia-arm64 .
docker run --gpus all -it torch-nvidia-arm64
docker run --gpus all -it torch-nvidia-arm64 python /app/src/verify_gpu.py
```

### Build & Run Apple Silicon M1:

```bash
docker build --build-arg arch=mac-aarch64 -f Dockerfile.macos -t torch-mps-m1 .
docker run --gpus=all -it torch-mps-m1
docker run --gpus=all -it torch-mps-m1 python /app/src/verify_gpu.py
```

## Features

- **Python 3.12** with `uv` package manager
- **PyTorch** with CUDA 12.x for NVIDIA or MPS for Apple Silicon
- **HuggingFace Transformers** GPU acceleration
- Multi-architecture Docker support
- GPU verification script included

## Project Structure

```
dscontainer/
├── Dockerfile              # Multi-architecture Dockerfile
├── Dockerfile.arm64        # NVIDIA GPU version
├── Dockerfile.macos        # Apple Silicon version
├── docker-compose.yml      # Docker Compose
├── docker-build.sh         # Build script
├── pyproject.toml          # uv configuration
├── .dockerignore
└── src/verify_gpu.py      # GPU verification script
```

## Detailed Documentation

See [README_FULL.md](README_FULL.md) for complete documentation, troubleshooting, performance optimization, and advanced usage.