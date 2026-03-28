# Multi-Architecture Docker Environment

A multi-architecture Docker environment for deep learning with Python 3.12, PyTorch, and HuggingFace Transformers, supporting both NVIDIA GPU (CUDA) and Apple Silicon (MPS) deployments.

## Quick Start

### Build the Docker Image

**For NVIDIA ARM64 GPU (CUDA):**
```bash
./docker-build.sh linux/arm64
# or
docker build --build-arg arch=linux/arm64 -f Dockerfile.arm64 -t torch-nvidia-arm64 --platform linux/arm64 .
```

**For Apple Silicon M1 (MPS):**
```bash
./docker-build.sh mac-aarch64
# or
docker build --build-arg arch=mac-aarch64 -f Dockerfile.macos -t torch-mps-m1 .
```

### Run the Container

**NVIDIA ARM64 GPU:**
```bash
docker run --gpus all -it torch-nvidia-arm64
```

**Apple Silicon M1:**
```bash
docker run --gpus=all -it torch-mps-m1
```

### GPU Verification

```bash
# Run GPU verification script
docker run --gpus all -it torch-nvidia-arm64 python /app/src/verify_gpu.py

# Run GPU verification script on Apple Silicon
docker run --gpus=all -it torch-mps-m1 python /app/src/verify_gpu.py
```

## Project Structure

```
dscontainer/
├── Dockerfile              # Main multi-architecture Dockerfile
├── Dockerfile.arm64        # NVIDIA GPU version for ARM64
├── Dockerfile.macos        # Apple Silicon M1 version
├── docker-compose.yml      # Docker Compose configuration
├── docker-build.sh         # Build script
├── pyproject.toml          # uv package configuration
├── .dockerignore          # Docker build exclusions
├── README.md               # This file
└── src/                    # Application source code
    └── verify_gpu.py      # GPU verification script
```

## Features

- **Python 3.12**: Latest Python version for deep learning applications
- **PyTorch with GPU Support**:
  - CUDA 12.x for NVIDIA GPU deployments
  - MPS (Metal Performance Shaders) for Apple Silicon M1/M2 chips
- **HuggingFace Transformers**: Latest transformers library with GPU acceleration
- **uv Package Manager**: Fast Python package management
- **Multi-architecture Support**: Builds for ARM64 on x86_64 hosts with Docker Buildx
- **Non-root User**: Security best practices with user permissions
- **GPU Verification**: Built-in scripts to verify GPU operations

## Architecture Support

### NVIDIA ARM64 GPU Container
- **Architecture**: Linux ARM64 with NVIDIA CUDA support
- **GPU**: CUDA 12.x with cuDNN
- **Use Case**: ARM64 servers with NVIDIA GPU attached
- **Docker Command**: `docker build --build-arg arch=linux/arm64`

### Apple Silicon M1 Container
- **Architecture**: macOS with Apple Silicon (ARM64)
- **GPU**: MPS (Metal Performance Shaders)
- **Use Case**: M1, M2, M3 Macs with Apple Silicon hardware
- **Docker Command**: `docker build --build-arg arch=mac-aarch64`

## Dependencies

### Core Dependencies
- Python 3.12
- PyTorch 2.0.1
  - CUDA 12.3 support for NVIDIA
  - MPS support for Apple Silicon
- HuggingFace Transformers 4.30.2
- Accelerate 0.20.3
- Datasets 2.13.0

### Docker Dependencies
- Docker 20.10+ with buildx support
- NVIDIA Container Toolkit (for NVIDIA GPU support)
- Apple Silicon compatible Docker Engine

## Configuration

### pyproject.toml
Configure your Python dependencies in `pyproject.toml` using uv:

```toml
[project]
name = "deep-learning-env"
version = "1.0.0"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "accelerate>=0.20.0",
    "datasets>=2.0.0",
]
```

### Environment Variables

**NVIDIA GPU:**
- `CUDA_VISIBLE_DEVICES`: GPU device mapping
- `CUDA_HOME`: CUDA installation directory
- `LD_LIBRARY_PATH`: CUDA library paths

**Apple Silicon M1:**
- `PYTORCH_ENABLE_MPS_FALLBACK`: Enable CPU fallback for MPS
- `CUDA_VISIBLE_DEVICES`: Disable CUDA (MPS preferred)

## Build Instructions

### Prerequisites

1. **Docker Installation:**
   ```bash
   # Install Docker Desktop with appropriate GPU support
   # NVIDIA: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
   # Apple Silicon: Docker Desktop for Mac with Silicon support
   ```

2. **Optional: Docker Buildx (for cross-platform builds):**
   ```bash
   docker buildx install
   ```

### Building the Images

**Using the build script:**
```bash
./docker-build.sh linux/arm64  # NVIDIA GPU version
./docker-build.sh mac-aarch64   # Apple Silicon version
```

**Using Docker directly:**
```bash
# NVIDIA GPU version
docker build --build-arg arch=linux/arm64 -f Dockerfile.arm64 -t torch-nvidia-arm64 .

# Apple Silicon version
docker build --build-arg arch=mac-aarch64 -f Dockerfile.macos -t torch-mps-m1 .
```

### Using Docker Compose

```bash
# Build and run NVIDIA GPU container
docker-compose up nvidia-arm64

# Build and run Apple Silicon container
docker-compose up mps-m1

# Run verification script with Docker Compose
docker-compose exec nvidia-arm64 python /app/src/verify_gpu.py
```

## Running Applications

### Python Scripts

1. **Copy your code to the container:**
   ```bash
   docker run -v $(pwd):/app --gpus all -it torch-nvidia-arm64
   cd /app
   python your_script.py
   ```

2. **Use the verification script:**
   ```bash
   docker run --gpus all -it torch-nvidia-arm64 python /app/src/verify_gpu.py
   ```

### Jupyter Notebooks

1. **Start Jupyter:**
   ```bash
   docker run --gpus all -p 8888:8888 -v $(pwd):/app -it torch-nvidia-arm64 bash
   cd /app && jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
   ```

2. **Access from host:**
   ```
   http://localhost:8888/?token=<your-token>
   ```

## Troubleshooting

### NVIDIA GPU Issues

1. **CUDA not detected:**
   ```bash
   # Check if NVIDIA container toolkit is installed
   nvidia-smi

   # Verify container has GPU access
   docker run --gpus all nvidia/cuda:12.0-base nvidia-smi
   ```

2. **Driver compatibility:**
   Ensure your host machine has compatible NVIDIA drivers and CUDA libraries.

### Apple Silicon Issues

1. **MPS not available:**
   ```bash
   # Verify MPS availability
   python -c "import torch; print(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())"

   # Check if PyTorch is built with MPS support
   python -c "import torch; print('MPS built:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_built())"
   ```

2. **Performance issues:**
   - Ensure running on actual Apple Silicon hardware, not Intel Mac
   - Use the appropriate `torch-mps` image

### Build Issues

1. **Cross-platform build:**
   ```bash
   # Use Docker Buildx for cross-platform builds
   docker buildx build --platform linux/arm64 --build-arg arch=linux/arm64 -t torch-nvidia-arm64 .
   ```

2. **Permission issues:**
   ```bash
   # Fix Docker permissions
   sudo usermod -aG docker $USER
   logout and login again
   ```

## Performance Optimization

### NVIDIA GPU

1. **Enable CUDA optimizations:**
   ```bash
   export TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6"
   ```

2. **Use data loaders for large datasets:**
   ```python
   from torch.utils.data import DataLoader
   dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
   ```

### Apple Silicon M1

1. **Enable MPS optimizations:**
   ```python
   if hasattr(torch.backends, 'mps'):
       torch.backends.mps.set_available(True)
   ```

2. **Use Metal optimized operations:**
   ```python
   # PyTorch automatically uses MPS when available
   device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
   ```

## Advanced Usage

### Custom Dependencies

1. **Add dependencies to pyproject.toml:**
   ```toml
   [project.optional-dependencies]
   dev = ["pytest", "black", "flake8"]
   ```

2. **Install additional packages:**
   ```bash
   docker run --gpus all -it torch-nvidia-arm64 pip install <package-name>
   ```

### Multi-GPU Support

1. **NVIDIA GPU:**
   ```python
   import torch
   device = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
   ```

2. **Apple Silicon M1:** Limited multi-GPU support - typically uses single GPU

### Model Optimization

1. **Use mixed precision:**
   ```python
   if torch.cuda.is_available():
       scaler = torch.cuda.amp.GradScaler()
   ```

2. **Quantization:**
   ```python
   model = transformers.AutoModelForSequenceClassification.from_pretrained(
       "bert-base-uncased",
       torch_dtype=torch.float16
   )
   ```

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [Docker GPU Support](https://docs.docker.com/engine/guides/containers/gpus/)
- [uv Package Manager](https://github.com/astral-sh/uv)
- [CUDA on Linux](https://docs.nvidia.com/cuda/)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metal/pytorch_supporting_metal_on_apple_silicon)

## License

This project is provided as-is for educational and development purposes.

## Contributing

Contributions are welcome! Please ensure:
- Your code follows best practices
- GPU operations are properly tested
- Documentation is updated when needed
- Cross-platform compatibility is maintained

## Support

For issues and questions:
- Check the [Troubleshooting](#troubleshooting) section
- Review PyTorch and HuggingFace documentation
- Check Docker and GPU driver documentation