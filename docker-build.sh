#!/bin/bash
set -e

ARCH=${ARCH:-linux/arm64}

case $ARCH in
    linux/arm64|linux/amd64|linux/ppc64le)
        IMAGE="torch-env-gpu:latest"
        docker build --build-arg architecture=$ARCH --build-arg gpu_type=cuda -t $IMAGE .
        echo "✅ GPU image built: $IMAGE"
        echo "Run with: docker run --gpus all -it $IMAGE"
        ;;
    mac-aarch64)
        IMAGE="torch-env-mps:latest"
        docker build --build-arg architecture=$ARCH --build-arg gpu_type=mps -t $IMAGE .
        echo "✅ MPS image built: $IMAGE"
        echo "Run with: docker run -it $IMAGE"
        ;;
    *)
        echo "❌ Unknown architecture: $ARCH"
        exit 1
        ;;
esac