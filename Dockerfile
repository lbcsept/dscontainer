ARG DEBIAN_FRONTEND=noninteractive
ARG architecture=linux/arm64
ARG gpu_type=cuda

# Builder stage - install dependencies and Python packages
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl wget && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install uv
RUN uv pip install --system -e .

# Runtime stage - single image based on architecture
FROM python:3.12-slim

ARG architecture
ARG gpu_type

ENV PYTHONUNBUFFERED=1
ENV PATH="/home/pytorch/.local/bin:${PATH}"
ENV CUDA_VISIBLE_DEVICES=

RUN useradd -m -u 1000 pytorch && \
    chown -R pytorch:pytorch /home/pytorch

USER pytorch
WORKDIR /home/pytorch

# Install PyTorch based on architecture
# Install CUDA dependencies if needed for CUDA runtime
COPY --from=builder /home/pytorch/.local /home/pytorch/.local

COPY pyproject.toml ./
RUN uv pip install --system -e . --no-deps || true

RUN python - <<'PY'
import torch
print('=' * 50)
print(f'PyTorch Version: {torch.__version__}')
print('=' * 50)
PY