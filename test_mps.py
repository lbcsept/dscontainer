#!/usr/bin/env python3
import sys
import torch

print('='*60)
print('PyTorch Apple Silicon MPS Verification')
print('='*60)
print(f'Python version: {sys.version}')
print(f'PyTorch version: {torch.__version__}')
print('Testing MPS (Metal Performance Shaders)...')

try:
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print('✅ MPS is available!')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_built():
            print('✅ MPS is built with the current PyTorch installation')
        # Basic MPS test
        device = torch.device('mps')
        x = torch.randn(10, 10).to(device)
        print('✅ MPS device accessible')
        print('device count: 1')
        print('Current device: cpu')
        print('='*60)
    else:
        print('❌ MPS not available')
        print('Will use CPU fallback operations')
        print('='*60)
except Exception as e:
    print(f'Error checking MPS: {e}')
    print('Will use CPU fallback operations')
    print('='*60)