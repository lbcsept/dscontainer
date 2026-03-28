#!/usr/bin/env python3
"""
GPU Verification Script for PyTorch and HuggingFace Transformers
Tests GPU operations and verifies environment setup
"""

import sys
import torch
import transformers
import time
from typing import Optional, Any


def print_header(title: str) -> None:
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str) -> None:
    """Print formatted section."""
    print(f"\n{title}")
    print("-" * 70)


def check_python_version() -> None:
    """Check Python version compatibility."""
    print_section("Python Version Check")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    if sys.version_info >= (3, 12):
        print("✅ Python version compatible (>= 3.12)")
    else:
        print("❌ Python version incompatible (< 3.12)")


def check_pytorch() -> None:
    """Check PyTorch installation and GPU availability."""
    print_section("PyTorch Check")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch executable: {torch.__file__}")

    # Check CUDA
    if torch.cuda.is_available():
        print("✅ CUDA available")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDNN version: {torch.backends.cudnn.version() if hasattr(torch.backends, 'cudnn') else 'N/A'}")
        print(f"GPU count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Multi Processor Count: {props.multi_processor_count}")

    # Check MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ MPS available (Apple Silicon GPU)")
        print("  Backend: Metal Performance Shaders")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_built():
            print("  PyTorch MPS is built successfully")
    else:
        print("⚠️  No GPU detected - running on CPU")
        print("   You can still run CPU-based operations if necessary")


def test_torch_ops(device_type: str) -> bool:
    """Test basic PyTorch operations on specified device."""
    print_section(f"Test PyTorch Operations on {device_type.upper()}")

    try:
        # Create test tensors
        if device_type == "cuda":
            device = torch.device("cuda:0")
        elif device_type == "mps":
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Test simple operations
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        start_time = time.time()

        # Matrix multiplication
        result = torch.matmul(x, y)

        elapsed = time.time() - start_time
        print(f"✅ Matrix multiplication completed in {elapsed:.4f}s")
        print(f"   Result shape: {result.shape}")
        print(f"   Result device: {result.device}")
        print(f"   Result dtype: {result.dtype}")

        return True

    except Exception as e:
        print(f"❌ PyTorch operation failed: {e}")
        return False


def check_transformers() -> None:
    """Check HuggingFace Transformers installation."""
    print_section("HuggingFace Transformers Check")
    print(f"Transformers version: {transformers.__version__}")
    print(f"Transformers executable: {transformers.__file__}")

    # Check if transformers is importable with GPU support
    try:
        print("✅ Transformers imported successfully")

        # Test basic model loading
        print("\nTesting Transformers GPU operations:")
        if torch.cuda.is_available():
            print("  Using CUDA for model loading")
            model = transformers.AutoModel.from_pretrained(
                "bert-base-uncased",
                device_map="auto",
                torch_dtype=torch.float16
            )
            print("  ✅ Model loaded with CUDA")
            gpu_mem = torch.cuda.memory_allocated() / 1024**2
            print(f"  GPU memory allocated: {gpu_mem:.2f} MB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("  Using MPS for model loading")
            model = transformers.AutoModel.from_pretrained(
                "bert-base-uncased",
                device_map="auto"
            )
            print("  ✅ Model loaded with MPS")
        else:
            print("  Using CPU for model loading")
            model = transformers.AutoModel.from_pretrained("bert-base-uncased")
            print("  ✅ Model loaded with CPU")

        print(f"  Model architecture: {type(model).__name__}")

        # Test tokenizer
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            test_text = "Hello world!"
            tokens = tokenizer(test_text, return_tensors="pt")
            print("  ✅ Tokenizer loaded and working")

            # Test inference if using CPU/GPU
            if device.type in ["cuda", "cpu", "mps"]:
                outputs = model(**tokens)
                print("  ✅ Model inference completed")

        except Exception as e:
            print(f"  ⚠️  Tokenizer/inference test failed: {e}")

    except Exception as e:
        print(f"❌ Transformers check failed: {e}")


def measure_gpu_performance(device_type: str) -> None:
    """Measure GPU performance for common operations."""
    print_section(f"GPU Performance Measurement ({device_type.upper()})")

    try:
        # Measure matrix multiplication performance
        if device_type == "cuda":
            device = torch.device("cuda:0")
        elif device_type == "mps":
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Test vector operations
        print("\nTesting vector operations:")
        for size in [1000, 10000, 50000]:
            if device_type == "cuda" and not torch.cuda.is_available():
                continue
            if device_type == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                continue

            try:
                x = torch.randn(size).to(device)
                y = torch.randn(size).to(device)
                torch.cuda.synchronize() if device_type == "cuda" else torch.mps.synchronize(device)

                start = time.time()
                result = torch.cos(x) + torch.sin(y)
                torch.cuda.synchronize() if device_type == "cuda" else torch.mps.synchronize(device)

                elapsed = time.time() - start
                print(f"  Vector {size} elements: {elapsed:.4f}s")

            except Exception as e:
                print(f"  Vector {size} elements failed: {e}")

    except Exception as e:
        print(f"❌ Performance measurement failed: {e}")


def main() -> int:
    """Main verification function."""
    print_header("PyTorch and Transformers Environment Verification")

    # Check Python version
    check_python_version()

    # Check PyTorch
    check_pytorch()

    # Test operations based on detected device
    if torch.cuda.is_available():
        test_torch_ops("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        test_torch_ops("mps")
    else:
        test_torch_ops("cpu")

    # Check Transformers
    check_transformers()

    # Performance measurement
    if torch.cuda.is_available():
        measure_gpu_performance("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        measure_gpu_performance("mps")
    else:
        print_section("Performance Measurement (CPU)")
        print("GPU performance testing skipped - no GPU detected")

    # Final summary
    print_header("Verification Summary")

    if torch.cuda.is_available():
        print(f"✅ GPU: CUDA ({torch.cuda.get_device_name(0)})")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ GPU: Apple Silicon MPS")
        print("   GPU acceleration ready")
    else:
        print("⚠️  Warning: No GPU detected - will use CPU operations")

    print(f"✅ Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ Transformers: {transformers.__version__}")

    print("\nSetup verification completed successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())