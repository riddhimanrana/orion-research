#!/usr/bin/env python3
"""
Validate Orion installation and system readiness.

Run this script after installation to check:
- Python version
- PyTorch & device availability (MPS/CUDA/CPU)
- Model dependencies
- Optional components (Depth, Memgraph)
"""

import sys
import os

def check_python():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python {version.major}.{version.minor} - Need Python 3.10+")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_torch():
    """Check PyTorch and device availability."""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        # Check device
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {gpu_name}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            print(f"✅ MPS (Apple Silicon) available")
        else:
            device = "cpu"
            print(f"⚠️  CPU only (no GPU acceleration)")
        
        # Test tensor creation
        test_tensor = torch.randn(10, 10).to(device)
        print(f"✅ Device test passed: {device}")
        return True
        
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False

def check_core_dependencies():
    """Check core dependencies."""
    deps = {
        "transformers": "HuggingFace Transformers",
        "ultralytics": "YOLO",
        "cv2": "OpenCV",
        "numpy": "NumPy",
        "sklearn": "Scikit-learn"
    }
    
    all_ok = True
    for module, name in deps.items():
        try:
            if module == "cv2":
                import cv2
                version = cv2.__version__
            else:
                mod = __import__(module)
                version = mod.__version__
            print(f"✅ {name} {version}")
        except ImportError:
            print(f"❌ {name} not installed")
            all_ok = False
    
    return all_ok

def check_orion():
    """Check Orion package."""
    try:
        import orion
        from orion.perception.engine import PerceptionEngine
        from orion.perception.config import PerceptionConfig
        print(f"✅ Orion package installed")
        return True
    except ImportError as e:
        print(f"❌ Orion package not found: {e}")
        print("   Run: pip install -e .")
        return False

def check_models():
    """Check if models can be loaded."""
    try:
        from orion.managers.model_manager import ModelManager
        manager = ModelManager.get_instance()
        print(f"✅ Model manager initialized")
        
        # Just check that manager works - actual model loading happens on demand
        print(f"✅ Models will auto-download on first use")
        
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def check_optional_depth():
    """Check depth estimation (optional)."""
    try:
        from orion.perception.depth import DepthEstimator
        # Don't initialize (slow), just check import
        print(f"✅ Depth estimation available (DepthAnythingV2/V3)")
        return True
    except ImportError:
        print(f"ℹ️  Depth estimation: optional (works without it)")
        return True

def check_optional_memgraph():
    """Check Memgraph connectivity (optional)."""
    try:
        from neo4j import GraphDatabase
        print(f"✅ Memgraph driver available")
        print(f"ℹ️  To use: docker run -p 7687:7687 memgraph/memgraph")
        return True
    except ImportError:
        print(f"ℹ️  Memgraph: optional, install with pip install orion[memgraph]")
        return True

def check_environment():
    """Check environment variables."""
    mps_fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    if mps_fallback == "1":
        print(f"✅ PYTORCH_ENABLE_MPS_FALLBACK=1 (recommended for Apple Silicon)")
    else:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"ℹ️  Tip: export PYTORCH_ENABLE_MPS_FALLBACK=1 (for better compatibility)")

def main():
    print("="*60)
    print("ORION SETUP VALIDATION")
    print("="*60)
    print()
    
    # Required checks
    required_checks = [
        ("Python Version", check_python),
        ("PyTorch & Device", check_torch),
        ("Core Dependencies", check_core_dependencies),
        ("Orion Package", check_orion),
        ("Model Loading", check_models),
    ]
    
    # Optional checks
    optional_checks = [
        ("Depth Estimation (optional)", check_optional_depth),
        ("Memgraph (optional)", check_optional_memgraph),
        ("Environment", check_environment),
    ]
    
    required_results = []
    for name, check_fn in required_checks:
        print(f"\n{name}:")
        print("-" * 60)
        try:
            result = check_fn()
            required_results.append(result)
        except Exception as e:
            print(f"❌ Check failed: {e}")
            required_results.append(False)
    
    for name, check_fn in optional_checks:
        print(f"\n{name}:")
        print("-" * 60)
        try:
            check_fn()  # Don't count optional checks as failures
        except Exception as e:
            print(f"ℹ️  Optional check skipped: {e}")
    
    print("\n" + "="*60)
    if all(required_results):
        print("✅ ALL CHECKS PASSED - Orion is ready!")
    else:
        print("⚠️  SOME CHECKS FAILED - See above for details")
    print("="*60)
    
    return 0 if all(required_results) else 1

if __name__ == "__main__":
    sys.exit(main())
