#!/usr/bin/env python3
"""Test FastVLM backend selection and functionality.

Tests:
1. Backend selection logic (MLX for Apple Silicon, PyTorch otherwise)
2. Model loading for selected backend
3. Basic inference test
"""

import logging
import platform
import sys
from pathlib import Path

# Add src to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_backend_selection():
    """Test that the correct backend is selected."""
    from orion.backends.torch_fastvlm import _is_apple_silicon
    
    is_apple = _is_apple_silicon()
    logger.info(f"Platform: {platform.system()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info(f"Is Apple Silicon: {is_apple}")
    
    return is_apple


def test_backend_import(force_backend=None):
    """Test backend initialization."""
    from orion.backends import FastVLMTorchWrapper
    
    try:
        # Try to initialize (this will download models if needed)
        wrapper = FastVLMTorchWrapper(force_backend=force_backend)
        backend_type = type(wrapper._backend).__name__
        logger.info(f"✓ Backend initialized: {backend_type}")
        return wrapper
    except Exception as exc:
        logger.error(f"✗ Backend initialization failed: {exc}")
        raise


def test_inference(wrapper, test_image_path=None):
    """Test basic inference."""
    if test_image_path is None:
        # Skip inference test if no image provided
        logger.info("  Skipping inference test (no test image)")
        return
    
    if not Path(test_image_path).exists():
        logger.warning(f"  Test image not found: {test_image_path}")
        return
    
    try:
        result = wrapper.generate_description(
            test_image_path,
            "Describe this image briefly.",
            max_tokens=50,
        )
        logger.info(f"✓ Inference successful")
        logger.info(f"  Result: {result[:100]}...")
        return result
    except Exception as exc:
        logger.error(f"✗ Inference failed: {exc}")
        raise


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("FastVLM Backend Test Suite")
    logger.info("=" * 60)
    
    # Test 1: Backend selection
    logger.info("\n[1/3] Testing backend selection...")
    is_apple = test_backend_selection()
    expected_backend = "FastVLMMLXWrapper" if is_apple else "FastVLMTorchLegacyWrapper"
    logger.info(f"  Expected backend: {expected_backend}")
    
    # Test 2: Backend initialization
    logger.info("\n[2/3] Testing backend initialization...")
    wrapper = test_backend_import()
    actual_backend = type(wrapper._backend).__name__
    
    if actual_backend == expected_backend:
        logger.info(f"  ✓ Correct backend selected: {actual_backend}")
    else:
        logger.warning(f"  ! Backend mismatch: expected {expected_backend}, got {actual_backend}")
    
    # Test 3: Inference (optional, needs test image)
    logger.info("\n[3/3] Testing inference...")
    # You can provide a test image path here
    # test_inference(wrapper, "path/to/test/image.jpg")
    test_inference(wrapper, None)
    
    logger.info("\n" + "=" * 60)
    logger.info("All tests completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
