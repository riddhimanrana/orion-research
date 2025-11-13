#!/usr/bin/env python3
"""
Test All Orion Backends
========================

Comprehensive test of all model backends to ensure they work correctly:
- YOLO (detection)
- CLIP (embeddings + text)
- DINO (vision embeddings)
- FastVLM (descriptions)

Usage:
    python scripts/test_all_backends.py [--device auto|cpu|mps|cuda]
"""

import argparse
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np


def test_yolo(mm, device: str):
    """Test YOLO detection"""
    print("\n" + "=" * 60)
    print("Testing YOLO Detection")
    print("=" * 60)
    
    try:
        yolo = mm.yolo
        print(f"✓ YOLO loaded: {mm.yolo_model_name}")
        
        # Create dummy image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Run detection
        results = yolo(img, verbose=False)
        
        # Check results
        if results and len(results) > 0:
            n_detections = len(results[0].boxes) if hasattr(results[0], 'boxes') else 0
            print(f"✓ Detection successful ({n_detections} detections on random image)")
        else:
            print("⚠ Detection returned no results (expected for random image)")
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO test failed: {e}")
        traceback.print_exc()
        return False


def test_clip(mm, device: str):
    """Test CLIP embeddings"""
    print("\n" + "=" * 60)
    print("Testing CLIP Embeddings")
    print("=" * 60)
    
    try:
        clip = mm.clip
        print("✓ CLIP loaded")
        
        # Create dummy image (RGB)
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Encode image
        emb = clip.encode_image(img)
        print(f"✓ Image encoding successful: {emb.shape}, dtype={emb.dtype}")
        
        # Check normalization
        norm = np.linalg.norm(emb)
        print(f"  L2 norm: {norm:.4f} (should be ~1.0 if normalized)")
        
        # Test text encoding
        text_emb = clip.encode_text("a photo of a cat")
        print(f"✓ Text encoding successful: {text_emb.shape}")
        
        # Test similarity
        sim = float(np.dot(emb, text_emb))
        print(f"  Image-text similarity: {sim:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ CLIP test failed: {e}")
        traceback.print_exc()
        return False


def test_dino(mm, device: str):
    """Test DINO embeddings"""
    print("\n" + "=" * 60)
    print("Testing DINO Embeddings")
    print("=" * 60)
    
    try:
        dino = mm.dino
        backend = getattr(dino, '_backend', 'unknown')
        print(f"✓ DINO loaded (backend: {backend})")
        
        # Create dummy image (RGB)
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Encode image
        emb = dino.encode_image(img)
        print(f"✓ Image encoding successful: {emb.shape}, dtype={emb.dtype}")
        
        # Check normalization
        norm = np.linalg.norm(emb)
        print(f"  L2 norm: {norm:.4f} (should be ~1.0 if normalized)")
        
        # Test with another image for similarity
        img2 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        emb2 = dino.encode_image(img2)
        
        sim = float(np.dot(emb, emb2))
        print(f"✓ Second encoding successful")
        print(f"  Random image similarity: {sim:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ DINO test failed: {e}")
        traceback.print_exc()
        return False


def test_fastvlm(mm, device: str):
    """Test FastVLM descriptions"""
    print("\n" + "=" * 60)
    print("Testing FastVLM Descriptions")
    print("=" * 60)
    
    try:
        from PIL import Image
        
        fastvlm = mm.fastvlm
        print("✓ FastVLM loaded")
        
        # Create dummy image and convert to PIL
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Generate description
        desc = fastvlm.generate(img, prompt="Describe this image.")
        print(f"✓ Description generated: {len(desc)} chars")
        print(f"  Preview: {desc[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ FastVLM test failed: {e}")
        traceback.print_exc()
        return False


def test_memory_usage(mm):
    """Test memory reporting"""
    print("\n" + "=" * 60)
    print("Memory Usage")
    print("=" * 60)
    
    try:
        usage = mm.get_memory_usage()
        for key, val in usage.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.2f}")
            else:
                print(f"  {key}: {val}")
        return True
    except Exception as e:
        print(f"❌ Memory usage check failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test all Orion backends")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                       help="Device to use")
    parser.add_argument("--skip", nargs="+", choices=["yolo", "clip", "dino", "fastvlm"],
                       help="Backends to skip")
    
    args = parser.parse_args()
    
    # Setup path
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))
    
    print("Orion Backend Test Suite")
    print("=" * 60)
    print(f"Device: {args.device}")
    
    # Import model manager
    try:
        from orion.managers.model_manager import ModelManager
        mm = ModelManager.get_instance()
        
        # Override device if specified
        if args.device != "auto":
            mm.device = args.device
            print(f"Device override: {mm.device}")
        else:
            print(f"Auto-detected device: {mm.device}")
            
    except Exception as e:
        print(f"❌ Failed to initialize ModelManager: {e}")
        traceback.print_exc()
        return 1
    
    # Run tests
    results = {}
    skip = set(args.skip or [])
    
    if "yolo" not in skip:
        results["yolo"] = test_yolo(mm, mm.device)
    
    if "clip" not in skip:
        results["clip"] = test_clip(mm, mm.device)
    
    if "dino" not in skip:
        results["dino"] = test_dino(mm, mm.device)
    
    if "fastvlm" not in skip:
        results["fastvlm"] = test_fastvlm(mm, mm.device)
    
    # Memory usage
    test_memory_usage(mm)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {name:12s} {status}")
    
    print()
    print(f"Overall: {passed}/{total} tests passed")
    
    # Cleanup
    try:
        mm.cleanup()
        print("\n✓ Cleanup complete")
    except Exception as e:
        print(f"\n⚠ Cleanup warning: {e}")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
