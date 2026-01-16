#!/usr/bin/env python3
"""
Pre-bake YOLO-World model with expanded vocabulary.

This script creates a custom YOLO-World model with the vocabulary pre-embedded,
eliminating the ~60s initialization overhead from set_classes().

Usage:
    python scripts/prebake_yoloworld_vocab.py
"""

import time
from pathlib import Path


def main():
    from ultralytics import YOLO
    from orion.perception.config import DetectionConfig
    
    # Get vocabulary from config
    config = DetectionConfig()
    vocab = config.yoloworld_categories()
    
    print("=" * 70)
    print("YOLO-World Vocabulary Pre-baking")
    print("=" * 70)
    print(f"\nVocabulary ({len(vocab)} classes):")
    for i, cls in enumerate(vocab):
        if cls:  # Skip empty background class for display
            print(f"  {i:3d}: {cls}")
        else:
            print(f"  {i:3d}: (background)")
    
    # Load base model
    print("\n1. Loading base model (yolov8l-worldv2.pt)...")
    t0 = time.time()
    model = YOLO("yolov8l-worldv2.pt")
    
    # Move to GPU if available to speed up CLIP encoding
    import torch
    if torch.cuda.is_available():
        print("   Moving model to CUDA for faster encoding...")
        model.to("cuda")
    
    print(f"   Done in {time.time() - t0:.2f}s")
    
    # Set classes (this is the slow part)
    print("\n2. Setting classes (this takes ~60s due to CLIP text encoding)...")
    t0 = time.time()
    model.set_classes(vocab)
    print(f"   Done in {time.time() - t0:.2f}s")
    
    # Save custom model
    output_path = Path("models/yolov8l-worldv2-general.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n3. Saving pre-baked model to {output_path}...")
    t0 = time.time()
    model.save(str(output_path))
    print(f"   Done in {time.time() - t0:.2f}s")
    
    # Verify the saved model loads fast
    print("\n4. Verifying fast load...")
    t0 = time.time()
    test_model = YOLO(str(output_path))
    load_time = time.time() - t0
    print(f"   Load time: {load_time:.2f}s (should be <5s)")
    
    # Verify classes
    print(f"\n5. Verifying classes...")
    print(f"   Model has {len(test_model.names)} classes")
    if len(test_model.names) == len(vocab):
        print("   ✓ Class count matches!")
    else:
        print(f"   ✗ Mismatch: expected {len(vocab)}, got {len(test_model.names)}")
    
    print("\n" + "=" * 70)
    print("SUCCESS!")
    print(f"Pre-baked model saved to: {output_path}")
    print(f"Model size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()
