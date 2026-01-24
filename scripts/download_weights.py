#!/usr/bin/env python3
"""
Setup script for Orion v3 models.

Ensures directories exist for:
1. DINOv3 (Gated model, requires manual download)
2. Depth Anything V3 (Managed via pip package)

Usage:
    python scripts/download_weights.py [--fake]
"""

import os
from pathlib import Path
import sys
import argparse

# Resolve to repo root from scripts/download_weights.py
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
DINOV3_DIR = MODELS_DIR / "dinov3-vitb16"

def setup_dinov3(create_fake: bool = False):
    print(f"\n[DINOv3 Setup]")
    print(f"Target directory: {DINOV3_DIR}")
    
    DINOV3_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check for weights in target dir
    weights = list(DINOV3_DIR.glob("*.pth"))
    
    # Check for weights in parent models dir (common mistake)
    parent_weights = list(MODELS_DIR.glob("dinov3*.pth"))
    
    if weights:
        print(f"✓ Found DINOv3 weights: {[w.name for w in weights]}")
    elif parent_weights:
        print(f"⚠ Found DINOv3 weights in parent directory: {[w.name for w in parent_weights]}")
        print(f"  Please move them to: {DINOV3_DIR}")
        print(f"  Command: mv {parent_weights[0]} {DINOV3_DIR}/")
    elif create_fake:
        print("⚠ Creating FAKE DINOv3 weights for testing...")
        fake_path = DINOV3_DIR / "FAKE_DINOV3"
        fake_path.touch()
        print(f"✓ Created marker file: {fake_path}")
        print("  The pipeline will now run in mock mode (random embeddings).")
    else:
        print("⚠ No weights found.")
        print(f"  Checked: {DINOV3_DIR.absolute()}")
        print("  DINOv3 is a gated model. You must download it manually:")
        print("  1. Go to https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/")
        print("  2. Download 'dinov3_vitb16_pretrain_lvd1689m.pth' (or similar)")
        print(f"  3. Place the .pth file in: {DINOV3_DIR.absolute()}")
        print("\n  (Or run with --fake to create dummy weights for testing)")

def setup_depth_anything_v3():
    print(f"\n[Depth Anything V3 Setup]")
    try:
        import depth_anything_3
        print("✓ depth_anything_3 package is installed.")
        print("  Weights will be downloaded automatically by the library on first run.")
    except ImportError:
        print("⚠ depth_anything_3 package NOT found.")
        print("  Please install it via pip:")
        print("  pip install git+https://github.com/LiheYoung/Depth-Anything-3")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true", help="Create fake weights for testing")
    args = parser.parse_args()

    print(f"=== Orion v3 Model Setup ===")
    print(f"Repo Root: {REPO_ROOT}")
    setup_dinov3(create_fake=args.fake)
    setup_depth_anything_v3()
    print("\nDone.")