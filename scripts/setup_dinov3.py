#!/usr/bin/env python3
"""
DINOv3 Setup Verification Script

Verifies that DINOv3 weights are properly downloaded and located.
"""

import sys
import json
from pathlib import Path


def verify_dinov3_weights(weights_dir: str) -> bool:
    """Verify DINOv3 weights structure and file sizes."""
    path = Path(weights_dir)
    
    print(f"\nVerifying DINOv3 weights at: {path}")
    
    if not path.exists():
        print(f"❌ Weights directory not found: {weights_dir}")
        return False
    
    print(f"✅ Directory exists")
    
    # Check required files
    required_files = {
        "pytorch_model.bin": (300 * 1024**2, 400 * 1024**2),  # 300-400 MB for ViT-B
        "config.json": (1024, 10 * 1024),  # 1KB - 10KB
        "preprocessor_config.json": (1024, 10 * 1024),  # 1KB - 10KB
    }
    
    all_valid = True
    for filename, (min_size, max_size) in required_files.items():
        filepath = path / filename
        if not filepath.exists():
            print(f"❌ Missing: {filename}")
            all_valid = False
        else:
            size = filepath.stat().st_size
            if not (min_size <= size <= max_size):
                print(f"⚠️  {filename}: {size / 1024**2:.1f}MB (expected {min_size / 1024**2:.0f}-{max_size / 1024**2:.0f}MB)")
            else:
                print(f"✅ {filename}: {size / 1024**2:.1f}MB")
    
    if not all_valid:
        return False
    
    # Try to load config
    try:
        with open(path / "config.json") as f:
            config = json.load(f)
        print(f"✅ Config loaded successfully")
        
        # Print architecture info
        if "hidden_size" in config:
            print(f"   Architecture: ViT with {config['hidden_size']} hidden dim")
        if "num_hidden_layers" in config:
            print(f"   Layers: {config['num_hidden_layers']}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False
    
    return True


def main():
    print("=" * 80)
    print("DINOv3 SETUP VERIFICATION")
    print("=" * 80)
    
    weights_dir = "models/dinov3-vitb16"
    
    if verify_dinov3_weights(weights_dir):
        print("\n" + "=" * 80)
        print("✅ DINOv3 IS READY TO USE!")
        print("=" * 80)
        print("\nUsage examples:")
        print("\n1. CLI:")
        print(f"   python -m orion.cli.run_showcase \\")
        print(f"     --embedding-backend dinov3 \\")
        print(f"     --dinov3-weights {weights_dir} \\")
        print(f"     --episode my_video --video video.mp4")
        
        print("\n2. Python:")
        print(f"   from orion.perception.config import get_dinov3_config")
        print(f"   config = get_dinov3_config()")
        print(f"   config.embedding.dinov3_weights_dir = '{weights_dir}'")
        print(f"   from orion.perception.engine import PerceptionEngine")
        print(f"   engine = PerceptionEngine(config=config)")
        
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ DINOv3 SETUP INCOMPLETE")
        print("=" * 80)
        print("\nSetup instructions:")
        print("1. Download DINOv3 weights from:")
        print("   https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/")
        print("\n2. Extract to:")
        print(f"   {weights_dir}/")
        print("\n3. Verify with:")
        print(f"   python scripts/setup_dinov3.py")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
