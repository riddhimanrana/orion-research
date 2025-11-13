#!/usr/bin/env python3
"""Quick backend status check"""
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from orion.managers.model_manager import ModelManager

mm = ModelManager.get_instance()

print("Orion Backend Status")
print("=" * 60)
print(f"Device: {mm.device}")
print()

# Check DINO specifically
print("DINO Backend:")
try:
    dino = mm.dino
    backend = getattr(dino, '_backend', 'unknown')
    model_name = getattr(dino, 'model_name', 'unknown')
    local_dir = getattr(dino, 'local_weights_dir', None)
    
    print(f"  Backend: {backend}")
    print(f"  Model: {model_name}")
    if local_dir:
        print(f"  Local weights: {local_dir}")
    else:
        print(f"  Source: Hugging Face Hub")
    
    # Test embedding
    import numpy as np
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    emb = dino.encode_image(img)
    print(f"  Embedding: {emb.shape}, L2 norm={np.linalg.norm(emb):.4f}")
    print(f"  ✓ DINO ready")
except Exception as e:
    print(f"  ❌ Error: {e}")

print()

# Check if DINOv3 is available
dinov3_path = mm.models_dir / "dinov3-vitb16"
if dinov3_path.exists():
    print(f"✓ DINOv3 local weights found at {dinov3_path}")
else:
    print(f"ℹ Using DINOv2 (public fallback)")
    print(f"  To use DINOv3, download weights and place in: {dinov3_path}")
    print(f"  See: python scripts/setup_dinov3_weights.py --instructions")

mm.cleanup()
