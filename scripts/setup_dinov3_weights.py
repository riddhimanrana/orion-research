#!/usr/bin/env python3
"""
Download DINOv3 Weights Helper
==============================

DINOv3 models are gated and require manual download after requesting access:
https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/

This script helps organize downloaded weights for Orion.

After getting access and receiving the download URLs via email:
1. Use wget to download the model files (NOT a browser)
2. Run this script to organize them

Example:
    # Download (replace URL with your actual link from Meta)
    wget -O dinov3_vitb16_pretrain.pth "YOUR_DINOV3_VITB16_URL"
    
    # Organize
    python scripts/setup_dinov3_weights.py --checkpoint dinov3_vitb16_pretrain.pth
"""

import argparse
import shutil
from pathlib import Path


def setup_dinov3_weights(checkpoint_path: Path, models_dir: Path):
    """
    Organize DINOv3 checkpoint into expected directory structure.
    
    For Transformers compatibility, we need:
    - config.json
    - pytorch_model.bin (or model.safetensors)
    - preprocessor_config.json
    
    Since DINOv3 raw checkpoints don't have these, we'll create a
    minimal structure for torch.hub compatibility instead.
    """
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    # Create target directory
    target_dir = models_dir / "dinov3-vitb16"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy checkpoint
    target_checkpoint = target_dir / "pytorch_model.bin"
    shutil.copy2(checkpoint_path, target_checkpoint)
    
    # Create minimal config for torch.hub loading
    config_json = target_dir / "config.json"
    config_json.write_text('''{
  "model_type": "dinov3",
  "architectures": ["Dinov3Model"],
  "hidden_size": 768,
  "num_hidden_layers": 12,
  "num_attention_heads": 12,
  "intermediate_size": 3072,
  "hidden_act": "gelu",
  "image_size": 224,
  "patch_size": 16
}''')
    
    # Create preprocessor config
    preprocessor_config = target_dir / "preprocessor_config.json"
    preprocessor_config.write_text('''{
  "do_resize": true,
  "size": {"height": 224, "width": 224},
  "do_normalize": true,
  "image_mean": [0.485, 0.456, 0.406],
  "image_std": [0.229, 0.224, 0.225]
}''')
    
    print(f"✓ DINOv3 weights organized in: {target_dir}")
    print(f"  - Checkpoint: {target_checkpoint.name}")
    print(f"  - Config: {config_json.name}")
    print(f"  - Preprocessor: {preprocessor_config.name}")
    print()
    print("Note: For full Transformers compatibility, you may need to convert")
    print("the checkpoint format. For now, use torch.hub or timm backends.")
    
    return True


def print_instructions():
    print("""
DINOv3 Download Instructions
=============================

1. Request access at:
   https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/

2. After approval, you'll receive an email with download links

3. Download using wget (NOT browser):
   wget -O dinov3_vitb16.pth "YOUR_URL_HERE"

4. Run this script:
   python scripts/setup_dinov3_weights.py --checkpoint dinov3_vitb16.pth

Available models:
- ViT-S/16 (21M params)
- ViT-B/16 (86M params) ← recommended
- ViT-L/16 (300M params)
- ViT-H+/16 (840M params)

For now, Orion will use public DINOv2 as fallback if DINOv3 is not available.
""")


def main():
    parser = argparse.ArgumentParser(description="Setup DINOv3 weights for Orion")
    parser.add_argument("--checkpoint", type=Path, help="Path to downloaded .pth checkpoint")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Models directory")
    parser.add_argument("--instructions", action="store_true", help="Print download instructions")
    
    args = parser.parse_args()
    
    if args.instructions or not args.checkpoint:
        print_instructions()
        if not args.checkpoint:
            return
    
    setup_dinov3_weights(args.checkpoint, args.models_dir)


if __name__ == "__main__":
    main()
