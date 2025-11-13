#!/usr/bin/env python3
"""
Download DINOv3 ViT-L/16 from Hugging Face
==========================================

Downloads the full DINOv3 ViT-L/16 model from Hugging Face Hub.
Requires: huggingface-cli login (or HF_TOKEN env var)
"""

import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Installing huggingface_hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import snapshot_download

def main():
    model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    local_dir = Path("models/dinov3-vitl16")
    
    print(f"Downloading {model_name}...")
    print(f"Target: {local_dir.absolute()}")
    print()
    print("This will download ~1.2GB of model files.")
    print()
    
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        print()
        print("✓ Download complete!")
        print(f"  Location: {local_dir.absolute()}")
        print()
        print("Verifying files...")
        
        required_files = ["config.json", "preprocessor_config.json"]
        model_files = list(local_dir.glob("*.safetensors")) or list(local_dir.glob("*.bin"))
        
        for f in required_files:
            if (local_dir / f).exists():
                print(f"  ✓ {f}")
            else:
                print(f"  ⚠ Missing: {f}")
        
        if model_files:
            print(f"  ✓ Model weights: {model_files[0].name}")
        else:
            print(f"  ⚠ No model weights found")
        
        print()
        print("Test the model:")
        print("  python scripts/check_dino.py")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print()
        print("Troubleshooting:")
        print("1. Login to Hugging Face:")
        print("   huggingface-cli login")
        print()
        print("2. Or set token:")
        print("   export HF_TOKEN=your_token_here")
        print()
        print("3. Verify access at:")
        print("   https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
