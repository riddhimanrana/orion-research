import torch
import timm
from pathlib import Path
import sys

def test_load():
    weights_dir = Path("models/dinov3-vitb16")
    if not weights_dir.exists():
        print(f"Weights dir {weights_dir} does not exist")
        return

    pth_files = list(weights_dir.glob("*.pth"))
    if not pth_files:
        print(f"No .pth files found in {weights_dir}")
        return

    weights_path = pth_files[0]
    print(f"Found weights: {weights_path}")

    # Try loading with timm
    model_name = "vit_base_patch16_224" # Guessing based on filename
    print(f"Attempting to load into {model_name}...")
    
    try:
        model = timm.create_model(model_name, pretrained=False)
        # Load state dict
        state_dict = torch.load(weights_path, map_location="cpu")
        
        # Check if state_dict has a 'model' or 'teacher' key (common in DINO checkpoints)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "teacher" in state_dict:
            state_dict = state_dict["teacher"]
            
        # Remove prefix if needed
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            k = k.replace("backbone.", "")
            new_state_dict[k] = v
            
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded with {len(missing)} missing keys and {len(unexpected)} unexpected keys")
        
        if len(missing) > 0:
            print(f"Missing keys (first 5): {missing[:5]}")
        if len(unexpected) > 0:
            print(f"Unexpected keys (first 5): {unexpected[:5]}")
            
    except Exception as e:
        print(f"Failed to load: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load()
