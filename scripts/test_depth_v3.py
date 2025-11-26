import sys
import os
from pathlib import Path
import numpy as np
import cv2
import torch

# Add workspace root to path
workspace_root = Path(__file__).resolve().parents[1]
sys.path.append(str(workspace_root))

from orion.perception.depth import DepthEstimator

def test_depth_v3():
    print("Initializing DepthEstimator (V3)...")
    try:
        estimator = DepthEstimator(model_name="depth_anything_v3", model_size="small")
    except Exception as e:
        print(f"Failed to initialize estimator: {e}")
        return

    print("Creating dummy image...")
    # Create a gradient image
    width, height = 640, 480
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        image[i, :, :] = (i / height) * 255
    
    print(f"Running inference on image of shape {image.shape}...")
    try:
        depth_map, _ = estimator.estimate(image)
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Inference successful!")
    print(f"Depth map shape: {depth_map.shape}")
    print(f"Depth map range: {depth_map.min()} - {depth_map.max()}")
    
    if depth_map.shape != (height, width):
        print("ERROR: Output shape mismatch!")
    else:
        print("SUCCESS: Output shape matches input.")

if __name__ == "__main__":
    test_depth_v3()
