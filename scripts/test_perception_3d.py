import sys
import os
from pathlib import Path
import numpy as np
import cv2
import logging

# Add workspace root to path
workspace_root = Path(__file__).resolve().parents[1]
sys.path.append(str(workspace_root))

# Configure logging
logging.basicConfig(level=logging.INFO)

from orion.perception.perception_3d import Perception3DEngine

def test_perception_3d():
    print("Initializing Perception3DEngine...")
    try:
        engine = Perception3DEngine(enable_depth=True, enable_slam=True)
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return

    print("Creating dummy frame...")
    width, height = 640, 480
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Add some texture
    cv2.randu(frame, 0, 255)
    
    print("Processing frame...")
    try:
        result = engine.process_frame(
            frame=frame,
            detections=[],
            frame_number=0,
            timestamp=0.0
        )
    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Verification:")
    
    # Check Depth
    if result.depth_map is not None:
        print(f"✓ Depth map generated: {result.depth_map.shape}")
    else:
        print("✗ Depth map missing")
        
    # Check Pose
    if result.camera_pose is not None:
        print(f"✓ Camera pose generated: \n{result.camera_pose}")
    else:
        print("✗ Camera pose missing")
        
    # Check SLAM Engine state
    if engine.slam_engine:
        print(f"✓ SLAM Engine poses count: {len(engine.slam_engine.poses)}")
        if len(engine.slam_engine.poses) > 0:
            print("✓ SLAM state updated")
        else:
            print("✗ SLAM state NOT updated")
    else:
        print("✗ SLAM Engine missing")

if __name__ == "__main__":
    # Set MPS fallback for DepthAnything3
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    test_perception_3d()
