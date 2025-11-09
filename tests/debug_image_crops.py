#!/usr/bin/env python3
"""
Debug Image Crops: Visualize what FastVLM actually sees
========================================================

This script extracts and saves the image crops that are being
fed to FastVLM to verify they match what YOLO detected.

Author: Orion Research Team
Date: October 2025
"""

import cv2
import logging
from pathlib import Path

from orion.perception.perception_3d import Perception3DEngine
from orion.perception.config import get_fast_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

def main():
    print("=" * 80)
    print("DEBUG: IMAGE CROPS VISUALIZATION")
    print("=" * 80)
    print()
    
    # Create output directory
    output_dir = Path("debug_crops")
    output_dir.mkdir(exist_ok=True)
    print(f"[INIT] Saving crops to: {output_dir}/")
    print()
    
    # Create perception engine
    config = get_fast_config()
    engine = PerceptionEngine(config=config, verbose=False)
    
    # Process video
    video_path = "data/examples/video.mp4"
    print(f"[VIDEO] Processing: {video_path}")
    print()
    
    result = engine.process_video(video_path)
    
    print()
    print("=" * 80)
    print("EXTRACTED CROPS")
    print("=" * 80)
    print()
    
    # Save crops for each entity
    for entity in result.entities:
        print(f"\n{entity.entity_id} - {entity.object_class.value}")
        print(f"  Observations: {len(entity.observations)}")
        
        # Get best observation
        best_obs = entity.get_best_observation()
        if best_obs and best_obs.image_patch is not None:
            crop = best_obs.image_patch
            print(f"  Best frame: {best_obs.frame_number}")
            print(f"  Crop shape: {crop.shape}")
            print(f"  Confidence: {best_obs.confidence:.2f}")
            print(f"  Bbox: ({best_obs.bounding_box.x1:.0f}, {best_obs.bounding_box.y1:.0f}) -> "
                  f"({best_obs.bounding_box.x2:.0f}, {best_obs.bounding_box.y2:.0f})")
            
            # Save crop
            crop_path = output_dir / f"{entity.entity_id}_{entity.object_class.value}_frame{best_obs.frame_number}.jpg"
            cv2.imwrite(str(crop_path), crop)
            print(f"  ✓ Saved: {crop_path}")
            
            # Also save with bbox drawn on original frame
            # (We need to reload the frame)
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, best_obs.frame_number)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Draw bounding box
                x1, y1 = int(best_obs.bounding_box.x1), int(best_obs.bounding_box.y1)
                x2, y2 = int(best_obs.bounding_box.x2), int(best_obs.bounding_box.y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"{entity.object_class.value} ({best_obs.confidence:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save full frame with bbox
                frame_path = output_dir / f"{entity.entity_id}_{entity.object_class.value}_fullframe{best_obs.frame_number}.jpg"
                cv2.imwrite(str(frame_path), frame)
                print(f"  ✓ Saved full frame: {frame_path}")
        else:
            print(f"  ✗ No image patch available")
    
    print()
    print("=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)
    print()
    print(f"Check the crops in: {output_dir}/")
    print("Verify:")
    print("  1. Are the crops centered on the correct objects?")
    print("  2. Is there sufficient context around each object?")
    print("  3. Are the crops high quality (not blurry/dark)?")
    print()

if __name__ == "__main__":
    main()
