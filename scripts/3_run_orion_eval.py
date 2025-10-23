#!/usr/bin/env python3
"""
Run Orion evaluation pipeline.
Guides user through running Orion and evaluating results.
"""

import json
import os

GROUND_TRUTH_FILE = 'data/tao_75_test/ground_truth.json'
RESULTS_DIR = 'data/tao_75_test/results'
FRAMES_DIR = 'data/tao_frames'

def main():
    print("="*70)
    print("STEP 3: Run Orion Scene Graph Generation")
    print("="*70)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Check ground truth exists
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"\n❌ Ground truth not found: {GROUND_TRUTH_FILE}")
        print("   Run script 2 first: python scripts/2_prepare_ground_truth.py")
        return
    
    # Load and display info
    print("\n1. Loading ground truth...")
    with open(GROUND_TRUTH_FILE, 'r') as f:
        gt_data = json.load(f)
    
    print(f"   ✓ Videos: {len(gt_data.get('videos', []))}")
    print(f"   ✓ Images/Frames: {len(gt_data.get('images', []))}")
    print(f"   ✓ Annotations: {len(gt_data.get('annotations', []))}")
    print(f"   ✓ Categories: {len(gt_data.get('categories', []))}")
    
    print("\n" + "="*70)
    print("RUNNING ORION PIPELINE")
    print("="*70)
    
    print(f"""
Now you need to run Orion on the TAO videos:

Video frames location:
   {FRAMES_DIR}/frames/validation/

Expected output:
   Save scene graph predictions to:
   {RESULTS_DIR}/predictions.json

Prediction format (JSON array):
   [
     {{
       "image_id": int,              # from ground truth images
       "category_id": int,           # object class
       "bbox": [x, y, width, height],
       "score": float,               # confidence 0-1
       "track_id": int,              # optional, for tracking
       "video_id": int               # optional
     }},
     ...
   ]

Steps:
1. Load frames from: {FRAMES_DIR}/frames/validation/
2. Run Orion scene graph generation on each frame/video
3. Extract object detections (bboxes + classes)
4. Save predictions in the format above

Once complete, run:
   python scripts/4_evaluate_predictions.py
""")

if __name__ == '__main__':
    main()
