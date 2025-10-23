#!/usr/bin/env python3
"""
Download 75 TAO test videos from HuggingFace.
"""

import json
import os
import sys

GROUND_TRUTH_FILE = 'data/tao_75_test/ground_truth.json'
OUTPUT_DIR = 'data/tao_frames'

def get_75_test_videos():
    """Extract first 75 videos from ground truth"""
    with open(GROUND_TRUTH_FILE, 'r') as f:
        data = json.load(f)
    
    videos = data.get('videos', [])[:75]
    video_paths = []
    
    for v in videos:
        name = v.get('name', '')
        if name:
            video_paths.append(name)
    
    return video_paths

def download_tao_frames():
    """Download TAO test frames using snapshot_download"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ huggingface_hub not installed")
        print("   Install with: pip install huggingface-hub")
        return False
    
    print("\n" + "="*70)
    print("📥 DOWNLOADING TAO TEST FRAMES")
    print("="*70)
    print(f"\n   Size: ~30GB")
    print(f"   Time: 10-30 minutes depending on connection")
    print(f"   Location: {OUTPUT_DIR}/\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        print("Starting download from HuggingFace...")
        snapshot_download(
            repo_id='chengyenhsieh/TAO-Amodal',
            repo_type='dataset',
            local_dir=OUTPUT_DIR,
            local_dir_use_symlinks=False
        )
        print("\n✓ Download complete!")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def main():
    print("="*70)
    print("STEP 2: Download TAO Test Video Frames")
    print("="*70)
    
    print("\n1. Checking ground truth...")
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"❌ Ground truth not found: {GROUND_TRUTH_FILE}")
        print("   Run script 1 first: python scripts/1_download_75_tao_test.py")
        return False
    
    video_paths = get_75_test_videos()
    print(f"✓ Found ground truth with {len(video_paths)} videos")
    
    print("\nSample videos to download:")
    for i, path in enumerate(video_paths[:3], 1):
        print(f"   {i}. {path}")
    if len(video_paths) > 3:
        print(f"   ... and {len(video_paths)-3} more")
    
    print("\n2. Starting download...")
    if not download_tao_frames():
        return False
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("""
✓ Test frames ready in: data/tao_frames/frames/test/

Now run Orion on these videos and save predictions to:
   data/tao_75_test/results/predictions.json

Then run evaluation:
   python scripts/3_run_orion_eval.py
   python scripts/4_evaluate_predictions.py
""")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
