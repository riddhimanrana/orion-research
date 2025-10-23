#!/usr/bin/env python3
"""
Download TAO test frames. Since allow_patterns doesn't work with this dataset,
use git-lfs or full download. This script provides the commands.
"""

import json
import os

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

def main():
    print("="*70)
    print("STEP 2: Download TAO Test Video Frames")
    print("="*70)
    
    print("\n1. Checking ground truth...")
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"❌ Ground truth not found: {GROUND_TRUTH_FILE}")
        print("   Run script 1 first: python scripts/1_download_75_tao_test.py")
        return
    
    video_paths = get_75_test_videos()
    print(f"✓ Found ground truth with {len(video_paths)} videos")
    
    print("\n" + "="*70)
    print("DOWNLOAD OPTIONS (choose one):")
    print("="*70)
    
    print("""
Option 1: Using git-lfs (RECOMMENDED - fastest)
   git lfs install
   git clone git@hf.co:datasets/chengyenhsieh/TAO-Amodal data/tao_frames
   
   Then extract test videos only:
   cd data/tao_frames
   python unzip_video.py  # (modify to extract test/ only)
   cd ../..

Option 2: Using Python snapshot_download (full test set ~30GB)
   python << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='chengyenhsieh/TAO-Amodal',
    repo_type='dataset',
    local_dir='data/tao_frames'
)
EOF

Option 3: Manual selective download (Python)
   python << 'EOF'
from huggingface_hub import hf_hub_download
import os

videos = {video_paths}

for i, video_path in enumerate(sorted(videos), 1):
    print(f"[{{i}}/{{len(videos)}}] {{video_path}}")
    try:
        # This may not work for directories, but worth trying
        hf_hub_download(
            repo_id='chengyenhsieh/TAO-Amodal',
            filename=f'frames/test/{{video_path}}',
            repo_type='dataset',
            local_dir='data/tao_frames'
        )
    except:
        pass
EOF
""")
    
    print("\n" + "="*70)
    print("AFTER DOWNLOADING:")
    print("="*70)
    print("""
1. Verify frames are in: data/tao_frames/frames/test/

2. Run Orion:
   python scripts/3_run_orion_eval.py

3. Save predictions to:
   data/tao_75_test/results/predictions.json

4. Evaluate:
   python scripts/4_evaluate_predictions.py

Videos to process ({} total):
""".format(len(video_paths)))
    
    for i, path in enumerate(video_paths, 1):
        if i <= 5:
            print(f"   {i}. {path}")
    if len(video_paths) > 5:
        print(f"   ... and {len(video_paths)-5} more")

if __name__ == '__main__':
    main()
