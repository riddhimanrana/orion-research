#!/usr/bin/env python3
"""
Download 75 TAO test videos from HuggingFace (minimal download - only what you need).
"""

import json
import os
from pathlib import Path

OUTPUT_DIR = 'data/tao_75_test'
METADATA_DIR = 'data/tao_metadata'

def download_tao_test_annotations():
    """Download TAO test annotations only"""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("❌ huggingface_hub not installed. Install with:")
        print("   pip install huggingface-hub")
        return False
    
    os.makedirs(METADATA_DIR, exist_ok=True)
    
    print("\n📥 Downloading TAO test annotations from HuggingFace...")
    
    try:
        print(f"   Downloading: amodal_annotations/test.json...")
        hf_hub_download(
            repo_id="chengyenhsieh/TAO-Amodal",
            filename='amodal_annotations/test.json',
            repo_type="dataset",
            local_dir=METADATA_DIR,
            local_dir_use_symlinks=False
        )
        print("✓ Test annotations downloaded")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def load_tao_test_data():
    """Load TAO test annotations"""
    anno_path = os.path.join(METADATA_DIR, 'amodal_annotations', 'test.json')
    
    if not os.path.exists(anno_path):
        print(f"❌ Annotations not found at: {anno_path}")
        return None
    
    try:
        with open(anno_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load annotations: {e}")
        return None

def extract_75_videos(data, max_count=75):
    """Extract first 75 videos from test set"""
    videos = data.get('videos', [])[:max_count]
    print(f"  Videos: {len(videos)}")
    print(f"  Images: {len(data.get('images', []))}")
    print(f"  Annotations: {len(data.get('annotations', []))}")
    return videos

def get_video_names(videos):
    """Extract video names/paths"""
    video_names = []
    for v in videos:
        name = v.get('name', '')
        if name:
            video_names.append(name)
    return video_names

def save_ground_truth(data, output_file):
    """Save test ground truth"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(data, f)
    
    file_size_mb = os.path.getsize(output_file) / (1024*1024)
    print(f"✓ Saved ground truth: {output_file} ({file_size_mb:.1f} MB)")

def main():
    print("="*70)
    print("STEP 1: Setup 75 TAO Test Videos for Evaluation")
    print("="*70)
    
    # Download metadata
    print("\n1. Downloading TAO test annotations...")
    if not download_tao_test_annotations():
        return
    
    # Load annotations
    print("\n2. Loading TAO test annotations...")
    data = load_tao_test_data()
    if data is None:
        return
    
    print("\n3. Extracting video info...")
    videos_75 = extract_75_videos(data, max_count=75)
    
    # Get video names
    print("\n4. Identifying 75 video paths...")
    video_names = get_video_names(videos_75)
    print(f"   Found {len(video_names)} videos")
    
    # Save ground truth
    print("\n5. Saving ground truth...")
    gt_file = f'{OUTPUT_DIR}/ground_truth.json'
    save_ground_truth(data, gt_file)
    
    print("\n" + "="*70)
    print("NEXT: Download 75 test videos")
    print("="*70)
    print(f"""
✓ Ground truth ready: {gt_file}

TO DOWNLOAD 75 TEST VIDEOS:

Option 1: Selective download (fastest, ~10-20GB)
   python scripts/2_download_75_frames.py

Option 2: Download all test frames via HuggingFace
   from huggingface_hub import snapshot_download
   snapshot_download(
       repo_id='chengyenhsieh/TAO-Amodal',
       allow_patterns='frames/test/*',
       local_dir='data/tao_frames'
   )

Option 3: Clone with git-lfs (full dataset)
   git lfs install
   git clone git@hf.co:datasets/chengyenhsieh/TAO-Amodal data/tao_frames

Then run:
   python scripts/3_run_orion_eval.py
   python scripts/4_evaluate_predictions.py
""")

if __name__ == '__main__':
    main()
