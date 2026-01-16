#!/usr/bin/env python3
"""
Batch recluster memory for all videos with YOLO-World detections.
Re-runs Phase 2 (Re-ID + memory clustering) for each video.
"""

import subprocess
import sys
from pathlib import Path

videos = [
    "0001_4164158586", "0003_3396832512", "0003_6141007489", "0004_11566980553",
    "0005_2505076295", "0006_2889117240", "0008_6225185844", "0008_8890945814",
    "0010_8610561401", "0018_3057666738",
    "0020_10793023296", "0020_5323209509", "0021_2446450580", "0021_4999665957",
    "0024_5224805531", "0026_2764832695", "0027_4571353789", "0028_3085751774",
    "0028_4021064662", "0029_5139813648"
]

base_video_path = "datasets/PVSG/VidOR/mnt/lustre/jkyang/CVPR23/openpvsg/data/vidor/videos"
results_dir = Path("results")

print("Batch reclustering memory for all videos...")
print(f"Total videos: {len(videos)}\n")

failed = []
for idx, vid in enumerate(videos, 1):
    video_file = Path(base_video_path) / f"{vid}.mp4"
    
    if not video_file.exists():
        print(f"[{idx}/{len(videos)}] {vid}... ⚠️  Video not found")
        failed.append(vid)
        continue
    
    print(f"[{idx}/{len(videos)}] {vid}... ", end="", flush=True)
    
    # Run memory clustering (Phase 2)
    result = subprocess.run([
        sys.executable, "-m", "orion.cli.run_showcase",
        "--episode", vid,
        "--video", str(video_file),
        "--skip-phase1",
        "--force-memory",
        "--skip-graph",
        "--no-overlay"
    ], capture_output=True, text=True, timeout=120)
    
    if result.returncode == 0:
        # Check how many objects were created
        mem_file = results_dir / vid / "memory.json"
        if mem_file.exists():
            import json
            with open(mem_file) as f:
                mem = json.load(f)
                num_objects = len(mem.get('objects', []))
            print(f"✓ ({num_objects} objects)")
        else:
            print("✓")
    else:
        print(f"✗ Error")
        failed.append(vid)

print(f"\n{'='*60}")
if failed:
    print(f"Failed ({len(failed)}):")
    for vid in failed:
        print(f"  - {vid}")
else:
    print("✓ All videos reclustered successfully!")

print(f"\nNext: python scripts/eval_sgg_filtered.py")
