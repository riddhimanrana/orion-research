#!/usr/bin/env python3
"""Reprocess all videos with YOLO-World PVSG vocabulary."""

import subprocess
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
yolo_prompt = Path("pvsg_yoloworld_prompt.txt").read_text()

print("Reprocessing all videos with YOLO-World PVSG vocabulary...")
print("This will take ~30-60 minutes (~2-3 min per video)\n")

for i, vid in enumerate(videos, 1):
    video_file = Path(base_video_path) / f"{vid}.mp4"
    
    if not video_file.exists():
        print(f"[{i:2d}/20] ⚠️  {vid} - video not found")
        continue
    
    print(f"[{i:2d}/20] {vid}...", end=" ", flush=True)
    
    result = subprocess.run([
        "python", "-m", "orion.cli.run_showcase",
        "--episode", vid,
        "--video", str(video_file),
        "--detector-backend", "yoloworld",
        "--yoloworld-prompt", yolo_prompt,
        "--force-phase1"  # Force re-detect with new vocab
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        # Count objects detected
        try:
            import json
            mem_file = Path("results") / vid / "memory.json"
            if mem_file.exists():
                with open(mem_file) as f:
                    mem = json.load(f)
                    obj_count = len(mem.get("objects", []))
                    print(f"✓ ({obj_count} objects)")
            else:
                print("✓")
        except:
            print("✓")
    else:
        print(f"✗ Error")

print("\n✓ All videos reprocessed with YOLO-World vocabulary")
print("Now run: python scripts/rebuild_all_graphs.py")
