#!/usr/bin/env python3
"""Rebuild all scene graphs with aggressive thresholds."""

import json
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

print("Rebuilding all scene graphs with AGGRESSIVE thresholds...")
for vid in videos:
    sg_file = results_dir / vid / "scene_graph.jsonl"
    video_file = Path(base_video_path) / f"{vid}.mp4"
    
    if not video_file.exists():
        print(f"⚠️  Video not found: {vid}")
        continue
    
    # Delete old scene graph
    if sg_file.exists():
        sg_file.unlink()
    
    print(f"  {vid}...", end=" ", flush=True)
    
    # Rebuild with aggressive mode
    result = subprocess.run([
        "python", "-m", "orion.cli.run_showcase",
        "--episode", vid,
        "--video", str(video_file),
        "--skip-phase1",
        "--force-graph",
        "--no-overlay",
        "--aggressive-sgg"  # NEW: aggressive mode
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓")
    else:
        print(f"✗ Error")

print("\n✓ All graphs rebuilt with aggressive thresholds")
