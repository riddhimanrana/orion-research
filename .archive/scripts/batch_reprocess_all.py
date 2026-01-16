#!/usr/bin/env python3
"""
Batch reprocess all 20 PVSG videos:
1. Re-cluster memory with V-JEPA2 embeddings from existing tracks
2. Rebuild scene graphs with expanded relations
3. Evaluate recall
"""

import subprocess
import sys
from pathlib import Path

VIDEOS = [
    "0001_4164158586", "0003_3396832512", "0003_6141007489", "0004_11566980553",
    "0005_2505076295", "0006_2889117240", "0008_6225185844", "0008_8890945814",
    "0010_8610561401", "0018_3057666738",
    "0020_10793023296", "0020_5323209509", "0021_2446450580", "0021_4999665957",
    "0024_5224805531", "0026_2764832695", "0027_4571353789", "0028_3085751774",
    "0028_4021064662", "0029_5139813648"
]

BASE_PATH = "datasets/PVSG/VidOR/mnt/lustre/jkyang/CVPR23/openpvsg/data/vidor/videos"

def process_video(vid: str, vid_num: int, total: int) -> bool:
    """Process a single video: re-cluster memory + rebuild scene graph."""
    print(f"[{vid_num}/{total}] {vid}... ", end="", flush=True)
    
    video_path = f"{BASE_PATH}/{vid}.mp4"
    
    # Delete old memory and scene graph to force rebuild
    results_dir = Path(f"results/{vid}")
    if (results_dir / "memory.json").exists():
        (results_dir / "memory.json").unlink()
    if (results_dir / "scene_graph.jsonl").exists():
        (results_dir / "scene_graph.jsonl").unlink()
    
    try:
        # Run showcase with skip-phase1 (reuse tracks), force memory rebuild
        result = subprocess.run([
            "python", "-m", "orion.cli.run_showcase",
            "--episode", vid,
            "--video", video_path,
            "--skip-phase1",  # Reuse existing tracks
            "--force-memory",  # Force rebuild memory with V-JEPA2
            "--no-overlay"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0 and "Built" in result.stderr:
            print("✓")
            return True
        else:
            print(f"✗ (code {result.returncode})")
            if "error" in result.stderr.lower():
                print(f"    Error: {result.stderr[-200:]}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ (timeout)")
        return False
    except Exception as e:
        print(f"✗ ({e})")
        return False

def main():
    print("Batch reprocessing all 20 PVSG videos")
    print("=" * 60)
    print("Phase 1: Re-cluster memory with V-JEPA2 embeddings")
    print("Phase 2: Rebuild scene graphs with expanded relations")
    print("=" * 60)
    print()
    
    success = 0
    failed = 0
    
    for i, vid in enumerate(VIDEOS):
        if process_video(vid, i + 1, len(VIDEOS)):
            success += 1
        else:
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Completed: {success} success, {failed} failed")
    print("=" * 60)
    
    if success > 0:
        print("\nNow run evaluation:")
        print("  python scripts/eval_sgg_filtered.py")

if __name__ == '__main__':
    main()
