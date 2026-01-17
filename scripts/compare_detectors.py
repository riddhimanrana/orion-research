#!/usr/bin/env python3
"""Quick comparison: YOLO11x vs YOLO-World on PVSG dataset"""

import json
import subprocess
import sys
from pathlib import Path

def run_showcase(detector_backend, model, episode_suffix):
    """Run showcase with given detector config"""
    video = "data/examples/episodes/1019_3004044251/video.mp4"
    episode = f"1019_3004044251_{episode_suffix}"
    
    cmd = [
        "conda", "run", "-n", "orion",
        "python", "-m", "orion.cli.run_showcase",
        "--episode", episode,
        "--video", video,
        "--detector-backend", detector_backend,
        "--fps", "2.0",
        "--no-overlay"
    ]
    
    if model:
        cmd.extend(["--detection-model", model])
    
    print(f"\n{'='*70}")
    print(f"Running: {detector_backend} (model={model})")
    print(f"Episode: {episode}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0, episode

def evaluate_episode(episode):
    """Run evaluation on episode"""
    cmd = [
        "conda", "run", "-n", "orion",
        "python", "scripts/eval_sgg_recall.py"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse results
    try:
        with open("sgg_recall_results.json") as f:
            data = json.load(f)
            for video in data["per_video"]:
                if episode in video["video_id"]:
                    return {
                        "pred_count": video["pred_count"],
                        "gt_count": video["gt_count"],
                        "matches": video["matches"],
                        "R@20": video["R@20"],
                    }
    except:
        pass
    return None

def main():
    """Run comparison"""
    
    # Test 1: YOLO11x baseline
    success1, ep1 = run_showcase("yolo", "yolo11x", "yolo11x")
    if success1:
        res1 = evaluate_episode(ep1)
        print(f"✅ YOLO11x results: {res1}")
    else:
        print(f"❌ YOLO11x failed")
        res1 = None
    
    # Test 2: YOLO-World (current)
    success2, ep2 = run_showcase("yoloworld", None, "yoloworld_baseline")
    if success2:
        res2 = evaluate_episode(ep2)
        print(f"✅ YOLO-World results: {res2}")
    else:
        print(f"❌ YOLO-World failed")
        res2 = None
    
    # Comparison
    if res1 and res2:
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        print(f"YOLO11x R@20:     {res1['R@20']:.2f}%")
        print(f"YOLO-World R@20:  {res2['R@20']:.2f}%")
        improvement = (res1['R@20'] - res2['R@20']) / max(res2['R@20'], 0.1) * 100
        print(f"Improvement:      {improvement:.0f}%")

if __name__ == "__main__":
    main()
