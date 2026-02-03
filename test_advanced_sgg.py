#!/usr/bin/env python3
"""
Test Advanced SGG Pipeline (GDINO + V-JEPA + CIS)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_ids", nargs="+", default=["0020_5323209509"], help="Video ID(s) to test")
    parser.add_argument("--backend", default="groundingdino", choices=["yoloworld", "groundingdino"], help="Detection backend")
    args = parser.parse_args()
    
    video_ids = args.video_ids
    backend = args.backend
    
    for video_id in video_ids:
        # 1. Find Video Path
        print(f"\n--- PROCESSING {video_id} ---")
        try:
            res = subprocess.run(
                ["find", "datasets/PVSG", "-name", f"{video_id}.mp4"],
                capture_output=True, text=True, check=True
            )
            video_path = res.stdout.strip().split('\n')[0] # Take first match
            if not video_path:
                print(f"Error: Video {video_id} not found in datasets/PVSG")
                continue
            print(f"Found video: {video_path}")
        except Exception as e:
            print(f"Error searching for video: {e}")
            continue

        # 2. Run Full Pipeline via run_and_eval.py
        print(f"Running pipeline (Backend: {backend})...")
        cmd = [
            sys.executable, "run_and_eval.py",
            "--video", video_path,
            "--backend", backend
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Pipeline failed for {video_id}: {e}")
            continue

        print(f"Results for {video_id} saved in results/{video_id}/")

    print(f"\n--- BATCH COMPLETE ---\n")
    print(f"Aggregate results in sgg_recall_results.json")

if __name__ == "__main__":
    main()
