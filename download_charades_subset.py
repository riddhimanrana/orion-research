#!/usr/bin/env python3
"""
Download a subset of Charades videos for Action Genome evaluation.

Charades videos are ~10 seconds each, ~3-5MB per video.
1000 videos ≈ 3-5 GB
2000 videos ≈ 6-10 GB

Usage:
    python download_charades_subset.py --num_videos 1000
    python download_charades_subset.py --num_videos 2000 --resume
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import ssl

# Charades download base URL (official)
# Videos are hosted at: http://ai2-website.s3.amazonaws.com/data/Charades/Charades_v1_480/{video_id}.mp4
# But that URL is now blocked. Alternative sources:
# 1. Academic Torrents
# 2. Direct from CMU (if available)
# 3. AWS S3 mirror

# Try multiple sources
VIDEO_SOURCES = [
    # Primary: Charades official mirror (scaled 480p)
    "https://prior-datasets.s3.us-east-2.amazonaws.com/charades/Charades_v1_480/{video_id}.mp4",
    # Backup: Original full res
    "http://ai2-website.s3.amazonaws.com/data/Charades_v1/{video_id}.mp4",
    # Backup 2: CMU mirror
    "https://www.cs.cmu.edu/~sigMDL/resources/Charades/{video_id}.mp4",
]


def download_video(video_id: str, output_dir: Path, timeout: int = 60) -> tuple:
    """Download a single video, trying multiple sources."""
    output_path = output_dir / f"{video_id}.mp4"
    
    if output_path.exists() and output_path.stat().st_size > 10000:
        return video_id, "skipped", "already exists"
    
    # Create SSL context that doesn't verify (some academic servers have cert issues)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    for source_url in VIDEO_SOURCES:
        url = source_url.format(video_id=video_id)
        try:
            # Use wget for better reliability
            result = subprocess.run(
                ["wget", "-q", "--timeout=30", "-O", str(output_path), url],
                capture_output=True,
                timeout=timeout
            )
            
            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 10000:
                return video_id, "success", url
            
            # Clean up failed download
            if output_path.exists():
                output_path.unlink()
                
        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            continue
    
    return video_id, "failed", "all sources failed"


def main():
    parser = argparse.ArgumentParser(description="Download Charades videos for Action Genome")
    parser.add_argument("--num_videos", type=int, default=1000, help="Number of videos to download")
    parser.add_argument("--output_dir", default="datasets/ActionGenome/videos", help="Output directory")
    parser.add_argument("--video_list", default="ag_video_ids.txt", help="File with video IDs")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download workers")
    parser.add_argument("--resume", action="store_true", help="Skip already downloaded")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load video IDs
    if not os.path.exists(args.video_list):
        print(f"Error: {args.video_list} not found. Run extraction first.")
        sys.exit(1)
    
    with open(args.video_list) as f:
        all_videos = [line.strip() for line in f if line.strip()]
    
    print(f"Total videos in AG: {len(all_videos)}")
    
    # Select subset
    videos_to_download = all_videos[:args.num_videos]
    print(f"Downloading {len(videos_to_download)} videos to {output_dir}")
    
    # Check existing
    if args.resume:
        existing = set(p.stem for p in output_dir.glob("*.mp4") if p.stat().st_size > 10000)
        videos_to_download = [v for v in videos_to_download if v not in existing]
        print(f"Skipping {len(existing)} already downloaded, {len(videos_to_download)} remaining")
    
    if not videos_to_download:
        print("All videos already downloaded!")
        return
    
    # Download with progress
    success = 0
    failed = 0
    skipped = 0
    
    print(f"\nDownloading with {args.workers} workers...")
    print("-" * 60)
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_video, vid, output_dir): vid 
            for vid in videos_to_download
        }
        
        for i, future in enumerate(as_completed(futures)):
            video_id, status, info = future.result()
            
            if status == "success":
                success += 1
                symbol = "✓"
            elif status == "skipped":
                skipped += 1
                symbol = "○"
            else:
                failed += 1
                symbol = "✗"
            
            # Progress
            total = success + failed + skipped
            pct = total / len(videos_to_download) * 100
            print(f"\r[{pct:5.1f}%] {symbol} {video_id}: {status} ({success} ok, {failed} failed)", end="")
            
            if (i + 1) % 100 == 0:
                print()  # Newline every 100
    
    print(f"\n\n{'='*60}")
    print(f"Download Complete!")
    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed:  {failed}")
    print(f"  Total:   {success + skipped} videos available")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
