#!/usr/bin/env python3
"""
Run Orion on a batch of PVSG videos with resume, per-episode logs, and failure handling.

Usage examples:
  python scripts/run_pvsg_batch.py --limit 100 --skip-existing
  python scripts/run_pvsg_batch.py --start 101 --limit 100 --skip-existing

Notes:
- Discovers videos under datasets/PVSG/*/mnt/lustre/jkyang/CVPR23/openpvsg/data/*/videos/*.mp4|*.MP4
- Creates results/<episode_id>/ and writes run.log per episode
- Episode IDs are pvsg_<index:03d> in discovery order
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def find_pvsg_videos(pvsg_root: Path) -> list[Path]:
    patterns = [
        "*/mnt/lustre/jkyang/CVPR23/openpvsg/data/*/videos/*.mp4",
        "*/mnt/lustre/jkyang/CVPR23/openpvsg/data/*/videos/*.MP4",
    ]
    videos: list[Path] = []
    for pat in patterns:
        videos.extend(pvsg_root.glob(pat))
    videos = sorted(videos)
    return videos


def already_done(results_root: Path, episode_id: str) -> bool:
    ep_dir = results_root / episode_id
    if not ep_dir.exists():
        return False
    # Consider episode done if tracks.jsonl exists and is non-empty
    tracks = ep_dir / "tracks.jsonl"
    return tracks.exists() and tracks.stat().st_size > 0


def run_one(repo_root: Path, episode_id: str, video_path: Path, timeout_s: int = 900) -> tuple[int, str]:
    cmd = [
        sys.executable,
        "-m",
        "orion.cli.run_showcase",
        "--episode",
        episode_id,
        "--video",
        str(video_path),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    out = proc.stdout or ""
    err = proc.stderr or ""
    return proc.returncode, out + ("\n" + err if err else "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-run Orion on PVSG videos")
    parser.add_argument("--pvsg-root", default="datasets/PVSG", help="Root of PVSG dataset")
    parser.add_argument("--results-root", default="results", help="Results root directory")
    parser.add_argument("--start", type=int, default=1, help="1-based start index in discovered videos")
    parser.add_argument("--limit", type=int, default=100, help="How many videos to process")
    parser.add_argument("--skip-existing", action="store_true", help="Skip episodes with existing outputs")
    parser.add_argument("--timeout", type=int, default=900, help="Per-video timeout (seconds)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    pvsg_root = (repo_root / args.pvsg_root).resolve()
    results_root = (repo_root / args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    videos = find_pvsg_videos(pvsg_root)
    if not videos:
        print(f"No videos found under {pvsg_root}")
        return 2

    start_idx = max(1, args.start)
    end_idx = min(len(videos), start_idx - 1 + max(0, args.limit))
    if end_idx < start_idx:
        print("Nothing to process (check --start/--limit)")
        return 0

    print(f"Found {len(videos)} videos. Processing range: {start_idx}-{end_idx}")

    successes = 0
    failures = 0
    for i in range(start_idx, end_idx + 1):
        episode_id = f"pvsg_{i:03d}"
        video_path = videos[i - 1]
        ep_dir = results_root / episode_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        log_path = ep_dir / "run.log"

        if args.skip_existing and already_done(results_root, episode_id):
            print(f"[{i}/{end_idx}] {episode_id}: SKIP (exists) -> {video_path.name}")
            continue

        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{i}/{end_idx}] {episode_id}: START {video_path.name} @ {stamp}")
        rc, output = run_one(repo_root, episode_id, video_path, timeout_s=args.timeout)
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(f"=== {stamp} | rc={rc} | video={video_path}\n")
            lf.write(output)
            lf.write("\n\n")

        if rc == 0:
            print(f"  ✓ Success")
            successes += 1
        else:
            print(f"  ✗ Failed (rc={rc}) — see {log_path}")
            failures += 1

    print(f"\nDone. Successes={successes}, Failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
