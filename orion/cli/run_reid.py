#!/usr/bin/env python3
"""
Run Re-ID + Memory (Phase 2)
============================

Consumes tracks.jsonl and the source video to compute V-JEPA2 embeddings,
cluster tracks into persistent objects, and save memory.json. Also updates
tracks.jsonl with embedding_id fields.

Usage:
    python -m orion.cli.run_reid --episode test_validation --video data/examples/test.mp4
    python -m orion.cli.run_reid --tracks results/test_validation/tracks.jsonl --video data/examples/test.mp4 --out results/test_validation
"""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orion.config import (
    get_episode_dir,
    get_episode_video_path,
    ensure_results_dir,
)
from orion.perception.reid.matcher import build_memory_from_tracks

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Re-ID + Memory Builder")
    parser.add_argument("--episode", type=str, help="Episode ID (under data/examples/episodes)", required=False)
    parser.add_argument("--video", type=str, help="Path to source video", required=False)
    parser.add_argument("--tracks", type=str, help="Path to tracks.jsonl (defaults to results/<episode>/tracks.jsonl)", required=False)
    parser.add_argument("--out", type=str, help="Output directory (defaults to results/<episode>)", required=False)
    parser.add_argument("--threshold", type=float, default=0.70, help="Default cosine similarity threshold")
    parser.add_argument("--class-thresholds", type=str, default=None, help="Path to JSON mapping {class: threshold}")
    parser.add_argument("--max-crops-per-track", type=int, default=5, help="Max crops to embed per track")
    args = parser.parse_args()

    if not args.episode and (not args.video or not args.tracks or not args.out):
        parser.error("Provide --episode or all of --video, --tracks, --out")

    if args.episode:
        episode_id = args.episode
        episode_dir = get_episode_dir(episode_id)
        if not episode_dir.exists():
            raise FileNotFoundError(f"Episode not found: {episode_dir}")
        video_path = Path(args.video) if args.video else get_episode_video_path(episode_id)
        if video_path is None:
            raise FileNotFoundError("Video not found. Provide --video explicitly.")
        results_dir = ensure_results_dir(episode_id)
        tracks_path = Path(args.tracks) if args.tracks else results_dir / "tracks.jsonl"
    else:
        episode_id = ""
        video_path = Path(args.video)
        results_dir = Path(args.out)
        tracks_path = Path(args.tracks)

    if not tracks_path.exists():
        raise FileNotFoundError(f"tracks.jsonl not found: {tracks_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("================================================================")
    logger.info("PHASE 2: RE-ID + MEMORY")
    logger.info("================================================================")
    logger.info("[1/3] Loading inputs…")
    logger.info(f"  Episode: {episode_id or '(custom)'}")
    logger.info(f"  Video:   {video_path}")
    logger.info(f"  Tracks:  {tracks_path}")
    logger.info(f"  Output:  {results_dir}")
    logger.info(f"  Threshold: {args.threshold}")
    if args.class_thresholds:
        logger.info(f"  Class thresholds: {args.class_thresholds}")
    logger.info(f"  Max crops/track: {args.max_crops_per_track}")

    logger.info("[2/3] Building memory.json…")
    # Load class thresholds if provided
    class_thresholds = None
    if args.class_thresholds:
        import json
        with open(args.class_thresholds, 'r') as f:
            class_thresholds = json.load(f)

    memory_path = build_memory_from_tracks(
        episode_id=episode_id or "custom",
        video_path=Path(video_path),
        tracks_path=Path(tracks_path),
        results_dir=Path(results_dir),
        cosine_threshold=args.threshold,
        max_crops_per_track=args.max_crops_per_track,
        class_thresholds=class_thresholds,
    )
    logger.info("[3/3] Done.")
    logger.info("================================================================")
    logger.info(f"  Saved: {memory_path}")
    logger.info(f"  Updated tracks: {tracks_path}")
    logger.info(f"  Clusters: {results_dir / 'reid_clusters.json'}")
    logger.info("================================================================")


if __name__ == "__main__":
    main()
