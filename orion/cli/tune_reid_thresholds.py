#!/usr/bin/env python3
"""
Auto-tune Re-ID class thresholds across one or more episodes using Otsu.

Usage:
  python -m orion.cli.tune_reid_thresholds \
    --pairs \
      data/examples/test.mp4:results/test_validation/tracks.jsonl \
      data/examples/video.mp4:results/video_validation/tracks.jsonl \
    --out orion/perception/reid/thresholds_autotuned.json
"""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orion.perception.reid.thresholds import tune_thresholds_across, save_thresholds

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_pairs(pairs: list[str]):
    parsed = []
    for p in pairs:
        if ':' not in p:
            raise ValueError(f"Invalid pair: {p}. Expected video:tracks")
        v, t = p.split(':', 1)
        parsed.append((Path(v), Path(t)))
    return parsed


def main():
    ap = argparse.ArgumentParser(description="Tune per-class re-ID thresholds across episodes")
    ap.add_argument('--pairs', nargs='+', required=True, help='List of video:tracks.jsonl pairs')
    ap.add_argument('--out', type=str, required=True, help='Output JSON path for tuned thresholds')
    ap.add_argument('--min', dest='min_v', type=float, default=0.65, help='Min clamp for thresholds')
    ap.add_argument('--max', dest='max_v', type=float, default=0.9, help='Max clamp for thresholds')
    args = ap.parse_args()

    pairs = parse_pairs(args.pairs)
    for v, t in pairs:
        if not v.exists():
            raise FileNotFoundError(f"Video not found: {v}")
        if not t.exists():
            raise FileNotFoundError(f"Tracks not found: {t}")

    logger.info("Tuning thresholds across episodes…")
    thresholds = tune_thresholds_across(pairs, clamp=(args.min_v, args.max_v))
    out_path = save_thresholds(thresholds, Path(args.out))
    logger.info(f"✓ Saved tuned thresholds → {out_path}")


if __name__ == '__main__':
    main()
