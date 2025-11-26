#!/usr/bin/env python3
"""
Run Phase 3: Memory Lifecycle Events
====================================

Generates events.jsonl from results/<episode>/ memory.json and tracks.jsonl

Usage:
  python -m orion.cli.run_memory --results results/test_validation
"""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orion.graph import (
    build_events,
    build_merge_suggestions,
    build_relation_events,
    build_split_events,
    build_state_events,
    load_memory,
    save_events_jsonl,
    save_merge_suggestions,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description="Phase 3: Memory lifecycle + heuristics events generator")
    ap.add_argument('--results', type=str, required=True, help='Path to results/<episode> directory')
    ap.add_argument('--no-state', action='store_true', help='Disable state heuristics (held_by_person)')
    ap.add_argument('--no-split', action='store_true', help='Disable split detection heuristics')
    ap.add_argument('--no-rel', action='store_true', help='Disable relation (near/on) events')
    ap.add_argument('--state-iou', type=float, default=0.3, help='IoU threshold for held_by_person state')
    ap.add_argument('--debounce', type=int, default=2, help='Debounce window (frames) for state/relation changes')
    ap.add_argument('--relations', nargs='+', default=['near','on'], help='Relations to emit: near on')
    ap.add_argument('--near-dist', type=float, default=0.08, help='Normalized centroid distance for near relation')
    ap.add_argument('--on-overlap', type=float, default=0.3, help='Min horizontal overlap ratio for on relation')
    ap.add_argument('--on-vgap', type=float, default=0.02, help='Max normalized vertical gap for on relation')
    ap.add_argument('--split-sim', type=float, default=0.9, help='Cosine similarity threshold for split detection')
    ap.add_argument('--split-gap', type=int, default=1800, help='Max frame gap for split detection')
    ap.add_argument('--split-dist', type=float, default=0.1, help='Normalized centroid distance for split detection')
    ap.add_argument('--no-suggest', action='store_true', help='Disable writing merge suggestions JSON')
    args = ap.parse_args()

    results_dir = Path(args.results)
    memory_path = results_dir / 'memory.json'
    events_path = results_dir / 'events.jsonl'
    tracks_path = results_dir / 'tracks.jsonl'

    if not memory_path.exists():
        raise FileNotFoundError(f"Missing memory.json: {memory_path}. Run Phase 2 first.")

    logger.info("==============================================================")
    logger.info("PHASE 3: MEMORY LIFECYCLE EVENTS")
    logger.info("==============================================================")
    logger.info(f"Results dir: {results_dir}")

    memory = load_memory(memory_path)
    events = build_events(memory)

    # Optional advanced heuristics
    if not args.no_state or not args.no_split or not args.no_rel or not args.no_suggest:
        if not tracks_path.exists():
            logger.warning("tracks.jsonl missing; skipping advanced heuristics (state/split/relations/suggestions)")
        else:
            import json
            with tracks_path.open('r') as f:
                tracks = [json.loads(line) for line in f if line.strip()]

            if not args.no_state:
                se = build_state_events(memory, tracks, iou_threshold=args.state_iou, debounce_window=args.debounce)
                logger.info(f"+ State events: {len(se)}")
                events.extend(se)
            if not args.no_rel:
                re = build_relation_events(
                    memory,
                    tracks,
                    relations=args.relations,
                    near_dist_norm=args.near_dist,
                    on_h_overlap=args.on_overlap,
                    on_vgap_norm=args.on_vgap,
                    debounce_window=args.debounce,
                )
                logger.info(f"+ Relation events: {len(re)}")
                events.extend(re)
            if not args.no_split:
                spe = build_split_events(
                    memory,
                    tracks,
                    split_sim_threshold=args.split_sim,
                    max_gap_frames=args.split_gap,
                    spatial_dist_norm=args.split_dist,
                )
                logger.info(f"+ Split events: {len(spe)}")
                events.extend(spe)
            if not args.no_suggest:
                sugg = build_merge_suggestions(
                    memory,
                    tracks,
                    split_sim_threshold=max(0.85, args.split_sim - 0.05),
                    max_gap_frames=int(args.split_gap * 1.3),
                    spatial_dist_norm=max(args.split_dist, 0.15),
                )
                out_sugg = results_dir / 'merge_suggestions.json'
                save_merge_suggestions(sugg, out_sugg)
                logger.info(f"+ Wrote {out_sugg} ({len(sugg)} suggestions)")
    save_events_jsonl(events, events_path)
    logger.info(f"✓ Wrote {events_path} ({len(events)} events)")


if __name__ == '__main__':
    main()
