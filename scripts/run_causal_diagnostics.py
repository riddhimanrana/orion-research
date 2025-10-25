#!/usr/bin/env python3
"""Quick diagnostics runner for the causal inference stack.

This script executes the perception + semantic pipeline on a video,
then prints a detailed breakdown of Causal Influence Scores (CIS).

Example:
    python scripts/run_causal_diagnostics.py --video data/examples/video_short.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure local imports work when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.perception.config import (
    PerceptionConfig,
    get_accurate_config as get_perception_accurate,
    get_balanced_config as get_perception_balanced,
    get_fast_config as get_perception_fast,
)
from orion.perception.engine import PerceptionEngine
from orion.semantic.config import (
    get_accurate_semantic_config,
    get_balanced_semantic_config,
    get_fast_semantic_config,
    SemanticConfig,
)
from orion.semantic.engine import SemanticEngine


PERCEPTION_PRESETS = {
    "fast": get_perception_fast,
    "balanced": get_perception_balanced,
    "accurate": get_perception_accurate,
}

SEMANTIC_PRESETS = {
    "fast": get_fast_semantic_config,
    "balanced": get_balanced_semantic_config,
    "accurate": get_accurate_semantic_config,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Orion causal diagnostics on a video")
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the input video clip",
    )
    parser.add_argument(
        "--perception-mode",
        choices=list(PERCEPTION_PRESETS.keys()),
        default="balanced",
        help="Perception preset to use",
    )
    parser.add_argument(
        "--semantic-mode",
        choices=list(SEMANTIC_PRESETS.keys()),
        default="balanced",
        help="Semantic preset to use",
    )
    parser.add_argument(
        "--max-links",
        type=int,
        default=10,
        help="Maximum number of causal links to display",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write diagnostics JSON",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for the run (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def resolve_configs(perception_mode: str, semantic_mode: str) -> Tuple[
    PerceptionConfig,
    SemanticConfig,
]:
    perception_config = PERCEPTION_PRESETS[perception_mode]()
    semantic_config = SEMANTIC_PRESETS[semantic_mode]()
    semantic_config.verbose = True
    return perception_config, semantic_config


def summarise_links(links: List) -> List[Dict[str, float]]:
    summary = []
    for link in links:
        features = link.features or {}
        summary.append(
            {
                "agent_id": link.agent_id,
                "patient_id": link.patient_id,
                "score": float(link.influence_score),
                "temporal": float(features.get("temporal", 0.0)),
                "spatial": float(features.get("spatial", 0.0)),
                "motion": float(features.get("motion", 0.0)),
                "semantic": float(features.get("semantic", 0.0)),
                "time_delta": float(features.get("time_delta", 0.0)),
                "distance": float(features.get("distance", 0.0)),
                "alignment": float(features.get("alignment", 0.0)),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="[%(levelname)s] %(message)s")

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    perception_config, semantic_config = resolve_configs(args.perception_mode, args.semantic_mode)

    perception_engine = PerceptionEngine(perception_config)
    semantic_engine = SemanticEngine(semantic_config)

    try:
        logging.info("Running perception stage (%s mode) ...", args.perception_mode)
        perception_result = perception_engine.process_video(str(args.video))
        logging.info(
            "Perception complete: %d entities, %d detections",
            perception_result.unique_entities,
            perception_result.total_detections,
        )

        logging.info("Running semantic stage (%s mode) ...", args.semantic_mode)
        semantic_result = semantic_engine.process(perception_result, video_path=str(args.video))
    finally:
        semantic_engine.close()

    causal_links = sorted(semantic_result.causal_links, key=lambda link: link.influence_score, reverse=True)

    print("\n" + "=" * 72)
    print("CAUSAL INFERENCE DIAGNOSTICS")
    print("=" * 72)
    print(f"Video: {args.video}")
    print(f"Entities: {len(semantic_result.entities)}")
    print(f"State changes: {len(semantic_result.state_changes)}")
    print(f"Temporal windows: {len(semantic_result.temporal_windows)}")
    print(f"Scenes: {len(semantic_result.scenes)}")
    print(f"Causal links: {len(causal_links)}")
    print("-" * 72)

    limit = max(1, args.max_links)
    top_links = causal_links[:limit]

    if not top_links:
        print("No causal links above threshold. Try lowering cis_threshold in the config.")
    else:
        print(f"Top {len(top_links)} links (score breakdown):")
        for rank, link in enumerate(top_links, start=1):
            features = link.features or {}
            print(
                f"{rank:02d}. {link.agent_id} -> {link.patient_id} | "
                f"score={link.influence_score:.3f} | "
                f"T={features.get('temporal', 0.0):.2f} "
                f"S={features.get('spatial', 0.0):.2f} "
                f"M={features.get('motion', 0.0):.2f} "
                f"Se={features.get('semantic', 0.0):.2f} | "
                f"Δt={features.get('time_delta', 0.0):.2f}s "
                f"dist={features.get('distance', 0.0):.1f}px"
            )

    if args.output:
        summary = {
            "video": str(args.video),
            "perception_mode": args.perception_mode,
            "semantic_mode": args.semantic_mode,
            "entities": len(semantic_result.entities),
            "state_changes": len(semantic_result.state_changes),
            "temporal_windows": len(semantic_result.temporal_windows),
            "scenes": len(semantic_result.scenes),
            "causal_links": len(causal_links),
            "links": summarise_links(causal_links[:limit]),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"\nSaved diagnostics summary to {args.output}")

    semantic_engine.close()


if __name__ == "__main__":
    main()
