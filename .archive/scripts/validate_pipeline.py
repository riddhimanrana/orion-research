#!/usr/bin/env python3
"""End-to-end validation script for Orion v2 pipeline.

Runs detection → embedding → tracking → canonicalization on a sample video
and prints diagnostics. Designed to run on Lambda A10 GPU.

Usage:
    python scripts/validate_pipeline.py --video data/examples/test.mp4 --output results/validation --fps 5

Outputs:
    - Logs: class counts, YOLO-World prompts, candidate labels, canonical labels
    - JSON: results/validation/pipeline_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

# Add orion to path if running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("validate_pipeline")


def validate_imports() -> bool:
    """Ensure all required modules can be imported."""
    logger.info("Checking imports...")
    errors = []

    try:
        import torch  # noqa: F401
        logger.info("  ✓ PyTorch available")
    except Exception as e:  # noqa: BLE001
        errors.append(f"torch: {e}")

    try:
        from ultralytics import YOLO  # noqa: F401
        logger.info("  ✓ Ultralytics YOLO available")
    except Exception as e:  # noqa: BLE001
        errors.append(f"ultralytics: {e}")

    try:
        from orion.perception.config import PerceptionConfig, DetectionConfig  # noqa: F401
        logger.info("  ✓ orion.perception.config available")
    except Exception as e:  # noqa: BLE001
        errors.append(f"orion.perception.config: {e}")

    try:
        from orion.perception.engine import PerceptionEngine  # noqa: F401
        logger.info("  ✓ orion.perception.engine available")
    except Exception as e:  # noqa: BLE001
        errors.append(f"orion.perception.engine: {e}")

    try:
        from orion.perception.canonical_labeler import CanonicalLabeler  # noqa: F401
        logger.info("  ✓ orion.perception.canonical_labeler available")
    except Exception as e:  # noqa: BLE001
        errors.append(f"orion.perception.canonical_labeler: {e}")

    try:
        import hdbscan  # noqa: F401
        logger.info("  ✓ hdbscan available")
    except Exception as e:  # noqa: BLE001
        errors.append(f"hdbscan: {e}")

    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        logger.info("  ✓ sentence_transformers available")
    except Exception as e:  # noqa: BLE001
        errors.append(f"sentence_transformers: {e}")

    if errors:
        logger.error("Import errors detected:")
        for err in errors:
            logger.error(f"  ✗ {err}")
        return False
    return True


def run_pipeline(video_path: str, output_dir: str, target_fps: float = 5.0):
    """Run the full perception pipeline and return summary."""
    from orion.perception.config import PerceptionConfig, DetectionConfig
    from orion.perception.engine import PerceptionEngine

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Write minimal episode metadata for downstream tools (e.g. gemini_pipeline_review.py)
    episode_meta_path = output_path / "episode_meta.json"
    try:
        episode_meta_path.write_text(
            json.dumps(
                {
                    "episode_id": output_path.name,
                    "video_path": str(Path(video_path).resolve()),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
        )
        logger.info("Episode meta written to: %s", episode_meta_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed writing episode_meta.json: %s", e)

    # Build detection config with coarse prompts (should NOT be 122 classes)
    det_config = DetectionConfig(
        backend="yoloworld",
        yoloworld_prompt_preset="coarse",  # ~17 classes
        yoloworld_use_custom_classes=True,
        yoloworld_enable_candidate_labels=True,
        yoloworld_candidate_top_k=5,
        confidence_threshold=0.15,
    )

    categories = det_config.yoloworld_categories()
    logger.info("YOLO-World prompt preset: %s", det_config.yoloworld_prompt_preset)
    logger.info("YOLO-World class count: %d (should be ~17 for 'coarse')", len(categories))
    if len(categories) > 30:
        logger.warning("⚠️ Class count > 30: may cause prompt collapse!")
    else:
        logger.info("  ✓ Class count within safe range")

    config = PerceptionConfig(
        target_fps=target_fps,
        detection=det_config,
        enable_tracking=True,
        use_memgraph=False,
    )

    logger.info("Embedding backend: V-JEPA2 (hardcoded)")
    logger.info("Embedding dim: %s", config.embedding.embedding_dim)

    # Create engine
    t0 = time.time()
    engine = PerceptionEngine(config=config, verbose=True)
    init_time = time.time() - t0
    logger.info("Engine initialized in %.2fs", init_time)

    # Run pipeline
    t0 = time.time()
    result = engine.process_video(video_path, save_visualizations=True, output_dir=str(output_path))
    pipeline_time = time.time() - t0
    logger.info("Pipeline completed in %.2fs", pipeline_time)

    # Summarize results
    summary = {
        "video_path": video_path,
        "output_dir": str(output_path),
        "config": {
            "yoloworld_prompt_preset": det_config.yoloworld_prompt_preset,
            "yoloworld_class_count": len(categories),
            "target_fps": target_fps,
            "embedding_dim": config.embedding.embedding_dim,
        },
        "results": {
            "total_entities": len(result.entities),
            "total_observations": len(result.raw_observations),
            "processing_time_seconds": pipeline_time,
        },
        "entity_summary": [],
        "canonical_labels": {},
        "candidate_labeling": {},
    }

    canonical_counts = Counter()
    for ent in result.entities:
        ent_info = {
            "entity_id": ent.entity_id,
            "object_class": getattr(ent.object_class, "value", str(ent.object_class)),
            "appearance_count": ent.appearance_count,
            "canonical_label": getattr(ent, "canonical_label", None),
            "canonical_confidence": getattr(ent, "canonical_confidence", None),
        }
        summary["entity_summary"].append(ent_info)
        if ent_info["canonical_label"]:
            canonical_counts[ent_info["canonical_label"]] += 1

    summary["canonical_labels"] = dict(canonical_counts.most_common(20))

    # Candidate label diagnostics
    obs_with_candidates = 0
    group_counts = Counter()
    candidate_label_counts = Counter()
    for obs in result.raw_observations:
        cands = getattr(obs, "candidate_labels", None)
        if cands:
            obs_with_candidates += 1
            grp = getattr(obs, "candidate_group", None)
            if grp:
                group_counts[str(grp)] += 1
            for c in cands:
                try:
                    lbl = c.get("label") if isinstance(c, dict) else None
                    if lbl:
                        candidate_label_counts[str(lbl)] += 1
                except Exception:
                    continue

    summary["candidate_labeling"] = {
        "observations_with_candidates": obs_with_candidates,
        "total_observations": len(result.raw_observations),
        "coverage": (obs_with_candidates / max(1, len(result.raw_observations))),
        "prompt_group_distribution": dict(group_counts.most_common(20)),
        "top_candidate_labels": dict(candidate_label_counts.most_common(30)),
    }

    summary_path = output_path / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary written to: %s", summary_path)

    logger.info(
        "Observations with candidate labels: %d/%d (%.1f%%)",
        obs_with_candidates,
        len(result.raw_observations),
        100.0 * obs_with_candidates / max(1, len(result.raw_observations)),
    )

    if summary["canonical_labels"]:
        logger.info("Top canonical labels:")
        for lbl, count in list(summary["canonical_labels"].items())[:10]:
            logger.info("  %s: %d", lbl, count)
    else:
        logger.warning("No canonical labels resolved (may need more observations)")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Validate Orion v2 pipeline end-to-end")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default="results/validation", help="Output directory")
    parser.add_argument("--fps", type=float, default=5.0, help="Target FPS for sampling")
    parser.add_argument("--skip-imports", action="store_true", help="Skip import validation")
    args = parser.parse_args()

    if not Path(args.video).exists():
        logger.error("Video not found: %s", args.video)
        sys.exit(1)

    if not args.skip_imports:
        if not validate_imports():
            logger.error("Import validation failed. Fix missing dependencies and retry.")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("ORION V2 PIPELINE VALIDATION")
    logger.info("=" * 60)
    logger.info("Video: %s", args.video)
    logger.info("Output: %s", args.output)
    logger.info("Target FPS: %s", args.fps)

    try:
        run_pipeline(args.video, args.output, args.fps)
        logger.info("=" * 60)
        logger.info("✓ VALIDATION COMPLETE")
        logger.info("=" * 60)
    except Exception as e:  # noqa: BLE001
        logger.exception("Pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
