#!/usr/bin/env python3
"""
Run Perception Engine (v2)
===========================

Full Phase 1 pipeline using PerceptionEngine with:
- YOLO-World open-vocabulary detection + crop refinement
- FastVLM semantic verification (candidates_only mode)
- V-JEPA2 embeddings for Re-ID
- HDBSCAN entity clustering
- Scene context from SceneContextManager

Usage:
    # Basic run with defaults
    python -m orion.cli.run_engine --video data/examples/test.mp4 --output results/engine_test

    # Full semantic verification pipeline
    python -m orion.cli.run_engine --video data/examples/video.mp4 --output results/full_test \\
        --enable-semantic-verifier --enable-crop-refinement

    # Fast mode (no VLM, no crop refinement)
    python -m orion.cli.run_engine --video data/examples/test.mp4 --output results/fast_test --fast
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orion.perception.config import (
    PerceptionConfig,
    SemanticVerificationConfig,
    get_balanced_config,
    get_fast_config,
)
from orion.perception.engine import PerceptionEngine, PerceptionResult

if TYPE_CHECKING:
    from orion.perception.types import Observation, PerceptionEntity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("orion.cli.run_engine")


def get_default_device() -> str:
    """Auto-detect the best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def build_config(args: argparse.Namespace) -> tuple[PerceptionConfig, str]:
    """Build PerceptionConfig from CLI arguments.
    
    Returns (config, device) tuple.
    """
    # Start from preset
    if args.fast:
        config = get_fast_config()
    else:
        config = get_balanced_config()

    # Determine device
    device = args.device if args.device != "auto" else get_default_device()

    # Detection backend
    config.detection.backend = args.detection_backend
    if args.detection_backend == "yoloworld":
        config.detection.yoloworld_model = args.yoloworld_model
        config.detection.yoloworld_prompt_preset = args.yoloworld_prompt
        # Crop refinement settings
        config.detection.yoloworld_enable_crop_refinement = args.enable_crop_refinement
        config.detection.yoloworld_refinement_every_n_sampled_frames = args.crop_refinement_every_n
        config.detection.yoloworld_refinement_max_crops_per_class = args.crop_refinement_max_per_class

    # Confidence threshold
    config.detection.confidence_threshold = args.confidence

    # Semantic verifier settings (via SemanticVerificationConfig)
    config.semantic_verification.enabled = args.enable_semantic_verifier
    if args.semantic_candidates_only:
        config.semantic_verification.mode = "candidates_only"
    config.semantic_verification.rerank_blend = args.rerank_blend

    # Embedding device
    config.embedding.device = device

    # Tracking
    config.enable_tracking = not args.no_tracking

    # Sampling FPS
    config.target_fps = args.fps

    return config, device


def observation_to_dict(obs: Any) -> dict:
    """Convert Observation object to serializable dict."""
    if isinstance(obs, dict):
        return obs
    
    # Handle Observation dataclass
    d = {
        "frame_number": obs.frame_number,
        "timestamp": obs.timestamp,
        "object_class": obs.object_class.value if hasattr(obs.object_class, "value") else str(obs.object_class),
        "confidence": float(obs.confidence),
        "bounding_box": {
            "x1": obs.bounding_box.x1,
            "y1": obs.bounding_box.y1,
            "x2": obs.bounding_box.x2,
            "y2": obs.bounding_box.y2,
        },
        "centroid": list(obs.centroid),
        "temp_id": obs.temp_id,
    }
    
    # Optional fields
    if obs.entity_id:
        d["entity_id"] = obs.entity_id
    if obs.candidate_labels:
        d["candidate_labels"] = obs.candidate_labels
    if obs.candidate_group:
        d["candidate_group"] = obs.candidate_group
    if obs.vlm_description:
        d["vlm_description"] = obs.vlm_description
    if obs.vlm_similarity is not None:
        d["vlm_similarity"] = float(obs.vlm_similarity)
    if obs.vlm_is_valid is not None:
        d["vlm_is_valid"] = obs.vlm_is_valid
    if obs.scene_similarity is not None:
        d["scene_similarity"] = float(obs.scene_similarity)
    if obs.scene_filter_reason:
        d["scene_filter_reason"] = obs.scene_filter_reason
    if obs.raw_yolo_class:
        d["raw_yolo_class"] = obs.raw_yolo_class
    
    return d


def save_tracks_jsonl(observations: list, output_dir: Path) -> Path:
    """Save observations as tracks.jsonl with all metadata."""
    tracks_path = output_dir / "tracks.jsonl"
    with open(tracks_path, "w") as f:
        for obs in observations:
            d = observation_to_dict(obs)
            f.write(json.dumps(d, default=str) + "\n")
    return tracks_path


def main():
    parser = argparse.ArgumentParser(
        description="Run Perception Engine v2 with semantic verification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument(
        "--output", type=str, default="results/engine_output", help="Output directory"
    )

    # Preset modes
    parser.add_argument("--fast", action="store_true", help="Use fast preset (no VLM, limited refinement)")

    # Detection settings
    parser.add_argument(
        "--detection-backend",
        type=str,
        default="yoloworld",
        choices=["yolo", "yoloworld"],
        help="Detection backend",
    )
    parser.add_argument(
        "--yoloworld-model",
        type=str,
        default="yolov8l-worldv2.pt",
        help="YOLO-World model file",
    )
    parser.add_argument(
        "--yoloworld-prompt",
        type=str,
        default="coarse",
        choices=["coarse", "coco", "indoor_full", "custom"],
        help="YOLO-World prompt preset",
    )
    parser.add_argument("--confidence", type=float, default=0.15, help="Detection confidence threshold")

    # Crop refinement
    parser.add_argument(
        "--enable-crop-refinement",
        action="store_true",
        help="Enable YOLO-World crop-level refinement for fine-grained labels",
    )
    parser.add_argument(
        "--crop-refinement-every-n",
        type=int,
        default=4,
        help="Run crop refinement every N sampled frames",
    )
    parser.add_argument(
        "--crop-refinement-max-per-class",
        type=int,
        default=3,
        help="Max crops per class for refinement",
    )

    # Semantic verifier (FastVLM)
    parser.add_argument(
        "--enable-semantic-verifier",
        action="store_true",
        help="Enable FastVLM semantic verification",
    )
    parser.add_argument(
        "--semantic-candidates-only",
        action="store_true",
        default=True,
        help="Only add VLM metadata without modifying labels",
    )
    parser.add_argument(
        "--rerank-blend",
        type=float,
        default=0.45,
        help="Blend factor for VLM reranking (0=detector only, 1=VLM only)",
    )

    # Tracking
    parser.add_argument("--no-tracking", action="store_true", help="Disable enhanced tracking")

    # Device & performance
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device",
    )
    parser.add_argument("--fps", type=float, default=5.0, help="Target sample FPS")

    # Visualization
    parser.add_argument(
        "--save-viz", action="store_true", help="Save visualization data"
    )

    args = parser.parse_args()

    # Validate inputs
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build config
    config, device = build_config(args)
    logger.info("="*80)
    logger.info("PERCEPTION ENGINE v2")
    logger.info("="*80)
    logger.info(f"Video: {video_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Backend: {config.detection.backend}")
    if config.detection.backend == "yoloworld":
        logger.info(f"  YOLO-World model: {config.detection.yoloworld_model}")
        logger.info(f"  Prompt preset: {config.detection.yoloworld_prompt_preset}")
        logger.info(f"  Crop refinement: {config.detection.yoloworld_enable_crop_refinement}")
    logger.info(f"Semantic verifier: {config.semantic_verification.enabled}")
    logger.info(f"Tracking: {config.enable_tracking}")
    logger.info(f"Sample FPS: {config.target_fps}")

    # Initialize engine
    logger.info("\n[1/3] Initializing Perception Engine...")
    t0 = time.time()
    engine = PerceptionEngine(config=config, verbose=True)
    init_time = time.time() - t0
    logger.info(f"  ✓ Engine initialized in {init_time:.2f}s")

    # Process video
    logger.info("\n[2/3] Processing video...")
    t0 = time.time()
    result: PerceptionResult = engine.process_video(
        str(video_path),
        save_visualizations=args.save_viz,
        output_dir=str(output_dir),
    )
    process_time = time.time() - t0

    # Save tracks.jsonl from raw observations
    logger.info("\n[3/3] Saving results...")
    tracks_path = save_tracks_jsonl(result.raw_observations, output_dir)
    logger.info(f"  ✓ Saved {len(result.raw_observations)} observations → {tracks_path}")

    # Save entities summary
    entities_path = output_dir / "entities.json"
    entities_data = []
    for entity in result.entities:
        entity_dict = {
            "entity_id": entity.entity_id,
            "object_class": entity.object_class.value if hasattr(entity.object_class, "value") else str(entity.object_class),
            "first_seen_frame": entity.first_seen_frame,
            "last_seen_frame": entity.last_seen_frame,
            "appearance_count": entity.appearance_count,
            "description": entity.description,
            "canonical_label": entity.canonical_label,
            "canonical_confidence": entity.canonical_confidence,
        }
        entities_data.append(entity_dict)

    with open(entities_path, "w") as f:
        json.dump(entities_data, f, indent=2)
    logger.info(f"  ✓ Saved {len(entities_data)} entities → {entities_path}")

    # Save run metadata
    meta_path = output_dir / "run_metadata.json"
    metadata = {
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "total_frames": result.total_frames,
        "total_detections": result.total_detections,
        "unique_entities": result.unique_entities,
        "processing_time_seconds": process_time,
        "fps_achieved": result.total_frames / process_time if process_time > 0 else 0,
        "scene_caption": result.scene_caption,
        "config": {
            "detection_backend": config.detection.backend,
            "device": device,
            "target_fps": config.target_fps,
            "confidence_threshold": config.detection.confidence_threshold,
            "semantic_verifier_enabled": config.semantic_verification.enabled,
            "semantic_verifier_mode": config.semantic_verification.mode,
            "crop_refinement": config.detection.yoloworld_enable_crop_refinement,
        },
        "metrics": result.metrics,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  ✓ Saved metadata → {meta_path}")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total frames: {result.total_frames}")
    logger.info(f"Detections: {result.total_detections}")
    logger.info(f"Unique entities: {result.unique_entities}")
    logger.info(f"Processing time: {process_time:.2f}s")
    if process_time > 0:
        logger.info(f"Effective FPS: {result.total_frames / process_time:.2f}")
    if result.scene_caption:
        logger.info(f"Scene: {result.scene_caption[:100]}...")
    logger.info(f"\nOutputs written to: {output_dir}")

    # Print sample of VLM metadata if available
    vlm_samples = [obs for obs in result.raw_observations[:30] if getattr(obs, "vlm_description", None)]
    if vlm_samples:
        logger.info("\n--- Sample VLM Descriptions ---")
        for obs in vlm_samples[:3]:
            label = obs.object_class.value if hasattr(obs.object_class, "value") else str(obs.object_class)
            vlm_desc = (obs.vlm_description or "")[:100]
            vlm_sim = obs.vlm_similarity or 0
            logger.info(f"  [{label}] sim={vlm_sim:.3f} → {vlm_desc}...")

    # Print candidate labels if available
    cand_samples = [obs for obs in result.raw_observations[:30] if getattr(obs, "candidate_labels", None)]
    if cand_samples:
        logger.info("\n--- Sample Candidate Labels ---")
        for obs in cand_samples[:3]:
            cands = (obs.candidate_labels or [])[:3]
            cand_str = ", ".join(f"{c['label']}={c['score']:.2f}" for c in cands)
            logger.info(f"  {cand_str}")


if __name__ == "__main__":
    main()
