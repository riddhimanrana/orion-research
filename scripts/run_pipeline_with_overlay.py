#!/usr/bin/env python3
"""Run the perception pipeline and immediately render an annotated overlay video.

This helper executes the full ``PerceptionEngine`` on the provided video, exports
minimal ``tracks.jsonl``/``memory.json`` artifacts understood by the insight
overlay renderer, and then invokes the overlay generator to produce an annotated
MP4 in a single command.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.perception.engine import PerceptionEngine
from orion.perception.config import PerceptionConfig
from orion.perception.types import Observation, PerceptionEntity
from orion.perception.spatial_zones import ZoneManager
from orion.perception.viz_overlay import OverlayOptions, render_insight_overlay

logger = logging.getLogger("orion.pipeline.overlay")


@dataclass
class VideoMetadata:
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run perception pipeline and render overlay video")
    parser.add_argument("--video", default=str(REPO_ROOT / "data/examples/video.mp4"), help="Input video path")
    parser.add_argument(
        "--results-dir",
        default=str(REPO_ROOT / "results/full_video_analysis"),
        help="Directory to store pipeline artifacts",
    )
    parser.add_argument("--overlay-output", help="Optional explicit path for overlay MP4 output")
    parser.add_argument("--target-fps", type=float, default=None, help="Override target FPS for sampling")
    parser.add_argument("--disable-depth", action="store_true", help="Skip depth/3D stages (useful on CPU-only hosts)")
    parser.add_argument(
        "--disable-descriptions",
        action="store_true",
        help="Skip FastVLM descriptions/class corrections (runs without VLM)",
    )
    parser.add_argument("--enable-sam", action="store_true", help="Enable SAM mask refinement for detector outputs")
    parser.add_argument(
        "--sam-model",
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM backbone to load when --enable-sam is provided",
    )
    parser.add_argument(
        "--sam-device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device override for SAM inference",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default=None,
        help="Path to SAM checkpoint (.pth). Required when enabling SAM unless weights already live under models/weights",
    )
    parser.add_argument("--skip-overlay", action="store_true", help="Run perception only; do not render overlay")
    parser.add_argument("--max-overlay-messages", type=int, default=5, help="Maximum concurrent narrative messages")
    parser.add_argument("--max-overlay-relations", type=int, default=4, help="Max relations listed per frame")
    parser.add_argument("--overlay-message-seconds", type=float, default=1.75, help="Seconds to display overlay state messages")
    parser.add_argument("--overlay-refind-gap", type=int, default=45, help="Frames required to treat a track as refound")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose perception logging")
    return parser.parse_args()


def configure_perception(args: argparse.Namespace) -> PerceptionConfig:
    config = PerceptionConfig()
    if args.target_fps:
        config.target_fps = args.target_fps
    if args.disable_depth:
        config.enable_depth = False
        config.enable_3d = False
        config.enable_occlusion = False
    if args.disable_descriptions:
        config.enable_descriptions = False
    if args.enable_sam:
        config.segmentation.enabled = True
        config.segmentation.model_type = args.sam_model
        config.segmentation.device = args.sam_device
        if args.sam_checkpoint:
            config.segmentation.checkpoint_path = args.sam_checkpoint
    config.enable_cis = True  # ensure causal influence scores exported
    return config


def _parse_track_id(entity_id: Optional[str]) -> Optional[int]:
    if not entity_id:
        return None
    match = re.search(r"(\d+)$", entity_id)
    if not match:
        return None
    return int(match.group(1))


def _encode_mask_rle(mask: np.ndarray) -> Optional[Dict[str, object]]:
    """Encode a binary mask into a compact run-length representation."""
    if mask is None:
        return None
    binary = (mask.astype(np.uint8) > 0).astype(np.uint8)
    if binary.size == 0:
        return None
    flat = binary.flatten(order="C")
    start_value = int(flat[0])
    counts: List[int] = []
    current_value = start_value
    run_length = 1
    for value in flat[1:]:
        value = int(value)
        if value == current_value:
            run_length += 1
        else:
            counts.append(int(run_length))
            current_value = value
            run_length = 1
    counts.append(int(run_length))
    return {
        "order": "row-major",
        "size": [int(binary.shape[0]), int(binary.shape[1])],
        "counts": counts,
        "start": int(start_value),
    }


def _build_segmentation_entry(obs: Observation) -> Optional[Dict[str, object]]:
    """Serialize an observation's segmentation mask for overlay consumption."""
    mask = getattr(obs, "segmentation_mask", None)
    if mask is None:
        return None
    encoded = _encode_mask_rle(mask)
    if not encoded:
        return None
    features = getattr(obs, "features", {}) or {}
    payload: Dict[str, object] = {
        "encoding": "rle",
        "version": 1,
        "data": encoded,
    }
    score = features.get("sam_score") or features.get("segmentation_score")
    if score is not None:
        payload["score"] = float(score)
    area = features.get("sam_area")
    if area is not None:
        payload["area"] = int(area)
    return payload


def build_entity_mapping(entities: Iterable[PerceptionEntity]) -> Dict[str, Dict[str, object]]:
    mapping: Dict[str, Dict[str, object]] = {}
    used_track_ids: set[int] = set()
    next_track_id = 1
    for idx, entity in enumerate(entities, start=1):
        track_id = _parse_track_id(entity.entity_id)
        if track_id is None or track_id in used_track_ids:
            while next_track_id in used_track_ids:
                next_track_id += 1
            track_id = next_track_id
            used_track_ids.add(track_id)
            next_track_id += 1
        else:
            used_track_ids.add(track_id)
        mapping[entity.entity_id] = {
            "track_id": track_id,
            "mem_id": f"mem_{idx:03d}",
            "emb_id": f"emb_{idx:03d}",
        }
    return mapping


def export_tracks(
    observations: Iterable[Observation],
    mapping: Dict[str, Dict[str, object]],
    results_dir: Path,
    video_meta: VideoMetadata,
) -> Path:
    tracks_path = results_dir / "tracks.jsonl"
    written = 0
    with tracks_path.open("w", encoding="utf-8") as fp:
        for obs in observations:
            if not obs.entity_id or obs.entity_id not in mapping:
                continue
            info = mapping[obs.entity_id]
            bbox = obs.bounding_box.to_list()
            centroid = list(obs.centroid)
            entry = {
                "bbox": [float(v) for v in bbox],
                "centroid": [float(centroid[0]), float(centroid[1])],
                "category": obs.object_class.value if hasattr(obs.object_class, "value") else str(obs.object_class),
                "confidence": float(obs.confidence),
                "frame_id": int(obs.frame_number),
                "timestamp": float(obs.timestamp),
                "frame_width": int(obs.frame_width or video_meta.width),
                "frame_height": int(obs.frame_height or video_meta.height),
                "track_id": int(info["track_id"]),
                "embedding_id": info["emb_id"],
            }
            segmentation_payload = _build_segmentation_entry(obs)
            if segmentation_payload is not None:
                entry["segmentation"] = segmentation_payload
            fp.write(json.dumps(entry) + "\n")
            written += 1
    logger.info("Saved %d track observations → %s", written, tracks_path)
    return tracks_path


def export_memory(
    entities: Iterable[PerceptionEntity],
    mapping: Dict[str, Dict[str, object]],
    results_dir: Path,
) -> Path:
    zone_manager = ZoneManager()
    objects: List[Dict[str, object]] = []
    embeddings: Dict[str, List[float]] = {}

    for entity in entities:
        if entity.entity_id not in mapping:
            continue
        info = mapping[entity.entity_id]
        emb_id = info["emb_id"]
        track_id = int(info["track_id"])
        mem_id = info["mem_id"]
        embedding = entity.average_embedding
        if embedding is None and entity.observations:
            try:
                embedding = entity.compute_average_embedding()
            except Exception:
                embedding = None
        if embedding is not None:
            embeddings[emb_id] = [float(v) for v in np.asarray(embedding).flatten()]
        description = entity.description or ""
        zone_id = zone_manager.assign_zone(entity)
        objects.append(
            {
                "memory_id": mem_id,
                "class": entity.display_class(),
                "first_seen_frame": int(entity.first_seen_frame),
                "last_seen_frame": int(entity.last_seen_frame),
                "total_observations": int(entity.appearance_count),
                "prototype_embedding": emb_id,
                "appearance_history": [
                    {
                        "track_id": track_id,
                        "first_frame": int(entity.first_seen_frame),
                        "last_frame": int(entity.last_seen_frame),
                        "observations": int(entity.appearance_count),
                    }
                ],
                "current_state": "visible",
                "description": description,
                "zone_id": zone_id,
                "zone_name": zone_manager.get_zone_name(zone_id),
            }
        )

    payload = {
        "objects": objects,
        "embeddings": embeddings,
        "statistics": {
            "total_objects": len(objects),
            "with_descriptions": sum(1 for obj in objects if obj.get("description")),
        },
    }
    memory_path = results_dir / "memory.json"
    memory_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved memory.json with %d objects → %s", len(objects), memory_path)
    return memory_path


def ensure_results_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _probe_video_metadata(video_path: Path) -> VideoMetadata:
    import cv2  # Local import to avoid OpenCV requirement for non-overlay workflows

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    cap.release()
    return VideoMetadata(width=width, height=height)


def run() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    video_path = Path(args.video).expanduser().resolve()
    results_dir = ensure_results_dir(Path(args.results_dir).expanduser().resolve())

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found at {video_path}")

    config = configure_perception(args)
    engine = PerceptionEngine(config=config, verbose=args.verbose)

    logger.info("Running perception pipeline on %s", video_path)
    result = engine.process_video(str(video_path), save_visualizations=True, output_dir=str(results_dir))

    # Export helper artifacts for overlay
    mapping = build_entity_mapping(result.entities)
    video_meta = _probe_video_metadata(video_path)
    export_tracks(result.raw_observations, mapping, results_dir, video_meta)
    export_memory(result.entities, mapping, results_dir)

    # Persist lightweight pipeline summary for downstream tooling
    summary_path = results_dir / "pipeline_output.json"
    summary = {
        "video_path": str(video_path),
        "results_dir": str(results_dir),
        "total_frames": result.total_frames,
        "fps": result.fps,
        "duration": result.duration_seconds,
        "entities": [entity.to_dict() for entity in result.entities],
        "metrics": result.metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved pipeline summary → %s", summary_path)

    if args.skip_overlay:
        logger.info("Overlay rendering skipped by request")
        return

    options = OverlayOptions(
        max_relations=args.max_overlay_relations,
        max_state_messages=args.max_overlay_messages,
        message_linger_seconds=args.overlay_message_seconds,
        gap_frames_for_refind=args.overlay_refind_gap,
        overlay_basename=(Path(args.overlay_output).name if args.overlay_output else "video_overlay_insights.mp4"),
    )
    overlay_path = Path(args.overlay_output) if args.overlay_output else None
    rendered = render_insight_overlay(video_path=video_path, results_dir=results_dir, output_path=overlay_path, options=options)
    logger.info("Overlay video saved to %s", rendered)


def main() -> None:
    try:
        run()
    except Exception as exc:
        logger.exception("Pipeline overlay run failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
