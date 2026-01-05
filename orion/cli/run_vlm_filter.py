#!/usr/bin/env python3
"""FastVLM-based semantic filtering for tracks.

Goal
----
Use FastVLM + Sentence Transformers to validate object detections.
1. Generates a visual description of the object crop.
2. Compares the description to the detected label using semantic similarity.
3. Removes tracks where the description strongly contradicts the label.

Outputs
-------
- `tracks_filtered.jsonl`: same schema as input, with removed tracks omitted
- `vlm_filter_audit.jsonl`: per-track decision log with reasons

Usage
-----
python -m orion.cli.run_vlm_filter \
  --video data/examples/test.mp4 \
  --tracks tracks.jsonl \
  --out-tracks tracks_filtered.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image

from orion.perception.filters import create_semantic_filter, SemanticFilter, FilterResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackSummary:
    track_id: int
    label: str
    best_confidence: float
    best_frame_id: int
    best_bbox: tuple[int, int, int, int]


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _get_bbox(rec: dict) -> Optional[tuple[int, int, int, int]]:
    bb = rec.get("bbox")
    if bb is None:
        bb = rec.get("bbox_2d")
    if bb is None:
        return None
    if isinstance(bb, dict):
        try:
            return int(bb["x1"]), int(bb["y1"]), int(bb["x2"]), int(bb["y2"])
        except Exception:
            return None
    if isinstance(bb, (list, tuple)) and len(bb) >= 4:
        return int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
    return None


def _crop_bgr(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int], pad: float = 0.08) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(bw * pad)
    py = int(bh * pad)
    x1p = max(0, x1 - px)
    y1p = max(0, y1 - py)
    x2p = min(w - 1, x2 + px)
    y2p = min(h - 1, y2 + py)
    if x2p <= x1p or y2p <= y1p:
        return frame_bgr
    return frame_bgr[y1p:y2p, x1p:x2p]


def _bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _summarize_tracks(records: Iterable[dict]) -> dict[int, TrackSummary]:
    best: dict[int, TrackSummary] = {}

    for rec in records:
        track_id = rec.get("track_id", rec.get("id"))
        if track_id is None:
            continue
        try:
            track_id_int = int(track_id)
        except Exception:
            continue

        label = rec.get("class_name") or rec.get("category") or rec.get("object_class") or "unknown"
        conf = float(rec.get("confidence", 0.0) or 0.0)
        frame_id = rec.get("frame_id")
        if frame_id is None:
            frame_id = rec.get("frame_number")
        if frame_id is None:
            continue
        frame_id_int = int(frame_id)

        bbox = _get_bbox(rec)
        if bbox is None:
            continue

        prev = best.get(track_id_int)
        if prev is None or conf > prev.best_confidence:
            best[track_id_int] = TrackSummary(
                track_id=track_id_int,
                label=str(label),
                best_confidence=conf,
                best_frame_id=frame_id_int,
                best_bbox=bbox,
            )

    return best


def _iter_frames(video_path: Path) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()


def run_vlm_filter(
    video_path: Path,
    tracks_path: Path,
    out_tracks_path: Path,
    audit_jsonl_path: Path,
    *,
    similarity_threshold: float = 0.25,
    device: str = "cuda",
    use_scene_context: bool = True,
) -> None:
    # 0. Generate scene context (v2)
    scene_caption = ""
    if use_scene_context:
        try:
            from orion.perception.scene_context import SceneContextManager, SceneContextConfig
            scene_config = SceneContextConfig(device=device)
            scene_mgr = SceneContextManager(config=scene_config)
            
            # Read first frame for scene context
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                ret, first_frame = cap.read()
                if ret:
                    scene_snapshot = scene_mgr.update(first_frame, frame_idx=0, force=True)
                    scene_caption = scene_snapshot.caption
                    logger.info(f"Scene Context: {scene_caption}")
                    logger.info(f"  Objects mentioned: {scene_snapshot.objects_mentioned}")
                cap.release()
        except Exception as e:
            logger.warning(f"Failed to generate scene context: {e}")

    # 1. Load tracks
    records = list(_iter_jsonl(tracks_path))
    if not records:
        raise RuntimeError(f"No records found in: {tracks_path}")

    summaries = _summarize_tracks(records)
    logger.info("Loaded %d track observations, %d unique tracks", len(records), len(summaries))

    needed_frames: set[int] = {s.best_frame_id for s in summaries.values()}
    logger.info("Need %d frames for object crops", len(needed_frames))

    # 2. Collect crops
    # We collect all crops first, then batch process them
    crops_data: List[Tuple[int, Image.Image, str, float]] = []  # (track_id, crop, label, conf)
    
    # Map track_id to summary for easy access
    track_map = summaries

    for frame_id, frame_bgr in _iter_frames(video_path):
        if frame_id not in needed_frames:
            continue

        # Find tracks that need this frame
        for track_id, summary in track_map.items():
            if summary.best_frame_id == frame_id:
                crop_bgr = _crop_bgr(frame_bgr, summary.best_bbox)
                crop_pil = _bgr_to_pil(crop_bgr)
                crops_data.append((track_id, crop_pil, summary.label, summary.best_confidence))

    logger.info(f"Collected {len(crops_data)} crops for filtering")

    # 3. Run Semantic Filter
    filter_engine = create_semantic_filter(
        device=device,
        similarity_threshold=similarity_threshold
    )

    # Unpack data for batch processing
    track_ids = [x[0] for x in crops_data]
    crops = [x[1] for x in crops_data]
    labels = [x[2] for x in crops_data]
    confidences = [x[3] for x in crops_data]

    results = filter_engine.filter_batch(
        crops=crops,
        labels=labels,
        track_ids=track_ids,
        confidences=confidences,
        scene_context=scene_caption if scene_caption else None,
        show_progress=True
    )

    # 4. Process results
    valid_track_ids = {r.track_id for r in results if r.is_valid}
    
    # Write audit log
    logger.info(f"Writing audit log to {audit_jsonl_path}")
    with audit_jsonl_path.open("w") as f:
        for r in results:
            audit_rec = {
                "track_id": r.track_id,
                "label": r.label,
                "description": r.description,
                "similarity": r.similarity,
                "confidence": r.confidence,
                "is_valid": r.is_valid,
                "reason": r.reason
            }
            f.write(json.dumps(audit_rec) + "\n")

    # Write filtered tracks
    logger.info(f"Writing filtered tracks to {out_tracks_path}")
    kept_count = 0
    removed_count = 0
    
    with out_tracks_path.open("w") as f:
        for rec in records:
            track_id = rec.get("track_id", rec.get("id"))
            if track_id is not None:
                try:
                    tid = int(track_id)
                    if tid in summaries: # Only filter tracks we summarized (some might be skipped if no bbox)
                        if tid not in valid_track_ids:
                            removed_count += 1
                            continue # Skip this record
                except ValueError:
                    pass
            
            f.write(json.dumps(rec) + "\n")
            kept_count += 1

    logger.info(f"Filtering complete. Kept {kept_count} records, removed {removed_count} records.")


def main():
    parser = argparse.ArgumentParser(description="Semantic filtering for tracks")
    parser.add_argument("--video", type=Path, required=True, help="Path to video file")
    parser.add_argument("--tracks", type=Path, required=True, help="Input tracks.jsonl")
    parser.add_argument("--out-tracks", type=Path, required=True, help="Output filtered tracks.jsonl")
    parser.add_argument("--audit", type=Path, default=Path("vlm_audit.jsonl"), help="Audit log path")
    parser.add_argument("--threshold", type=float, default=0.25, help="Similarity threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/mps/cpu)")
    parser.add_argument("--no-scene-context", action="store_true", help="Disable scene context")
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_vlm_filter(
        video_path=args.video,
        tracks_path=args.tracks,
        out_tracks_path=args.out_tracks,
        audit_jsonl_path=args.audit,
        similarity_threshold=args.threshold,
        device=args.device,
        use_scene_context=not args.no_scene_context,
    )


if __name__ == "__main__":
    main()
