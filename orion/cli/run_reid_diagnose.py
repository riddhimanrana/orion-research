#!/usr/bin/env python3
"""Re-ID similarity diagnostic.

Purpose
-------
Given a `tracks.jsonl` and source video, compute the cosine similarity between
embeddings for two observations of the same `track_id`.

This uses the same crop + portrait un-rotation logic as Phase-2 ReID
(`orion.perception.reid.matcher`).

Usage
-----
python -m orion.cli.run_reid_diagnose \
  --video data/examples/test.mp4 \
  --tracks results/<episode>/tracks.jsonl \
  --track-id 12

Optional:
  --frame-a / --frame-b   Choose exact frames (defaults to first+last obs)
  --out-dir               Save debug crops to a folder
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Keep parity with other CLI entrypoints that can be run as scripts.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orion.perception.reid.matcher import (  # noqa: E402
    compute_observation_embedding,
    extract_observation_crop,
)

logger = logging.getLogger(__name__)


def _iter_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _get_frame_id(rec: Dict[str, Any]) -> Optional[int]:
    fid = rec.get("frame_id")
    if fid is None:
        fid = rec.get("frame_number")
    if fid is None:
        return None
    try:
        return int(fid)
    except Exception:
        return None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _save_crop(video_path: Path, rec: Dict[str, Any], out_path: Path) -> None:
    frame_id = _get_frame_id(rec)
    if frame_id is None:
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx == frame_id:
                break
            idx += 1
        if idx != frame_id:
            return

        bbox = rec.get("bbox_2d") or rec.get("bbox")
        if isinstance(bbox, dict):
            bbox = [bbox.get("x1"), bbox.get("y1"), bbox.get("x2"), bbox.get("y2")]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return

        # NOTE: we intentionally do not re-implement un-rotation here; the
        # diagnostic embedding uses the canonical implementation.
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1:
            x2 = min(w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(h - 1, y1 + 1)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), crop)
    finally:
        cap.release()


def _save_canonical_crop(video_path: Path, rec: Dict[str, Any], out_path: Path) -> None:
    try:
        crop, _, _ = extract_observation_crop(video_path, rec)
    except Exception:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)


def _evenly_sample(items: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    if n <= 0 or len(items) <= n:
        return items
    if n == 1:
        return [items[len(items) // 2]]
    idxs = [int(round(i * (len(items) - 1) / (n - 1))) for i in range(n)]
    out: List[Dict[str, Any]] = []
    seen = set()
    for idx in idxs:
        if idx in seen:
            continue
        seen.add(idx)
        out.append(items[idx])
    return out


def _auto_pick_pair(
    video_path: Path,
    candidates: List[Dict[str, Any]],
    *,
    max_samples: int,
    min_frame_gap: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sampled = _evenly_sample(candidates, max_samples)
    embs: List[np.ndarray] = []
    kept: List[Dict[str, Any]] = []
    for rec in sampled:
        try:
            embs.append(compute_observation_embedding(video_path, rec))
            kept.append(rec)
        except Exception as e:
            logger.warning("Skip sample due to embedding error: %s", e)

    if len(kept) < 2:
        raise ValueError("Not enough valid samples to auto-pick a pair")

    best_i, best_j = 0, 1
    best_sim = 1.0
    for i in range(len(kept)):
        fi = _get_frame_id(kept[i]) or 0
        for j in range(i + 1, len(kept)):
            fj = _get_frame_id(kept[j]) or 0
            if abs(fj - fi) < int(min_frame_gap):
                continue
            sim = _cosine(embs[i], embs[j])
            if sim < best_sim:
                best_sim = sim
                best_i, best_j = i, j

    return kept[best_i], kept[best_j]


def _pick_records(
    records: List[Dict[str, Any]],
    track_id: int,
    frame_a: Optional[int],
    frame_b: Optional[int],
    *,
    video_path: Optional[Path] = None,
    auto_pair: bool = False,
    max_samples: int = 12,
    min_frame_gap: int = 30,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    candidates = [r for r in records if int(r.get("track_id", -1)) == track_id and _get_frame_id(r) is not None]
    if not candidates:
        raise ValueError(f"No records found for track_id={track_id}")

    candidates.sort(key=lambda r: _get_frame_id(r) or 0)

    if auto_pair:
        if video_path is None:
            raise ValueError("auto_pair requires video_path")
        return _auto_pick_pair(
            video_path,
            candidates,
            max_samples=int(max_samples),
            min_frame_gap=int(min_frame_gap),
        )

    if frame_a is None and frame_b is None:
        return candidates[0], candidates[-1]

    if frame_a is None or frame_b is None:
        raise ValueError("Provide both --frame-a and --frame-b, or neither")

    def find(fid: int) -> Dict[str, Any]:
        # pick the record at that frame, prefer highest confidence
        matches = [r for r in candidates if _get_frame_id(r) == fid]
        if not matches:
            raise ValueError(f"No record for track_id={track_id} at frame_id={fid}")
        matches.sort(key=lambda r: float(r.get("confidence", 0.0) or 0.0), reverse=True)
        return matches[0]

    return find(frame_a), find(frame_b)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute ReID embedding cosine similarity for two observations")
    ap.add_argument("--video", type=str, required=True, help="Path to source video")
    ap.add_argument("--tracks", type=str, required=True, help="Path to tracks.jsonl")
    ap.add_argument("--track-id", type=int, required=True, help="Track ID to analyze")
    ap.add_argument("--frame-a", type=int, default=None, help="First frame_id (optional)")
    ap.add_argument("--frame-b", type=int, default=None, help="Second frame_id (optional)")
    ap.add_argument(
        "--auto-pair",
        action="store_true",
        help="Auto-select two observations (among sampled frames) with the lowest cosine similarity",
    )
    ap.add_argument("--max-samples", type=int, default=12, help="Max observations to sample for --auto-pair")
    ap.add_argument("--min-frame-gap", type=int, default=30, help="Min frame distance between auto-picked pair")
    ap.add_argument("--out-dir", type=str, default=None, help="Optional folder to write debug crops")

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    video_path = Path(args.video)
    tracks_path = Path(args.tracks)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not tracks_path.exists():
        raise FileNotFoundError(tracks_path)

    records = _iter_jsonl(tracks_path)
    rec_a, rec_b = _pick_records(
        records,
        int(args.track_id),
        args.frame_a,
        args.frame_b,
        video_path=video_path,
        auto_pair=bool(args.auto_pair),
        max_samples=int(args.max_samples),
        min_frame_gap=int(args.min_frame_gap),
    )

    fid_a = _get_frame_id(rec_a)
    fid_b = _get_frame_id(rec_b)
    assert fid_a is not None and fid_b is not None

    emb_a = compute_observation_embedding(video_path, rec_a)
    emb_b = compute_observation_embedding(video_path, rec_b)

    sim = _cosine(emb_a, emb_b)

    label = rec_a.get("class_name") or rec_a.get("category") or rec_a.get("object_class")
    conf_a = float(rec_a.get("confidence", 0.0) or 0.0)
    conf_b = float(rec_b.get("confidence", 0.0) or 0.0)

    logger.info("track_id=%s label=%s", args.track_id, label)
    logger.info("A: frame_id=%s conf=%.3f", fid_a, conf_a)
    logger.info("B: frame_id=%s conf=%.3f", fid_b, conf_b)
    logger.info("cosine_similarity=%.6f", sim)

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_crop(video_path, rec_a, out_dir / f"track_{args.track_id}_frame_{fid_a}_rawcrop.jpg")
        _save_crop(video_path, rec_b, out_dir / f"track_{args.track_id}_frame_{fid_b}_rawcrop.jpg")
        _save_canonical_crop(video_path, rec_a, out_dir / f"track_{args.track_id}_frame_{fid_a}_canoncrop.jpg")
        _save_canonical_crop(video_path, rec_b, out_dir / f"track_{args.track_id}_frame_{fid_b}_canoncrop.jpg")
        logger.info("Wrote debug crops to %s", out_dir)


if __name__ == "__main__":
    main()
