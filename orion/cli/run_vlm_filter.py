#!/usr/bin/env python3
"""FastVLM-based sanity filtering for tracks.

Goal
----
Use FastVLM as a lightweight, running "visual ground truth" to identify
implausible detector labels and remove those tracks from `tracks.jsonl`.

This is intentionally conservative by default:
- High-confidence tracks are kept.
- Low-confidence tracks are removed only when FastVLM strongly contradicts them.

Outputs
-------
- `tracks_filtered.jsonl`: same schema as input, with removed tracks omitted
- `vlm_scene.jsonl`: per-frame scene captions (optional)
- `vlm_filter_audit.jsonl`: per-track decision log with reasons

Usage
-----
python -m orion.cli.run_vlm_filter \
  --video data/examples/test.mp4 \
  --results results/<episode> \
  --tracks tracks.jsonl \
  --out-tracks tracks_filtered.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import cv2
import numpy as np
from PIL import Image

from orion.managers.model_manager import ModelManager

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
        # Some schemas store as {x1,y1,x2,y2}
        try:
            return int(bb["x1"]), int(bb["y1"]), int(bb["x2"]), int(bb["y2"])
        except Exception:
            return None
    if isinstance(bb, (list, tuple)) and len(bb) >= 4:
        return int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
    return None


def _normalize_label(label: str) -> str:
    return " ".join(label.lower().replace("_", " ").split())


_SYNONYMS: dict[str, set[str]] = {
    "tv": {"television", "monitor", "screen", "display"},
    "couch": {"sofa", "settee", "loveseat"},
    "cell phone": {"phone", "smartphone", "mobile"},
    "remote": {"controller", "control"},
    "laptop": {"notebook", "computer"},
}


def _label_matches_text(label: str, text: str) -> bool:
    if not label or not text:
        return False
    lbl = _normalize_label(label)
    t = text.lower()

    if lbl in t:
        return True

    # token-level match for single-word labels
    if " " not in lbl:
        tokens = {tok.strip(".,;:!?()[]{}\"'") for tok in t.split()}
        if lbl in tokens or (lbl + "s") in tokens:
            return True

    # synonym match
    if lbl in _SYNONYMS:
        if any(s in t for s in _SYNONYMS[lbl]):
            return True

    return False


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


def _frame_embedding_small(frame_bgr: np.ndarray, size: int = 16) -> np.ndarray:
    """Cheap frame embedding for scene-change detection.

    Produces a normalized vector from a downsampled grayscale frame.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    vec = small.reshape(-1)
    vec = vec - float(vec.mean())
    n = float(np.linalg.norm(vec) + 1e-8)
    return (vec / n).astype(np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def run_vlm_filter(
    video_path: Path,
    tracks_path: Path,
    out_tracks_path: Path,
    scene_jsonl_path: Optional[Path],
    audit_jsonl_path: Path,
    *,
    caption_every_n: int = 1,
    keep_confidence: float = 0.70,
    remove_confidence_below: float = 0.60,
    include_scene_captions: bool = True,
    scene_trigger: str = "none",
    scene_change_threshold: float = 0.985,
) -> None:
    records = list(_iter_jsonl(tracks_path))
    if not records:
        raise RuntimeError(f"No records found in: {tracks_path}")

    summaries = _summarize_tracks(records)
    logger.info("Loaded %d track observations, %d unique tracks", len(records), len(summaries))

    needed_scene_frames: set[int] = set()
    if include_scene_captions and scene_jsonl_path is not None:
        # only caption frames that actually exist in tracks
        all_frame_ids = sorted({int(r.get("frame_id")) for r in records if r.get("frame_id") is not None})
        for i, fid in enumerate(all_frame_ids):
            if caption_every_n <= 1 or (i % caption_every_n == 0):
                needed_scene_frames.add(fid)

    needed_object_frames: set[int] = {s.best_frame_id for s in summaries.values()}
    needed_frames = needed_scene_frames | needed_object_frames

    logger.info(
        "Need %d frames for object crops, %d candidate frames for scene captions (trigger=%s)",
        len(needed_object_frames),
        len(needed_scene_frames),
        scene_trigger,
    )

    mm = ModelManager.get_instance()
    vlm = mm.fastvlm

    track_desc: dict[int, str] = {}
    scene_caps: dict[int, str] = {}

    last_scene_emb: Optional[np.ndarray] = None

    # Sequential decode; trigger VLM only for needed frames
    for frame_id, frame_bgr in _iter_frames(video_path):
        if frame_id not in needed_frames:
            continue

        if include_scene_captions and frame_id in needed_scene_frames:
            should_caption = True
            if scene_trigger == "cosine":
                try:
                    cur_emb = _frame_embedding_small(frame_bgr)
                    if last_scene_emb is not None:
                        sim = _cosine_sim(cur_emb, last_scene_emb)
                        # Caption only when similarity drops below threshold (scene change)
                        should_caption = sim < float(scene_change_threshold)
                    if should_caption:
                        last_scene_emb = cur_emb
                except Exception as e:
                    logger.warning("Scene-change embedding failed at frame %d: %s", frame_id, e)
                    should_caption = True

            if should_caption:
                prompt = (
                    "Describe this scene in one short sentence. "
                    "Mention the main visible objects. "
                    "Avoid guessing about hidden objects."
                )
                try:
                    scene_caps[frame_id] = vlm.generate_description(
                        _bgr_to_pil(frame_bgr), prompt, max_tokens=64, temperature=0.0
                    )
                except Exception as e:
                    logger.warning("Scene caption failed at frame %d: %s", frame_id, e)

        # object crops: describe only the tracks that chose this best frame
        for track_id, summary in summaries.items():
            if summary.best_frame_id != frame_id:
                continue
            crop = _crop_bgr(frame_bgr, summary.best_bbox)
            prompt = (
                "Describe the object in the image crop in one short sentence. "
                "Be concrete about what it is."
            )
            try:
                track_desc[track_id] = vlm.generate_description(_bgr_to_pil(crop), prompt, max_tokens=64, temperature=0.0)
            except Exception as e:
                logger.warning("Track description failed for track %d frame %d: %s", track_id, frame_id, e)

        if len(track_desc) >= len(summaries) and (not include_scene_captions or len(scene_caps) >= len(needed_scene_frames)):
            break

    # Decide keep/remove per track
    decisions: dict[int, dict[str, Any]] = {}
    for track_id, summary in summaries.items():
        label = summary.label
        conf = summary.best_confidence
        desc = track_desc.get(track_id, "")

        keep = True
        reason = "default_keep"

        if conf >= keep_confidence:
            keep = True
            reason = f"keep_high_confidence_{conf:.2f}"
        else:
            # If FastVLM agrees, keep.
            if desc and _label_matches_text(label, desc):
                keep = True
                reason = "keep_vlm_agrees"
            else:
                # If FastVLM strongly contradicts and confidence is low -> remove
                if conf < remove_confidence_below and desc:
                    keep = False
                    reason = "remove_vlm_mismatch_low_confidence"
                else:
                    # Scene caption as weak prior
                    if conf < remove_confidence_below and scene_caps:
                        cap = scene_caps.get(summary.best_frame_id) or ""
                        if cap and (not _label_matches_text(label, cap)):
                            keep = False
                            reason = "remove_not_in_scene_caption_low_confidence"

        decisions[track_id] = {
            "track_id": track_id,
            "label": label,
            "best_confidence": conf,
            "best_frame_id": summary.best_frame_id,
            "vlm_object": desc,
            "vlm_scene": scene_caps.get(summary.best_frame_id),
            "keep": bool(keep),
            "reason": reason,
        }

    keep_ids = {tid for tid, d in decisions.items() if d["keep"]}
    removed_ids = sorted(set(decisions.keys()) - keep_ids)
    logger.info("Keep %d tracks; remove %d tracks", len(keep_ids), len(removed_ids))

    # Write outputs
    with out_tracks_path.open("w") as f:
        for rec in records:
            tid = rec.get("track_id", rec.get("id"))
            if tid is None:
                continue
            if int(tid) in keep_ids:
                f.write(json.dumps(rec) + "\n")

    with audit_jsonl_path.open("w") as f:
        for tid in sorted(decisions.keys()):
            f.write(json.dumps(decisions[tid]) + "\n")

    if include_scene_captions and scene_jsonl_path is not None:
        with scene_jsonl_path.open("w") as f:
            for fid in sorted(scene_caps.keys()):
                f.write(json.dumps({"frame_id": fid, "caption": scene_caps[fid]}) + "\n")

    logger.info("Wrote filtered tracks: %s", out_tracks_path)
    logger.info("Wrote audit: %s", audit_jsonl_path)
    if include_scene_captions and scene_jsonl_path is not None:
        logger.info("Wrote scene captions: %s", scene_jsonl_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="FastVLM-based sanity filter for tracks.jsonl")
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--results", type=str, required=True, help="Results directory (default output location)")
    ap.add_argument("--tracks", type=str, default="tracks.jsonl", help="Tracks filename (within results dir) or absolute path")
    ap.add_argument("--out-tracks", type=str, default="tracks_filtered.jsonl")
    ap.add_argument("--scene-jsonl", type=str, default="vlm_scene.jsonl")
    ap.add_argument("--audit-jsonl", type=str, default="vlm_filter_audit.jsonl")
    ap.add_argument("--caption-every-n", type=int, default=1, help="Caption every N sampled frames (1=all)")
    ap.add_argument(
        "--scene-trigger",
        choices=["none", "cosine"],
        default="none",
        help="Optional trigger to reduce scene caption calls",
    )
    ap.add_argument(
        "--scene-change-threshold",
        type=float,
        default=0.985,
        help="For --scene-trigger=cosine: caption when cosine similarity < threshold",
    )
    ap.add_argument("--keep-confidence", type=float, default=0.70)
    ap.add_argument("--remove-confidence-below", type=float, default=0.60)
    ap.add_argument("--no-scene-captions", action="store_true")

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results_dir = Path(args.results)
    tracks_path = Path(args.tracks)
    if not tracks_path.is_absolute():
        tracks_path = results_dir / tracks_path

    out_tracks_path = results_dir / args.out_tracks
    scene_jsonl_path = None if args.no_scene_captions else (results_dir / args.scene_jsonl)
    audit_jsonl_path = results_dir / args.audit_jsonl

    run_vlm_filter(
        video_path=Path(args.video),
        tracks_path=tracks_path,
        out_tracks_path=out_tracks_path,
        scene_jsonl_path=scene_jsonl_path,
        audit_jsonl_path=audit_jsonl_path,
        caption_every_n=max(1, int(args.caption_every_n)),
        keep_confidence=float(args.keep_confidence),
        remove_confidence_below=float(args.remove_confidence_below),
        include_scene_captions=(not args.no_scene_captions),
        scene_trigger=str(args.scene_trigger),
        scene_change_threshold=float(args.scene_change_threshold),
    )


if __name__ == "__main__":
    main()
