#!/usr/bin/env python3
"""Gemini-powered pipeline review for Orion results.

This script is meant to answer: "Where are we lacking and how do we fix it?" across:
- Detection (misses / false positives / prompt issues)
- Classification (label confusion)
- Tracking (fragmentation / ID switches)
- Re-identification (wrong merges / missed merges)
- FastVLM semantics (caption/description consistency)

It works by:
1) Loading existing Orion artifacts (tracks.jsonl + optional entities.json/pipeline_summary.json)
2) Sampling informative frames and rendering an overlay (bbox + label + track_id)
3) Asking Gemini to critique each frame and propose concrete fixes
4) Optionally auditing track identity by comparing distant crops for the same track_id

Usage:
  python scripts/gemini_pipeline_review.py \
    --results results/validation \
    --video data/examples/test.mp4 \
    --num-frames 8 \
    --track-pairs 10

Output:
  <results>/gemini_pipeline_review.json

Environment:
  Set GOOGLE_API_KEY or GEMINI_API_KEY (supports .env in repo root).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orion.utils.gemini_client import GeminiClientError, get_gemini_model


# ----------------------------
# Utilities
# ----------------------------


def resolve_video_path(video_arg: str | None, results_dir: Path) -> Path | None:
    """Resolve a usable video path.

    Supports:
    - explicit --video <path>
    - results_dir/episode_meta.json: {"video_path": ...}
    - common repo-relative shortcuts like `test.mp4` -> `data/examples/test.mp4`
    """

    repo_root = Path(__file__).resolve().parent.parent

    candidates: list[Path] = []

    if video_arg:
        p = Path(video_arg)
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(repo_root / p)

        # If user passed a bare filename, try common locations.
        if "/" not in video_arg and "\\" not in video_arg:
            candidates.append(repo_root / "data" / "examples" / video_arg)
            candidates.append(repo_root / "data" / video_arg)

    # Try to infer from episode_meta.json if present
    meta = load_optional_json(results_dir / "episode_meta.json")
    vp = (meta or {}).get("video_path")
    if vp:
        vpp = Path(vp)
        candidates.append(vpp)
        if not vpp.is_absolute():
            candidates.append(repo_root / vpp)

    for c in candidates:
        try:
            if c.exists():
                return c.resolve()
        except OSError:
            continue

    return None


def to_base64_jpeg(frame_bgr) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode image")
    return base64.b64encode(buf).decode("utf-8")


def safe_json_extract(text: str) -> Any:
    """Parse JSON even if the model wraps it in markdown code fences."""
    t = (text or "").strip()
    if t.startswith("```"):
        # try ```json ... ``` or ``` ... ```
        parts = t.split("```")
        if len(parts) >= 3:
            t = parts[1]
            # strip an optional language hint
            if t.lstrip().startswith("json"):
                t = t.lstrip()[4:]
            t = t.strip("\n")
    # last resort: find first '{'..last '}'
    if "{" in t and "}" in t:
        start = t.find("{")
        end = t.rfind("}")
        t = t[start : end + 1]
    return json.loads(t)


@dataclass(frozen=True)
class TrackDet:
    frame_id: int
    track_id: int | None
    bbox: list[float]  # [x1,y1,x2,y2]
    label: str
    score: float | None
    raw: dict[str, Any]


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_track_row(row: dict[str, Any]) -> TrackDet | None:
    # Frame id
    frame = row.get("frame_id")
    if frame is None:
        frame = row.get("frame_number")
    if frame is None:
        return None

    # Bbox
    # Orion tracks.jsonl commonly uses bbox_2d=[x1,y1,x2,y2].
    bbox = row.get("bbox")
    if bbox is None:
        bb2d = row.get("bbox_2d")
        if isinstance(bb2d, list) and len(bb2d) == 4:
            bbox = bb2d
        elif isinstance(bb2d, dict) and {"x1", "y1", "x2", "y2"}.issubset(bb2d.keys()):
            bbox = [bb2d["x1"], bb2d["y1"], bb2d["x2"], bb2d["y2"]]

    if bbox is None:
        bb = row.get("bounding_box")
        # Some formats store dict or list; try best-effort.
        if isinstance(bb, list) and len(bb) == 4:
            bbox = bb
        elif isinstance(bb, dict) and {"x1", "y1", "x2", "y2"}.issubset(bb.keys()):
            bbox = [bb["x1"], bb["y1"], bb["x2"], bb["y2"]]

    if bbox is None or not isinstance(bbox, list) or len(bbox) != 4:
        return None

    # Track id
    track_id = row.get("track_id")
    if track_id is None:
        track_id = row.get("id")

    # Label
    label = (
        row.get("category")
        or row.get("label")
        or row.get("class_name")
        or row.get("object_class")
        or "unknown"
    )

    # Score
    score = row.get("score")
    if score is None:
        score = row.get("confidence")

    try:
        frame_i = int(frame)
    except Exception:
        return None

    try:
        tid_i = int(track_id) if track_id is not None else None
    except Exception:
        tid_i = None

    bbox_f = [float(x) for x in bbox]
    score_f = float(score) if score is not None else None

    return TrackDet(frame_id=frame_i, track_id=tid_i, bbox=bbox_f, label=str(label), score=score_f, raw=row)


def load_tracks(results_dir: Path) -> list[TrackDet]:
    tracks_path = results_dir / "tracks.jsonl"
    if not tracks_path.exists():
        raise FileNotFoundError(f"Missing tracks.jsonl at {tracks_path}")

    out: list[TrackDet] = []
    for row in read_jsonl(tracks_path):
        det = normalize_track_row(row)
        if det is not None:
            out.append(det)
    return out


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def draw_overlay(frame_bgr, dets: list[TrackDet]) -> Any:
    """Draw bbox + label + track_id onto a frame."""
    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 255),
        (255, 128, 0),
    ]

    out = frame_bgr.copy()
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = [int(round(v)) for v in d.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(out.shape[1] - 1, x2), min(out.shape[0] - 1, y2)

        color = colors[i % len(colors)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

        tid = "?" if d.track_id is None else str(d.track_id)
        score = "" if d.score is None else f" {d.score:.2f}"
        label = f"{d.label}#{tid}{score}"

        font_scale = 0.6
        thickness = 2
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(out, (x1, max(0, y1 - h - 8)), (x1 + w + 4, y1), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 2, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    return out


def crop_from_bbox(frame_bgr, bbox: list[float], pad: float = 0.12):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    x1p = int(max(0, round(x1 - pad * bw)))
    y1p = int(max(0, round(y1 - pad * bh)))
    x2p = int(min(w - 1, round(x2 + pad * bw)))
    y2p = int(min(h - 1, round(y2 + pad * bh)))

    if x2p <= x1p or y2p <= y1p:
        return frame_bgr
    return frame_bgr[y1p:y2p, x1p:x2p]


def read_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    return frame


def pick_informative_frames(by_frame: dict[int, list[TrackDet]], num_frames: int) -> list[int]:
    frames = sorted(by_frame.keys())
    if not frames:
        return []

    # Top-K busiest frames
    busiest = sorted(frames, key=lambda f: len(by_frame[f]), reverse=True)
    k_busy = max(1, num_frames // 2)
    chosen = list(dict.fromkeys(busiest[:k_busy]))

    # Remaining evenly spaced
    remaining = num_frames - len(chosen)
    if remaining > 0:
        step = max(1, len(frames) // (remaining + 1))
        for i in range(step, len(frames), step):
            chosen.append(frames[i])
            if len(chosen) >= num_frames:
                break

    return sorted(list(dict.fromkeys(chosen)))[:num_frames]


# ----------------------------
# Gemini prompts
# ----------------------------

def ask_gemini_frame_audit(model, frame_b64: str, frame_idx: int, dets: list[TrackDet], global_context: str):
    # Trim detections to keep prompt size stable
    dets_sorted = sorted(dets, key=lambda d: (d.score is None, -(d.score or 0.0)))
    dets_small = [
        {
            "track_id": d.track_id,
            "label": d.label,
            "score": d.score,
            "bbox": [round(x, 1) for x in d.bbox],
        }
        for d in dets_sorted[:30]
    ]

    schema = r'''{
    "visible_objects": ["..."] ,
    "missed_objects": ["..."] ,
    "false_positives": ["..."] ,
    "misclassifications": [{"predicted": "...", "should_be": "...", "why": "..."}],
    "bbox_quality": "...",
    "tracking_notes": "...",
    "recommended_fixes": {
         "detection": ["..."],
         "classification": ["..."],
         "tracking": ["..."],
         "reid": ["..."]
    },
    "confidence": 0.0
}'''

    prompt = f"""You are auditing an object tracking pipeline output for a video.

{global_context}

FRAME_AUDIT
- frame_idx: {frame_idx}
- The image has colored boxes with text labels `label#track_id score`.
- Below is the system's structured detection list for the frame (may be truncated to top items):
{json.dumps(dets_small, indent=2)}

Task:
1) List the main visible objects.
2) Compare vs the system detections: call out obvious misses, false positives, and misclassifications.
3) Comment briefly on bbox quality (too loose/tight, drift).
4) Suggest concrete fixes, grouped by stage:
   - detection (add/remove classes, prompt size, threshold)
   - classification (label set / confusion pairs)
   - tracking (IoU/gating, motion, NMS effects)
   - re-id (thresholds, embedding issues, merges)

Return STRICT JSON in this schema:
{schema}
"""

    resp = model.generate_content([
        {"mime_type": "image/jpeg", "data": frame_b64},
        prompt,
    ])
    return safe_json_extract(resp.text)


def ask_gemini_track_pair_audit(model, crop_a_b64: str, crop_b_b64: str, track_meta: dict[str, Any]):
    schema = r'''{
    "same_object": true/false,
    "confidence": 0.0,
    "reasoning": "...",
    "likely_failure_mode": "fragmentation|id_switch|wrong_merge|unknown",
    "recommended_fixes": ["..."],
    "notes": "..."
}'''

    prompt = f"""You are checking whether two crops correspond to the same physical object.

Track meta (from the tracking system):
{json.dumps(track_meta, indent=2)}

Questions:
1) Do these two crops look like the SAME physical object instance?
2) If not, this indicates a likely ID-switch or track contamination.
3) Provide concrete suggestions to reduce this issue (tracking gating, re-id thresholding, candidate labels, etc.).

Return STRICT JSON:
{schema}
"""

    resp = model.generate_content([
        {"mime_type": "image/jpeg", "data": crop_a_b64},
        {"mime_type": "image/jpeg", "data": crop_b_b64},
        prompt,
    ])
    return safe_json_extract(resp.text)


def ask_gemini_overall_review(model, context: dict[str, Any], frame_audits: list[dict[str, Any]], track_audits: list[dict[str, Any]]):
    schema = r'''{
    "prioritized_issues": [
        {
            "title": "...",
            "stage": "detection|classification|tracking|reid|vlm|other",
            "severity": "low|medium|high|critical",
            "evidence": ["..."],
            "likely_root_cause": "...",
            "concrete_fixes": ["..."],
            "validation_experiments": ["..."]
        }
    ],
    "quick_wins": ["..."],
    "longer_term_work": ["..."],
    "confidence": 0.0
}'''

    prompt = f"""You are producing a comprehensive evaluation + action plan for a video perception pipeline.

Context (system-produced metrics & summaries):
{json.dumps(context, indent=2)}

Frame-level audits (Gemini outputs):
{json.dumps(frame_audits, indent=2)[:12000]}

Track identity audits (Gemini outputs):
{json.dumps(track_audits, indent=2)[:12000]}

Deliver:
1) A prioritized list of issues with severity/impact.
2) For each issue: symptoms, most likely root cause, and a concrete fix (config knobs and/or code locations).
3) A short "next experiments" list to validate improvements.

Return STRICT JSON:
{schema}
"""

    resp = model.generate_content(prompt)
    return safe_json_extract(resp.text)


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Gemini-powered audit of Orion pipeline outputs")
    parser.add_argument("--results", required=True, help="Results directory (contains tracks.jsonl)")
    parser.add_argument("--video", type=str, default=None, help="Optional video path for frame extraction")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames to audit")
    parser.add_argument("--track-pairs", type=int, default=10, help="Number of track identity pairs to audit")
    parser.add_argument("--max-dets-per-frame", type=int, default=30, help="Max detections per frame to include")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep seconds between Gemini calls")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Gemini model name (overrides GEMINI_MODEL env). Default: gemini-3-flash-preview",
    )
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    model_name = args.model or os.environ.get("GEMINI_MODEL") or "gemini-3-flash-preview"
    try:
        model = get_gemini_model(model_name)
    except GeminiClientError as exc:
        raise RuntimeError(str(exc)) from exc

    tracks = load_tracks(results_dir)
    by_frame: dict[int, list[TrackDet]] = defaultdict(list)
    by_track: dict[int, list[TrackDet]] = defaultdict(list)

    for d in tracks:
        by_frame[d.frame_id].append(d)
        if d.track_id is not None:
            by_track[d.track_id].append(d)

    label_counts = Counter(d.label for d in tracks)
    track_lengths = {
        tid: len({d.frame_id for d in dets}) for tid, dets in by_track.items()
    }

    # Optional artifacts
    entities = load_optional_json(results_dir / "entities.json")
    pipeline_summary = load_optional_json(results_dir / "pipeline_summary.json")
    gemini_comparison = load_optional_json(results_dir / "gemini_comparison.json")

    sample_entity_descriptions = None
    try:
        if isinstance(entities, dict) and isinstance(entities.get("entities"), list):
            sample_entity_descriptions = [
                {
                    "id": e.get("id"),
                    "class": e.get("class"),
                    "confidence": e.get("confidence"),
                    "observation_count": e.get("observation_count"),
                    "description": e.get("description"),
                }
                for e in entities.get("entities", [])[:20]
            ]
    except Exception:
        sample_entity_descriptions = None

    context = {
        "results_dir": str(results_dir),
        "tracks": {
            "total_detections": len(tracks),
            "sampled_frames": len(by_frame),
            "unique_track_ids": len(by_track),
            "top_labels": dict(label_counts.most_common(30)),
            "top_tracks_by_length": sorted(track_lengths.items(), key=lambda kv: kv[1], reverse=True)[:20],
        },
        "artifacts_present": {
            "entities.json": entities is not None,
            "pipeline_summary.json": pipeline_summary is not None,
            "gemini_comparison.json": gemini_comparison is not None,
        },
        "pipeline_summary": pipeline_summary,
        "entities_summary": {
            "total_entities": (entities or {}).get("total_entities"),
            "tracking_stats": (entities or {}).get("tracking_stats"),
            "sample_descriptions": sample_entity_descriptions,
        } if entities else None,
        "previous_gemini_comparison": gemini_comparison,
    }

    # Determine video path (required for frame-level audits)
    video_path = resolve_video_path(args.video, results_dir)
    if video_path is None:
        hint = "data/examples/test.mp4" if args.video in {None, "test.mp4"} else None
        extra = f" Try: --video {hint}" if hint else ""
        raise FileNotFoundError(
            "Video path not provided or not found. Pass --video <path> (or ensure results/episode_meta.json has video_path)."
            + extra
        )

    # Frame audits
    chosen_frames = pick_informative_frames(by_frame, args.num_frames)

    global_context = (
        f"System summary: total_detections={len(tracks)}, sampled_frames={len(by_frame)}, "
        f"unique_tracks={len(by_track)}. Top labels: {dict(label_counts.most_common(10))}."
    )

    frame_audits: list[dict[str, Any]] = []
    for frame_idx in chosen_frames:
        dets = sorted(by_frame[frame_idx], key=lambda d: (d.score is None, -(d.score or 0.0)))[: args.max_dets_per_frame]
        frame = read_frame(video_path, frame_idx)
        overlay = draw_overlay(frame, dets)
        frame_b64 = to_base64_jpeg(overlay)

        audit = ask_gemini_frame_audit(model, frame_b64, frame_idx, dets, global_context)
        audit["frame_idx"] = frame_idx
        audit["system_detections_included"] = len(dets)
        frame_audits.append(audit)
        time.sleep(args.sleep)

    # Track identity audits
    track_audits: list[dict[str, Any]] = []
    candidate_tids = [tid for tid, ln in sorted(track_lengths.items(), key=lambda kv: kv[1], reverse=True) if ln >= 2]
    random.shuffle(candidate_tids)

    for tid in candidate_tids[: args.track_pairs]:
        dets = sorted(by_track[tid], key=lambda d: d.frame_id)
        a = dets[0]
        b = dets[-1]

        frame_a = read_frame(video_path, a.frame_id)
        frame_b = read_frame(video_path, b.frame_id)

        crop_a = crop_from_bbox(frame_a, a.bbox)
        crop_b = crop_from_bbox(frame_b, b.bbox)

        crop_a_b64 = to_base64_jpeg(crop_a)
        crop_b_b64 = to_base64_jpeg(crop_b)

        meta = {
            "track_id": tid,
            "label": a.label,
            "first_frame": a.frame_id,
            "last_frame": b.frame_id,
            "track_length_frames": track_lengths.get(tid),
        }

        audit = ask_gemini_track_pair_audit(model, crop_a_b64, crop_b_b64, meta)
        audit["track_meta"] = meta
        track_audits.append(audit)
        time.sleep(args.sleep)

    overall = ask_gemini_overall_review(model, context, frame_audits, track_audits)

    output_path = results_dir / "gemini_pipeline_review.json"
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "video": str(video_path),
        "context": context,
        "frame_audits": frame_audits,
        "track_identity_audits": track_audits,
        "overall_review": overall,
    }

    output_path.write_text(json.dumps(payload, indent=2))
    print(f"✓ Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
