#!/usr/bin/env python3
"""Stage 1–3 sweep runner (Lambda-side).

This script automates iterative tuning for:
- Stage 1: Detection (YOLO / YOLO-World)
- Stage 1.5: CLIP-based candidate labels (open-vocab hypotheses)
- Stage 2: Tracking/Re-ID (EnhancedTracker + V-JEPA2 embeddings)
- Stage 3: Canonical label resolution (HDBSCAN over candidate labels)

It is intentionally designed to run on a GPU box (e.g., Lambda). In the Codespace
we typically do *not* execute the pipeline.

Outputs one subdirectory per run under --out-root, each containing:
- tracks.jsonl
- entities.json (if save_visualizations=True)
- camera_intrinsics.json
- profiling_stats.json

You can then run `scripts/gemini_pipeline_review.py` on any run directory.

Usage:
  python scripts/stage123_sweep.py --video data/examples/video.mp4 --out-root results/sweeps

  # Provide explicit sweep definitions:
  python scripts/stage123_sweep.py --video ... --out-root ... --sweep-json sweeps.json

Sweep JSON format:
[
  {
    "name": "yoloworld_coarse_t015",
    "target_fps": 5.0,
    "detection": {
      "backend": "yoloworld",
      "confidence_threshold": 0.15,
      "yoloworld_prompt_preset": "coarse",
      "yoloworld_use_custom_classes": true,
      "yoloworld_enable_candidate_labels": true,
      "yoloworld_candidate_top_k": 5,
      "yoloworld_candidate_rotate_every_frames": 4
    },
    "tracking": {
      "max_age": 30,
      "iou_threshold": 0.3,
      "appearance_threshold": 0.65
    }
  }
]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Allow running as a script from repo root (so `import orion` works without installation).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _apply_overrides(obj: Any, overrides: dict[str, Any]) -> None:
    """Apply dict overrides to a dataclass-like object (shallow)."""
    for k, v in overrides.items():
        if not hasattr(obj, k):
            raise AttributeError(f"Unknown config field: {obj.__class__.__name__}.{k}")
        setattr(obj, k, v)


def _default_sweep() -> list[dict[str, Any]]:
    # A small, safe sweep that tends to expose the main failure modes quickly.
    return [
        {
            "name": "yoloworld_coarse_t015",
            "target_fps": 5.0,
            "detection": {
                "backend": "yoloworld",
                "confidence_threshold": 0.15,
                "yoloworld_prompt_preset": "coarse",
                "yoloworld_use_custom_classes": True,
                "yoloworld_enable_candidate_labels": True,
                "yoloworld_candidate_top_k": 5,
                "yoloworld_candidate_rotate_every_frames": 4,
            },
            "tracking": {"max_age": 30, "iou_threshold": 0.3, "appearance_threshold": 0.65},
        },
        {
            "name": "yoloworld_coarse_t020",
            "target_fps": 5.0,
            "detection": {
                "backend": "yoloworld",
                "confidence_threshold": 0.20,
                "yoloworld_prompt_preset": "coarse",
                "yoloworld_use_custom_classes": True,
                "yoloworld_enable_candidate_labels": True,
                "yoloworld_candidate_top_k": 5,
                "yoloworld_candidate_rotate_every_frames": 4,
            },
            "tracking": {"max_age": 30, "iou_threshold": 0.3, "appearance_threshold": 0.65},
        },
        {
            "name": "yolo11m_t035",
            "target_fps": 5.0,
            "detection": {
                "backend": "yolo",
                "model": "yolo11m",
                "confidence_threshold": 0.35,
                # Candidate labels still work for YOLO backend.
                "yoloworld_enable_candidate_labels": True,
                "yoloworld_candidate_top_k": 5,
                "yoloworld_candidate_rotate_every_frames": 4,
            },
            "tracking": {"max_age": 30, "iou_threshold": 0.3, "appearance_threshold": 0.65},
        },
    ]


def run_one(video: Path, out_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    from orion.perception.config import PerceptionConfig, DetectionConfig, EmbeddingConfig
    from orion.perception.engine import PerceptionEngine

    out_dir.mkdir(parents=True, exist_ok=True)

    # Start from a conservative baseline.
    det = DetectionConfig()
    emb = EmbeddingConfig()
    cfg = PerceptionConfig(
        detection=det,
        embedding=emb,
        target_fps=float(spec.get("target_fps", 5.0)),
        enable_tracking=True,
        use_memgraph=False,
    )

    # Apply overrides.
    if "detection" in spec:
        _apply_overrides(cfg.detection, dict(spec["detection"]))
    if "tracking" in spec:
        _apply_overrides(cfg.tracking, dict(spec["tracking"]))

    # Persist run config for reproducibility.
    (out_dir / "run_spec.json").write_text(json.dumps(spec, indent=2))
    (out_dir / "perception_config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    (out_dir / "episode_meta.json").write_text(
        json.dumps(
            {
                "episode_id": out_dir.name,
                "video_path": str(video.resolve()),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
    )

    engine = PerceptionEngine(config=cfg, verbose=True)
    t0 = time.time()
    result = engine.process_video(str(video), save_visualizations=True, output_dir=str(out_dir))
    elapsed = time.time() - t0

    summary = {
        "name": spec.get("name"),
        "out_dir": str(out_dir),
        "elapsed_s": elapsed,
        "entities": len(result.entities),
        "observations": len(result.raw_observations),
        "metrics": result.metrics,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a Stage 1–3 sweep (Lambda-side)")
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--out-root", required=True, help="Output root directory")
    ap.add_argument("--sweep-json", default=None, help="Optional JSON file with sweep specs")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.sweep_json:
        sweep = json.loads(Path(args.sweep_json).read_text())
    else:
        sweep = _default_sweep()

    scoreboard: list[dict[str, Any]] = []
    for i, spec in enumerate(sweep):
        name = spec.get("name") or f"run_{i:03d}"
        out_dir = out_root / name
        scoreboard.append(run_one(video, out_dir, spec))

    (out_root / "SCOREBOARD.json").write_text(json.dumps(scoreboard, indent=2, default=str))
    print(f"✓ Wrote {out_root / 'SCOREBOARD.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
