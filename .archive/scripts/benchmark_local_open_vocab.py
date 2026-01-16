#!/usr/bin/env python3
"""Benchmark local open-vocab detector backends.

This script runs `python -m orion.cli.run_tracks` for a set of detector backends
on the same video and summarizes key stats from `run_metadata.json`.

Why this exists:
- You asked to avoid cloud APIs.
- GroundingDINO is available locally in Orion already.
- This provides a repeatable bakeoff between:
  - yoloworld (prompted)
  - yoloworld (open vocab / no set_classes)
  - groundingdino
  - hybrid
  - openvocab (propose→label)

Example:
  python scripts/benchmark_local_open_vocab.py --video data/examples/video_short.mp4 --fps 2 --device cuda

Notes:
- Runs are sequential and can be slow on first run due to model downloads.
- Each run writes to `results/<episode_id>/`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_run_meta(results_dir: Path) -> dict[str, Any] | None:
    meta = results_dir / "run_metadata.json"
    if not meta.exists():
        return None
    return json.loads(meta.read_text())


def main() -> int:
    root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Benchmark local open-vocab detector backends")
    parser.add_argument("--video", type=str, default="data/examples/video_short.mp4")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])

    parser.add_argument(
        "--prompt-preset",
        type=str,
        default="coco",
        choices=["coco", "coarse", "indoor_full", "custom"],
        help="Prompt preset used for yoloworld + groundingdino category lists (ignored for yoloworld-open-vocab).",
    )
    parser.add_argument(
        "--yoloworld-prompt",
        type=str,
        default=None,
        help="Custom prompt when --prompt-preset custom (dot-separated, e.g. 'chair . table . lamp').",
    )

    parser.add_argument("--yolo-model", type=str, default="yolo11m", choices=["yolo11n", "yolo11s", "yolo11m", "yolo11x"])
    parser.add_argument(
        "--gdino-model",
        type=str,
        default="IDEA-Research/grounding-dino-tiny",
        choices=["IDEA-Research/grounding-dino-tiny", "IDEA-Research/grounding-dino-base"],
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (shared across runs).",
    )

    parser.add_argument(
        "--base-episode",
        type=str,
        default="bench_local_ov",
        help="Prefix for results episode ids.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag appended to episode ids. Default: timestamp.",
    )

    parser.add_argument(
        "--runs",
        type=str,
        default="yoloworld,yoloworld_open_vocab,groundingdino,hybrid,openvocab",
        help=(
            "Comma-separated run set. Supported: yoloworld, yoloworld_open_vocab, groundingdino, hybrid, openvocab"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run if results/<episode>/run_metadata.json already exists.",
    )

    args = parser.parse_args()

    video = (root / args.video).resolve() if not Path(args.video).is_absolute() else Path(args.video)
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    if args.prompt_preset == "custom" and not args.yoloworld_prompt:
        raise SystemExit("--prompt-preset custom requires --yoloworld-prompt")

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    runs = [r.strip() for r in str(args.runs).split(",") if r.strip()]
    supported = {"yoloworld", "yoloworld_open_vocab", "groundingdino", "hybrid", "openvocab"}
    unknown = [r for r in runs if r not in supported]
    if unknown:
        raise SystemExit(f"Unknown runs: {unknown}. Supported: {sorted(supported)}")

    summaries: list[dict[str, Any]] = []

    for r in runs:
        episode = f"{args.base_episode}_{r}_{tag}"
        results_dir = root / "results" / episode

        if args.skip_existing:
            meta = _load_run_meta(results_dir)
            if meta is not None:
                summaries.append({"run": r, "episode": episode, "skipped": True, "meta": meta})
                continue

        cmd = [
            sys.executable,
            "-m",
            "orion.cli.run_tracks",
            "--video",
            str(video),
            "--episode",
            episode,
            "--fps",
            str(args.fps),
            "--device",
            str(args.device),
            "--conf-threshold",
            str(args.conf),
            "--model",
            str(args.yolo_model),
        ]

        # Configure detector backend for this run
        if r == "yoloworld":
            cmd += ["--detector-backend", "yoloworld"]
            if args.prompt_preset:
                cmd += ["--prompt-preset", str(args.prompt_preset)]
            if args.yoloworld_prompt:
                cmd += ["--yoloworld-prompt", str(args.yoloworld_prompt)]
        elif r == "yoloworld_open_vocab":
            cmd += ["--detector-backend", "yoloworld", "--yoloworld-open-vocab"]
        elif r == "groundingdino":
            cmd += ["--detector-backend", "groundingdino", "--gdino-model", str(args.gdino_model)]
            if args.prompt_preset:
                cmd += ["--prompt-preset", str(args.prompt_preset)]
            if args.yoloworld_prompt:
                cmd += ["--yoloworld-prompt", str(args.yoloworld_prompt)]
        elif r == "hybrid":
            cmd += ["--detector-backend", "hybrid", "--gdino-model", str(args.gdino_model)]
        elif r == "openvocab":
            cmd += ["--detector-backend", "openvocab"]
        else:
            raise AssertionError(r)

        results_dir.mkdir(parents=True, exist_ok=True)
        log_path = results_dir / "benchmark_run.log"
        with open(log_path, "w") as f:
            f.write("Command:\n")
            f.write(" ".join(cmd) + "\n\n")
            f.flush()
            proc = subprocess.run(cmd, cwd=str(root), stdout=f, stderr=subprocess.STDOUT)

        meta = _load_run_meta(results_dir)
        summaries.append({"run": r, "episode": episode, "returncode": proc.returncode, "meta": meta})

    # Print and persist a compact summary
    out = {
        "video": str(video),
        "fps": args.fps,
        "device": args.device,
        "prompt_preset": args.prompt_preset,
        "yoloworld_prompt": args.yoloworld_prompt,
        "tag": tag,
        "runs": [],
    }

    for item in summaries:
        meta = item.get("meta") or {}
        stats = (meta.get("statistics") or {}) if isinstance(meta, dict) else {}
        det = (meta.get("detector") or {}) if isinstance(meta, dict) else {}
        out["runs"].append(
            {
                "run": item.get("run"),
                "episode": item.get("episode"),
                "returncode": item.get("returncode"),
                "skipped": bool(item.get("skipped")),
                "detector": det,
                "statistics": {
                    "total_detections": stats.get("total_detections"),
                    "unique_tracks": stats.get("unique_tracks"),
                    "frames_processed": stats.get("frames_processed"),
                },
            }
        )

    summary_path = root / "results" / f"{args.base_episode}_summary_{tag}.json"
    summary_path.write_text(json.dumps(out, indent=2))

    # Human-readable table
    print("\n=== Local open-vocab benchmark summary ===")
    print(f"video: {video}")
    print(f"fps: {args.fps}  device: {args.device}  prompt_preset: {args.prompt_preset}")
    for r in out["runs"]:
        st = r.get("statistics") or {}
        print(
            f"- {r['run']:<20} dets={st.get('total_detections')}  tracks={st.get('unique_tracks')}  frames={st.get('frames_processed')}  episode={r['episode']}"
        )
    print(f"\nSaved summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
