"""Batch-run Orion on selected Action Genome videos.

This helper automates the "mini dataset" workflow discussed in the QA conversation:

1. Pick a small subset of videos from the Action Genome dataset.
2. Optionally extract their frames into ``dataset/ag/frames/<VIDEO>.mp4/`` so the
   frame-based loaders can reuse them later.
3. Invoke ``orion analyze`` for each video and drop the results under
   ``results/ag_eval/<VIDEO_ID>/`` (or a custom directory).

Example (process 5 videos in fast mode and leave existing runs untouched)::

    conda activate orion
    python scripts/run_orion_on_ag_videos.py \
        --dataset-root "Action Genome Dataset/ActionGenome/dataset/ag" \
        --limit 5 \
        --mode fast \
        --skip-existing

"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    cv2 = None


def _read_video_list(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _select_videos(video_dir: Path, explicit: Iterable[str] | None, list_file: Path | None, limit: int | None) -> List[Path]:
    names: List[str]
    if explicit:
        names = [name if name.endswith(".mp4") else f"{name}.mp4" for name in explicit]
    elif list_file:
        names = _read_video_list(list_file)
    else:
        names = sorted(p.name for p in video_dir.glob("*.mp4"))

    if limit is not None:
        names = names[:limit]

    selected: List[Path] = []
    for name in names:
        candidate = video_dir / name
        if not candidate.is_file():
            print(f"[WARN] Skipping missing video: {candidate}")
            continue
        selected.append(candidate)
    return selected


def _extract_frames(video_path: Path, frames_dir: Path, *, overwrite: bool, stride: int = 1) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for frame extraction but is not available.")

    frames_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        existing = any(frames_dir.glob("*.png"))
        if existing:
            print(f"[INFO] Frames already exist for {video_path.name}; skipping extraction")
            return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for extraction: {video_path}")

    frame_idx = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            out_path = frames_dir / f"{frame_idx:06d}.png"
            cv2.imwrite(str(out_path), frame)
            saved_idx += 1
        frame_idx += 1

    cap.release()
    print(f"[INFO] Extracted {saved_idx} frame(s) from {video_path.name} → {frames_dir}")


def _run_orion(video_path: Path, output_dir: Path, *, mode: str, verbose: bool, skip_graph: bool, extra_args: List[str]) -> None:
    cmd = [
        sys.executable,
        "-m",
        "orion",
        "analyze",
        str(video_path),
        "--output",
        str(output_dir),
    ]

    if mode == "fast":
        cmd.append("--fast")
    elif mode == "accurate":
        cmd.append("--accurate")

    if verbose:
        cmd.append("--verbose")
    if skip_graph:
        cmd.append("--skip-graph")

    cmd.extend(extra_args)

    print(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Orion on a batch of Action Genome videos")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Action Genome Dataset/ActionGenome/dataset/ag"),
        help="Path to the Action Genome dataset root (contains videos/ and frames/)",
    )
    parser.add_argument(
        "--videos",
        nargs="*",
        help="Explicit list of video IDs (with or without .mp4). Overrides --limit",
    )
    parser.add_argument(
        "--video-list",
        type=Path,
        help="Text file containing video IDs (one per line)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of videos to process when no explicit list is provided",
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "balanced", "accurate"],
        default="balanced",
        help="Which Orion perception mode to use",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/ag_eval"),
        help="Directory where each Orion run will be stored",
    )
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Extract raw frames into dataset/ag/frames/<video>.mp4/ using OpenCV",
    )
    parser.add_argument(
        "--overwrite-frames",
        action="store_true",
        help="Re-extract frames even if PNGs already exist",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Only save every Nth frame when extracting (default: 1 = all frames)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip Orion runs whose output directory already contains scene_graph.jsonl",
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Pass --skip-graph to Orion (set only if you do not need graph exports)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose Orion output",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments forwarded to 'orion analyze'",
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    videos_dir = dataset_root / "videos"
    frames_root = dataset_root / "frames"

    if not videos_dir.is_dir():
        parser.error(f"Could not find videos directory at {videos_dir}")

    frames_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    selected_videos = _select_videos(
        videos_dir,
        args.videos,
        args.video_list,
        None if args.videos else args.limit,
    )
    if not selected_videos:
        parser.error("No videos selected. Provide --videos, --video-list, or ensure videos/ is populated.")

    summary = []
    for video_path in selected_videos:
        video_id = video_path.name.replace(".mp4", "")
        run_dir = args.output_root / video_id
        scene_graph = run_dir / "scene_graph.jsonl"

        print("=" * 80)
        print(f"[INFO] Processing {video_path.name}")
        print("=" * 80)

        if args.extract_frames:
            target_frames = frames_root / video_path.name
            try:
                _extract_frames(
                    video_path,
                    target_frames,
                    overwrite=args.overwrite_frames,
                    stride=max(1, args.frame_stride),
                )
            except Exception as exc:
                print(f"[ERROR] Frame extraction failed for {video_path.name}: {exc}")
                continue

        if args.skip_existing and scene_graph.exists():
            print(f"[INFO] Skipping Orion run for {video_path.name}; outputs already exist at {run_dir}")
            summary.append({"video": video_path.name, "status": "skipped"})
            continue

        try:
            _run_orion(
                video_path,
                run_dir,
                mode=args.mode,
                verbose=args.verbose,
                skip_graph=args.skip_graph,
                extra_args=args.extra_args,
            )
            summary.append({"video": video_path.name, "status": "ok", "output": str(run_dir)})
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] Orion analyze failed for {video_path.name}: {exc}")
            summary.append({"video": video_path.name, "status": "failed", "code": exc.returncode})

    print("\nBatch summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
