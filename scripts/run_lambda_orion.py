#!/usr/bin/env python3
"""Run Orion end-to-end on `lambda-orion` via SSH and pull artifacts.

This script is intentionally minimal: it shells out to `ssh`/`scp`.

Example
-------
python scripts/run_lambda_orion.py \
  --episode test_demo \
  --video data/examples/test.mp4 \
  --pull-to results/_pulled

Notes
-----
- Assumes `lambda-orion` is a working SSH host alias.
- Assumes the repo exists on the remote machine at `--remote-repo`.
- Assumes the remote shell can run the `--remote-prefix` snippet to activate env.

This script defaults to the shared Lambda NFS layout used by Orion:
`~/orion-core-fs/orion-research` (a symlink to `/lambda/nfs/orion-core-fs/orion-research`).
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _ssh(host: str, remote_cmd: str) -> None:
    # Use bash -lc so conda activation works if configured.
    _run(["ssh", host, "bash", "-lc", remote_cmd])


def _scp_dir(host: str, remote_path: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    _run(["scp", "-r", f"{host}:{remote_path}", str(local_dir)])


def main() -> None:
    ap = argparse.ArgumentParser(description="SSH-run Orion pipeline on lambda-orion and pull results")
    ap.add_argument("--host", type=str, default="lambda-orion", help="SSH host (alias or user@host)")
    ap.add_argument(
        "--remote-repo",
        type=str,
        default="~/orion-core-fs/orion-research",
        help="Remote repo path (default matches Lambda NFS layout)",
    )
    ap.add_argument(
        "--remote-prefix",
        type=str,
        default=(
            # Run after `cd <remote_repo>`.
            "(" \
            "if [ -f venv/bin/activate ]; then source venv/bin/activate; " \
            "elif [ -f .venv/bin/activate ]; then source .venv/bin/activate; " \
            "elif [ -f $HOME/miniconda3/etc/profile.d/conda.sh ]; then " \
            "  source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate orion; " \
            "elif command -v conda >/dev/null 2>&1; then conda activate orion; " \
            "fi" \
            ")"
        ),
        help=(
            "Shell snippet to activate env on remote (executed after cd into repo). "
            "Defaults to auto-detect venv/.venv/conda."
        ),
    )

    ap.add_argument("--episode", type=str, required=True, help="Episode id (results/<episode>)")
    ap.add_argument("--video", type=str, required=True, help="Video path on remote (relative to repo or absolute)")

    ap.add_argument("--no-showcase", action="store_true", help="Skip Orion showcase run")
    ap.add_argument("--vlm-filter", action="store_true", help="Run FastVLM filtering on produced tracks")
    ap.add_argument("--reid", action="store_true", help="Run Phase-2 ReID (memory builder)")
    ap.add_argument("--reid-use-filtered", action="store_true", help="Run ReID on tracks_filtered.jsonl")
    ap.add_argument("--gemini", action="store_true", help="Run Gemini comparison against tracks.jsonl")

    ap.add_argument("--pull-to", type=str, default="results/_pulled", help="Local folder to pull results into")

    args = ap.parse_args()

    remote_repo = args.remote_repo
    episode = args.episode
    results_dir = f"results/{episode}"

    # Build remote video path. If relative, treat as relative to repo.
    video_arg = args.video
    if not video_arg.startswith("/") and not video_arg.startswith("~"):
        video_arg = f"{remote_repo.rstrip('/')}/{video_arg}"

    prefix = args.remote_prefix.strip()
    cd_repo = f"cd {shlex.quote(remote_repo)}"

    def runcmd(cmd: str) -> None:
        # Run activation after we are in the repo so relative venv paths work.
        full = f"{cd_repo} && {prefix} && {cmd}" if prefix else f"{cd_repo} && {cmd}"
        _ssh(args.host, full)

    if not args.no_showcase:
        runcmd(
            "python -m orion.cli.run_showcase "
            f"--episode {shlex.quote(episode)} "
            f"--video {shlex.quote(video_arg)}"
        )

    if args.vlm_filter:
        runcmd(
            "python -m orion.cli.run_vlm_filter "
            f"--video {shlex.quote(video_arg)} "
            f"--results {shlex.quote(results_dir)} "
            "--scene-trigger cosine --scene-change-threshold 0.985"
        )

    if args.reid:
        if args.reid_use_filtered:
            runcmd(
                "python -m orion.cli.run_reid "
                f"--video {shlex.quote(video_arg)} "
                f"--tracks {shlex.quote(results_dir + '/tracks_filtered.jsonl')} "
                f"--out {shlex.quote(results_dir + '/reid_filtered')}"
            )
        else:
            runcmd(
                "python -m orion.cli.run_reid "
                f"--episode {shlex.quote(episode)} "
                f"--video {shlex.quote(video_arg)}"
            )

    if args.gemini:
        # Use existing tracks.jsonl under results/<episode>.
        runcmd(
            "python scripts/test_gemini_comparison.py "
            f"--video {shlex.quote(video_arg)} "
            f"--output {shlex.quote(results_dir)} "
            "--skip-orion"
        )

    # Pull entire episode folder (simple + robust).
    local_pull_root = Path(args.pull_to)
    _scp_dir(args.host, f"{remote_repo.rstrip('/')}/{results_dir}", local_pull_root)


if __name__ == "__main__":
    main()
