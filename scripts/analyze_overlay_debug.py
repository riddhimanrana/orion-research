#!/usr/bin/env python3
"""Analyze overlay debug JSONL produced by orion.perception.viz_overlay_v2.

Usage:
  python scripts/analyze_overlay_debug.py results/<episode>/overlay_debug.jsonl

Prints a small timing summary (delta between track timestamp and computed frame/FPS timestamp).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/analyze_overlay_debug.py <overlay_debug.jsonl>")
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        return 2

    deltas = []
    frames = 0
    det_counts = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            frames += 1
            det_counts.append(int(obj.get("num_detections", 0)))
            delta = obj.get("delta_ms")
            if delta is not None:
                try:
                    deltas.append(float(delta))
                except Exception:
                    pass

    print(f"Frames logged: {frames}")
    if det_counts:
        print(f"Detections/frame: mean={np.mean(det_counts):.2f}, max={np.max(det_counts)}")

    if not deltas:
        print("No delta_ms values present (track timestamps missing?)")
        return 0

    arr = np.asarray(deltas, dtype=np.float32)
    abs_arr = np.abs(arr)
    print(
        "delta_ms: "
        f"p50={np.percentile(arr, 50):.1f} "
        f"p95={np.percentile(arr, 95):.1f} "
        f"max_abs={np.max(abs_arr):.1f}"
    )

    # Quick outlier count
    for thresh in (50, 100, 250):
        cnt = int(np.sum(abs_arr > thresh))
        print(f"|delta_ms| > {thresh}ms: {cnt}/{len(arr)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
