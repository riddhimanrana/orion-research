#!/usr/bin/env python3
"""Sample a reproducible subset of Ego4D video UIDs.

This script is intentionally lightweight and does NOT require the Ego4D package.
It samples IDs from either:
  1) Ego4D metadata JSON (ego4d.json), or
  2) a manifest CSV downloaded by the Ego4D CLI.

Outputs a whitespace-delimited UID file suitable for:
  ego4d --video_uid_file <file>

Note: The caller is responsible for honoring Ego4D license constraints.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import List, Optional, Sequence


def _read_uids_from_manifest(manifest_csv: Path, uid_column: Optional[str]) -> List[str]:
    with manifest_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {manifest_csv}")
        candidates = [c for c in (uid_column, "video_uid", "clip_uid", "uid") if c]
        chosen = None
        for c in candidates:
            if c in reader.fieldnames:
                chosen = c
                break
        if chosen is None:
            raise ValueError(
                f"Could not find UID column in {manifest_csv}. "
                f"Columns: {reader.fieldnames}. "
                f"Pass --uid-column to specify the correct one."
            )
        uids: List[str] = []
        for row in reader:
            v = (row.get(chosen) or "").strip()
            if v:
                uids.append(v)
        return sorted(set(uids))


def _read_uids_from_ego4d_json(ego4d_json: Path) -> List[str]:
    obj = json.loads(ego4d_json.read_text())

    # ego4d.json structure evolves; try common layouts.
    # We only need a stable list of canonical video_uids.
    uids: List[str] = []

    if isinstance(obj, dict):
        # Common: {"videos": [ {"video_uid": ...}, ... ]}
        videos = obj.get("videos")
        if isinstance(videos, list):
            for v in videos:
                if isinstance(v, dict) and v.get("video_uid"):
                    uids.append(str(v["video_uid"]))

        # Some releases store by benchmark sections; fall back to deep scan.
        if not uids:
            stack: List[object] = [obj]
            while stack:
                cur = stack.pop()
                if isinstance(cur, dict):
                    if "video_uid" in cur and cur["video_uid"]:
                        uids.append(str(cur["video_uid"]))
                    stack.extend(cur.values())
                elif isinstance(cur, list):
                    stack.extend(cur)

    return sorted(set(uids))


def _sample(uids: Sequence[str], n: int, seed: int) -> List[str]:
    if n <= 0:
        return []
    if n >= len(uids):
        return list(uids)
    rng = random.Random(seed)
    return sorted(rng.sample(list(uids), n))


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample Ego4D UIDs reproducibly")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ego4d-json", type=Path, help="Path to ego4d.json metadata")
    src.add_argument("--manifest", type=Path, help="Path to Ego4D CLI manifest.csv")
    ap.add_argument("--uid-column", help="UID column name in manifest (optional)")

    ap.add_argument("-n", type=int, default=100, help="Number of UIDs to sample")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--out", type=Path, required=True, help="Output UID file (whitespace delimited)")

    args = ap.parse_args()

    if args.ego4d_json:
        uids = _read_uids_from_ego4d_json(args.ego4d_json)
    else:
        uids = _read_uids_from_manifest(args.manifest, args.uid_column)

    if not uids:
        raise SystemExit("No UIDs found in source input")

    sampled = _sample(uids, args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sampled) + "\n")

    print(f"Found {len(uids)} total UIDs")
    print(f"Wrote {len(sampled)} UIDs to {args.out}")


if __name__ == "__main__":
    main()
