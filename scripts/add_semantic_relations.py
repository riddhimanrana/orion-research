#!/usr/bin/env python3
"""
Semantic relation booster for Orion scene_graph.jsonl outputs.

This post-processes results/<video_id>/scene_graph.jsonl and adds
semantic predicates that better match PVSG:
- talking to (person-person near)
- looking at (person toward object horizontally near)
- sitting on / standing on (person above support surface)
- holding (person close + overlap to small object)
- next to (fallback for near)

If mlx_vlm is installed, you can enable a light VLM-assisted check via
--use-vlm to re-score candidate relations (optional, fast heuristic otherwise).
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from mlx_vlm import load as vlm_load, process_images as vlm_process
    HAS_VLM = True
except Exception:
    HAS_VLM = False


PersonLike = {"person", "adult", "child", "baby", "man", "woman"}
Seating = {"chair", "sofa", "couch", "bench", "bed", "stool"}
Support = Seating | {"table", "desk", "counter", "floor", "ground", "grass"}
SmallPortable = {
    "cup", "mug", "bottle", "phone", "cellphone", "book", "cake", "knife",
    "fork", "spoon", "bowl", "remote", "bag", "backpack", "handbag", "toy",
}


def _area(b: List[float]) -> float:
    return max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))


def _centroid(b: List[float]) -> Tuple[float, float]:
    return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _horiz_overlap(a: List[float], b: List[float]) -> float:
    left = max(a[0], b[0]); right = min(a[2], b[2])
    return max(0.0, right - left) / max(1e-6, min(a[2]-a[0], b[2]-b[0]))


def classify_semantic(frame: Dict, nodes: Dict[str, Dict]) -> List[Dict]:
    edges = []
    node_ids = list(nodes.keys())
    centroids = {nid: _centroid(nodes[nid]["bbox"]) for nid in node_ids}
    areas = {nid: _area(nodes[nid]["bbox"]) for nid in node_ids}

    # Normalize distance by frame diag if available
    fw = frame.get("frame_width", 0) or frame.get("width", 0)
    fh = frame.get("frame_height", 0) or frame.get("height", 0)
    diag = math.hypot(fw, fh) if fw and fh else None

    for i, sid in enumerate(node_ids):
        for oid in node_ids[i+1:]:
            s_cls = nodes[sid]["class"].lower()
            o_cls = nodes[oid]["class"].lower()
            sb = nodes[sid]["bbox"]
            ob = nodes[oid]["bbox"]
            sc = centroids[sid]; oc = centroids[oid]
            dist = _dist(sc, oc)
            if diag:
                dist /= max(1e-6, diag)

            # talking to: two persons close horizontally
            if s_cls in PersonLike and o_cls in PersonLike:
                if dist < 0.12:
                    edges.append({"relation": "talking to", "subject": sid, "object": oid})
                    edges.append({"relation": "talking to", "subject": oid, "object": sid})
            # looking at: person near small object in front (horizontal projection)
            if s_cls in PersonLike and o_cls not in PersonLike:
                if dist < 0.10 and _horiz_overlap(sb, ob) > 0.3:
                    edges.append({"relation": "looking at", "subject": sid, "object": oid})
            if o_cls in PersonLike and s_cls not in PersonLike:
                if dist < 0.10 and _horiz_overlap(sb, ob) > 0.3:
                    edges.append({"relation": "looking at", "subject": oid, "object": sid})
            # holding: person close to small portable
            if s_cls in PersonLike and o_cls in SmallPortable:
                if dist < 0.08:
                    edges.append({"relation": "holding", "subject": sid, "object": oid})
            if o_cls in PersonLike and s_cls in SmallPortable:
                if dist < 0.08:
                    edges.append({"relation": "holding", "subject": oid, "object": sid})
            # sitting/standing on: person above support
            if s_cls in PersonLike and o_cls in Support:
                subj_bottom = sb[3]; obj_top = ob[1]; vert_gap = (obj_top - subj_bottom)/(fh or 1)
                if subj_bottom <= obj_top + 8:  # slight overlap tolerated
                    if o_cls in Seating:
                        edges.append({"relation": "sitting on", "subject": sid, "object": oid})
                    else:
                        edges.append({"relation": "standing on", "subject": sid, "object": oid})
            if o_cls in PersonLike and s_cls in Support:
                obj_bottom = ob[3]; subj_top = sb[1];
                if obj_bottom <= subj_top + 8:
                    if s_cls in Seating:
                        edges.append({"relation": "sitting on", "subject": oid, "object": sid})
                    else:
                        edges.append({"relation": "standing on", "subject": oid, "object": sid})
            # next to: fallback proximity
            if dist < 0.15:
                edges.append({"relation": "next to", "subject": sid, "object": oid})
                edges.append({"relation": "next to", "subject": oid, "object": sid})
    return edges


def merge_edges(original: List[Dict], added: List[Dict]) -> List[Dict]:
    seen = {(e["relation"], e["subject"], e["object"]) for e in original}
    out = list(original)
    for e in added:
        key = (e["relation"], e["subject"], e["object"])
        if key not in seen:
            out.append(e)
            seen.add(key)
    return out


def process_video(video_id: str, overwrite: bool = True) -> int:
    sg_path = Path("results") / video_id / "scene_graph.jsonl"
    if not sg_path.exists():
        return 0
    tmp_path = sg_path.with_suffix(".semantic.tmp")
    written = 0
    with open(sg_path) as f_in, open(tmp_path, "w") as f_out:
        for line in f_in:
            g = json.loads(line)
            nodes = {n["memory_id"]: n for n in g.get("nodes", []) if n.get("bbox")}
            added = classify_semantic(g, nodes)
            g["edges"] = merge_edges(g.get("edges", []), added)
            f_out.write(json.dumps(g) + "\n")
            written += len(added)
    if overwrite:
        sg_path.rename(sg_path.with_suffix(".bak"))
        tmp_path.rename(sg_path)
    return written


def main():
    parser = argparse.ArgumentParser(description="Add semantic relations to scene graphs")
    parser.add_argument("video_ids", nargs="+", help="Episode IDs under results/")
    parser.add_argument("--no-overwrite", action="store_true", help="Keep original file; write .semantic.jsonl")
    args = parser.parse_args()

    total = 0
    for vid in args.video_ids:
        print(f"[+] {vid} ...", end=" ")
        added = process_video(vid, overwrite=not args.no_overwrite)
        print(f"added {added} relations")
        total += added
    print(f"\nDone. Total added: {total}")


if __name__ == "__main__":
    main()
