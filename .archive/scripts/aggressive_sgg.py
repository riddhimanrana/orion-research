#!/usr/bin/env python3
"""
Aggressive scene graph generator - lowers thresholds to maximize triplet generation.
Trades some precision for recall.
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple

def _iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aw, ah = max(0.0, ax2 - ax1), max(0.0, ay2 - ay1)
    bw, bh = max(0.0, bx2 - bx1), max(0.0, by2 - by1)
    ua = aw * ah + bw * bh - inter + 1e-8
    return float(inter / ua)

def _centroid(b: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def _frame_diag(wh: Tuple[int, int]) -> float:
    w, h = wh
    if not w or not h:
        return 1.0
    return float(math.hypot(w, h))

def _horiz_overlap_ratio(a: List[float], b: List[float]) -> float:
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    left = max(ax1, bx1)
    right = min(ax2, bx2)
    ov = max(0.0, right - left)
    aw = max(1e-8, ax2 - ax1)
    bw = max(1e-8, bx2 - bx1)
    return float(ov / min(aw, bw))

def generate_aggressive_relations(
    memory: Dict[str, Any],
    tracks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Generate scene graphs with AGGRESSIVE thresholds (low precision, high recall).
    """
    mem_to_class: Dict[str, str] = {}
    for obj in memory.get("objects", []):
        mem_to_class[obj["memory_id"]] = obj.get("class", "object")

    # Group tracks by frame
    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for r in tracks:
        category = r.get("category", r.get("class_name", "object"))
        # Find memory object with matching class
        mem_id = None
        for mem_obj in memory.get("objects", []):
            if mem_obj.get("class", "").lower() == category.lower():
                mem_id = mem_obj["memory_id"]
                break
        
        if not mem_id:
            continue
        
        bbox = r.get("bbox") or r.get("bbox_2d")
        item = {
            "memory_id": mem_id,
            "class": mem_to_class.get(mem_id, category),
            "bbox": bbox,
            "frame_width": int(r.get("frame_width", 0)),
            "frame_height": int(r.get("frame_height", 0)),
        }
        by_frame.setdefault(int(r.get("frame_id", 0)), []).append(item)

    graphs: List[Dict[str, Any]] = []
    for frame_id in sorted(by_frame.keys()):
        obs = by_frame[frame_id]
        edges = []
        n = len(obs)

        for i in range(n):
            A = obs[i]
            for j in range(n):
                if i == j:
                    continue
                B = obs[j]
                if not A.get("bbox") or not B.get("bbox"):
                    continue
                
                a_b = A["bbox"]
                b_b = B["bbox"]
                wh = (A.get("frame_width", 0), A.get("frame_height", 0))
                diag = _frame_diag(wh)
                iou_val = _iou(a_b, b_b)

                cxA, cyA = _centroid(a_b)
                cxB, cyB = _centroid(b_b)
                dist = math.hypot(cxA - cxB, cyA - cyB) / max(1e-6, diag)
                bottomA = a_b[3]
                topB = b_b[1]
                topA = a_b[1]
                bottomB = b_b[3]
                h_ov = _horiz_overlap_ratio(a_b, b_b)

                ax1, ay1, ax2, ay2 = a_b
                bx1, by1, bx2, by2 = b_b
                aw = max(1e-8, ax2 - ax1)
                ah = max(1e-8, ay2 - ay1)
                bw = max(1e-8, bx2 - bx1)
                bh = max(1e-8, by2 - by1)
                vgap = abs(topB - bottomA) / max(1.0, wh[1])

                # AGGRESSIVE THRESHOLDS (high recall, low precision)
                # near: very loose distance threshold
                if dist <= 0.25:  # was 0.08
                    edges.append({
                        "relation": "near",
                        "subject": A["memory_id"],
                        "object": B["memory_id"],
                    })

                # on: very loose spatial threshold
                if h_ov >= 0.1 and vgap <= 0.1 and cyA <= cyB:  # was h_ov >= 0.3, vgap <= 0.02
                    edges.append({
                        "relation": "on",
                        "subject": A["memory_id"],
                        "object": B["memory_id"],
                    })

                # held_by: loose IoU threshold
                if A["class"] != "person" and B["class"] == "person":
                    if iou_val >= 0.1:  # was 0.3
                        edges.append({
                            "relation": "held_by",
                            "subject": A["memory_id"],
                            "object": B["memory_id"],
                        })

                # beside: looser distance
                if dist <= 0.25 and h_ov < 0.3 and iou_val < 0.1:
                    edges.append({
                        "relation": "beside",
                        "subject": A["memory_id"],
                        "object": B["memory_id"],
                    })

                # above/below: looser gap thresholds
                cyA_in_B = (cxA >= bx1) and (cxA <= bx2)
                if cyA_in_B:
                    above_gap = (topB - bottomA) / max(1.0, wh[1]) if wh[1] else 1.0
                    below_gap = (topA - bottomB) / max(1.0, wh[1]) if wh[1] else 1.0
                    
                    if -0.1 <= above_gap <= 0.5:  # was 0.02-0.3
                        edges.append({
                            "relation": "above",
                            "subject": A["memory_id"],
                            "object": B["memory_id"],
                        })
                    
                    if -0.1 <= below_gap <= 0.5:  # was 0.02-0.3
                        edges.append({
                            "relation": "below",
                            "subject": A["memory_id"],
                            "object": B["memory_id"],
                        })

        graphs.append({
            "frame": frame_id,
            "nodes": [{"memory_id": o["memory_id"], "class": o["class"], "bbox": o["bbox"]} for o in obs],
            "edges": edges,
        })

    return graphs


def main():
    # Test on first few videos
    videos = ["0001_4164158586", "0003_3396832512"]
    results_dir = Path("results")

    for vid in videos:
        memory_file = results_dir / vid / "memory.json"
        tracks_file = results_dir / vid / "tracks.jsonl"
        
        if not memory_file.exists() or not tracks_file.exists():
            print(f"⚠️  {vid}: missing files")
            continue
        
        # Load data
        with open(memory_file) as f:
            memory = json.load(f)
        
        tracks = []
        with open(tracks_file) as f:
            for line in f:
                tracks.append(json.loads(line))
        
        # Generate aggressive graphs
        graphs = generate_aggressive_relations(memory, tracks)
        
        # Count triplets
        triplet_count = sum(len(g.get("edges", [])) for g in graphs)
        
        print(f"{vid}: {triplet_count} relations ({len(graphs)} frames)")


if __name__ == "__main__":
    main()
