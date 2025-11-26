#!/usr/bin/env python3
"""
Visualize scene graph samples with detailed relation metrics for validation.

Usage:
  python scripts/visualize_scene_graph.py --results results/video_validation --num-samples 5
"""

import argparse
import json
import math
from pathlib import Path


def _centroid(b):
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _iou(a, b):
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


def _horiz_overlap_ratio(a, b):
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    left = max(ax1, bx1)
    right = min(ax2, bx2)
    ov = max(0.0, right - left)
    aw = max(1e-8, ax2 - ax1)
    bw = max(1e-8, bx2 - bx1)
    return float(ov / min(aw, bw))


def visualize_graph_sample(graph, frame_wh):
    frame = graph.get("frame", 0)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    print(f"\n{'='*70}")
    print(f"Frame {frame}: {len(nodes)} nodes, {len(edges)} edges")
    print(f"{'='*70}")

    node_map = {n["memory_id"]: n for n in nodes}

    for n in nodes:
        print(f"  Node {n['memory_id']:10s} ({n['class']:10s})  bbox={n['bbox']}")

    if not edges:
        print("  (No edges)")
        return

    diag = math.hypot(frame_wh[0], frame_wh[1]) if frame_wh[0] and frame_wh[1] else 1.0

    for e in edges:
        rel = e.get("relation", "?")
        subj_id = e.get("subject", "?")
        obj_id = e.get("object", "?")

        subj = node_map.get(subj_id)
        obj = node_map.get(obj_id)
        if not subj or not obj:
            continue

        s_b = subj.get("bbox")
        o_b = obj.get("bbox")
        if not s_b or not o_b:
            continue

        # Compute metrics
        iou = _iou(s_b, o_b)
        cxS, cyS = _centroid(s_b)
        cxO, cyO = _centroid(o_b)
        dist = math.hypot(cxS - cxO, cyS - cyO)
        norm_dist = dist / diag if diag > 0 else 0.0
        h_ov = _horiz_overlap_ratio(s_b, o_b)
        vgap = abs(o_b[1] - s_b[3]) / max(1.0, frame_wh[1])

        print(f"  Edge: {subj_id:10s} --[{rel:8s}]--> {obj_id:10s}")
        print(f"        IoU={iou:.3f}, norm_dist={norm_dist:.3f}, h_overlap={h_ov:.3f}, vgap={vgap:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=str, required=True, help='Results dir')
    ap.add_argument('--num-samples', type=int, default=5, help='Number of sample frames to show')
    args = ap.parse_args()

    results_dir = Path(args.results)
    graph_path = results_dir / 'scene_graph.jsonl'
    summary_path = results_dir / 'graph_summary.json'

    if not graph_path.exists():
        print(f"ERROR: {graph_path} not found. Run run_scene_graph.py first.")
        return

    # Load summary
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        print(f"\nScene Graph Summary for {results_dir.name}:")
        print(f"  Total frames: {summary.get('total_frames', 0)}")
        print(f"  Avg nodes/frame: {summary.get('avg_nodes_per_frame', 0)}")
        print(f"  Avg edges/frame: {summary.get('avg_edges_per_frame', 0)}")
        print(f"  Relation counts: {summary.get('relation_counts', {})}")

    # Load graphs
    graphs = []
    with open(graph_path, 'r') as f:
        for line in f:
            if line.strip():
                graphs.append(json.loads(line))

    # Filter graphs with edges
    graphs_with_edges = [g for g in graphs if len(g.get("edges", [])) > 0]

    if not graphs_with_edges:
        print("\n(No frames with edges found)")
        return

    # Sample evenly
    step = max(1, len(graphs_with_edges) // args.num_samples)
    samples = graphs_with_edges[::step][:args.num_samples]

    # Infer frame dimensions from first node
    frame_wh = (1920, 1080)  # default
    for g in graphs:
        for n in g.get("nodes", []):
            if n.get("bbox"):
                # Approximate from bbox scale (heuristic)
                bbox = n["bbox"]
                if bbox[2] > 100 or bbox[3] > 100:
                    frame_wh = (1920, 1080)
                    break

    print(f"\nShowing {len(samples)} sample frames with edges (from {len(graphs_with_edges)} total):")
    for g in samples:
        visualize_graph_sample(g, frame_wh)


if __name__ == '__main__':
    main()
