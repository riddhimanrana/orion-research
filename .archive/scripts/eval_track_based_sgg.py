#!/usr/bin/env python3
"""
Direct track-based SGG evaluation: Build triplets from tracks directly, skip memory clustering.
This bypasses the memory bottleneck and builds relations at native track granularity.
"""

import json
from pathlib import Path
from typing import List, Set, Tuple, Dict, Optional
from collections import defaultdict
import math

def normalize_class(cls: str) -> str:
    """Normalize class names."""
    cls = cls.lower().strip()
    mappings = {
        'person': 'adult',
        'adult': 'adult',
        'child': 'child',
        'baby': 'baby',
        'dining table': 'table',
        'table': 'table',
        'coffee table': 'table',
        'couch': 'sofa',
        'chair': 'chair',
        'tv': 'television',
        'cellphone': 'phone',
        'knife': 'knife',
        'plate': 'plate',
        'cup': 'cup',
        'fridge': 'refrigerator',
        'refrigerator': 'refrigerator',
        'door': 'door',
        'cabinet': 'cabinet',
        'cake': 'cake',
        'microwave': 'microwave',
        'sink': 'sink',
    }
    return mappings.get(cls, cls)

def normalize_predicate(pred: str) -> str:
    """Normalize predicate names."""
    pred = pred.lower().strip()
    mappings = {
        'on': 'on',
        'near': 'near',
        'held_by': 'holding',
        'holding': 'holding',
    }
    return mappings.get(pred, pred)

def _centroid(b: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def _frame_diag(wh: Tuple[int, int]) -> float:
    if wh[0] == 0 or wh[1] == 0:
        return 1.0
    return math.sqrt(wh[0] ** 2 + wh[1] ** 2)

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

def build_track_triplets(tracks: List[Dict]) -> List[Tuple[str, str, str]]:
    """Build triplets directly from tracks using spatial heuristics."""
    # Group tracks by frame
    by_frame = defaultdict(list)
    for t in tracks:
        frame_id = int(t.get("frame_id", 0))
        bbox = t.get("bbox") or t.get("bbox_2d")
        if not bbox:
            continue
        by_frame[frame_id].append({
            "class": normalize_class(t.get("category", "unknown")),
            "bbox": bbox,
            "frame_width": int(t.get("frame_width", 800)),
            "frame_height": int(t.get("frame_height", 600)),
        })
    
    # Build triplets from all frame pairs
    triplets = set()  # Use set to avoid duplicates
    
    for frame_id in by_frame:
        obs = by_frame[frame_id]
        
        for i in range(len(obs)):
            for j in range(len(obs)):
                if i == j:
                    continue
                
                A, B = obs[i], obs[j]
                wh = (A["frame_width"], A["frame_height"])
                diag = _frame_diag(wh)
                
                cxA, cyA = _centroid(A["bbox"])
                cxB, cyB = _centroid(B["bbox"])
                dist = math.hypot(cxA - cxB, cyA - cyB) / max(1e-6, diag)
                iou_val = _iou(A["bbox"], B["bbox"])
                
                # Near relation
                if dist <= 0.1 and iou_val < 0.1:
                    triplets.add((A["class"], "near", B["class"]))
                
                # On relation: A on B (higher object on lower)
                bottomA = A["bbox"][3]
                topB = B["bbox"][1]
                vgap = abs(topB - bottomA) / max(1.0, wh[1])
                h_ov = max(0.0, min(A["bbox"][2], B["bbox"][2]) - max(A["bbox"][0], B["bbox"][0])) / max(A["bbox"][2] - A["bbox"][0], 1e-8)
                
                if h_ov >= 0.2 and vgap <= 0.05 and bottomA <= topB:
                    triplets.add((A["class"], "on", B["class"]))
                
                # Holding relation: A held by B (person holds objects)
                if B["class"] in ["adult", "person"] and A["class"] != "adult":
                    if iou_val >= 0.2:
                        triplets.add((A["class"], "holding", B["class"]))
    
    return list(triplets)

def load_pvsg_ground_truth(pvsg_json_path: str) -> Dict[str, Dict]:
    """Load PVSG ground truth."""
    with open(pvsg_json_path) as f:
        pvsg = json.load(f)
    return {v['video_id']: v for v in pvsg['data']}

def evaluate_video(video_id: str, gt_video: Dict) -> Dict:
    """Evaluate a single video."""
    results_dir = Path("results")
    tracks_file = results_dir / video_id / "tracks.jsonl"
    
    if not tracks_file.exists():
        return None
    
    # Load tracks
    tracks = []
    with open(tracks_file) as f:
        for line in f:
            tracks.append(json.loads(line))
    
    # Build track-based triplets
    pred_triplets = build_track_triplets(tracks)
    
    # Load GT triplets
    objects = {obj['object_id']: normalize_class(obj['category']) for obj in gt_video.get('objects', [])}
    gt_triplets = set()
    for rel in gt_video.get('relations', []):
        subj_id, obj_id, pred, _ = rel
        if subj_id in objects and obj_id in objects:
            subj = objects[subj_id]
            obj = objects[obj_id]
            pred_norm = normalize_predicate(pred)
            if pred_norm in ['holding', 'on', 'near']:
                gt_triplets.add((subj, pred_norm, obj))
    
    # Count matches
    matches = len(set(pred_triplets) & gt_triplets)
    
    return {
        'video_id': video_id,
        'predicted': len(pred_triplets),
        'gt': len(gt_triplets),
        'matches': matches,
        'recall': (matches / len(gt_triplets) * 100) if gt_triplets else 0
    }

def main():
    """Evaluate all videos with track-based triplets."""
    pvsg_gt = load_pvsg_ground_truth('datasets/PVSG/pvsg.json')
    
    videos = [
        "0001_4164158586", "0003_3396832512", "0003_6141007489", "0004_11566980553",
        "0005_2505076295", "0006_2889117240", "0008_6225185844", "0008_8890945814",
        "0010_8610561401", "0018_3057666738",
        "0020_10793023296", "0020_5323209509", "0021_2446450580", "0021_4999665957",
        "0024_5224805531", "0026_2764832695", "0027_4571353789", "0028_3085751774",
        "0028_4021064662", "0029_5139813648"
    ]
    
    print("Track-Based SGG Evaluation (Direct Triplets)\n")
    print(f"{'Video':<20} | {'Pred':<5} | {'GT':<5} | {'Match':<5} | {'R@20':<6}")
    print("-" * 65)
    
    total_matches = 0
    total_gt = 0
    
    for vid in videos:
        if vid not in pvsg_gt:
            continue
        
        result = evaluate_video(vid, pvsg_gt[vid])
        if result:
            print(f"{result['video_id']:<20} | {result['predicted']:<5} | {result['gt']:<5} | "
                  f"{result['matches']:<5} | {result['recall']:>5.1f}%")
            total_matches += result['matches']
            total_gt += result['gt']
    
    print("-" * 65)
    avg_recall = (total_matches / total_gt * 100) if total_gt else 0
    print(f"{'AVERAGE':<20} | {'':5} | {total_gt:<5} | {total_matches:<5} | {avg_recall:>5.1f}%")

if __name__ == '__main__':
    main()
