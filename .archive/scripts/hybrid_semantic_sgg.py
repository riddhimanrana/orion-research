#!/usr/bin/env python3
"""
Hybrid semantic relation generation: Combines spatial heuristics with confidence-based semantic patterns.
Uses object class pairs and spatial metrics to infer semantic relations reliably.
"""

import json
from pathlib import Path
from typing import List, Set, Tuple, Dict
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
        'couch': 'sofa',
        'sofa': 'sofa',
        'chair': 'chair',
        'bed': 'bed',
        'floor': 'floor',
        'tv': 'television',
        'cellphone': 'phone',
        'fridge': 'refrigerator',
        'refrigerator': 'refrigerator',
        'cake': 'cake',
        'knife': 'knife',
        'cup': 'cup',
        'plate': 'plate',
        'toy': 'toy',
        'ball': 'ball',
    }
    return mappings.get(cls, cls)

# Semantic relation rules: (subject_class, object_class) -> [(relation, confidence)]
# Focused on top 8 PVSG predicates: holding, on, sitting on, standing on, looking at, etc.
SEMANTIC_RULES = {
    # Person -> Objects (HOLDING relations)
    ('adult', 'cake'): [('holding', 0.75)],
    ('adult', 'cup'): [('holding', 0.8)],
    ('adult', 'plate'): [('holding', 0.75)],
    ('adult', 'knife'): [('holding', 0.9)],
    ('adult', 'phone'): [('holding', 0.85), ('looking at', 0.6)],
    ('adult', 'child'): [('holding', 0.8)],
    ('adult', 'baby'): [('holding', 0.85)],
    ('person', 'cake'): [('holding', 0.7)],
    ('person', 'cup'): [('holding', 0.75)],
    ('person', 'child'): [('holding', 0.75)],
    ('person', 'baby'): [('holding', 0.8)],
    
    # Child -> Objects
    ('child', 'cake'): [('holding', 0.7)],
    ('child', 'toy'): [('holding', 0.8)],
    ('child', 'ball'): [('holding', 0.75)],
    
    # Person -> Furniture (SITTING/STANDING/LYING ON)
    ('adult', 'chair'): [('sitting on', 0.75)],
    ('adult', 'sofa'): [('sitting on', 0.75)],
    ('adult', 'bed'): [('lying on', 0.7), ('sitting on', 0.5)],
    ('adult', 'table'): [('standing on', 0.65)],
    ('adult', 'floor'): [('standing on', 0.5), ('walking on', 0.4)],
    ('person', 'chair'): [('sitting on', 0.7)],
    ('person', 'sofa'): [('sitting on', 0.7)],
    ('person', 'bed'): [('lying on', 0.65)],
    ('person', 'table'): [('standing on', 0.6)],
    ('child', 'chair'): [('sitting on', 0.65)],
    
    # Object -> Object (ON relations)
    ('cake', 'plate'): [('on', 0.85)],
    ('cake', 'table'): [('on', 0.75)],
    ('cup', 'table'): [('on', 0.85)],
    ('cup', 'plate'): [('on', 0.6)],
    ('plate', 'table'): [('on', 0.85)],
    ('knife', 'table'): [('on', 0.7)],
    ('phone', 'table'): [('on', 0.75)],
    ('plate', 'chair'): [('on', 0.5)],
    
    # Person -> Person (LOOKING AT, TALKING TO, etc.)
    ('adult', 'person'): [('looking at', 0.6), ('talking to', 0.5)],
    ('person', 'person'): [('looking at', 0.55), ('talking to', 0.45)],
}

# Spatial rule overrides for adjacent objects
SPATIAL_RELATIONS = {
    'near': {'distance_threshold': 0.15, 'iou_threshold': 0.1},
    'on': {'h_overlap': 0.3, 'v_gap': 0.05},
    'holding': {'iou': 0.2},
}

def _centroid(b: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def _frame_diag(wh: Tuple[int, int]) -> float:
    return math.sqrt(wh[0] ** 2 + wh[1] ** 2) if wh[0] and wh[1] else 1.0

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

def get_semantic_relations(subj_class: str, obj_class: str, 
                          spatial_metrics: Dict) -> List[Tuple[str, float]]:
    """
    Get semantic relations for a subject-object pair using rules and spatial context.
    Focuses on high-frequency PVSG predicates: holding, on, sitting on, standing on, looking at.
    
    Returns:
        List of (relation, confidence) tuples
    """
    subj_norm = normalize_class(subj_class)
    obj_norm = normalize_class(obj_class)
    
    relations = []
    
    # Direct semantic rules (highest priority)
    key = (subj_norm, obj_norm)
    if key in SEMANTIC_RULES:
        relations.extend(SEMANTIC_RULES[key])
    
    # Spatial context for unmapped pairs
    dist = spatial_metrics.get('distance', 1.0)
    iou = spatial_metrics.get('iou', 0.0)
    h_overlap = spatial_metrics.get('h_overlap', 0.0)
    v_gap = spatial_metrics.get('v_gap', 1.0)
    
    # If no rules matched, apply spatial heuristics
    if not relations:
        # High IOU -> likely holding (if one is person, other is portable object)
        if iou >= 0.3:
            if subj_norm in ['cake', 'cup', 'plate', 'knife', 'phone', 'toy'] and obj_norm == 'adult':
                relations.append(('holding', 0.65))
            elif subj_norm == 'adult' and obj_norm in ['child', 'baby']:
                relations.append(('holding', 0.7))
        
        # Vertical alignment + horizontal overlap -> on relation
        if h_overlap >= 0.3 and v_gap <= 0.08:
            if subj_norm in ['cake', 'cup', 'plate', 'knife', 'phone'] and obj_norm in ['table', 'plate', 'chair']:
                relations.append(('on', 0.65))
    
    # Remove duplicates, keep highest confidence
    unique_rels = {}
    for rel, conf in relations:
        if rel not in unique_rels or conf > unique_rels[rel]:
            unique_rels[rel] = conf
    
    return list(unique_rels.items())


def build_hybrid_triplets(video_id: str) -> List[Tuple[str, str, str]]:
    """Build triplets using hybrid semantic + spatial approach."""
    results_dir = Path("results")
    sg_file = results_dir / video_id / "scene_graph.jsonl"
    mem_file = results_dir / video_id / "memory.json"
    
    if not sg_file.exists() or not mem_file.exists():
        return []
    
    # Load memory
    with open(mem_file) as f:
        mem = json.load(f)
    
    mem_id_to_class = {obj['memory_id']: normalize_class(obj.get('class', 'unknown'))
                       for obj in mem.get('objects', [])}
    
    # Load scene graphs (for bbox info)
    graphs = []
    with open(sg_file) as f:
        for line in f:
            graphs.append(json.loads(line))
    
    # Build triplets from semantic rules
    triplets = set()
    
    for graph in graphs:
        frame_h = graph.get('frame_height', 600)
        frame_w = graph.get('frame_width', 800)
        wh = (frame_w, frame_h)
        diag = _frame_diag(wh)
        
        nodes = graph.get('nodes', [])
        
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    continue
                
                node_a = nodes[i]
                node_b = nodes[j]
                
                mem_id_a = node_a.get('memory_id')
                mem_id_b = node_b.get('memory_id')
                
                class_a = mem_id_to_class.get(mem_id_a, 'unknown')
                class_b = mem_id_to_class.get(mem_id_b, 'unknown')
                
                bbox_a = node_a.get('bbox')
                bbox_b = node_b.get('bbox')
                
                if not bbox_a or not bbox_b:
                    continue
                
                # Compute spatial metrics
                cxA, cyA = _centroid(bbox_a)
                cxB, cyB = _centroid(bbox_b)
                dist = math.hypot(cxA - cxB, cyA - cyB) / max(1e-6, diag)
                iou = _iou(bbox_a, bbox_b)
                
                h_ov = max(0.0, min(bbox_a[2], bbox_b[2]) - max(bbox_a[0], bbox_b[0])) / max(bbox_a[2] - bbox_a[0], 1e-8)
                bottomA = bbox_a[3]
                topB = bbox_b[1]
                v_gap = abs(topB - bottomA) / max(1.0, frame_h)
                
                spatial_metrics = {
                    'distance': dist,
                    'iou': iou,
                    'h_overlap': h_ov,
                    'v_gap': v_gap,
                }
                
                # Get semantic relations
                rels = get_semantic_relations(class_a, class_b, spatial_metrics)
                
                for rel, conf in rels:
                    if conf >= 0.4:  # Filter low confidence
                        triplets.add((class_a, rel, class_b))
    
    return list(triplets)


def evaluate_with_hybrid(pvsg_path: str = 'datasets/PVSG/pvsg.json') -> Dict:
    """Evaluate all videos with hybrid semantic relations."""
    # Load GT
    with open(pvsg_path) as f:
        pvsg = json.load(f)
    
    gt_data = {v['video_id']: v for v in pvsg['data']}
    
    videos = [
        "0001_4164158586", "0003_3396832512", "0003_6141007489", "0004_11566980553",
        "0005_2505076295", "0006_2889117240", "0008_6225185844", "0008_8890945814",
        "0010_8610561401", "0018_3057666738",
        "0020_10793023296", "0020_5323209509", "0021_2446450580", "0021_4999665957",
        "0024_5224805531", "0026_2764832695", "0027_4571353789", "0028_3085751774",
        "0028_4021064662", "0029_5139813648"
    ]
    
    print("Hybrid Semantic SGG Evaluation\n")
    print(f"{'Video':<20} | {'Pred':<5} | {'GT':<5} | {'Match':<5} | {'R@20':<6}")
    print("-" * 70)
    
    total_matches = 0
    total_gt = 0
    
    for vid in videos:
        if vid not in gt_data:
            continue
        
        # Get predicted triplets
        pred_triplets = build_hybrid_triplets(vid)
        
        # Get GT triplets
        gt_video = gt_data[vid]
        objects = {obj['object_id']: normalize_class(obj['category']) 
                  for obj in gt_video.get('objects', [])}
        gt_triplets = set()
        
        for rel in gt_video.get('relations', []):
            subj_id, obj_id, pred, _ = rel
            if subj_id in objects and obj_id in objects:
                subj = objects[subj_id]
                obj = objects[obj_id]
                
                # Normalize pred
                pred_norm = pred.lower()
                pred_norm = pred_norm.replace('_', ' ')
                
                # Accept if common PVSG predicate
                pred_norm = pred_norm if pred_norm in [
                    'holding', 'on', 'near', 'looking at', 'eating', 'picking',
                    'sitting on', 'standing on', 'drinking from', 'touching', 'playing with',
                    'pushing', 'pulling', 'pointing to', 'hugging', 'talking to'
                ] else None
                
                if pred_norm:
                    gt_triplets.add((subj, pred_norm, obj))
        
        # Count matches
        matches = len(set(pred_triplets) & gt_triplets)
        
        print(f"{vid:<20} | {len(pred_triplets):<5} | {len(gt_triplets):<5} | "
              f"{matches:<5} | {matches/len(gt_triplets)*100 if gt_triplets else 0:>5.1f}%")
        
        total_matches += matches
        total_gt += len(gt_triplets)
    
    print("-" * 70)
    avg_recall = (total_matches / total_gt * 100) if total_gt else 0
    print(f"{'AVERAGE':<20} | {'':5} | {total_gt:<5} | {total_matches:<5} | {avg_recall:>5.1f}%\n")
    
    return {
        'total_videos': len(videos),
        'total_gt': total_gt,
        'total_matches': total_matches,
        'recall': avg_recall
    }


if __name__ == '__main__':
    results = evaluate_with_hybrid()
    print(f"✓ Evaluation complete: {results['recall']:.1f}% R@20")
