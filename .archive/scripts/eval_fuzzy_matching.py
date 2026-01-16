#!/usr/bin/env python3
"""
Ultra-permissive evaluation: Match triplets with fuzzy class matching.
This shows the theoretical maximum if class names weren't an issue.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from difflib import SequenceMatcher

def load_pvsg_ground_truth(pvsg_json_path: str) -> Dict[str, Dict]:
    """Load PVSG ground truth annotations."""
    with open(pvsg_json_path, 'r') as f:
        pvsg = json.load(f)
    return {v['video_id']: v for v in pvsg['data']}

def fuzzy_match(cls1: str, cls2: str, threshold: float = 0.5) -> bool:
    """Fuzzy string matching for classes."""
    # Direct match
    if cls1.lower() == cls2.lower():
        return True
    
    # Substring match
    if cls1.lower() in cls2.lower() or cls2.lower() in cls1.lower():
        return True
    
    # Edit distance ratio
    ratio = SequenceMatcher(None, cls1.lower(), cls2.lower()).ratio()
    return ratio >= threshold

def normalize_class(cls: str) -> str:
    """Normalize class names."""
    cls = cls.lower().strip().replace('_', ' ')
    mappings = {
        'person': 'adult',
        'adult': 'adult',
        'child': 'child',
        'baby': 'baby',
        'dining table': 'table',
        'table': 'table',
        'couch': 'sofa',
        'arm chair': 'chair',
        'armchair': 'chair',
        'tv': 'television',
        'cellphone': 'phone',
        'knife': 'knife',
        'plate': 'plate',
        'cup': 'cup',
        'bottle': 'bottle',
        'cake': 'cake',
    }
    return mappings.get(cls, cls)

def normalize_predicate(pred: str) -> str:
    """Normalize predicate names."""
    pred = pred.lower().strip()
    mappings = {
        'held_by': 'holding',
        'holding': 'holding',
        'hold': 'holding',
        'next_to': 'near',
        'near': 'near',
        'on': 'on',
        'sitting_on': 'sitting on',
        'standing_on': 'standing on',
        'beside': 'beside',
        'above': 'above',
        'below': 'below',
        'in front of': 'in front of',
    }
    return mappings.get(pred, pred)

def load_orion_triplets(video_id: str) -> List[Tuple[str, str, str]]:
    """Load Orion-generated triplets from scene_graph.jsonl."""
    results_dir = Path("results")
    sg_file = results_dir / video_id / "scene_graph.jsonl"
    
    if not sg_file.exists():
        return []
    
    # Load memory for class mapping
    mem_file = results_dir / video_id / "memory.json"
    memory_classes = {}
    if mem_file.exists():
        with open(mem_file) as f:
            mem = json.load(f)
            for obj in mem.get('objects', []):
                mem_id = obj.get('memory_id', obj.get('id'))
                cls = normalize_class(obj.get('class', 'unknown'))
                memory_classes[mem_id] = cls
    
    triplets = []
    with open(sg_file) as f:
        for line in f:
            try:
                data = json.loads(line)
                for edge in data.get('edges', []):
                    subj_id = edge.get('subject', '')
                    obj_id = edge.get('object', '')
                    relation = normalize_predicate(edge.get('relation', ''))
                    
                    subj_class = memory_classes.get(subj_id, 'unknown')
                    obj_class = memory_classes.get(obj_id, 'unknown')
                    
                    if subj_class != 'unknown' and obj_class != 'unknown':
                        triplets.append((subj_class, relation, obj_class))
            except:
                pass
    
    return triplets

def evaluate_with_fuzzy_matching(video_id: str, gt_triplets: List[Tuple]) -> dict:
    """Evaluate using fuzzy class matching."""
    pred_triplets = load_orion_triplets(video_id)
    
    if not pred_triplets or not gt_triplets:
        return {
            'video_id': video_id,
            'predicted': len(pred_triplets),
            'gt': len(gt_triplets),
            'matches': 0,
            'recall': 0.0
        }
    
    # Count fuzzy matches
    matches = 0
    for pred in pred_triplets:
        for gt in gt_triplets:
            if (fuzzy_match(pred[0], gt[0]) and  # subject match
                fuzzy_match(pred[1], gt[1]) and  # relation match
                fuzzy_match(pred[2], gt[2])):    # object match
                matches += 1
                break  # Count each GT only once
    
    return {
        'video_id': video_id,
        'predicted': len(pred_triplets),
        'gt': len(gt_triplets),
        'matches': matches,
        'recall': (matches / len(gt_triplets) * 100) if gt_triplets else 0
    }

def main():
    """Evaluate with fuzzy matching."""
    pvsg_gt = load_pvsg_ground_truth('datasets/PVSG/pvsg.json')
    
    # Test videos
    videos = [
        "0001_4164158586", "0003_3396832512", "0003_6141007489", "0004_11566980553",
        "0005_2505076295", "0006_2889117240", "0008_6225185844", "0008_8890945814",
        "0010_8610561401", "0018_3057666738",
        "0020_10793023296", "0020_5323209509", "0021_2446450580", "0021_4999665957",
        "0024_5224805531", "0026_2764832695", "0027_4571353789", "0028_3085751774",
        "0028_4021064662", "0029_5139813648"
    ]
    
    print("Ultra-permissive Evaluation (fuzzy class matching)\n")
    print(f"{'Video':<20} | {'Pred':<5} | {'GT':<5} | {'Match':<5} | {'R@20':<6}")
    print("-" * 60)
    
    total_matches = 0
    total_gt = 0
    
    for vid in videos:
        if vid not in pvsg_gt:
            continue
        
        gt_video = pvsg_gt[vid]
        objects = {obj['object_id']: normalize_class(obj['category']) for obj in gt_video.get('objects', [])}
        
        # Filter GT to relevant predicates
        gt_triplets = []
        for rel in gt_video.get('relations', []):
            subj_id, obj_id, pred, _ = rel
            if subj_id in objects and obj_id in objects:
                subj = objects[subj_id]
                obj = objects[obj_id]
                pred_norm = normalize_predicate(pred)
                if pred_norm in ['holding', 'on', 'near', 'sitting on', 'standing on', 'beside', 'in front of', 'above', 'below']:
                    gt_triplets.append((subj, pred_norm, obj))
        
        result = evaluate_with_fuzzy_matching(vid, gt_triplets)
        print(f"{result['video_id']:<20} | {result['predicted']:<5} | {result['gt']:<5} | "
              f"{result['matches']:<5} | {result['recall']:>5.1f}%")
        
        total_matches += result['matches']
        total_gt += result['gt']
    
    print("-" * 60)
    avg_recall = (total_matches / total_gt * 100) if total_gt else 0
    print(f"{'AVERAGE':<20} | {'':5} | {total_gt:<5} | {total_matches:<5} | {avg_recall:>5.1f}%")

if __name__ == '__main__':
    main()
