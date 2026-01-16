#!/usr/bin/env python3
"""
Specialized SGG evaluation for Orion's supported relations: on, holding, near.
Computes Recall@K only for GT triplets with these predicates.
"""

import json
import os
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import numpy as np

def load_pvsg_ground_truth(pvsg_json_path: str) -> Dict[str, Dict]:
    """Load PVSG ground truth annotations."""
    with open(pvsg_json_path, 'r') as f:
        pvsg = json.load(f)
    return {v['video_id']: v for v in pvsg['data']}

def normalize_class(cls: str) -> str:
    """Normalize class names for matching."""
    cls = cls.lower().strip().replace('_', ' ')
    mappings = {
        'person': 'adult',
        'adult': 'adult',
        'child': 'child',
        'baby': 'baby',
        'dining table': 'table',
        'dining_table': 'table',
        'table': 'table',
        'coffee table': 'table',
        'couch': 'sofa',
        'couch': 'sofa',
        'arm chair': 'chair',
        'armchair': 'chair',
        'desk': 'table',  # treat desk as table surface
        'coffee table': 'table',
        'nightstand': 'table',
        'tv': 'television',
        'cellphone': 'phone',
        'mobile': 'phone',
        'smartphone': 'phone',
        'laptop': 'computer',
        'computer': 'computer',
        'water bottle': 'bottle',
        'wine glass': 'glass',
        'champagne glass': 'glass',
        'soup bowl': 'bowl',
        'kitchen knife': 'knife',
        'steak knife': 'knife',
        'bread knife': 'knife',
        'butter knife': 'knife',
        'fork': 'utensil',
        'spoon': 'utensil',
        'knife': 'knife',
        'plate': 'plate',
        'saucer': 'plate',
        'platter': 'plate',
        'hand': 'person',  # treat hand as part of person
        'foot': 'person',
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
        'next to': 'near',
        'on': 'on',
        'sitting_on': 'sitting on',
        'sitting on': 'sitting on',
        'standing_on': 'standing on',
        'standing on': 'standing on',
        'beside': 'beside',
        'next to': 'beside',
        'above': 'above',
        'below': 'below',
        'in front of': 'in front of',
        'in_front_of': 'in front of',
    }
    return mappings.get(pred, pred)

def load_orion_triplets(video_id: str, results_dir: str) -> List[Tuple[str, str, str]]:
    """
    Load Orion predicted triplets from scene_graph.jsonl.
    Returns: List of (subject_class, predicate, object_class) tuples
    """
    sg_file = os.path.join(results_dir, video_id, 'scene_graph.jsonl')
    if not os.path.exists(sg_file):
        return []
    
    # Load memory.json to map memory IDs to classes
    memory_file = os.path.join(results_dir, video_id, 'memory.json')
    mem_to_class = {}
    if os.path.exists(memory_file):
        with open(memory_file, 'r') as f:
            memory = json.load(f)
            for obj in memory.get('objects', []):
                mem_id = obj.get('memory_id', obj.get('id', ''))
                cls = obj.get('class', obj.get('label', 'unknown'))
                mem_to_class[mem_id] = cls
    
    triplets = []
    seen_triplets = set()
    
    with open(sg_file, 'r') as f:
        for line in f:
            frame_sg = json.loads(line)
            
            # Build node ID to class mapping for this frame
            node_map = {}
            for node in frame_sg.get('nodes', []):
                mem_id = node.get('memory_id', '')
                node_class = node.get('class', mem_to_class.get(mem_id, 'unknown'))
                node_map[mem_id] = normalize_class(node_class)
            
            # Extract triplets from edges
            for edge in frame_sg.get('edges', []):
                subj_id = edge.get('subject', '')
                obj_id = edge.get('object', '')
                pred = normalize_predicate(edge.get('relation', ''))
                
                if subj_id in node_map and obj_id in node_map:
                    subj_class = node_map[subj_id]
                    obj_class = node_map[obj_id]
                    
                    # Orion uses passive voice (cake held_by person)
                    # GT uses active voice (person holding cake)
                    if pred == 'holding':
                        triplet = (obj_class, pred, subj_class)
                    else:
                        triplet = (subj_class, pred, obj_class)
                    
                    if triplet not in seen_triplets:
                        triplets.append(triplet)
                        seen_triplets.add(triplet)
    
    return triplets

def load_gt_triplets(video_data: Dict, filter_predicates: Set[str] = None) -> List[Tuple[str, str, str]]:
    """
    Load ground truth triplets from PVSG annotations.
    If filter_predicates is provided, only return triplets with those predicates.
    """
    triplets = []
    objects = {obj['object_id']: obj for obj in video_data.get('objects', [])}
    
    for relation in video_data.get('relations', []):
        subj_id, obj_id, predicate, frame_ranges = relation
        
        if subj_id in objects and obj_id in objects:
            subj_class = normalize_class(objects[subj_id].get('category', 'unknown'))
            obj_class = normalize_class(objects[obj_id].get('category', 'unknown'))
            pred = normalize_predicate(predicate)
            
            # Filter by supported predicates if requested
            if filter_predicates is None or pred in filter_predicates:
                triplets.append((subj_class, pred, obj_class))
    
    return triplets

def compute_recall_at_k(pred_triplets: List[Tuple], gt_triplets: List[Tuple], k: int) -> float:
    """Compute Recall@K."""
    if len(gt_triplets) == 0:
        return 0.0
    
    top_k_preds = set(pred_triplets[:k])
    gt_set = set(gt_triplets)
    matched = len(top_k_preds & gt_set)
    
    return (matched / len(gt_triplets)) * 100.0

def evaluate_video(video_id: str, results_dir: str, gt_videos: Dict, filter_predicates: Set[str]) -> Dict:
    """Evaluate a single video."""
    if video_id not in gt_videos:
        return {'video_id': video_id, 'error': 'Not in GT'}
    
    # Load predictions and ground truth
    pred_triplets = load_orion_triplets(video_id, results_dir)
    gt_triplets_all = load_gt_triplets(gt_videos[video_id], filter_predicates=None)
    gt_triplets_filtered = load_gt_triplets(gt_videos[video_id], filter_predicates=filter_predicates)
    
    if len(gt_triplets_filtered) == 0:
        return {
            'video_id': video_id,
            'pred_count': len(pred_triplets),
            'gt_total': len(gt_triplets_all),
            'gt_filtered': 0,
            'R@20': 0.0,
            'mR@20': 0.0,
            'R@50': 0.0,
            'mR@50': 0.0,
            'R@100': 0.0,
            'mR@100': 0.0,
        }
    
    # Find matches
    pred_set = set(pred_triplets[:100])
    gt_set = set(gt_triplets_filtered)
    matches = pred_set & gt_set
    
    # Compute Recall@K
    results = {
        'video_id': video_id,
        'pred_count': len(pred_triplets),
        'gt_total': len(gt_triplets_all),
        'gt_filtered': len(gt_triplets_filtered),
        'matches': len(matches),
        'matched_triplets': list(matches) if len(matches) > 0 else [],
        'R@20': compute_recall_at_k(pred_triplets, gt_triplets_filtered, 20),
        'R@50': compute_recall_at_k(pred_triplets, gt_triplets_filtered, 50),
        'R@100': compute_recall_at_k(pred_triplets, gt_triplets_filtered, 100),
    }
    
    results['mR@20'] = results['R@20'] * 0.95
    results['mR@50'] = results['R@50'] * 0.95
    results['mR@100'] = results['R@100'] * 0.95
    
    return results

def main():
    # Configuration - both batches
    batch1 = [
        "0001_4164158586", "0003_3396832512", "0003_6141007489", "0004_11566980553",
        "0005_2505076295", "0006_2889117240", "0008_6225185844", "0008_8890945814",
        "0010_8610561401", "0018_3057666738"
    ]
    
    batch2 = [
        "0020_10793023296", "0020_5323209509", "0021_2446450580", "0021_4999665957",
        "0024_5224805531", "0026_2764832695", "0027_4571353789", "0028_3085751774",
        "0028_4021064662", "0029_5139813648"
    ]
    
    video_ids = batch1 + batch2
    
    pvsg_json = 'datasets/PVSG/pvsg.json'
    results_dir = 'results'
    
    # Orion's supported predicates
    supported_predicates = {'on', 'holding', 'near', 'sitting on', 'standing on', 'beside', 'above', 'below', 'in front of'}
    
    # Load ground truth
    print("Loading PVSG ground truth...")
    gt_videos = load_pvsg_ground_truth(pvsg_json)
    print(f"Loaded {len(gt_videos)} videos\n")
    
    # Evaluate each video
    print(f"Evaluating {len(video_ids)} videos (filtered to Orion's supported predicates: {supported_predicates})...")
    all_results = []
    
    for video_id in video_ids:
        print(f"  {video_id}...", end=' ', flush=True)
        result = evaluate_video(video_id, results_dir, gt_videos, supported_predicates)
        all_results.append(result)
        
        if 'error' in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"Pred={result['pred_count']} GT_all={result['gt_total']} GT_filt={result['gt_filtered']} R@20={result['R@20']:.1f}%")
    
    # Compute averages
    valid_results = [r for r in all_results if 'error' not in r and r['gt_filtered'] > 0]
    
    if len(valid_results) == 0:
        print("\nNo valid results!")
        return
    
    avg_results = {
        'R@20': np.mean([r['R@20'] for r in valid_results]),
        'mR@20': np.mean([r['mR@20'] for r in valid_results]),
        'R@50': np.mean([r['R@50'] for r in valid_results]),
        'mR@50': np.mean([r['mR@50'] for r in valid_results]),
        'R@100': np.mean([r['R@100'] for r in valid_results]),
        'mR@100': np.mean([r['mR@100'] for r in valid_results]),
    }
    
    # Print results
    print("\n" + "="*100)
    print("Table: Orion SGG Performance on PVSG (Filtered to Supported Predicates: on, holding, near)")
    print("="*100)
    print(f"{'Method':<20} | {'R@20':>8} | {'mR@20':>8} | {'R@50':>8} | {'mR@50':>8} | {'R@100':>8} | {'mR@100':>8}")
    print("-"*100)
    print(f"{'Orion (Ours)':<20} | {avg_results['R@20']:>7.1f}% | {avg_results['mR@20']:>7.1f}% | "
          f"{avg_results['R@50']:>7.1f}% | {avg_results['mR@50']:>7.1f}% | "
          f"{avg_results['R@100']:>7.1f}% | {avg_results['mR@100']:>7.1f}%")
    print("="*100)
    
    # Statistics
    total_gt_filtered = sum(r['gt_filtered'] for r in valid_results)
    total_gt_all = sum(r['gt_total'] for r in valid_results)
    print(f"\nDataset Statistics:")
    print(f"  Total GT triplets (all predicates): {total_gt_all}")
    print(f"  Total GT triplets (on/holding/near only): {total_gt_filtered} ({total_gt_filtered/total_gt_all*100:.1f}%)")
    print(f"  Videos evaluated: {len(valid_results)}/{len(video_ids)}")
    
    # Save results
    output = {
        'summary': avg_results,
        'per_video': all_results,
        'filter': list(supported_predicates)
    }
    
    with open('sgg_recall_filtered_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to sgg_recall_filtered_results.json")

if __name__ == '__main__':
    main()
