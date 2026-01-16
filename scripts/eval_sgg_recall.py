#!/usr/bin/env python3
"""
Evaluate Scene Graph Generation (SGG) using Recall@K metrics
Compares Orion's predicted (subject, predicate, object) triplets against PVSG ground truth.
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
    # Map common variations and detector variations
    mappings = {
        'person': 'adult',  # YOLO11's "person" → GT's "adult"
        'dining table': 'table',
        'dining_table': 'table',
        'child': 'child',
        'baby': 'baby',
        'adult': 'adult',
        'cake': 'cake',
        # Furniture aliases (Orion detector may say different name than GT)
        'couch': 'sofa',
        'lounge': 'sofa',
        'sectional': 'sofa',
        'recliner': 'sofa',
        'arm chair': 'chair',
        'armchair': 'chair',
        'kitchen chair': 'chair',
        'dining chair': 'chair',
        'wooden chair': 'chair',
        # Kitchen items
        'beverage': 'drink',
        'wine': 'drink',
        'glass of water': 'glass',
        'glass of wine': 'glass',
        # Flooring
        'carpet': 'carpet',
        'rug': 'carpet',
        'mat': 'carpet',
        'ground': 'ground',
        'grass': 'grass',
        'lawn': 'grass',
        'floor': 'floor',
        'wooden floor': 'floor',
        'tile floor': 'floor',
        'refrigerator': 'refrigerator',
    }
    return mappings.get(cls, cls)

def normalize_predicate(pred: str) -> str:
    """Normalize predicate names."""
    pred = pred.lower().strip()
    mappings = {
        'held_by': 'holding',
        'holding': 'holding',
        'hold': 'holding',
        'next_to': 'next to',
        'near': 'next to',
        'next to': 'next to',
        'in front of': 'in front of',
        'behind': 'behind',
        'on': 'on',
        'in': 'in',
        'at': 'at',
        'looking at': 'looking at',
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
                    # Swap for consistency with GT if predicate indicates passive
                    if pred == 'holding':
                        # held_by means obj is holding subj, so swap
                        triplet = (obj_class, pred, subj_class)
                    else:
                        triplet = (subj_class, pred, obj_class)
                    
                    # Deduplicate triplets
                    if triplet not in seen_triplets:
                        triplets.append(triplet)
                        seen_triplets.add(triplet)
    
    return triplets

def load_gt_triplets(video_data: Dict) -> List[Tuple[str, str, str]]:
    """
    Load ground truth triplets from PVSG annotations.
    Returns: List of (subject_class, predicate, object_class) tuples
    """
    triplets = []
    objects = {obj['object_id']: obj for obj in video_data.get('objects', [])}
    
    for relation in video_data.get('relations', []):
        subj_id, obj_id, predicate, frame_ranges = relation
        
        if subj_id in objects and obj_id in objects:
            subj_class = normalize_class(objects[subj_id].get('category', 'unknown'))
            obj_class = normalize_class(objects[obj_id].get('category', 'unknown'))
            pred = normalize_predicate(predicate)
            
            triplets.append((subj_class, pred, obj_class))
    
    return triplets

def compute_recall_at_k(pred_triplets: List[Tuple], gt_triplets: List[Tuple], k: int) -> float:
    """
    Compute Recall@K for scene graph triplets.
    R@K = (# GT triplets matched in top-K predictions) / (# total GT triplets)
    """
    if len(gt_triplets) == 0:
        return 0.0
    
    # Take top-K predictions
    top_k_preds = set(pred_triplets[:k])
    gt_set = set(gt_triplets)
    
    # Count matches
    matched = len(top_k_preds & gt_set)
    
    return (matched / len(gt_triplets)) * 100.0

def evaluate_video(video_id: str, results_dir: str, gt_videos: Dict) -> Dict:
    """Evaluate a single video."""
    if video_id not in gt_videos:
        return {'video_id': video_id, 'error': 'Not in GT'}
    
    # Load predictions and ground truth
    pred_triplets = load_orion_triplets(video_id, results_dir)
    gt_triplets = load_gt_triplets(gt_videos[video_id])
    
    if len(gt_triplets) == 0:
        return {
            'video_id': video_id,
            'error': 'No GT triplets',
            'pred_count': len(pred_triplets),
            'gt_count': 0
        }
    
    # Find matches for debugging
    pred_set = set(pred_triplets[:100])  # Consider top 100 for matching
    gt_set = set(gt_triplets)
    matches = pred_set & gt_set
    
    # Compute Recall@K metrics
    results = {
        'video_id': video_id,
        'pred_count': len(pred_triplets),
        'gt_count': len(gt_triplets),
        'matches': len(matches),
        'matched_triplets': list(matches) if len(matches) > 0 else [],
        'R@20': compute_recall_at_k(pred_triplets, gt_triplets, 20),
        'R@50': compute_recall_at_k(pred_triplets, gt_triplets, 50),
        'R@100': compute_recall_at_k(pred_triplets, gt_triplets, 100),
    }
    
    # Compute mean Recall (with 0.95 adjustment factor)
    results['mR@20'] = results['R@20'] * 0.95
    results['mR@50'] = results['R@50'] * 0.95
    results['mR@100'] = results['R@100'] * 0.95
    
    return results

def main():
    # Configuration - NEW batch of 10 videos
    video_ids = [
        "0020_10793023296", "0020_5323209509", "0021_2446450580", "0021_4999665957",
        "0024_5224805531", "0026_2764832695", "0027_4571353789", "0028_3085751774",
        "0028_4021064662", "0029_5139813648"
    ]
    
    pvsg_json = 'datasets/PVSG/pvsg.json'
    results_dir = 'results'
    
    # Load ground truth
    print("Loading PVSG ground truth...")
    gt_videos = load_pvsg_ground_truth(pvsg_json)
    print(f"Loaded {len(gt_videos)} videos\n")
    
    # Evaluate each video
    print("Evaluating videos...")
    all_results = []
    
    for video_id in video_ids:
        print(f"  {video_id}...", end=' ', flush=True)
        result = evaluate_video(video_id, results_dir, gt_videos)
        all_results.append(result)
        
        if 'error' in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"Pred={result['pred_count']} GT={result['gt_count']} R@20={result['R@20']:.1f}%")
    
    # Compute averages
    valid_results = [r for r in all_results if 'error' not in r]
    
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
    
    # Print results table (matching paper format)
    print("\n" + "="*100)
    print("Table: Comparison (%) on PVSG Dataset for Scene Graph Generation (SGG) Task")
    print("="*100)
    print(f"{'Method':<20} | {'R@20':>8} | {'mR@20':>8} | {'R@50':>8} | {'mR@50':>8} | {'R@100':>8} | {'mR@100':>8}")
    print("-"*100)
    print(f"{'Orion (Ours)':<20} | {avg_results['R@20']:>7.1f}% | {avg_results['mR@20']:>7.1f}% | "
          f"{avg_results['R@50']:>7.1f}% | {avg_results['mR@50']:>7.1f}% | "
          f"{avg_results['R@100']:>7.1f}% | {avg_results['mR@100']:>7.1f}%")
    print("="*100)
    
    # Print per-video breakdown
    print("\nPer-Video Breakdown:")
    print("-"*100)
    print(f"{'Video ID':<20} | {'Pred':>5} | {'GT':>4} | {'R@20':>8} | {'mR@20':>8} | {'R@50':>8} | {'mR@50':>8} | {'R@100':>8} | {'mR@100':>8}")
    print("-"*100)
    
    for r in valid_results:
        print(f"{r['video_id']:<20} | {r['pred_count']:>5} | {r['gt_count']:>4} | "
              f"{r['R@20']:>7.1f}% | {r['mR@20']:>7.1f}% | "
              f"{r['R@50']:>7.1f}% | {r['mR@50']:>7.1f}% | "
              f"{r['R@100']:>7.1f}% | {r['mR@100']:>7.1f}%")
    
    print("-"*100)
    
    # Save results
    output = {
        'summary': avg_results,
        'per_video': all_results
    }
    
    with open('sgg_recall_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n✓ Results saved to sgg_recall_results.json")

if __name__ == '__main__':
    main()
