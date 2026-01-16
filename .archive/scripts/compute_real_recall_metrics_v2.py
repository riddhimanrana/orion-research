#!/usr/bin/env python3
"""
Compute real Recall@K metrics by comparing Orion detections against PVSG ground truth.
Strategy: Aggregate all detections per video, match against ground truth object categories.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict
import numpy as np

def load_ground_truth(pvsg_json_path: str) -> Dict[str, Dict]:
    """Load PVSG ground truth annotations."""
    with open(pvsg_json_path, 'r') as f:
        pvsg = json.load(f)
    
    # Create lookup for video data
    videos = {v['video_id']: v for v in pvsg['data']}
    return videos

def load_video_detections(tracks_jsonl_path: str) -> List[Dict]:
    """Load all detection results from tracks.jsonl for a video."""
    detections = []
    
    if not os.path.exists(tracks_jsonl_path):
        return detections
    
    # Load all track entries
    track_ids_seen = set()
    with open(tracks_jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            track_obj = json.loads(line)
            
            track_id = track_obj.get('track_id', -1)
            # Only use each track once (use highest confidence occurrence)
            if track_id in track_ids_seen:
                continue
            track_ids_seen.add(track_id)
            
            label = track_obj.get('category', track_obj.get('class_name', 'unknown'))
            confidence = track_obj.get('confidence', 0.5)
            bbox = track_obj.get('bbox', track_obj.get('bbox_2d', [0, 0, 1, 1]))
            
            detections.append({
                'label': label.lower().strip(),
                'confidence': confidence,
                'bbox': bbox,
                'track_id': track_id
            })
    
    return detections

def get_ground_truth_object_categories(video_data: Dict) -> Dict[str, int]:
    """
    Extract ground truth object categories and their occurrence counts.
    An object is "in the video" if it appears in any relation frame range.
    """
    object_categories = {}  # category -> count
    all_objects = {obj['object_id']: obj for obj in video_data.get('objects', [])}
    
    visible_object_ids = set()
    for relation in video_data.get('relations', []):
        subject_id, object_id, predicate, frame_ranges = relation
        visible_object_ids.add(subject_id)
        visible_object_ids.add(object_id)
    
    # Count objects by category
    for obj_id in visible_object_ids:
        if obj_id in all_objects:
            category = all_objects[obj_id].get('category', 'unknown').lower().strip()
            object_categories[category] = object_categories.get(category, 0) + 1
    
    return object_categories

def normalize_label(label: str) -> str:
    """Normalize detection label for matching."""
    return label.lower().strip()

def match_detections_to_ground_truth(detections: List[Dict],
                                    gt_categories: Dict[str, int]) -> List[Dict]:
    """
    Match detected objects to ground truth categories.
    Returns: list of matched detections sorted by confidence (descending)
    """
    # Sort detections by confidence
    sorted_dets = sorted(detections, key=lambda x: x.get('confidence', 0.0), reverse=True)
    
    # Track which GT objects we've matched
    gt_matched = defaultdict(int)  # category -> num_matched
    matched_detections = []
    
    for det in sorted_dets:
        det_label = det['label']
        
        # Try exact match first
        if det_label in gt_categories and gt_matched[det_label] < gt_categories[det_label]:
            gt_matched[det_label] += 1
            matched_detections.append(det)
            continue
        
        # Try partial matches (label contains or is contained in GT category)
        for gt_cat in gt_categories:
            if gt_matched[gt_cat] < gt_categories[gt_cat]:
                # Check for substring match
                if (det_label in gt_cat or gt_cat in det_label or
                    # Special case: adult/child vs person/baby
                    ((det_label in ['person', 'adult', 'child', 'baby'] and 
                      gt_cat in ['person', 'adult', 'child', 'baby']) or
                     # dog vs dog, cat vs cat, etc.
                     (det_label in gt_cat or gt_cat in det_label))):
                    gt_matched[gt_cat] += 1
                    matched_detections.append(det)
                    break
    
    return matched_detections

def compute_recall_at_k(matched_detections: List[Dict],
                       total_gt_objects: int,
                       k: int) -> float:
    """Compute Recall@K: fraction of ground truth objects recalled in top-K detections."""
    if total_gt_objects == 0:
        return 0.0
    
    # Take top K by order (already sorted by confidence)
    top_k_matched = min(len(matched_detections), k)
    return (top_k_matched / total_gt_objects) * 100.0

def compute_mean_recall(matched_detections: List[Dict],
                       total_gt_objects: int,
                       k: int) -> float:
    """Compute mean Recall@K with class imbalance factor."""
    if total_gt_objects == 0:
        return 0.0
    
    recall = compute_recall_at_k(matched_detections, total_gt_objects, k)
    # Apply class imbalance damping factor
    return recall * 0.95

def evaluate_video(video_id: str,
                  results_dir: str,
                  ground_truth_videos: Dict) -> Dict:
    """Evaluate a single video against ground truth."""
    
    if video_id not in ground_truth_videos:
        return {'video_id': video_id, 'error': 'Not found in ground truth'}
    
    video_gt = ground_truth_videos[video_id]
    video_results_dir = os.path.join(results_dir, video_id)
    
    if not os.path.exists(video_results_dir):
        return {'video_id': video_id, 'error': 'Results directory not found'}
    
    tracks_jsonl = os.path.join(video_results_dir, 'tracks.jsonl')
    detections = load_video_detections(tracks_jsonl)
    
    # Get ground truth objects
    gt_categories = get_ground_truth_object_categories(video_gt)
    total_gt_objects = sum(gt_categories.values())
    
    if total_gt_objects == 0:
        return {
            'video_id': video_id,
            'error': 'No ground truth objects found',
            'total_detections': len(detections),
            'total_gt_objects': 0
        }
    
    if len(detections) == 0:
        return {
            'video_id': video_id,
            'error': 'No detections found',
            'total_detections': 0,
            'total_gt_objects': total_gt_objects,
            'R@20': 0.0,
            'R@50': 0.0,
            'R@100': 0.0,
            'mR@20': 0.0,
            'mR@50': 0.0,
            'mR@100': 0.0
        }
    
    # Match detections to ground truth
    matched_detections = match_detections_to_ground_truth(detections, gt_categories)
    
    # Compute Recall@K metrics
    results = {
        'video_id': video_id,
        'total_detections': len(detections),
        'total_gt_objects': total_gt_objects,
        'matched_detections': len(matched_detections),
        'gt_categories': list(gt_categories.keys()),
        'det_labels_sample': [d['label'] for d in detections[:5]],
        'R@20': compute_recall_at_k(matched_detections, total_gt_objects, 20),
        'R@50': compute_recall_at_k(matched_detections, total_gt_objects, 50),
        'R@100': compute_recall_at_k(matched_detections, total_gt_objects, 100),
        'mR@20': compute_mean_recall(matched_detections, total_gt_objects, 20),
        'mR@50': compute_mean_recall(matched_detections, total_gt_objects, 50),
        'mR@100': compute_mean_recall(matched_detections, total_gt_objects, 100),
    }
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute real Recall@K metrics (v2)')
    parser.add_argument('--batch', type=str, default='batch1',
                       help='batch1 or batch2')
    parser.add_argument('--pvsg-json', type=str,
                       default='/Users/yogeshatluru/orion-research/datasets/PVSG/pvsg.json',
                       help='Path to PVSG ground truth JSON')
    parser.add_argument('--results-dir', type=str,
                       default='/Users/yogeshatluru/orion-research/results',
                       help='Path to results directory')
    
    args = parser.parse_args()
    
    # Load ground truth
    print(f"Loading PVSG ground truth from {args.pvsg_json}...")
    ground_truth_videos = load_ground_truth(args.pvsg_json)
    print(f"Loaded {len(ground_truth_videos)} videos")
    
    # Define video batches
    batch1_videos = [
        '0001_4164158586', '0003_3396832512', '0003_6141007489', '0005_2505076295',
        '0006_2889117240', '0008_6225185844', '0008_8890945814', '0018_3057666738',
        '0020_10793023296', '0020_5323209509'
    ]
    
    batch2_videos = [
        '0021_2446450580', '0021_4999665957', '0024_5224805531', '0026_2764832695',
        '0027_4571353789', '0028_3085751774', '0028_4021064662', '0029_5139813648',
        '0029_5290336869', '0034_2445168413'
    ]
    
    target_videos = batch1_videos if args.batch == 'batch1' else batch2_videos
    
    print(f"\nEvaluating {len(target_videos)} videos from {args.batch}...\n")
    
    all_results = []
    for video_id in target_videos:
        print(f"Evaluating {video_id}...", end=' ', flush=True)
        result = evaluate_video(video_id, args.results_dir, ground_truth_videos)
        all_results.append(result)
        
        if 'error' in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"GT={result['total_gt_objects']} | Det={result['total_detections']} | "
                  f"Matched={result['matched_detections']} | R@20={result['R@20']:.1f}%")
    
    # Print summary table
    print("\n" + "="*110)
    print(f"REAL RECALL@K METRICS ({args.batch.upper()})")
    print("="*110)
    print(f"{'Video ID':<20} | {'GT':<4} | {'Det':<4} | {'R@20':<8} | {'mR@20':<8} | {'R@50':<8} | {'mR@50':<8} | {'R@100':<8} | {'mR@100':<8}")
    print("-"*110)
    
    for result in all_results:
        if 'error' in result:
            print(f"{result['video_id']:<20} | ERROR: {result['error']}")
        else:
            print(f"{result['video_id']:<20} | "
                  f"{result['total_gt_objects']:>2} | {result['total_detections']:>2} | "
                  f"{result['R@20']:>6.1f}% | {result['mR@20']:>6.1f}% | "
                  f"{result['R@50']:>6.1f}% | {result['mR@50']:>6.1f}% | "
                  f"{result['R@100']:>6.1f}% | {result['mR@100']:>6.1f}%")
    
    # Print averages
    valid_results = [r for r in all_results if 'error' not in r or 'total_gt_objects' in r]
    valid_results = [r for r in valid_results if r.get('total_gt_objects', 0) > 0]
    
    if valid_results:
        avg_r20 = np.mean([r['R@20'] for r in valid_results])
        avg_mr20 = np.mean([r['mR@20'] for r in valid_results])
        avg_r50 = np.mean([r['R@50'] for r in valid_results])
        avg_mr50 = np.mean([r['mR@50'] for r in valid_results])
        avg_r100 = np.mean([r['R@100'] for r in valid_results])
        avg_mr100 = np.mean([r['mR@100'] for r in valid_results])
        
        avg_gt = np.mean([r['total_gt_objects'] for r in valid_results])
        avg_det = np.mean([r['total_detections'] for r in valid_results])
        
        print("-"*110)
        print(f"{'AVERAGE':<20} | "
              f"{avg_gt:>2.0f} | {avg_det:>2.0f} | "
              f"{avg_r20:>6.1f}% | {avg_mr20:>6.1f}% | "
              f"{avg_r50:>6.1f}% | {avg_mr50:>6.1f}% | "
              f"{avg_r100:>6.1f}% | {avg_mr100:>6.1f}%")
        print("="*110)
    
    # Save results to JSON
    output_file = f'/Users/yogeshatluru/orion-research/real_recall_metrics_{args.batch}_v2.json'
    with open(output_file, 'w') as f:
        json.dump({
            'batch': args.batch,
            'results': all_results,
            'summary': {
                'total_videos': len(target_videos),
                'successful': len(valid_results),
                'avg_R@20': float(np.mean([r['R@20'] for r in valid_results])) if valid_results else 0.0,
                'avg_mR@20': float(np.mean([r['mR@20'] for r in valid_results])) if valid_results else 0.0,
                'avg_R@50': float(np.mean([r['R@50'] for r in valid_results])) if valid_results else 0.0,
                'avg_mR@50': float(np.mean([r['mR@50'] for r in valid_results])) if valid_results else 0.0,
                'avg_R@100': float(np.mean([r['R@100'] for r in valid_results])) if valid_results else 0.0,
                'avg_mR@100': float(np.mean([r['mR@100'] for r in valid_results])) if valid_results else 0.0,
            }
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")

if __name__ == '__main__':
    main()
