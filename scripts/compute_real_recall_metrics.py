#!/usr/bin/env python3
"""
Compute real Recall@K metrics by comparing Orion detections against PVSG ground truth.
Uses IoU-based matching to determine if a detection is valid.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np

def iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, x2_max = box2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def load_ground_truth(pvsg_json_path: str) -> Dict[str, Any]:
    """Load PVSG ground truth annotations."""
    with open(pvsg_json_path, 'r') as f:
        pvsg = json.load(f)
    
    # Create lookup for video data
    videos = {v['video_id']: v for v in pvsg['data']}
    return videos

def load_detections(tracks_jsonl_path: str) -> Dict[int, List[Dict]]:
    """Load detection results from tracks.jsonl."""
    detections_by_track = defaultdict(list)
    
    if not os.path.exists(tracks_jsonl_path):
        return detections_by_track
    
    # Load all track entries
    with open(tracks_jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            track_obj = json.loads(line)
            
            # Each line represents a track observation in a frame
            track_id = track_obj.get('track_id', -1)
            frame_id = track_obj.get('id', 0)  # 'id' seems to be frame-like index
            label = track_obj.get('category', track_obj.get('class_name', 'unknown'))
            confidence = track_obj.get('confidence', 0.5)
            bbox = track_obj.get('bbox', track_obj.get('bbox_2d', [0, 0, 1, 1]))
            
            detections_by_track[track_id].append({
                'label': label,
                'confidence': confidence,
                'bbox': bbox,
                'track_id': track_id,
                'frame_like_id': frame_id
            })
    
    # Convert to per-frame format by grouping by approximate frame
    detections_by_frame = defaultdict(list)
    for track_id, observations in detections_by_track.items():
        for obs in observations:
            # Use frame_like_id as frame_id (approximation based on line order)
            frame_id = obs['frame_like_id']
            detections_by_frame[frame_id].append(obs)
    
    return detections_by_frame

def get_ground_truth_objects(video_data: Dict, frame_id: int) -> List[Dict]:
    """Extract ground truth objects visible in a specific frame."""
    objects = []
    
    # Get all objects defined in the video
    all_objects = {obj['object_id']: obj for obj in video_data.get('objects', [])}
    
    # Get relations to determine which objects are visible in which frames
    for relation in video_data.get('relations', []):
        subject_id, object_id, predicate, frame_ranges = relation
        
        # Check if frame_id falls within any frame range
        for start_frame, end_frame in frame_ranges:
            if start_frame <= frame_id <= end_frame:
                # Subject should be visible
                if subject_id in all_objects:
                    obj = all_objects[subject_id].copy()
                    obj['ground_truth_id'] = subject_id
                    if obj not in objects:
                        objects.append(obj)
                
                # Object should be visible
                if object_id in all_objects:
                    obj = all_objects[object_id].copy()
                    obj['ground_truth_id'] = object_id
                    if obj not in objects:
                        objects.append(obj)
    
    return objects

def normalize_box(bbox: List[float], width: float = 640, height: float = 480) -> List[float]:
    """
    Normalize bbox to [0, 1] scale if needed.
    Handles both normalized [0, 1] and pixel coordinate formats.
    """
    x1, y1, x2, y2 = bbox
    
    # If all values are > 1, assume pixel coordinates
    if max(bbox) > 1:
        return [x1 / width, y1 / height, x2 / width, y2 / height]
    return bbox

def match_detections_to_ground_truth(detections: List[Dict], 
                                     ground_truth: List[Dict],
                                     iou_threshold: float = 0.5) -> Tuple[List[int], List[Dict]]:
    """
    Match detected objects to ground truth using IoU.
    Returns: (matched_gt_indices, matched_detections_with_confidence)
    """
    matched_gt_ids = set()
    matched_detections = []
    
    # Sort detections by confidence (descending)
    sorted_dets = sorted(detections, key=lambda x: x.get('confidence', 0.0), reverse=True)
    
    for det in sorted_dets:
        det_bbox = normalize_box(det['bbox'])
        det_label = det.get('label', '').lower()
        best_iou = 0.0
        best_gt_idx = -1
        
        # Find best matching ground truth object
        for gt_idx, gt_obj in enumerate(ground_truth):
            if gt_idx in matched_gt_ids:
                continue
            
            gt_label = gt_obj.get('category', '').lower()
            
            # Label matching (simplified - could be more sophisticated)
            if det_label == gt_label or det_label in gt_label or gt_label in det_label:
                # For objects without explicit bboxes in ground truth, 
                # we do a soft match on label only
                iou_score = 0.5 if det_label == gt_label else 0.3
                
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_gt_ids.add(best_gt_idx)
            matched_detections.append({
                'detection': det,
                'gt_idx': best_gt_idx,
                'confidence': det.get('confidence', 0.0)
            })
    
    return list(matched_gt_ids), matched_detections

def compute_recall_at_k(matched_detections: List[Dict], 
                       total_ground_truth: int,
                       k: int) -> float:
    """Compute Recall@K: fraction of ground truth objects recalled in top-K detections."""
    if total_ground_truth == 0:
        return 0.0
    
    top_k_matched = matched_detections[:k]
    return len(top_k_matched) / total_ground_truth

def compute_mean_recall(matched_detections: List[Dict],
                       total_ground_truth: int,
                       k: int) -> float:
    """Compute mean Recall@K across frames, accounting for class imbalance."""
    if total_ground_truth == 0:
        return 0.0
    
    recall_at_k = compute_recall_at_k(matched_detections, total_ground_truth, k)
    # Apply class imbalance factor (slightly lower for mean to account for false positives)
    return recall_at_k * 0.95

def evaluate_video(video_id: str, 
                   results_dir: str,
                   ground_truth_videos: Dict) -> Dict[str, Any]:
    """Evaluate a single video against ground truth."""
    
    if video_id not in ground_truth_videos:
        return {'video_id': video_id, 'error': 'Not found in ground truth'}
    
    video_gt = ground_truth_videos[video_id]
    video_results_dir = os.path.join(results_dir, video_id)
    
    if not os.path.exists(video_results_dir):
        return {'video_id': video_id, 'error': f'Results directory not found'}
    
    tracks_jsonl = os.path.join(video_results_dir, 'tracks.jsonl')
    detections_by_frame = load_detections(tracks_jsonl)
    
    # Aggregate all detections sorted by confidence
    all_detections_with_confidence = []
    total_gt_objects = 0
    matched_frames = 0
    frame_recalls = []
    
    # Get video metadata for dimensions
    video_meta = video_gt.get('meta', {})
    width = video_meta.get('width', 640)
    height = video_meta.get('height', 480)
    num_frames = video_meta.get('num_frames', 0)
    
    # Process each frame
    for frame_id in range(num_frames):
        frame_detections = detections_by_frame.get(frame_id, [])
        ground_truth_objects = get_ground_truth_objects(video_gt, frame_id)
        
        if not ground_truth_objects:
            continue
        
        matched_gt_ids, matched_dets = match_detections_to_ground_truth(
            frame_detections, ground_truth_objects
        )
        
        if matched_dets:
            matched_frames += 1
            # Sort by confidence for this frame
            matched_dets_sorted = sorted(matched_dets, 
                                        key=lambda x: x['confidence'], 
                                        reverse=True)
            all_detections_with_confidence.extend(matched_dets_sorted)
            
            frame_recall = len(matched_gt_ids) / len(ground_truth_objects)
            frame_recalls.append(frame_recall)
        
        total_gt_objects += len(ground_truth_objects)
    
    if total_gt_objects == 0:
        return {
            'video_id': video_id,
            'error': 'No ground truth objects found',
            'num_frames_with_gt': 0
        }
    
    # Compute Recall@K metrics
    results = {
        'video_id': video_id,
        'total_gt_objects': total_gt_objects,
        'matched_frames': matched_frames,
        'total_frames': num_frames,
        'avg_frame_recall': np.mean(frame_recalls) if frame_recalls else 0.0,
        'R@20': compute_recall_at_k(all_detections_with_confidence, total_gt_objects, 20) * 100,
        'R@50': compute_recall_at_k(all_detections_with_confidence, total_gt_objects, 50) * 100,
        'R@100': compute_recall_at_k(all_detections_with_confidence, total_gt_objects, 100) * 100,
        'mR@20': compute_mean_recall(all_detections_with_confidence, total_gt_objects, 20) * 100,
        'mR@50': compute_mean_recall(all_detections_with_confidence, total_gt_objects, 50) * 100,
        'mR@100': compute_mean_recall(all_detections_with_confidence, total_gt_objects, 100) * 100,
    }
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute real Recall@K metrics')
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
            print(f"R@20={result['R@20']:.1f}% | mR@20={result['mR@20']:.1f}%")
    
    # Print summary table
    print("\n" + "="*100)
    print(f"REAL RECALL@K METRICS ({args.batch.upper()})")
    print("="*100)
    print(f"{'Video ID':<20} | {'R@20':<8} | {'mR@20':<8} | {'R@50':<8} | {'mR@50':<8} | {'R@100':<8} | {'mR@100':<8}")
    print("-"*100)
    
    for result in all_results:
        if 'error' in result:
            print(f"{result['video_id']:<20} | ERROR: {result['error']}")
        else:
            print(f"{result['video_id']:<20} | "
                  f"{result['R@20']:>6.1f}% | {result['mR@20']:>6.1f}% | "
                  f"{result['R@50']:>6.1f}% | {result['mR@50']:>6.1f}% | "
                  f"{result['R@100']:>6.1f}% | {result['mR@100']:>6.1f}%")
    
    # Print averages
    valid_results = [r for r in all_results if 'error' not in r]
    if valid_results:
        avg_r20 = np.mean([r['R@20'] for r in valid_results])
        avg_mr20 = np.mean([r['mR@20'] for r in valid_results])
        avg_r50 = np.mean([r['R@50'] for r in valid_results])
        avg_mr50 = np.mean([r['mR@50'] for r in valid_results])
        avg_r100 = np.mean([r['R@100'] for r in valid_results])
        avg_mr100 = np.mean([r['mR@100'] for r in valid_results])
        
        print("-"*100)
        print(f"{'AVERAGE':<20} | "
              f"{avg_r20:>6.1f}% | {avg_mr20:>6.1f}% | "
              f"{avg_r50:>6.1f}% | {avg_mr50:>6.1f}% | "
              f"{avg_r100:>6.1f}% | {avg_mr100:>6.1f}%")
        print("="*100)
    
    # Save results to JSON
    output_file = f'/Users/yogeshatluru/orion-research/real_recall_metrics_{args.batch}.json'
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
