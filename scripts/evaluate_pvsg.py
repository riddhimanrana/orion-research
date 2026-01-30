#!/usr/bin/env python3
"""
PVSG Relation Evaluation Script

Computes standard Scene Graph Generation metrics:
- Recall@K (R@20, R@50, R@100)
- Mean Recall@K (mR@20, mR@50, mR@100)
"""

import json
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def load_pvsg_data(path: str, video_id: str) -> Dict:
    """Load PVSG data (GT or predictions) for a specific video."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, dict) and 'video_id' in data:
        # Single video format (predictions)
        if data['video_id'] == video_id:
            return data
        raise ValueError(f"Video {video_id} not found in predictions")
    elif isinstance(data, dict) and 'data' in data:
        # PVSG dataset format (GT)
        for video_data in data['data']:
            if video_data['video_id'] == video_id:
                return video_data
        raise ValueError(f"Video {video_id} not found in GT")
    else:
        raise ValueError("Unknown data format")


def compute_iou_temporal(pred_ranges: List[List[int]], gt_ranges: List[List[int]]) -> float:
    """
    Compute temporal IoU between predicted and GT frame ranges.
    
    Args:
        pred_ranges: List of [start, end] frame ranges
        gt_ranges: List of [start, end] frame ranges
    
    Returns:
        IoU score (intersection / union)
    """
    # Convert ranges to sets of frames
    pred_frames = set()
    for start, end in pred_ranges:
        pred_frames.update(range(start, end + 1))
    
    gt_frames = set()
    for start, end in gt_ranges:
        gt_frames.update(range(start, end + 1))
    
    if not gt_frames:
        return 0.0
    
    intersection = len(pred_frames & gt_frames)
    union = len(pred_frames | gt_frames)
    
    return intersection / union if union > 0 else 0.0


def match_relations(
    predictions: List[List],
    ground_truth: List[List],
    iou_threshold: float = 0.5
) -> Tuple[Set[int], Dict[str, int]]:
    """
    Match predicted relations to ground truth.
    
    Args:
        predictions: List of [subject_id, object_id, predicate, ranges]
        ground_truth: List of [subject_id, object_id, predicate, ranges]
        iou_threshold: Minimum temporal IoU for a match
    
    Returns:
        - Set of matched GT indices
        - Dict of predicate -> count of matches
    """
    matched_gt = set()
    predicate_matches = defaultdict(int)
    
    for pred in predictions:
        pred_subj, pred_obj, pred_pred, pred_ranges = pred
        
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            
            gt_subj, gt_obj, gt_pred, gt_ranges = gt
            
            # Check if subject, object, and predicate match
            if pred_subj == gt_subj and pred_obj == gt_obj and pred_pred == gt_pred:
                # Compute temporal IoU
                iou = compute_iou_temporal(pred_ranges, gt_ranges)
                
                if iou >= iou_threshold:
                    matched_gt.add(gt_idx)
                    predicate_matches[gt_pred] += 1
                    break  # Move to next prediction
    
    return matched_gt, predicate_matches


def compute_recall_at_k(
    predictions: List[List],
    ground_truth: List[List],
    k: int,
    iou_threshold: float = 0.5
) -> float:
    """
    Compute Recall@K.
    
    Args:
        predictions: Predicted relations
        ground_truth: Ground truth relations
        k: Top-k predictions to consider
        iou_threshold: Minimum temporal IoU for a match
    
    Returns:
        Recall@K score
    """
    if not ground_truth:
        return 0.0
    
    # Take top-k predictions (in practice, we'd sort by confidence)
    top_k_preds = predictions[:k]
    
    matched_gt, _ = match_relations(top_k_preds, ground_truth, iou_threshold)
    
    return len(matched_gt) / len(ground_truth)


def compute_mean_recall_at_k(
    predictions: List[List],
    ground_truth: List[List],
    k: int,
    iou_threshold: float = 0.5
) -> float:
    """
    Compute Mean Recall@K (per-predicate recall).
    
    Args:
        predictions: Predicted relations
        ground_truth: Ground truth relations
        k: Top-k predictions to consider
        iou_threshold: Minimum temporal IoU for a match
    
    Returns:
        Mean Recall@K score
    """
    if not ground_truth:
        return 0.0
    
    # Group GT by predicate
    gt_by_predicate = defaultdict(list)
    for gt in ground_truth:
        predicate = gt[2]
        gt_by_predicate[predicate].append(gt)
    
    # Take top-k predictions
    top_k_preds = predictions[:k]
    
    # Compute recall for each predicate
    predicate_recalls = []
    
    for predicate, gt_rels in gt_by_predicate.items():
        # Filter predictions for this predicate
        pred_rels = [p for p in top_k_preds if p[2] == predicate]
        
        if not pred_rels:
            # No predictions for this predicate -> recall = 0
            predicate_recalls.append(0.0)
            continue
        
        matched_gt, _ = match_relations(pred_rels, gt_rels, iou_threshold)
        recall = len(matched_gt) / len(gt_rels)
        predicate_recalls.append(recall)
    
    return sum(predicate_recalls) / len(predicate_recalls) if predicate_recalls else 0.0


def print_evaluation_results(results: Dict):
    """Print evaluation results in a formatted table."""
    print("\n" + "="*80)
    print("PVSG RELATION EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nVideo: {results['video_id']}")
    print(f"Ground Truth Relations: {results['num_gt_relations']}")
    print(f"Predicted Relations: {results['num_pred_relations']}")
    
    print("\n" + "-"*80)
    print(f"{'Metric':<30} {'Score':<15}")
    print("-"*80)
    
    for metric, score in results['metrics'].items():
        print(f"{metric:<30} {score:>14.4f}")
    
    print("="*80 + "\n")
    
    # Print per-predicate breakdown
    if 'predicate_breakdown' in results:
        print("\nPer-Predicate Breakdown:")
        print("-"*80)
        print(f"{'Predicate':<30} {'GT Count':<15} {'Matched':<15}")
        print("-"*80)
        
        for pred, stats in sorted(results['predicate_breakdown'].items()):
            print(f"{pred:<30} {stats['gt_count']:<15} {stats['matched']:<15}")
        
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate PVSG relation predictions')
    parser.add_argument('--gt', required=True, help='Path to ground truth (pvsg.json)')
    parser.add_argument('--predictions', required=True, help='Path to predictions (aligned format)')
    parser.add_argument('--video_id', required=True, help='Video ID to evaluate')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='Temporal IoU threshold')
    parser.add_argument('--output', help='Output path for results JSON')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading ground truth for video {args.video_id}...")
    gt_data = load_pvsg_data(args.gt, args.video_id)
    
    print(f"Loading predictions...")
    pred_data = load_pvsg_data(args.predictions, args.video_id)
    
    gt_relations = gt_data['relations']
    pred_relations = pred_data['relations']
    
    print(f"Ground truth: {len(gt_relations)} relations")
    print(f"Predictions: {len(pred_relations)} relations")
    
    # Compute metrics
    print("\nComputing metrics...")
    
    results = {
        'video_id': args.video_id,
        'num_gt_relations': len(gt_relations),
        'num_pred_relations': len(pred_relations),
        'metrics': {}
    }
    
    # Recall@K
    for k in [20, 50, 100]:
        recall = compute_recall_at_k(pred_relations, gt_relations, k, args.iou_threshold)
        results['metrics'][f'R@{k}'] = recall
    
    # Mean Recall@K
    for k in [20, 50, 100]:
        mean_recall = compute_mean_recall_at_k(pred_relations, gt_relations, k, args.iou_threshold)
        results['metrics'][f'mR@{k}'] = mean_recall
    
    # Per-predicate breakdown
    _, predicate_matches = match_relations(pred_relations, gt_relations, args.iou_threshold)
    
    gt_by_predicate = defaultdict(int)
    for gt in gt_relations:
        gt_by_predicate[gt[2]] += 1
    
    results['predicate_breakdown'] = {}
    for predicate, gt_count in gt_by_predicate.items():
        results['predicate_breakdown'][predicate] = {
            'gt_count': gt_count,
            'matched': predicate_matches.get(predicate, 0)
        }
    
    # Print results
    print_evaluation_results(results)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
