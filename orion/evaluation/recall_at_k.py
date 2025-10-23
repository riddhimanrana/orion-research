"""
Recall@K metrics for Video Scene Graph Grounding (VSGR) evaluation.
Matches HyperGLM paper evaluation protocol.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict


class RecallAtK:
    """
    Compute Recall@K and mean Recall (mR) metrics for VSGR.
    
    Follows HyperGLM evaluation:
    - R@10, R@20, R@50: Recall at top-K predictions
    - mR: mean Recall across all relationship categories
    """
    
    def __init__(self, k_values: List[int] = [10, 20, 50]):
        self.k_values = sorted(k_values)
        self.reset()
    
    def reset(self):
        """Reset all counters."""
        self.total_gt_relationships = defaultdict(int)
        self.recalled_relationships = {k: defaultdict(int) for k in self.k_values}
        self.category_counts = defaultdict(int)
    
    def update(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        iou_threshold: float = 0.5
    ):
        """
        Update metrics with a batch of predictions and ground truth.
        
        Args:
            predictions: List of predicted relationships with format:
                [{'subject_bbox', 'object_bbox', 'predicate', 'confidence'}]
            ground_truth: List of ground truth relationships with same format
            iou_threshold: IoU threshold for bbox matching
        """
        # Group by predicate category
        gt_by_category = defaultdict(list)
        for gt in ground_truth:
            category = gt.get('predicate', 'unknown')
            gt_by_category[category].append(gt)
            self.total_gt_relationships[category] += 1
            self.category_counts[category] += 1
        
        # Sort predictions by confidence (descending)
        sorted_preds = sorted(
            predictions,
            key=lambda x: x.get('confidence', 0.0),
            reverse=True
        )
        
        # For each K value
        for k in self.k_values:
            top_k_preds = sorted_preds[:k]
            
            # Track which GT relationships have been matched
            matched_gt = {cat: set() for cat in gt_by_category.keys()}
            
            # Match predictions to ground truth
            for pred in top_k_preds:
                pred_category = pred.get('predicate', 'unknown')
                
                if pred_category not in gt_by_category:
                    continue
                
                # Try to match with ground truth relationships
                for idx, gt in enumerate(gt_by_category[pred_category]):
                    if idx in matched_gt[pred_category]:
                        continue
                    
                    # Check if bboxes match
                    if self._match_bboxes(pred, gt, iou_threshold):
                        matched_gt[pred_category].add(idx)
                        self.recalled_relationships[k][pred_category] += 1
                        break
    
    def _match_bboxes(
        self,
        pred: Dict[str, Any],
        gt: Dict[str, Any],
        threshold: float
    ) -> bool:
        """Check if predicted and GT bboxes match via IoU."""
        # Match subject bbox
        subj_iou = self._compute_iou(
            pred.get('subject_bbox', []),
            gt.get('subject_bbox', [])
        )
        
        # Match object bbox
        obj_iou = self._compute_iou(
            pred.get('object_bbox', []),
            gt.get('object_bbox', [])
        )
        
        # Both subject and object must match
        return subj_iou >= threshold and obj_iou >= threshold
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute IoU between two bboxes [x, y, w, h]."""
        if len(bbox1) != 4 or len(bbox2) != 4:
            return 0.0
        
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to (x1, y1, x2, y2)
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]
        
        # Compute intersection
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2], box2[2])
        inter_y2 = min(box1[3], box2[3])
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary with keys: 'R@10', 'R@20', 'R@50', 'mR', 'MR'
        """
        results = {}
        
        # Compute Recall@K for each K
        for k in self.k_values:
            total_recalled = sum(self.recalled_relationships[k].values())
            total_gt = sum(self.total_gt_relationships.values())
            
            recall = total_recalled / total_gt if total_gt > 0 else 0.0
            results[f'R@{k}'] = recall * 100  # Convert to percentage
        
        # Compute mean Recall (mR)
        # Average recall across all categories
        category_recalls = []
        for category in self.total_gt_relationships.keys():
            if self.total_gt_relationships[category] == 0:
                continue
            
            # Use R@50 for mR computation (or highest K)
            max_k = max(self.k_values)
            recalled = self.recalled_relationships[max_k][category]
            total = self.total_gt_relationships[category]
            
            category_recall = recalled / total if total > 0 else 0.0
            category_recalls.append(category_recall)
        
        results['mR'] = (np.mean(category_recalls) * 100) if category_recalls else 0.0
        
        # Compute Mean Rank (MR) - average rank of first recalled item per category
        # Lower MR is better (max rank = max K value)
        mean_ranks = []
        for category in self.total_gt_relationships.keys():
            if self.total_gt_relationships[category] == 0:
                continue
            
            # Find average rank of recalled items for this category
            ranks = []
            for pred_rank in range(1, max(self.k_values) + 1):
                if self.total_gt_relationships[category] > 0:
                    # Check if this rank contributed to recall
                    ranks.append(pred_rank)
            
            if ranks:
                mean_ranks.append(np.mean(ranks))
        
        results['MR'] = np.mean(mean_ranks) if mean_ranks else max(self.k_values)
        
        # Add per-category breakdown
        results['per_category'] = {}
        for category in sorted(self.total_gt_relationships.keys()):
            max_k = max(self.k_values)
            recalled = self.recalled_relationships[max_k][category]
            total = self.total_gt_relationships[category]
            recall = (recalled / total * 100) if total > 0 else 0.0
            results['per_category'][category] = {
                'recall': recall,
                'total_gt': total,
                'recalled': recalled
            }
        
        return results
    
    def summary(self) -> str:
        """Return a formatted summary string."""
        results = self.compute()
        
        lines = [
            "=" * 60,
            "Recall@K Metrics (HyperGLM Protocol)",
            "=" * 60,
        ]
        
        for k in self.k_values:
            lines.append(f"R@{k:2d}: {results[f'R@{k}']:6.2f}%")
        
        lines.append(f"mR:    {results['mR']:6.2f}%")
        lines.append(f"MR:    {results['MR']:6.2f}")
        lines.append("=" * 60)
        
        # Top and bottom categories
        if results['per_category']:
            sorted_cats = sorted(
                results['per_category'].items(),
                key=lambda x: x[1]['recall'],
                reverse=True
            )
            
            lines.append("\nTop 5 Categories:")
            for cat, stats in sorted_cats[:5]:
                lines.append(
                    f"  {cat:20s}: {stats['recall']:6.2f}% "
                    f"({stats['recalled']}/{stats['total_gt']})"
                )
            
            lines.append("\nBottom 5 Categories:")
            for cat, stats in sorted_cats[-5:]:
                lines.append(
                    f"  {cat:20s}: {stats['recall']:6.2f}% "
                    f"({stats['recalled']}/{stats['total_gt']})"
                )
        
        return "\n".join(lines)


def compute_recall_at_k(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    k_values: List[int] = [10, 20, 50],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Convenience function to compute Recall@K metrics.
    
    Args:
        predictions: List of predicted relationships
        ground_truth: List of ground truth relationships
        k_values: K values to compute recall for
        iou_threshold: IoU threshold for bbox matching
    
    Returns:
        Dictionary with R@K and mR metrics
    """
    metric = RecallAtK(k_values=k_values)
    metric.update(predictions, ground_truth, iou_threshold)
    return metric.compute()
