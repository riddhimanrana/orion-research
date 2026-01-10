"""
ActionGenome Scene Graph Anticipation (SGA) Evaluator
======================================================

Evaluates the ability to predict future scene graphs given a pruned video.

Task: Given frames 0 to t (e.g., first 50% of video), predict scene graphs
      at frames t+1, t+2, ... (remaining frames)

Uses Recall@K and anticipation accuracy metrics.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnticipationMetrics:
    """Metrics for scene graph anticipation task."""
    video_id: str
    prune_ratio: float  # 0.5 = first 50% given, predict remaining 50%
    predicted_future_sgs: Dict[int, List['SceneGraphTriplet']]
    ground_truth_future_sgs: Dict[int, List['SceneGraphTriplet']]
    
    # Metrics per frame
    recall_at_k_per_frame: Dict[int, Dict[int, float]]  # frame -> {k -> recall}
    precision_per_frame: Dict[int, float]
    
    # Aggregated metrics
    mean_recall_at_k: Dict[int, float]  # {1: 0.5, 5: 0.7, 10: 0.8}
    mean_precision: float
    anticipation_success_rate: float  # % of frames where predictions were close


class ActionGenomeDataLoader:
    """Loads ActionGenome dataset annotations."""
    
    def __init__(self, dataset_root: Path = None):
        """
        Initialize ActionGenome loader.
        
        Args:
            dataset_root: Root directory of ActionGenome dataset
        """
        if dataset_root is None:
            dataset_root = Path("/Users/yogeshatluru/orion-research/datasets/ActionGenome")
        
        self.root = Path(dataset_root)
        self.videos = {}
        
        if self.root.exists():
            self._load_annotations()
        else:
            logger.warning(f"ActionGenome dataset not found at {self.root}")
    
    def _load_annotations(self):
        """Load scene graph annotations from ActionGenome."""
        # This would load the actual ActionGenome data structure
        # For now, providing interface
        pass
    
    def get_video(self, video_id: str) -> Dict[int, Any]:
        """Get scene graphs for all frames in a video."""
        return self.videos.get(video_id, {})


class SGAEvaluator:
    """
    Evaluates Scene Graph Anticipation (predicting future scene graphs).
    
    Strategy:
    1. Given frames 0 to t (prune_ratio % of frames)
    2. Model predicts scene graphs for remaining frames
    3. Compare predictions against ground truth
    4. Measure how well future scenes are anticipated
    """
    
    def __init__(self, dataset_root: Path = None):
        """Initialize SGA evaluator."""
        self.loader = ActionGenomeDataLoader(dataset_root)
        self.videos = self.loader.videos
    
    def compute_anticipation_recall_at_k(
        self,
        predicted: List['SceneGraphTriplet'],
        ground_truth: List['SceneGraphTriplet'],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[int, float]:
        """
        Compute Recall@K for anticipated future scene graphs.
        
        Similar to PVSG but focuses on how well model predicts
        what WILL happen (not just what is happening).
        """
        if not ground_truth:
            return {k: 1.0 for k in k_values}
        
        gt_set = set(ground_truth)
        results = {}
        
        for k in k_values:
            top_k_preds = set(predicted[:k])
            matches = len(gt_set & top_k_preds)
            recall = matches / len(gt_set)
            results[k] = recall
        
        return results
    
    def compute_anticipation_success(
        self,
        predicted: List['SceneGraphTriplet'],
        ground_truth: List['SceneGraphTriplet'],
        threshold: float = 0.5
    ) -> bool:
        """
        Check if anticipation was successful (Recall@10 >= threshold).
        
        Args:
            predicted: Predicted future triplets
            ground_truth: Ground truth future triplets
            threshold: Success threshold (default: 50% recall)
            
        Returns:
            True if anticipation was successful
        """
        if not ground_truth:
            return True
        
        recall_at_10 = self.compute_anticipation_recall_at_k(
            predicted, ground_truth, k_values=[10]
        )[10]
        
        return recall_at_10 >= threshold
    
    def evaluate_video_anticipation(
        self,
        video_id: str,
        prune_ratio: float = 0.5,
        predicted_future_sgs: Optional[Dict[int, List['SceneGraphTriplet']]] = None
    ) -> AnticipationMetrics:
        """
        Evaluate anticipation for a single video.
        
        Args:
            video_id: Video identifier
            prune_ratio: Ratio of frames given (rest must be anticipated)
                        0.5 = first 50% given, predict remaining 50%
            predicted_future_sgs: Model's future predictions
                                 Dict[frame_id -> List[triplets]]
            
        Returns:
            Anticipation metrics
        """
        # Get video scene graphs
        video_sgs = self.loader.get_video(video_id)
        
        # Split into past (known) and future (to anticipate)
        total_frames = len(video_sgs)
        cutoff_frame = int(total_frames * prune_ratio)
        
        future_sgs = {
            fid: triplets for fid, triplets in video_sgs.items()
            if fid > cutoff_frame
        }
        
        # Compute metrics
        recall_at_k_per_frame = {}
        precision_per_frame = {}
        success_frames = 0
        
        for frame_id, gt_triplets in future_sgs.items():
            predicted = predicted_future_sgs.get(frame_id, [])
            
            # Recall@K
            recall_at_k = self.compute_anticipation_recall_at_k(
                predicted, gt_triplets, k_values=[1, 5, 10]
            )
            recall_at_k_per_frame[frame_id] = recall_at_k
            
            # Precision
            if predicted:
                correct = sum(1 for p in predicted if p in set(gt_triplets))
                precision = correct / len(predicted)
            else:
                precision = 1.0 if not gt_triplets else 0.0
            
            precision_per_frame[frame_id] = precision
            
            # Success
            if self.compute_anticipation_success(predicted, gt_triplets):
                success_frames += 1
        
        # Aggregate
        mean_recall_at_k = defaultdict(list)
        for frame_id, recalls in recall_at_k_per_frame.items():
            for k, recall in recalls.items():
                mean_recall_at_k[k].append(recall)
        
        mean_recall_at_k = {
            k: np.mean(v) if v else 0.0
            for k, v in mean_recall_at_k.items()
        }
        
        mean_precision = np.mean(list(precision_per_frame.values())) if precision_per_frame else 0.0
        success_rate = success_frames / len(future_sgs) if future_sgs else 0.0
        
        return AnticipationMetrics(
            video_id=video_id,
            prune_ratio=prune_ratio,
            predicted_future_sgs=predicted_future_sgs or {},
            ground_truth_future_sgs=future_sgs,
            recall_at_k_per_frame=recall_at_k_per_frame,
            precision_per_frame=precision_per_frame,
            mean_recall_at_k=mean_recall_at_k,
            mean_precision=mean_precision,
            anticipation_success_rate=success_rate
        )
    
    def evaluate_batch(
        self,
        predictions: List[Tuple[str, float, Dict[int, List['SceneGraphTriplet']]]]
    ) -> Dict[str, Any]:
        """
        Evaluate anticipation on a batch of videos.
        
        Args:
            predictions: List of (video_id, prune_ratio, future_sgs)
            
        Returns:
            Aggregate metrics
        """
        results = []
        recall_at_k_list = defaultdict(list)
        precisions = []
        success_rates = []
        
        for video_id, prune_ratio, predicted_future in predictions:
            result = self.evaluate_video_anticipation(
                video_id, prune_ratio, predicted_future
            )
            results.append(result)
            
            # Aggregate
            for k, recall in result.mean_recall_at_k.items():
                recall_at_k_list[k].append(recall)
            precisions.append(result.mean_precision)
            success_rates.append(result.anticipation_success_rate)
        
        # Summary
        aggregate = {
            'num_videos': len(results),
            'mean_recall_at_1': np.mean(recall_at_k_list[1]) if recall_at_k_list[1] else 0.0,
            'mean_recall_at_5': np.mean(recall_at_k_list[5]) if recall_at_k_list[5] else 0.0,
            'mean_recall_at_10': np.mean(recall_at_k_list[10]) if recall_at_k_list[10] else 0.0,
            'mean_precision': np.mean(precisions) if precisions else 0.0,
            'anticipation_success_rate': np.mean(success_rates) if success_rates else 0.0,
        }
        
        logger.info(f"✓ Evaluated anticipation for {len(results)} videos")
        logger.info(f"  - Recall@1: {aggregate['mean_recall_at_1']:.3f}")
        logger.info(f"  - Recall@5: {aggregate['mean_recall_at_5']:.3f}")
        logger.info(f"  - Recall@10: {aggregate['mean_recall_at_10']:.3f}")
        logger.info(f"  - Mean Precision: {aggregate['mean_precision']:.3f}")
        logger.info(f"  - Success Rate (Recall@10 >= 0.5): {aggregate['anticipation_success_rate']:.3f}")
        
        return aggregate
