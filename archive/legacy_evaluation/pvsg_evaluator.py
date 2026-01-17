"""
PVSG Scene Graph Generation Evaluator
======================================

Evaluates scene graph generation quality on PVSG dataset.
Uses Recall@K metric to compare with HyperGLM baseline.

Supports:
- Gemini 3.5-flash (paper version - stronger results)
- FastVLM (lightweight version - faster inference)
- DINOv3 + Faster-RCNN detection backends
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
class SceneGraphTriplet:
    """Scene graph as (subject, predicate, object) triplet."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    
    def __hash__(self):
        return hash((self.subject, self.predicate, self.object))
    
    def __eq__(self, other):
        return (self.subject == other.subject and 
                self.predicate == other.predicate and 
                self.object == other.object)


@dataclass
class PVSGEvaluationResult:
    """Results for a single video evaluation."""
    video_id: str
    frame_id: int
    predicted_triplets: List[SceneGraphTriplet]
    ground_truth_triplets: List[SceneGraphTriplet]
    recall_at_k: Dict[int, float]  # recall@1, recall@5, recall@10
    precision: float
    f1_score: float


class PVSGEvaluator:
    """
    Evaluates scene graph generation using PVSG dataset.
    
    Metrics:
    - Recall@K: What % of ground truth triplets are in top-K predictions
    - Precision: What % of predictions are correct
    - F1-Score: Harmonic mean of precision/recall
    """
    
    def __init__(self, pvsg_root: Path = None):
        """
        Initialize PVSG evaluator.
        
        Args:
            pvsg_root: Path to PVSG dataset root directory
                      (default: datasets/PVSG)
        """
        if pvsg_root is None:
            pvsg_root = Path("/Users/yogeshatluru/orion-research/datasets/PVSG")
        
        self.pvsg_root = Path(pvsg_root)
        self.pvsg_json_path = self.pvsg_root / "pvsg.json"
        
        # Load PVSG annotations
        self.pvsg_data = self._load_pvsg()
        self.videos = list(self.pvsg_data.keys())
        
        logger.info(f"✓ Loaded PVSG with {len(self.videos)} videos")
    
    def _load_pvsg(self) -> Dict[str, Any]:
        """Load PVSG annotations."""
        if not self.pvsg_json_path.exists():
            raise FileNotFoundError(f"PVSG data not found at {self.pvsg_json_path}")
        
        with open(self.pvsg_json_path, 'r') as f:
            return json.load(f)
    
    def get_video_scene_graphs(self, video_id: str) -> Dict[int, List[SceneGraphTriplet]]:
        """
        Extract ground truth scene graphs for a video.
        
        Returns:
            Dict mapping frame_id -> list of triplets
        """
        if video_id not in self.pvsg_data:
            raise ValueError(f"Video {video_id} not found in PVSG")
        
        video_data = self.pvsg_data[video_id]
        frame_sgs = defaultdict(list)
        
        # Parse relationship annotations
        for relationship in video_data.get('relationships', []):
            frame_id = relationship.get('frame_id', 0)
            
            triplet = SceneGraphTriplet(
                subject=relationship.get('subject_name', 'unknown'),
                predicate=relationship.get('predicate', 'unknown'),
                object=relationship.get('object_name', 'unknown'),
                confidence=relationship.get('confidence', 1.0)
            )
            
            frame_sgs[frame_id].append(triplet)
        
        return dict(frame_sgs)
    
    def compute_recall_at_k(
        self,
        predicted: List[SceneGraphTriplet],
        ground_truth: List[SceneGraphTriplet],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[int, float]:
        """
        Compute Recall@K metric.
        
        Recall@K = (# of GT triplets in top-K predictions) / (total GT triplets)
        
        Args:
            predicted: Predicted triplets (should be sorted by confidence desc)
            ground_truth: Ground truth triplets
            k_values: K values to compute [1, 5, 10, ...]
            
        Returns:
            Dict mapping k -> recall@k score (0-1)
        """
        if not ground_truth:
            return {k: 1.0 for k in k_values}  # Perfect if no GT
        
        gt_set = set(ground_truth)
        results = {}
        
        for k in k_values:
            top_k_preds = set(predicted[:k])
            matches = len(gt_set & top_k_preds)
            recall = matches / len(gt_set)
            results[k] = recall
        
        return results
    
    def compute_precision(
        self,
        predicted: List[SceneGraphTriplet],
        ground_truth: List[SceneGraphTriplet]
    ) -> float:
        """
        Compute precision: (# correct predictions) / (total predictions)
        """
        if not predicted:
            return 1.0 if not ground_truth else 0.0
        
        gt_set = set(ground_truth)
        correct = sum(1 for p in predicted if p in gt_set)
        return correct / len(predicted)
    
    def compute_f1(self, precision: float, recall: float) -> float:
        """Compute F1-score."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def evaluate_predictions(
        self,
        video_id: str,
        frame_id: int,
        predicted_triplets: List[SceneGraphTriplet]
    ) -> PVSGEvaluationResult:
        """
        Evaluate predicted scene graph against ground truth.
        
        Args:
            video_id: Video identifier
            frame_id: Frame number
            predicted_triplets: Model predictions (should be sorted by confidence)
            
        Returns:
            Evaluation result with metrics
        """
        # Get ground truth
        video_sgs = self.get_video_scene_graphs(video_id)
        gt_triplets = video_sgs.get(frame_id, [])
        
        # Compute metrics
        recall_at_k = self.compute_recall_at_k(predicted_triplets, gt_triplets)
        precision = self.compute_precision(predicted_triplets, gt_triplets)
        
        # Use recall@10 for F1
        recall = recall_at_k.get(10, 0.0)
        f1 = self.compute_f1(precision, recall)
        
        return PVSGEvaluationResult(
            video_id=video_id,
            frame_id=frame_id,
            predicted_triplets=predicted_triplets,
            ground_truth_triplets=gt_triplets,
            recall_at_k=recall_at_k,
            precision=precision,
            f1_score=f1
        )
    
    def evaluate_batch(
        self,
        predictions: List[Tuple[str, int, List[SceneGraphTriplet]]]
    ) -> Tuple[Dict[str, float], List[PVSGEvaluationResult]]:
        """
        Evaluate a batch of predictions.
        
        Args:
            predictions: List of (video_id, frame_id, triplets)
            
        Returns:
            (aggregate_metrics, detailed_results)
        """
        results = []
        recall_at_k_list = defaultdict(list)
        precisions = []
        f1_scores = []
        
        for video_id, frame_id, predicted in predictions:
            result = self.evaluate_predictions(video_id, frame_id, predicted)
            results.append(result)
            
            # Aggregate metrics
            for k, recall in result.recall_at_k.items():
                recall_at_k_list[k].append(recall)
            precisions.append(result.precision)
            f1_scores.append(result.f1_score)
        
        # Compute averages
        aggregate = {
            'num_samples': len(results),
            'mean_precision': np.mean(precisions) if precisions else 0.0,
            'mean_f1': np.mean(f1_scores) if f1_scores else 0.0,
        }
        
        # Add recall@k averages
        for k in [1, 5, 10]:
            if recall_at_k_list[k]:
                aggregate[f'recall@{k}'] = np.mean(recall_at_k_list[k])
        
        logger.info(f"✓ Evaluated {len(results)} predictions")
        logger.info(f"  - Recall@1: {aggregate.get('recall@1', 0):.3f}")
        logger.info(f"  - Recall@5: {aggregate.get('recall@5', 0):.3f}")
        logger.info(f"  - Recall@10: {aggregate.get('recall@10', 0):.3f}")
        logger.info(f"  - Mean Precision: {aggregate['mean_precision']:.3f}")
        logger.info(f"  - Mean F1: {aggregate['mean_f1']:.3f}")
        
        return aggregate, results


if __name__ == "__main__":
    # Example usage
    evaluator = PVSGEvaluator()
    
    # Get available videos
    print(f"Available videos: {evaluator.videos[:5]}...")
    
    # Example: Evaluate on first video
    if evaluator.videos:
        video_id = evaluator.videos[0]
        sgs = evaluator.get_video_scene_graphs(video_id)
        print(f"\n{video_id}: {len(sgs)} frames with scene graphs")
        
        # Show example
        if sgs:
            frame_id = list(sgs.keys())[0]
            triplets = sgs[frame_id]
            print(f"  Frame {frame_id}: {len(triplets)} triplets")
            for t in triplets[:3]:
                print(f"    - ({t.subject}, {t.predicate}, {t.object})")
