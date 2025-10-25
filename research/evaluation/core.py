"""
Orion Evaluation Framework - Core Implementation
================================================

This module provides the core evaluation infrastructure for benchmarking
Orion against baselines and computing standard metrics.

Author: Orion Research Team
Date: October 2025
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation metrics"""
    precision: float
    recall: float
    f1: float
    accuracy: float
    confusion_matrix: np.ndarray
    per_class_metrics: Dict[str, Dict[str, float]]
    
    def __str__(self):
        return (f"Precision: {self.precision:.3f}, "
                f"Recall: {self.recall:.3f}, "
                f"F1: {self.f1:.3f}, "
                f"Accuracy: {self.accuracy:.3f}")


class ClassificationEvaluator:
    """Evaluates object classification performance"""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        
    def evaluate(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> EvaluationResult:
        """
        Evaluate classification predictions against ground truth
        
        Args:
            predictions: List of predicted class names
            ground_truth: List of ground truth class names
            
        Returns:
            EvaluationResult with metrics
        """
        # Convert to indices for sklearn
        class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        pred_indices = [class_to_idx.get(p, -1) for p in predictions]
        gt_indices = [class_to_idx.get(g, -1) for g in ground_truth]
        
        # Filter out unknown classes
        valid_mask = np.logical_and(
            np.array(pred_indices) >= 0,
            np.array(gt_indices) >= 0
        )
        pred_indices = np.array(pred_indices)[valid_mask]
        gt_indices = np.array(gt_indices)[valid_mask]
        
        # Compute metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            gt_indices,
            pred_indices,
            average='weighted',
            zero_division=0
        )
        
        accuracy = accuracy_score(gt_indices, pred_indices)
        conf_matrix = confusion_matrix(gt_indices, pred_indices)
        
        # Per-class metrics
        per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
            gt_indices,
            pred_indices,
            average=None,
            zero_division=0
        )
        
        per_class_metrics = {}
        for idx, class_name in enumerate(self.class_names):
            if idx < len(per_class_p):
                per_class_metrics[class_name] = {
                    'precision': float(per_class_p[idx]),
                    'recall': float(per_class_r[idx]),
                    'f1': float(per_class_f1[idx])
                }
        
        return EvaluationResult(
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            accuracy=float(accuracy),
            confusion_matrix=conf_matrix,
            per_class_metrics=per_class_metrics
        )
    
    def plot_confusion_matrix(
        self,
        result: EvaluationResult,
        save_path: Optional[Path] = None,
        normalize: bool = True
    ):
        """Plot confusion matrix"""
        cm = result.confusion_matrix
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=False,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        else:
            plt.show()
        plt.close()


class CorrectionEvaluator:
    """Evaluates correction quality (Orion-specific)"""
    
    def evaluate_corrections(
        self,
        corrections: List[Dict],
        ground_truth: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Evaluate correction quality
        
        Args:
            corrections: List of corrections with format:
                {
                    'entity_id': str,
                    'original_class': str,
                    'corrected_class': str,
                    'was_corrected': bool,
                    'correction_confidence': float
                }
            ground_truth: Dict mapping entity_id -> true class
            
        Returns:
            Dict of correction metrics
        """
        true_corrections = 0
        false_corrections = 0
        missed_corrections = 0
        correct_no_correction = 0
        
        for corr in corrections:
            entity_id = corr['entity_id']
            original = corr['original_class']
            corrected = corr.get('corrected_class', original)
            was_corrected = corr.get('was_corrected', False)
            true_class = ground_truth.get(entity_id)
            
            if true_class is None:
                continue
            
            if was_corrected:
                if corrected == true_class:
                    true_corrections += 1
                else:
                    false_corrections += 1
            else:
                if original == true_class:
                    correct_no_correction += 1
                else:
                    missed_corrections += 1
        
        # Compute metrics
        total_corrections = true_corrections + false_corrections
        correction_precision = (
            true_corrections / total_corrections if total_corrections > 0 else 0
        )
        
        total_misclassifications = true_corrections + missed_corrections
        correction_recall = (
            true_corrections / total_misclassifications 
            if total_misclassifications > 0 else 0
        )
        
        correction_f1 = (
            2 * correction_precision * correction_recall / 
            (correction_precision + correction_recall)
            if (correction_precision + correction_recall) > 0 else 0
        )
        
        return {
            'correction_precision': correction_precision,
            'correction_recall': correction_recall,
            'correction_f1': correction_f1,
            'true_corrections': true_corrections,
            'false_corrections': false_corrections,
            'missed_corrections': missed_corrections,
            'correct_no_correction': correct_no_correction,
            'total_corrections': total_corrections
        }


class SceneGraphEvaluator:
    """Evaluates scene graph generation (VSGR-style)"""
    
    def __init__(self):
        pass
    
    def compute_recall_at_k(
        self,
        predictions: List[Tuple[str, str, str]],
        ground_truth: List[Tuple[str, str, str]],
        k: int = 20
    ) -> float:
        """
        Compute Recall@K for scene graph triplets
        
        Args:
            predictions: List of (subject, predicate, object) triplets, ranked by confidence
            ground_truth: List of ground truth triplets
            k: Top-K to consider
            
        Returns:
            Recall@K score
        """
        if not ground_truth:
            return 0.0
        
        # Take top-K predictions
        top_k_predictions = set(predictions[:k])
        gt_set = set(ground_truth)
        
        # Count matches
        matches = len(top_k_predictions & gt_set)
        
        # Recall = matches / total ground truth
        recall = matches / len(gt_set)
        
        return recall
    
    def evaluate_scene_graphs(
        self,
        predictions: Dict[str, List[Tuple]],
        ground_truth: Dict[str, List[Tuple]],
        k_values: List[int] = [20, 50, 100]
    ) -> Dict[str, float]:
        """
        Evaluate scene graphs across multiple videos
        
        Args:
            predictions: Dict mapping video_id -> list of triplets
            ground_truth: Dict mapping video_id -> list of ground truth triplets
            k_values: List of K values to compute recall for
            
        Returns:
            Dict of Recall@K metrics
        """
        results = {f'recall@{k}': [] for k in k_values}
        
        for video_id in ground_truth.keys():
            if video_id not in predictions:
                # No predictions for this video
                for k in k_values:
                    results[f'recall@{k}'].append(0.0)
                continue
            
            pred_triplets = predictions[video_id]
            gt_triplets = ground_truth[video_id]
            
            for k in k_values:
                recall = self.compute_recall_at_k(pred_triplets, gt_triplets, k)
                results[f'recall@{k}'].append(recall)
        
        # Average across videos
        avg_results = {
            metric: np.mean(values) if values else 0.0
            for metric, values in results.items()
        }
        
        return avg_results


class BaselineComparator:
    """Compare Orion against baselines"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def compare_methods(
        self,
        methods_results: Dict[str, EvaluationResult]
    ) -> Dict:
        """
        Compare multiple methods
        
        Args:
            methods_results: Dict mapping method_name -> EvaluationResult
            
        Returns:
            Comparison dict with improvements
        """
        baseline_name = list(methods_results.keys())[0]
        baseline = methods_results[baseline_name]
        
        comparison = {
            'baseline': baseline_name,
            'methods': {}
        }
        
        for method_name, result in methods_results.items():
            comparison['methods'][method_name] = {
                'precision': result.precision,
                'recall': result.recall,
                'f1': result.f1,
                'accuracy': result.accuracy,
            }
            
            if method_name != baseline_name:
                # Compute improvements
                comparison['methods'][method_name]['improvements'] = {
                    'precision_improvement': (
                        (result.precision - baseline.precision) / baseline.precision * 100
                    ),
                    'recall_improvement': (
                        (result.recall - baseline.recall) / baseline.recall * 100
                    ),
                    'f1_improvement': (
                        (result.f1 - baseline.f1) / baseline.f1 * 100
                    ),
                }
        
        return comparison
    
    def save_comparison(self, comparison: Dict, filename: str = "comparison.json"):
        """Save comparison results to JSON"""
        output_path = self.results_dir / filename
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Saved comparison to {output_path}")
    
    def plot_comparison(
        self,
        methods_results: Dict[str, EvaluationResult],
        metrics: List[str] = ['precision', 'recall', 'f1', 'accuracy'],
        save_path: Optional[Path] = None
    ):
        """Plot comparison bar chart"""
        method_names = list(methods_results.keys())
        metric_values = {
            metric: [getattr(methods_results[m], metric) for m in method_names]
            for metric in metrics
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = metric_values[metric]
            bars = ax.bar(method_names, values, alpha=0.7)
            
            # Color best method
            best_idx = np.argmax(values)
            bars[best_idx].set_color('green')
            
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.set_ylim([0, 1.0])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9
                )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")
        else:
            plt.show()
        plt.close()


# Example usage
if __name__ == "__main__":
    # Example: Evaluate classification
    from orion.config import OrionConfig
    
    logging.basicConfig(level=logging.INFO)
    
    # Mock data for demonstration
    class_names = ['person', 'car', 'bicycle', 'dog', 'cat']
    
    # Ground truth
    ground_truth = ['person', 'car', 'bicycle', 'dog', 'cat'] * 20
    
    # Predictions (with some errors)
    predictions_baseline = ground_truth.copy()
    # Add 20% errors
    for i in range(0, len(predictions_baseline), 5):
        predictions_baseline[i] = class_names[(class_names.index(predictions_baseline[i]) + 1) % len(class_names)]
    
    predictions_orion = ground_truth.copy()
    # Add only 5% errors
    for i in range(0, len(predictions_orion), 20):
        predictions_orion[i] = class_names[(class_names.index(predictions_orion[i]) + 1) % len(class_names)]
    
    # Evaluate
    evaluator = ClassificationEvaluator(class_names)
    
    result_baseline = evaluator.evaluate(predictions_baseline, ground_truth)
    result_orion = evaluator.evaluate(predictions_orion, ground_truth)
    
    print("Baseline Results:")
    print(result_baseline)
    print("\nOrion Results:")
    print(result_orion)
    
    # Compare
    comparator = BaselineComparator(Path("evaluation_results"))
    comparison = comparator.compare_methods({
        'YOLO Baseline': result_baseline,
        'Orion': result_orion
    })
    
    print("\nComparison:")
    print(json.dumps(comparison, indent=2))
    
    # Plot
    comparator.plot_comparison(
        {'YOLO Baseline': result_baseline, 'Orion': result_orion},
        save_path=Path("evaluation_results/comparison.png")
    )
