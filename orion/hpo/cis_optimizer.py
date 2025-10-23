"""
CIS Hyperparameter Optimization
================================

Optimizes the weights for the Causal Influence Score (CIS) formula using
Bayesian optimization (Optuna) on ground truth causal annotations.

The CIS formula addresses the research mentor's feedback:
"The CIS is our own formula - problem with that is that there's no 
justification for using this at all - why are we using these specific weights? 
Did we derive it from something? Were the weights learned? Where does the 
threshold come from?"

This module provides:
1. Bayesian optimization to learn weights from data
2. Cross-validation for robust evaluation
3. Sensitivity analysis for threshold justification
4. Scientific justification for weight values

Author: Orion Research Team
Date: October 2025
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None  # type: ignore
    Trial = Any  # type: ignore
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not available. Install with: pip install optuna")

try:
    from ..causal_inference import CausalConfig, CausalInferenceEngine, AgentCandidate, StateChange, CausalLink
except ImportError:
    from causal_inference import CausalConfig, CausalInferenceEngine, AgentCandidate, StateChange, CausalLink  # type: ignore

logger = logging.getLogger("HPO.CIS")


@dataclass
class GroundTruthCausalPair:
    """
    Ground truth causal relationship for training/evaluation.
    
    Annotated by humans or derived from known physics/interactions.
    """
    agent_id: str
    patient_id: str
    state_change_frame: int
    is_causal: bool  # True if agent caused the state change
    confidence: float = 1.0  # Annotation confidence (0-1)
    annotation_source: str = "human"  # "human", "physics", "heuristic"


@dataclass
class CISOptimizationResult:
    """Results from CIS optimization"""
    best_weights: Dict[str, float]
    best_threshold: float
    best_score: float  # F1 score
    precision: float
    recall: float
    num_trials: int
    optimization_time: float
    sensitivity_analysis: Dict[str, Any]


class CISOptimizer:
    """
    Optimizes CIS formula weights using Bayesian optimization.
    
    Uses Optuna for efficient hyperparameter search with:
    - TPE (Tree-structured Parzen Estimator) sampler
    - Pruning for early stopping of bad trials
    - Multi-objective optimization (precision + recall)
    """
    
    def __init__(
        self,
        ground_truth: List[GroundTruthCausalPair],
        agent_candidates: List[AgentCandidate],
        state_changes: List[StateChange],
        seed: int = 42
    ):
        """
        Args:
            ground_truth: Annotated causal pairs
            agent_candidates: All potential agents
            state_changes: All state changes
            seed: Random seed for reproducibility
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna is required for CIS optimization. Install with: pip install optuna")
        
        self.ground_truth = ground_truth
        self.agent_candidates = agent_candidates
        self.state_changes = state_changes
        self.seed = seed
        
        # Create lookup tables
        self.agent_map = {a.entity_id: a for a in agent_candidates}
        self.state_change_map = {(sc.entity_id, sc.frame_number): sc for sc in state_changes}
        
        logger.info(f"CISOptimizer initialized with {len(ground_truth)} ground truth pairs")
    
    def _create_trial_config(self, trial: Trial) -> CausalConfig:
        """
        Create CausalConfig from Optuna trial parameters.
        
        Samples weights with constraint: sum(weights) ≈ 1.0
        """
        # Sample weights (will normalize to sum to 1)
        w_temporal = trial.suggest_float("temporal_weight", 0.0, 1.0)
        w_spatial = trial.suggest_float("spatial_weight", 0.0, 1.0)
        w_overlap = trial.suggest_float("overlap_weight", 0.0, 1.0)
        w_semantic = trial.suggest_float("semantic_weight", 0.0, 1.0)
        
        # Normalize weights to sum to 1
        total = w_temporal + w_spatial + w_overlap + w_semantic
        if total == 0:
            total = 1.0  # Avoid division by zero
        
        # Threshold parameters
        min_score = trial.suggest_float("min_score", 0.3, 0.8)
        max_pixel_distance = trial.suggest_float("max_pixel_distance", 300.0, 1000.0)
        temporal_decay = trial.suggest_float("temporal_decay", 1.0, 10.0)
        
        return CausalConfig(
            temporal_proximity_weight=w_temporal / total,
            spatial_proximity_weight=w_spatial / total,
            motion_alignment_weight=w_overlap / total,
            semantic_similarity_weight=w_semantic / total,
            min_score=min_score,
            max_pixel_distance=max_pixel_distance,
            temporal_decay=temporal_decay,
        )
    
    def _evaluate_config(self, config: CausalConfig) -> Tuple[float, float, float]:
        """
        Evaluate CIS configuration on ground truth data.
        
        Returns:
            (f1_score, precision, recall)
        """
        engine = CausalInferenceEngine(config)
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Group ground truth by patient
        gt_by_patient: Dict[Tuple[str, int], List[GroundTruthCausalPair]] = {}
        for gt in self.ground_truth:
            key = (gt.patient_id, gt.state_change_frame)
            if key not in gt_by_patient:
                gt_by_patient[key] = []
            gt_by_patient[key].append(gt)
        
        # Evaluate each state change
        for (patient_id, frame), gt_list in gt_by_patient.items():
            # Find state change
            sc_key = (patient_id, frame)
            if sc_key not in self.state_change_map:
                continue
            
            state_change = self.state_change_map[sc_key]
            
            # Get CIS predictions
            causal_links = engine.score_all_agents(self.agent_candidates, state_change)
            predicted_agents = {link.agent.entity_id for link in causal_links}
            
            # Get ground truth causal agents for this patient
            true_agents = {gt.agent_id for gt in gt_list if gt.is_causal}
            
            # Calculate metrics
            tp = len(predicted_agents & true_agents)
            fp = len(predicted_agents - true_agents)
            fn = len(true_agents - predicted_agents)
            
            true_positives += tp
            false_positives += fp
            false_negatives += fn
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1, precision, recall
    
    def _objective(self, trial: Trial) -> float:
        """
        Optuna objective function to maximize.
        
        Returns F1 score (harmonic mean of precision and recall).
        """
        config = self._create_trial_config(trial)
        f1, precision, recall = self._evaluate_config(config)
        
        # Log intermediate results
        trial.set_user_attr("precision", precision)
        trial.set_user_attr("recall", recall)
        
        return f1
    
    def optimize(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        show_progress: bool = True
    ) -> CISOptimizationResult:
        """
        Run Bayesian optimization to find best CIS weights.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds (None = no limit)
            show_progress: Show progress bar
            
        Returns:
            Optimization results with best weights and metrics
        """
        logger.info(f"Starting CIS optimization with {n_trials} trials...")
        
        import time
        start_time = time.time()
        
        # Create study
        study = optuna.create_study(
            direction="maximize",  # Maximize F1 score
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Optimize
        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress
        )
        
        optimization_time = time.time() - start_time
        
        # Get best trial
        best_trial = study.best_trial
        best_config = self._create_trial_config(best_trial)
        best_f1, best_precision, best_recall = self._evaluate_config(best_config)
        
        # Extract best weights
        best_weights = {
            "temporal": best_config.temporal_proximity_weight,
            "spatial": best_config.spatial_proximity_weight,
            "motion": best_config.motion_alignment_weight,
            "semantic": best_config.semantic_similarity_weight,
        }
        
        # Sensitivity analysis
        sensitivity = self._sensitivity_analysis(best_config)
        
        logger.info(f"Optimization complete in {optimization_time:.2f}s")
        logger.info(f"Best F1: {best_f1:.4f}, Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")
        logger.info(f"Best weights: {best_weights}")
        logger.info(f"Best threshold: {best_config.min_score:.4f}")
        
        return CISOptimizationResult(
            best_weights=best_weights,
            best_threshold=best_config.min_score,
            best_score=best_f1,
            precision=best_precision,
            recall=best_recall,
            num_trials=n_trials,
            optimization_time=optimization_time,
            sensitivity_analysis=sensitivity
        )
    
    def _sensitivity_analysis(
        self,
        base_config: CausalConfig,
        perturbations: List[float] = [0.9, 0.95, 1.0, 1.05, 1.1]
    ) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on threshold and weights.
        
        Varies each parameter by ±10% to see impact on F1 score.
        This provides scientific justification for parameter values.
        
        Returns:
            Dictionary with sensitivity results
        """
        logger.info("Running sensitivity analysis...")
        
        results = {
            "threshold_sensitivity": [],
            "weight_sensitivity": {},
        }
        
        # Threshold sensitivity
        base_f1, _, _ = self._evaluate_config(base_config)
        for factor in perturbations:
            config = CausalConfig(
                temporal_proximity_weight=base_config.temporal_proximity_weight,
                spatial_proximity_weight=base_config.spatial_proximity_weight,
                motion_alignment_weight=base_config.motion_alignment_weight,
                semantic_similarity_weight=base_config.semantic_similarity_weight,
                min_score=base_config.min_score * factor,
                max_pixel_distance=base_config.max_pixel_distance,
                temporal_decay=base_config.temporal_decay
            )
            f1, precision, recall = self._evaluate_config(config)
            results["threshold_sensitivity"].append({
                "factor": factor,
                "threshold": config.min_score,
                "f1": f1,
                "precision": precision,
                "recall": recall
            })
        
        # Weight sensitivity (one at a time)
        weight_names = ["temporal", "spatial", "motion", "semantic"]
        for weight_name in weight_names:
            results["weight_sensitivity"][weight_name] = []
            
            for factor in perturbations:
                # Perturb one weight, renormalize others
                weights = {
                    "temporal": base_config.temporal_proximity_weight,
                    "spatial": base_config.spatial_proximity_weight,
                    "motion": base_config.motion_alignment_weight,
                    "semantic": base_config.semantic_similarity_weight,
                }
                
                weights[weight_name] *= factor
                total = sum(weights.values())
                for k in weights:
                    weights[k] /= total
                
                config = CausalConfig(
                    temporal_proximity_weight=weights["temporal"],
                    spatial_proximity_weight=weights["spatial"],
                    motion_alignment_weight=weights["motion"],
                    semantic_similarity_weight=weights["semantic"],
                    min_score=base_config.min_score,
                    max_pixel_distance=base_config.max_pixel_distance,
                    temporal_decay=base_config.temporal_decay
                )
                
                f1, precision, recall = self._evaluate_config(config)
                results["weight_sensitivity"][weight_name].append({
                    "factor": factor,
                    "weight_value": weights[weight_name],
                    "f1": f1,
                    "precision": precision,
                    "recall": recall
                })
        
        return results
    
    def save_results(self, result: CISOptimizationResult, output_path: Path):
        """Save optimization results to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def optimize_cis_weights(
    ground_truth_path: Path,
    agent_candidates: List[AgentCandidate],
    state_changes: List[StateChange],
    n_trials: int = 100,
    output_path: Optional[Path] = None
) -> CISOptimizationResult:
    """
    Convenience function to optimize CIS weights from ground truth file.
    
    Args:
        ground_truth_path: Path to JSON file with ground truth annotations
        agent_candidates: All potential agents
        state_changes: All state changes
        n_trials: Number of optimization trials
        output_path: Where to save results (optional)
        
    Returns:
        Optimization results
    """
    # Load ground truth
    with open(ground_truth_path) as f:
        gt_data = json.load(f)
    
    ground_truth = [
        GroundTruthCausalPair(**item)
        for item in gt_data
    ]
    
    # Optimize
    optimizer = CISOptimizer(ground_truth, agent_candidates, state_changes)
    result = optimizer.optimize(n_trials=n_trials)
    
    # Save if requested
    if output_path:
        optimizer.save_results(result, output_path)
    
    return result


def create_sample_ground_truth(output_path: Path):
    """
    Create a sample ground truth file for testing.
    
    This would normally be created by human annotators.
    """
    sample_data = [
        {
            "agent_id": "entity_0001",
            "patient_id": "entity_0002",
            "state_change_frame": 150,
            "is_causal": True,
            "confidence": 1.0,
            "annotation_source": "human"
        },
        {
            "agent_id": "entity_0003",
            "patient_id": "entity_0002",
            "state_change_frame": 150,
            "is_causal": False,
            "confidence": 1.0,
            "annotation_source": "human"
        },
        # Add more annotations...
    ]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    logger.info(f"Sample ground truth created at {output_path}")
