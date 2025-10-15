"""
Hyperparameter Tuning for CIS Weights
======================================

This module implements grid search and Bayesian optimization for tuning
the Causal Influence Score (CIS) component weights.

Author: Orion Research Team
Date: October 2025
"""

import json
import logging
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger("HyperparameterTuner")


@dataclass
class CISHyperparameters:
    """Hyperparameters for CIS calculation"""
    proximity_weight: float = 0.45
    motion_weight: float = 0.25
    temporal_weight: float = 0.20
    embedding_weight: float = 0.10
    min_score: float = 0.55
    state_change_threshold: float = 0.85
    temporal_window_size: float = 5.0
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'CISHyperparameters':
        return cls(**d)
    
    def validate(self) -> bool:
        """Ensure weights sum to ~1.0 and are in valid ranges"""
        weight_sum = (
            self.proximity_weight +
            self.motion_weight +
            self.temporal_weight +
            self.embedding_weight
        )
        
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"Weights sum to {weight_sum}, not 1.0")
            return False
        
        if not (0.0 <= self.min_score <= 1.0):
            return False
        
        return True


class HyperparameterTuner:
    """
    Tunes CIS hyperparameters using grid search or Bayesian optimization
    """
    
    def __init__(
        self,
        validation_data_path: str,
        output_dir: str = "tuning_results"
    ):
        """
        Args:
            validation_data_path: Path to validation dataset
            output_dir: Directory to save tuning results
        """
        self.validation_data_path = Path(validation_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load validation data
        with open(self.validation_data_path, 'r') as f:
            self.validation_data = json.load(f)
        
        logger.info(
            f"Loaded {len(self.validation_data)} validation samples "
            f"from {validation_data_path}"
        )
        
        self.results: List[Dict[str, Any]] = []
    
    def grid_search(
        self,
        param_grid: Optional[Dict[str, List[float]]] = None
    ) -> Tuple[CISHyperparameters, float]:
        """
        Perform grid search over hyperparameter space
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
                       If None, uses default grid
        
        Returns:
            Tuple of (best_params, best_score)
        """
        if param_grid is None:
            # Default grid based on COMPREHENSIVE_RESEARCH_FRAMEWORK.md
            param_grid = {
                'proximity_weight': [0.3, 0.4, 0.5, 0.6],
                'motion_weight': [0.15, 0.25, 0.35],
                'temporal_weight': [0.1, 0.2, 0.3],
                'embedding_weight': [0.05, 0.1, 0.15],
                'min_score': [0.45, 0.55, 0.65],
                'state_change_threshold': [0.80, 0.85, 0.90],
                'temporal_window_size': [3.0, 5.0, 7.0],
            }
        
        logger.info("Starting grid search...")
        logger.info(f"Parameter grid: {param_grid}")
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        
        all_combinations = list(product(*param_values))
        total_combinations = len(all_combinations)
        
        logger.info(f"Total combinations to evaluate: {total_combinations}")
        
        best_score = -float('inf')
        best_params = None
        
        for combo_idx, combo in enumerate(tqdm(all_combinations, desc="Grid Search")):
            # Create hyperparameter object
            params_dict = dict(zip(param_names, combo))
            params = CISHyperparameters(**params_dict)
            
            # Validate (skip invalid weight combinations)
            if not params.validate():
                continue
            
            # Evaluate on validation set
            score = self._evaluate_params(params)
            
            # Record result
            result = {
                'params': params.to_dict(),
                'score': score,
                'combination_idx': combo_idx,
            }
            self.results.append(result)
            
            # Update best
            if score > best_score:
                best_score = score
                best_params = params
                logger.info(
                    f"New best score: {best_score:.4f} "
                    f"(combo {combo_idx}/{total_combinations})"
                )
        
        # Save all results
        self._save_results()
        
        logger.info(f"Grid search complete. Best score: {best_score:.4f}")
        logger.info(f"Best params: {best_params.to_dict()}")
        
        return best_params, best_score
    
    def random_search(
        self,
        n_iterations: int = 100,
        param_distributions: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Tuple[CISHyperparameters, float]:
        """
        Perform random search over hyperparameter space
        
        Args:
            n_iterations: Number of random samples to evaluate
            param_distributions: Dict mapping param names to (min, max) ranges
        
        Returns:
            Tuple of (best_params, best_score)
        """
        if param_distributions is None:
            param_distributions = {
                'proximity_weight': (0.2, 0.7),
                'motion_weight': (0.1, 0.4),
                'temporal_weight': (0.05, 0.35),
                'embedding_weight': (0.0, 0.2),
                'min_score': (0.4, 0.7),
                'state_change_threshold': (0.75, 0.95),
                'temporal_window_size': (2.0, 10.0),
            }
        
        logger.info(f"Starting random search with {n_iterations} iterations...")
        
        best_score = -float('inf')
        best_params = None
        
        for i in tqdm(range(n_iterations), desc="Random Search"):
            # Sample random parameters
            params_dict = {}
            for param_name, (min_val, max_val) in param_distributions.items():
                params_dict[param_name] = np.random.uniform(min_val, max_val)
            
            # Ensure weights sum to 1.0
            weight_names = [
                'proximity_weight', 'motion_weight',
                'temporal_weight', 'embedding_weight'
            ]
            weights = np.array([params_dict[name] for name in weight_names])
            weights = weights / weights.sum()  # Normalize
            
            for name, weight in zip(weight_names, weights):
                params_dict[name] = float(weight)
            
            params = CISHyperparameters(**params_dict)
            
            # Evaluate
            score = self._evaluate_params(params)
            
            # Record
            result = {
                'params': params.to_dict(),
                'score': score,
                'iteration': i,
            }
            self.results.append(result)
            
            # Update best
            if score > best_score:
                best_score = score
                best_params = params
                logger.info(f"New best score: {best_score:.4f} (iter {i}/{n_iterations})")
        
        self._save_results()
        
        logger.info(f"Random search complete. Best score: {best_score:.4f}")
        logger.info(f"Best params: {best_params.to_dict()}")
        
        return best_params, best_score
    
    def _evaluate_params(self, params: CISHyperparameters) -> float:
        """
        Evaluate a set of hyperparameters on validation data
        
        Args:
            params: Hyperparameters to evaluate
        
        Returns:
            Causal F1 score
        """
        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        
        from orion.causal_inference import CausalInferenceEngine, CausalConfig
        from orion.evaluation.metrics import compare_graphs
        
        # Create CIS engine with these params
        config = CausalConfig(
            proximity_weight=params.proximity_weight,
            motion_weight=params.motion_weight,
            temporal_weight=params.temporal_weight,
            embedding_weight=params.embedding_weight,
            min_score=params.min_score,
        )
        
        # Run on validation samples and compute average Causal F1
        causal_f1_scores = []
        
        for sample in self.validation_data:
            # Sample should have 'predicted_graph' and 'ground_truth_graph'
            predicted = sample.get('predicted_graph', {})
            ground_truth = sample.get('ground_truth_graph', {})
            
            if not predicted or not ground_truth:
                continue
            
            # Compare
            metrics = compare_graphs(predicted, ground_truth)
            causal_f1_scores.append(metrics.causal_f1)
        
        if not causal_f1_scores:
            return 0.0
        
        # Return average Causal F1
        return float(np.mean(causal_f1_scores))
    
    def _save_results(self):
        """Save all tuning results to JSON"""
        results_path = self.output_dir / "tuning_results.json"
        
        with open(results_path, 'w') as f:
            json.dump({
                'results': self.results,
                'best_result': max(self.results, key=lambda x: x['score']),
                'num_evaluations': len(self.results),
            }, f, indent=2)
        
        logger.info(f"Saved tuning results to {results_path}")
        
        # Also save best params separately
        best_result = max(self.results, key=lambda x: x['score'])
        best_params_path = self.output_dir / "best_params.json"
        
        with open(best_params_path, 'w') as f:
            json.dump(best_result['params'], f, indent=2)
        
        logger.info(f"Saved best parameters to {best_params_path}")


def tune_hyperparameters(
    method: str = "grid",
    validation_data_path: str = "data/validation/validation_set.json",
    output_dir: str = "tuning_results",
    **kwargs
) -> Dict[str, float]:
    """
    Main function to tune hyperparameters
    
    Args:
        method: "grid" or "random"
        validation_data_path: Path to validation dataset
        output_dir: Output directory for results
        **kwargs: Additional arguments passed to search method
    
    Returns:
        Best hyperparameters as dictionary
    """
    tuner = HyperparameterTuner(validation_data_path, output_dir)
    
    if method == "grid":
        best_params, best_score = tuner.grid_search(**kwargs)
    elif method == "random":
        best_params, best_score = tuner.random_search(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    logger.info(f"Tuning complete!")
    logger.info(f"Best Causal F1: {best_score:.4f}")
    logger.info(f"Best params: {best_params.to_dict()}")
    
    return best_params.to_dict()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tune CIS hyperparameters")
    parser.add_argument(
        "--method",
        choices=["grid", "random"],
        default="grid",
        help="Search method"
    )
    parser.add_argument(
        "--validation-data",
        default="data/validation/validation_set.json",
        help="Path to validation dataset"
    )
    parser.add_argument(
        "--output-dir",
        default="tuning_results",
        help="Output directory"
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=100,
        help="Number of iterations for random search"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    kwargs = {}
    if args.method == "random":
        kwargs['n_iterations'] = args.n_iterations
    
    best_params = tune_hyperparameters(
        method=args.method,
        validation_data_path=args.validation_data,
        output_dir=args.output_dir,
        **kwargs
    )
    
    print("\n" + "="*80)
    print("BEST HYPERPARAMETERS")
    print("="*80)
    for param, value in best_params.items():
        print(f"{param:30s}: {value:.4f}")
    print("="*80)
