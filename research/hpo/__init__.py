"""
HPO (Hyperparameter Optimization) Package
==========================================

This package contains tools for optimizing Orion's hyperparameters using
scientific methods (Optuna, Bayesian optimization, grid search, etc.).

Modules:
- cis_optimizer: Optimize CIS formula weights
- tao_data_loader: Load TAO-Amodal annotations for CIS training

Author: Orion Research Team
Date: October 2025
"""

from .cis_optimizer import (
    CISOptimizer,
    GroundTruthCausalPair,
    CISOptimizationResult,
    optimize_cis_weights,
)

from .tao_data_loader import (
    TAODataLoader,
    TAOTrack,
    VSGRGroundTruth,
    load_tao_training_data,
)

__all__ = [
    "CISOptimizer",
    "GroundTruthCausalPair",
    "CISOptimizationResult",
    "optimize_cis_weights",
    "TAODataLoader",
    "TAOTrack",
    "VSGRGroundTruth",
    "load_tao_training_data",
]
