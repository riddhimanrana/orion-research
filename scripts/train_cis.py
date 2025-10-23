#!/usr/bin/env python3
"""
CIS Hyperparameter Optimization Training Script
================================================

Trains the Causal Inference Score (CIS) formula weights using:
- TAO-Amodal bounding box annotations  
- VSGR ground truth causal labels

No video files required - works directly with annotations.

Usage:
    python scripts/train_cis.py --trials 100 --videos 10

Author: Orion Research Team
Date: October 2025
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add orion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.hpo.tao_data_loader import load_tao_training_data
from orion.hpo.cis_optimizer import CISOptimizer, GroundTruthCausalPair

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TrainCIS")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CIS weights using TAO-Amodal + VSGR annotations"
    )
    
    parser.add_argument(
        "--tao-json",
        type=str,
        default="data/aspire_train.json",
        help="Path to TAO-Amodal annotations (default: data/aspire_train.json)"
    )
    
    parser.add_argument(
        "--vsgr-json",
        type=str,
        default="data/benchmarks/ground_truth/vsgr_aspire_train_sample.json",
        help="Path to VSGR ground truth (default: data/benchmarks/ground_truth/vsgr_aspire_train_sample.json)"
    )
    
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of Optuna trials (default: 100)"
    )
    
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum videos to use for training (default: all)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="hpo_results/cis_weights.json",
        help="Output path for optimized weights (default: hpo_results/cis_weights.json)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("CIS Hyperparameter Optimization")
    logger.info("=" * 80)
    logger.info(f"TAO annotations: {args.tao_json}")
    logger.info(f"VSGR ground truth: {args.vsgr_json}")
    logger.info(f"Optimization trials: {args.trials}")
    logger.info(f"Max videos: {args.max_videos or 'all'}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("=" * 80)
    
    # Step 1: Load TAO-Amodal + VSGR data
    logger.info("\n[1/3] Loading training data...")
    agent_candidates, state_changes, ground_truth_dicts = load_tao_training_data(
        tao_json_path=args.tao_json,
        vsgr_json_path=args.vsgr_json,
        max_videos=args.max_videos
    )
    
    if not agent_candidates:
        logger.error("No agent candidates found! Check your data paths.")
        return 1
    
    if not state_changes:
        logger.error("No state changes found! Check your data paths.")
        return 1
    
    if not ground_truth_dicts:
        logger.error("No ground truth pairs found! Check your data paths.")
        return 1
    
    # Convert ground truth dicts to GroundTruthCausalPair objects
    ground_truth = [
        GroundTruthCausalPair(**gt) for gt in ground_truth_dicts
    ]
    
    logger.info(f"✓ Loaded {len(agent_candidates)} agent candidates")
    logger.info(f"✓ Loaded {len(state_changes)} state changes")
    logger.info(f"✓ Loaded {len(ground_truth)} ground truth pairs")
    
    # Step 2: Initialize optimizer
    logger.info("\n[2/3] Initializing CIS optimizer...")
    optimizer = CISOptimizer(
        ground_truth=ground_truth,
        agent_candidates=agent_candidates,
        state_changes=state_changes,
        seed=args.seed
    )
    
    # Step 3: Run optimization
    logger.info("\n[3/3] Running Bayesian optimization...")
    logger.info(f"This may take several minutes for {args.trials} trials...")
    
    result = optimizer.optimize(
        n_trials=args.trials,
        timeout=None,
        show_progress=True
    )
    
    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Best F1 Score:    {result.best_score:.4f}")
    logger.info(f"Precision:        {result.precision:.4f}")
    logger.info(f"Recall:           {result.recall:.4f}")
    logger.info(f"Optimization Time: {result.optimization_time:.2f}s")
    logger.info("\nOptimized Weights:")
    for name, weight in result.best_weights.items():
        logger.info(f"  {name:12s}: {weight:.4f}")
    logger.info(f"\nOptimized Threshold: {result.best_threshold:.4f}")
    logger.info("=" * 80)
    
    # Save results
    output_path = Path(args.output)
    optimizer.save_results(result, output_path)
    
    logger.info(f"\n✓ Results saved to {output_path}")
    logger.info("\nTo use these weights in Orion:")
    logger.info(f"  1. Run: export ORION_CIS_WEIGHTS={output_path}")
    logger.info(f"  2. Or pass: --cis-weights {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
