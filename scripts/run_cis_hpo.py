#!/usr/bin/env python3
"""
Run CIS Hyperparameter Optimization
====================================

Complete end-to-end script for optimizing CIS weights.

Usage:
    # From extracted data and ground truth
    python scripts/run_cis_hpo.py \
        --extracted-data data/hpo/extracted_data.pkl \
        --ground-truth data/hpo/annotations.json \
        --trials 100 \
        --output results/cis_hpo

    # Or from perception log directly
    python scripts/run_cis_hpo.py \
        --perception-log data/testing/perception_log_video_TIMESTAMP.json \
        --ground-truth data/hpo/annotations.json \
        --trials 100
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from orion.hpo.cis_optimizer import CISOptimizer, GroundTruthCausalPair
    from orion.causal_inference import AgentCandidate, StateChange
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ground_truth(ground_truth_path: Path):
    """Load ground truth annotations"""
    logger.info(f"Loading ground truth from {ground_truth_path}")
    
    with open(ground_truth_path) as f:
        gt_data = json.load(f)
    
    ground_truth = [GroundTruthCausalPair(**item) for item in gt_data]
    logger.info(f"Loaded {len(ground_truth)} ground truth pairs")
    
    return ground_truth


def load_extracted_data(extracted_data_path: Path):
    """Load pre-extracted agent candidates and state changes"""
    logger.info(f"Loading extracted data from {extracted_data_path}")
    
    with open(extracted_data_path, 'rb') as f:
        data = pickle.load(f)
    
    agent_candidates = data['agent_candidates']
    state_changes = data['state_changes']
    
    logger.info(f"Loaded {len(agent_candidates)} agent candidates")
    logger.info(f"Loaded {len(state_changes)} state changes")
    
    return agent_candidates, state_changes


def main():
    parser = argparse.ArgumentParser(
        description='Run CIS Hyperparameter Optimization'
    )
    
    # Data input options
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        '--extracted-data',
        type=str,
        help='Path to extracted data pickle file'
    )
    data_group.add_argument(
        '--perception-log',
        type=str,
        help='Path to perception log JSON (will extract on-the-fly)'
    )
    
    # Required arguments
    parser.add_argument(
        '--ground-truth',
        type=str,
        required=True,
        help='Path to ground truth annotations JSON'
    )
    
    # Optimization parameters
    parser.add_argument(
        '--trials',
        type=int,
        default=100,
        help='Number of optimization trials (default: 100)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Timeout in seconds (default: no limit)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default='hpo_results',
        help='Output directory (default: hpo_results)'
    )
    
    args = parser.parse_args()
    
    # Check imports
    if not IMPORTS_OK:
        logger.error(f"Import failed: {IMPORT_ERROR}")
        logger.error("\nTo run CIS optimization, install:")
        logger.error("  pip install optuna")
        return 1
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("="*60)
    logger.info("CIS HYPERPARAMETER OPTIMIZATION")
    logger.info("="*60)
    
    # Load ground truth
    ground_truth_path = Path(args.ground_truth)
    if not ground_truth_path.exists():
        logger.error(f"Ground truth file not found: {ground_truth_path}")
        return 1
    
    ground_truth = load_ground_truth(ground_truth_path)
    
    # Load agent candidates and state changes
    if args.extracted_data:
        extracted_data_path = Path(args.extracted_data)
        if not extracted_data_path.exists():
            logger.error(f"Extracted data file not found: {extracted_data_path}")
            return 1
        
        agent_candidates, state_changes = load_extracted_data(extracted_data_path)
    
    elif args.perception_log:
        # Extract on-the-fly
        logger.info("Extracting data from perception log...")
        from extract_data_for_hpo import (
            load_perception_log,
            extract_agent_candidates,
            extract_state_changes
        )
        
        perception_log_path = Path(args.perception_log)
        if not perception_log_path.exists():
            logger.error(f"Perception log not found: {perception_log_path}")
            return 1
        
        observations = load_perception_log(perception_log_path)
        agent_candidates = extract_agent_candidates(observations)
        state_changes = extract_state_changes(observations)
    
    # Validate data
    if not agent_candidates:
        logger.error("No agent candidates found!")
        return 1
    
    if not state_changes:
        logger.error("No state changes found!")
        return 1
    
    if not ground_truth:
        logger.error("No ground truth annotations found!")
        return 1
    
    # Create optimizer
    logger.info("\n" + "="*60)
    logger.info("CREATING OPTIMIZER")
    logger.info("="*60)
    
    optimizer = CISOptimizer(
        ground_truth=ground_truth,
        agent_candidates=agent_candidates,
        state_changes=state_changes,
        seed=args.seed
    )
    
    # Run optimization
    logger.info("\n" + "="*60)
    logger.info(f"RUNNING OPTIMIZATION ({args.trials} trials)")
    logger.info("="*60)
    
    result = optimizer.optimize(
        n_trials=args.trials,
        timeout=args.timeout,
        show_progress=True
    )
    
    # Save results
    output_path = output_dir / f"optimization_{timestamp}.json"
    optimizer.save_results(result, output_path)
    
    # Also save as "latest" for easy reference
    latest_path = output_dir / "optimization_latest.json"
    optimizer.save_results(result, latest_path)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZATION COMPLETE!")
    logger.info("="*60)
    
    print("\n" + "="*70)
    print(" 🎉 CIS OPTIMIZATION RESULTS 🎉")
    print("="*70)
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   F1 Score:   {result.best_score:.4f}")
    print(f"   Precision:  {result.precision:.4f}")
    print(f"   Recall:     {result.recall:.4f}")
    
    print(f"\n⚖️  LEARNED WEIGHTS:")
    for name, value in result.best_weights.items():
        print(f"   {name:12s}: {value:.4f}")
    
    print(f"\n🎯 THRESHOLD:")
    print(f"   Min Score:  {result.best_threshold:.4f}")
    
    print(f"\n⏱️  OPTIMIZATION:")
    print(f"   Trials:     {result.num_trials}")
    print(f"   Time:       {result.optimization_time:.2f}s")
    
    print(f"\n💾 SAVED TO:")
    print(f"   {output_path}")
    print(f"   {latest_path}")
    
    print("\n" + "="*70)
    print(" NEXT STEPS")
    print("="*70)
    print("\n1. Review sensitivity analysis:")
    print(f"   cat {output_path} | jq '.sensitivity_analysis'")
    
    print("\n2. Use optimized weights in your config:")
    print(f"   from orion.causal_inference import CausalConfig")
    print(f"   config = CausalConfig.from_hpo_result('{latest_path}')")
    
    print("\n3. Update main pipeline to use learned weights:")
    print(f"   orion analyze video.mp4 --cis-config {latest_path}")
    
    print("\n4. For your paper/report:")
    print("   - Weights were learned via Bayesian optimization")
    print(f"   - Achieved F1={result.best_score:.3f} on validation data")
    print(f"   - Based on {len(ground_truth)} human-annotated pairs")
    print("   - Optimized over {args.trials} trials using TPE sampler")
    
    print("\n" + "="*70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
