#!/usr/bin/env python3
"""
Quick Start Evaluation Script
==============================

Runs a complete end-to-end test of the evaluation system.

Author: Orion Research Team
Date: October 2025
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QuickStart")


def run_command(cmd, check=True):
    """Run a command and log output"""
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)
    
    if check and result.returncode != 0:
        logger.error(f"Command failed with return code {result.returncode}")
        sys.exit(1)
    
    return result


def main():
    logger.info("="*80)
    logger.info("ORION EVALUATION QUICK START")
    logger.info("="*80)
    
    # Step 1: Check dependencies
    logger.info("\n[Step 1/7] Checking dependencies...")
    try:
        import torch
        import numpy
        import tqdm
        logger.info("✓ Core dependencies installed")
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Run: pip install -e .")
        sys.exit(1)
    
    # Step 2: Run unit tests
    logger.info("\n[Step 2/7] Running unit tests...")
    result = run_command([
        sys.executable, "-m", "pytest",
        "tests/unit/",
        "-v",
        "--tb=short"
    ], check=False)
    
    if result.returncode == 0:
        logger.info("✓ All unit tests passed")
    else:
        logger.warning("⚠ Some tests failed, but continuing...")
    
    # Step 3: Create sample dataset
    logger.info("\n[Step 3/7] Creating sample dataset...")
    run_command([
        sys.executable,
        "scripts/download_datasets.py",
        "--dataset", "sample",
        "--data-root", "data/benchmarks"
    ])
    logger.info("✓ Sample dataset created")
    
    # Step 4: Verify dataset
    logger.info("\n[Step 4/7] Verifying dataset...")
    run_command([
        sys.executable,
        "scripts/download_datasets.py",
        "--verify-only",
        "--data-root", "data/benchmarks"
    ])
    
    # Step 5: Run quick evaluation (if sample video exists)
    logger.info("\n[Step 5/7] Checking for sample video...")
    sample_video = Path("data/benchmarks/sample/videos/sample_001.mp4")
    
    if sample_video.exists():
        logger.info(f"Found sample video at {sample_video}")
        logger.info("Running evaluation...")
        
        run_command([
            sys.executable,
            "scripts/run_evaluation.py",
            "--mode", "video",
            "--video", str(sample_video),
            "--output-dir", "results/quickstart"
        ])
        logger.info("✓ Evaluation complete")
        
        # Step 6: View results
        logger.info("\n[Step 6/7] Checking results...")
        results_file = Path("results/quickstart/comparison_report.json")
        
        if results_file.exists():
            import json
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            logger.info("Results summary:")
            if 'pairwise_comparisons' in results:
                for comparison, metrics in results['pairwise_comparisons'].items():
                    logger.info(f"  {comparison}:")
                    logger.info(f"    Edge F1: {metrics.get('edges', {}).get('f1', 0):.3f}")
                    logger.info(f"    Causal F1: {metrics.get('causal', {}).get('f1', 0):.3f}")
        else:
            logger.warning("No results file found")
    else:
        logger.warning(f"Sample video not found at {sample_video}")
        logger.info("Place a test video at that location and re-run")
        logger.info("Or run: python scripts/run_evaluation.py --video YOUR_VIDEO.mp4")
    
    # Step 7: Integration test
    logger.info("\n[Step 7/7] Running integration tests...")
    result = run_command([
        sys.executable, "-m", "pytest",
        "tests/test_evaluation_integration.py",
        "-v"
    ], check=False)
    
    if result.returncode == 0:
        logger.info("✓ Integration tests passed")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("QUICK START COMPLETE")
    logger.info("="*80)
    logger.info("\nWhat you can do next:")
    logger.info("1. Download real datasets:")
    logger.info("   python scripts/download_datasets.py --dataset action_genome")
    logger.info("\n2. Tune hyperparameters:")
    logger.info("   python -m orion.evaluation.hyperparameter_tuning --method grid")
    logger.info("\n3. Run full evaluation:")
    logger.info("   python scripts/run_evaluation.py --mode benchmark --benchmark action_genome")
    logger.info("\n4. Visualize results:")
    logger.info("   python scripts/visualize_results.py --results-file results/quickstart/comparison_report.json")
    logger.info("\n5. Read the docs:")
    logger.info("   cat EVALUATION_README.md")
    logger.info("="*80)


if __name__ == "__main__":
    main()
