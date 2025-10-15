#!/usr/bin/env python3
"""
Orion Research Evaluation Script
=================================

This script demonstrates the complete evaluation pipeline:
1. Run perception engine on a video
2. Build knowledge graph with CIS + LLM (our method)
3. Build knowledge graph with heuristic baseline
4. Compare the two approaches
5. Optionally evaluate against VSGR benchmark

Usage:
    python scripts/run_evaluation.py --video path/to/video.mp4
    python scripts/run_evaluation.py --benchmark vsgr --dataset-path /path/to/vsgr

Author: Orion Research Team
Date: October 2025
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orion.perception_engine import AsynchronousPerceptionEngine
from orion.semantic_uplift import SemanticUpliftEngine
from orion.evaluation import HeuristicBaseline, GraphComparator
from orion.evaluation.benchmarks import VSGRBenchmark

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EvaluationRunner")


def run_perception_on_video(video_path: str, output_dir: Path) -> Path:
    """
    Run perception engine on video and save perception log
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save outputs
        
    Returns:
        Path to saved perception log JSON
    """
    logger.info(f"Running perception engine on {video_path}")
    
    # Initialize perception engine
    engine = AsynchronousPerceptionEngine()
    
    # Process video
    perception_log = engine.process_video(video_path)
    
    # Save perception log
    perception_log_path = output_dir / "perception_log.json"
    with open(perception_log_path, 'w') as f:
        json.dump([obj.to_dict() for obj in perception_log], f, indent=2)
    
    logger.info(f"Saved perception log to {perception_log_path}")
    logger.info(f"Total objects detected: {len(perception_log)}")
    
    return perception_log_path


def build_cis_llm_graph(perception_log_path: Path, output_dir: Path) -> Path:
    """
    Build knowledge graph using CIS + LLM method
    
    Args:
        perception_log_path: Path to perception log JSON
        output_dir: Directory to save outputs
        
    Returns:
        Path to saved graph JSON
    """
    logger.info("Building knowledge graph with CIS + LLM method")
    
    # Load perception log
    with open(perception_log_path, 'r') as f:
        perception_objects = json.load(f)
    
    # Initialize semantic uplift engine
    uplift_engine = SemanticUpliftEngine()
    
    # Process perception log
    uplift_engine.process_perception_log(perception_objects)
    
    # Export graph
    graph_path = output_dir / "graph_cis_llm.json"
    uplift_engine.export_knowledge_graph(str(graph_path))
    
    logger.info(f"Saved CIS+LLM graph to {graph_path}")
    
    return graph_path


def build_heuristic_graph(perception_log_path: Path, output_dir: Path) -> Path:
    """
    Build knowledge graph using heuristic baseline
    
    Args:
        perception_log_path: Path to perception log JSON
        output_dir: Directory to save outputs
        
    Returns:
        Path to saved graph JSON
    """
    logger.info("Building knowledge graph with heuristic baseline")
    
    # Load perception log
    with open(perception_log_path, 'r') as f:
        perception_objects = json.load(f)
    
    # Initialize heuristic baseline
    baseline = HeuristicBaseline()
    
    # Process perception log
    graph_data = baseline.process_perception_log(perception_objects)
    
    # Export graph
    graph_path = output_dir / "graph_heuristic.json"
    baseline.export_to_json(str(graph_path))
    
    logger.info(f"Saved heuristic graph to {graph_path}")
    
    return graph_path


def compare_graphs(cis_graph_path: Path, heuristic_graph_path: Path, output_dir: Path):
    """
    Compare CIS+LLM and heuristic graphs
    
    Args:
        cis_graph_path: Path to CIS+LLM graph
        heuristic_graph_path: Path to heuristic graph
        output_dir: Directory to save comparison
    """
    logger.info("Comparing graphs...")
    
    # Initialize comparator
    comparator = GraphComparator()
    
    # Load graphs
    comparator.load_from_json("cis_llm", str(cis_graph_path))
    comparator.load_from_json("heuristic", str(heuristic_graph_path))
    
    # Generate report
    report_path = output_dir / "comparison_report.json"
    comparator.generate_report(str(report_path))
    
    # Print summary
    comparator.print_summary()
    
    logger.info(f"Saved comparison report to {report_path}")


def evaluate_on_vsgr(dataset_path: str, output_dir: Path):
    """
    Evaluate on VSGR benchmark dataset
    
    Args:
        dataset_path: Path to VSGR dataset root
        output_dir: Directory to save outputs
    """
    logger.info(f"Loading VSGR benchmark from {dataset_path}")
    
    # Load benchmark
    benchmark = VSGRBenchmark(dataset_path)
    
    logger.info(f"Found {len(benchmark.list_clips())} clips in VSGR dataset")
    
    # Process each clip
    all_predictions = {}
    
    for clip_id in benchmark.list_clips():
        logger.info(f"Processing clip: {clip_id}")
        
        clip = benchmark.get_clip(clip_id)
        video_path = str(clip.video_path)
        
        # Run perception
        clip_output_dir = output_dir / clip_id
        clip_output_dir.mkdir(parents=True, exist_ok=True)
        
        perception_log_path = run_perception_on_video(video_path, clip_output_dir)
        
        # Build CIS+LLM graph
        cis_graph_path = build_cis_llm_graph(perception_log_path, clip_output_dir)
        
        # Load prediction
        with open(cis_graph_path, 'r') as f:
            prediction = json.load(f)
        
        all_predictions[clip_id] = prediction
    
    # Batch evaluate
    logger.info("Running batch evaluation...")
    results = benchmark.batch_evaluate(all_predictions)
    
    # Save results
    results_path = output_dir / "vsgr_evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"VSGR Evaluation Results:")
    logger.info(f"  Average Edge F1: {results['edge_f1']:.3f}")
    logger.info(f"  Average Event F1: {results['event_f1']:.3f}")
    logger.info(f"  Average Causal F1: {results['causal_f1']:.3f}")
    logger.info(f"Saved results to {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Orion Research Evaluation Pipeline"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["video", "benchmark"],
        default="video",
        help="Evaluation mode: single video or benchmark dataset"
    )
    
    # Video mode options
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file (for video mode)"
    )
    
    # Benchmark mode options
    parser.add_argument(
        "--benchmark",
        choices=["vsgr"],
        help="Benchmark dataset name (for benchmark mode)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to benchmark dataset root (for benchmark mode)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_output",
        help="Directory to save evaluation outputs"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("Orion Research Evaluation Pipeline")
    logger.info("="*80)
    
    if args.mode == "video":
        if not args.video:
            parser.error("--video is required for video mode")
        
        logger.info(f"Mode: Single Video Evaluation")
        logger.info(f"Video: {args.video}")
        logger.info(f"Output: {output_dir}")
        logger.info("="*80)
        
        # Run perception
        perception_log_path = run_perception_on_video(args.video, output_dir)
        
        # Build CIS+LLM graph
        cis_graph_path = build_cis_llm_graph(perception_log_path, output_dir)
        
        # Build heuristic graph
        heuristic_graph_path = build_heuristic_graph(perception_log_path, output_dir)
        
        # Compare
        compare_graphs(cis_graph_path, heuristic_graph_path, output_dir)
        
    elif args.mode == "benchmark":
        if not args.benchmark or not args.dataset_path:
            parser.error("--benchmark and --dataset-path are required for benchmark mode")
        
        logger.info(f"Mode: Benchmark Evaluation")
        logger.info(f"Benchmark: {args.benchmark}")
        logger.info(f"Dataset: {args.dataset_path}")
        logger.info(f"Output: {output_dir}")
        logger.info("="*80)
        
        if args.benchmark == "vsgr":
            evaluate_on_vsgr(args.dataset_path, output_dir)
        else:
            logger.error(f"Unknown benchmark: {args.benchmark}")
            sys.exit(1)
    
    logger.info("="*80)
    logger.info("Evaluation complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
