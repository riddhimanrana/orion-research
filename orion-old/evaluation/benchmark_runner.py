"""
Benchmark Runner for Orion Evaluation
======================================

Command-line tool to run Orion on benchmark datasets and compute metrics.

Supported datasets:
- Action Genome
- VSGR (Video Scene Graph)
- PVSG (Panoptic Video Scene Graph)

Usage:
    orion benchmark --dataset action-genome --data-dir /path/to/data --output-dir results/

Author: Orion Research Team
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

from orion.evaluation.benchmark_evaluator import (
    BenchmarkEvaluator,
    EvaluationMetrics,
    GroundTruthGraph,
    PredictionGraph,
    print_evaluation_results,
)
from orion.evaluation.orion_adapter import OrionKGAdapter
from orion.evaluation.ag_adapter import ActionGenomeAdapter

logger = logging.getLogger("orion.benchmark")
console = Console()


class BenchmarkRunner:
    """
    Run Orion on benchmark datasets and evaluate
    """
    
    def __init__(
        self,
        dataset_name: str,
        data_dir: Path,
        output_dir: Path,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        iou_threshold: float = 0.5,
        tiou_threshold: float = 0.3,
    ):
        """
        Args:
            dataset_name: Name of benchmark dataset (action-genome, vsgr, pvsg)
            data_dir: Path to dataset directory
            output_dir: Path to output directory for results
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            iou_threshold: IoU threshold for entity matching
            tiou_threshold: Temporal IoU threshold for event matching
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Orion adapter
        self.orion_adapter = OrionKGAdapter(neo4j_uri, neo4j_user, neo4j_password)
        
        # Initialize evaluator
        self.evaluator = BenchmarkEvaluator(iou_threshold, tiou_threshold)
        
        # Load benchmark dataset
        self.ground_truth_loader = self._load_benchmark()
    
    def _load_benchmark(self):
        """Load benchmark dataset based on dataset_name"""
        if self.dataset_name == "action-genome":
            from orion.evaluation.benchmarks.action_genome_loader import ActionGenomeBenchmark
            return ActionGenomeBenchmark(str(self.data_dir))
        
        elif self.dataset_name == "vsgr":
            from orion.evaluation.benchmarks.vsgr_loader import VSGRBenchmark
            return VSGRBenchmark(str(self.data_dir))
        
        elif self.dataset_name == "pvsg":
            from orion.evaluation.benchmarks.pvsg_loader import PVSGBenchmark
            return PVSGBenchmark(str(self.data_dir))
        
        else:
            raise ValueError(
                f"Unknown dataset: {self.dataset_name}. "
                f"Supported: action-genome, vsgr, pvsg"
            )
    
    def run(
        self,
        video_ids: Optional[List[str]] = None,
        max_videos: Optional[int] = None,
    ) -> Dict[str, EvaluationMetrics]:
        """
        Run benchmark evaluation
        
        Args:
            video_ids: Optional list of specific video IDs to evaluate
            max_videos: Optional maximum number of videos to evaluate
        
        Returns:
            Dict mapping video_id -> EvaluationMetrics
        """
        console.print(Panel.fit(
            f"[bold cyan]Orion Benchmark Evaluation[/bold cyan]\n"
            f"Dataset: {self.dataset_name}\n"
            f"Data Dir: {self.data_dir}\n"
            f"Output Dir: {self.output_dir}",
            border_style="cyan"
        ))
        
        # Get video list
        if self.dataset_name == "action-genome":
            all_video_ids = list(self.ground_truth_loader.clips.keys())
        else:
            all_video_ids = self._get_video_ids_from_loader()
        
        # Filter if needed
        if video_ids:
            all_video_ids = [vid for vid in all_video_ids if vid in video_ids]
        
        if max_videos:
            all_video_ids = all_video_ids[:max_videos]
        
        console.print(f"\n[yellow]Evaluating {len(all_video_ids)} videos...[/yellow]\n")
        
        # Run evaluation on each video
        results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating...", total=len(all_video_ids))
            
            for video_id in all_video_ids:
                try:
                    # Load ground truth
                    gt = self._load_ground_truth(video_id)
                    
                    # Load Orion prediction
                    pred = self.orion_adapter.export_prediction_graph(video_id)
                    
                    # Evaluate
                    metrics = self.evaluator.evaluate(gt, pred)
                    results[video_id] = metrics
                    
                    progress.update(task, advance=1, description=f"Evaluated {video_id}")
                
                except Exception as e:
                    logger.error(f"Failed to evaluate {video_id}: {e}", exc_info=True)
                    progress.update(task, advance=1, description=f"Failed {video_id}")
        
        # Save results
        self._save_results(results)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _get_video_ids_from_loader(self) -> List[str]:
        """Get video IDs from benchmark loader"""
        # This depends on the specific loader structure
        # For now, return empty list if not action-genome
        logger.warning(f"Video ID extraction not implemented for {self.dataset_name}")
        return []
    
    def _load_ground_truth(self, video_id: str) -> GroundTruthGraph:
        """Load ground truth for a video"""
        if self.dataset_name == "action-genome":
            ag_dataset = self.ground_truth_loader.clips[video_id]
            adapter = ActionGenomeAdapter()
            return adapter.convert_to_ground_truth(ag_dataset)
        
        else:
            # Implement for other datasets
            raise NotImplementedError(
                f"Ground truth loading not implemented for {self.dataset_name}"
            )
    
    def _save_results(self, results: Dict[str, EvaluationMetrics]):
        """Save evaluation results to JSON"""
        output_file = self.output_dir / f"{self.dataset_name}_results.json"
        
        results_json = {
            "dataset": self.dataset_name,
            "num_videos": len(results),
            "per_video": {
                video_id: metrics.to_dict()
                for video_id, metrics in results.items()
            },
            "aggregate": self._compute_aggregate_metrics(results),
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        console.print(f"\n[green]Results saved to {output_file}[/green]")
    
    def _compute_aggregate_metrics(
        self,
        results: Dict[str, EvaluationMetrics],
    ) -> Dict:
        """Compute aggregate metrics across all videos"""
        if not results:
            return {}
        
        # Collect all metrics
        rel_precisions = [m.rel_precision for m in results.values()]
        rel_recalls = [m.rel_recall for m in results.values()]
        rel_f1s = [m.rel_f1 for m in results.values()]
        
        event_precisions = [m.event_precision for m in results.values()]
        event_recalls = [m.event_recall for m in results.values()]
        event_f1s = [m.event_f1 for m in results.values()]
        event_tious = [m.event_tiou for m in results.values()]
        
        causal_precisions = [m.causal_precision for m in results.values()]
        causal_recalls = [m.causal_recall for m in results.values()]
        causal_f1s = [m.causal_f1 for m in results.values()]
        
        entity_precisions = [m.entity_precision for m in results.values()]
        entity_recalls = [m.entity_recall for m in results.values()]
        
        geds = [m.graph_edit_distance for m in results.values()]
        
        return {
            "relationships": {
                "precision": {
                    "mean": round(np.mean(rel_precisions), 3),
                    "std": round(np.std(rel_precisions), 3),
                },
                "recall": {
                    "mean": round(np.mean(rel_recalls), 3),
                    "std": round(np.std(rel_recalls), 3),
                },
                "f1": {
                    "mean": round(np.mean(rel_f1s), 3),
                    "std": round(np.std(rel_f1s), 3),
                },
            },
            "events": {
                "precision": {
                    "mean": round(np.mean(event_precisions), 3),
                    "std": round(np.std(event_precisions), 3),
                },
                "recall": {
                    "mean": round(np.mean(event_recalls), 3),
                    "std": round(np.std(event_recalls), 3),
                },
                "f1": {
                    "mean": round(np.mean(event_f1s), 3),
                    "std": round(np.std(event_f1s), 3),
                },
                "temporal_iou": {
                    "mean": round(np.mean(event_tious), 3),
                    "std": round(np.std(event_tious), 3),
                },
            },
            "causal": {
                "precision": {
                    "mean": round(np.mean(causal_precisions), 3),
                    "std": round(np.std(causal_precisions), 3),
                },
                "recall": {
                    "mean": round(np.mean(causal_recalls), 3),
                    "std": round(np.std(causal_recalls), 3),
                },
                "f1": {
                    "mean": round(np.mean(causal_f1s), 3),
                    "std": round(np.std(causal_f1s), 3),
                },
            },
            "entities": {
                "precision": {
                    "mean": round(np.mean(entity_precisions), 3),
                    "std": round(np.std(entity_precisions), 3),
                },
                "recall": {
                    "mean": round(np.mean(entity_recalls), 3),
                    "std": round(np.std(entity_recalls), 3),
                },
            },
            "graph_edit_distance": {
                "mean": round(np.mean(geds), 2),
                "std": round(np.std(geds), 2),
            },
        }
    
    def _print_summary(self, results: Dict[str, EvaluationMetrics]):
        """Print summary table of results"""
        console.print("\n" + "="*80)
        console.print(f"[bold cyan]Aggregate Results ({len(results)} videos)[/bold cyan]")
        console.print("="*80 + "\n")
        
        aggregate = self._compute_aggregate_metrics(results)
        
        # Overall summary table
        table = Table(title="Overall Performance", show_header=True)
        table.add_column("Category", style="cyan", width=20)
        table.add_column("Precision", style="green", justify="right")
        table.add_column("Recall", style="green", justify="right")
        table.add_column("F1 Score", style="green", justify="right")
        
        # Relationships
        rel = aggregate["relationships"]
        table.add_row(
            "Relationships",
            f"{rel['precision']['mean']:.3f} ± {rel['precision']['std']:.3f}",
            f"{rel['recall']['mean']:.3f} ± {rel['recall']['std']:.3f}",
            f"{rel['f1']['mean']:.3f} ± {rel['f1']['std']:.3f}",
        )
        
        # Events
        evt = aggregate["events"]
        table.add_row(
            "Events",
            f"{evt['precision']['mean']:.3f} ± {evt['precision']['std']:.3f}",
            f"{evt['recall']['mean']:.3f} ± {evt['recall']['std']:.3f}",
            f"{evt['f1']['mean']:.3f} ± {evt['f1']['std']:.3f}",
        )
        
        # Causal
        caus = aggregate["causal"]
        table.add_row(
            "Causal Links",
            f"{caus['precision']['mean']:.3f} ± {caus['precision']['std']:.3f}",
            f"{caus['recall']['mean']:.3f} ± {caus['recall']['std']:.3f}",
            f"{caus['f1']['mean']:.3f} ± {caus['f1']['std']:.3f}",
        )
        
        # Entities
        ent = aggregate["entities"]
        table.add_row(
            "Entities",
            f"{ent['precision']['mean']:.3f} ± {ent['precision']['std']:.3f}",
            f"{ent['recall']['mean']:.3f} ± {ent['recall']['std']:.3f}",
            "—",
        )
        
        console.print(table)
        
        # Additional metrics
        console.print(f"\n[yellow]Event Temporal IoU:[/yellow] "
                     f"{evt['temporal_iou']['mean']:.3f} ± {evt['temporal_iou']['std']:.3f}")
        
        ged = aggregate["graph_edit_distance"]
        console.print(f"[yellow]Graph Edit Distance:[/yellow] "
                     f"{ged['mean']:.2f} ± {ged['std']:.2f}\n")
    
    def close(self):
        """Clean up resources"""
        self.orion_adapter.close()


def main():
    """CLI entry point for benchmark runner"""
    parser = argparse.ArgumentParser(
        description="Run Orion benchmark evaluation on standard datasets"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["action-genome", "vsgr", "pvsg"],
        help="Benchmark dataset to evaluate on",
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to benchmark dataset directory",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results (default: results/)",
    )
    
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default="bolt://localhost:7687",
        help="Neo4j connection URI",
    )
    
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default="neo4j",
        help="Neo4j username",
    )
    
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default="password",
        help="Neo4j password",
    )
    
    parser.add_argument(
        "--video-ids",
        nargs="+",
        help="Specific video IDs to evaluate (optional)",
    )
    
    parser.add_argument(
        "--max-videos",
        type=int,
        help="Maximum number of videos to evaluate (optional)",
    )
    
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for entity matching (default: 0.5)",
    )
    
    parser.add_argument(
        "--tiou-threshold",
        type=float,
        default=0.3,
        help="Temporal IoU threshold for event matching (default: 0.3)",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    # Run benchmark
    runner = BenchmarkRunner(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        iou_threshold=args.iou_threshold,
        tiou_threshold=args.tiou_threshold,
    )
    
    try:
        results = runner.run(
            video_ids=args.video_ids,
            max_videos=args.max_videos,
        )
        
        console.print("\n[bold green]✓ Benchmark evaluation complete![/bold green]\n")
        return 0
    
    except Exception as e:
        console.print(f"\n[bold red]✗ Benchmark evaluation failed:[/bold red] {e}\n")
        logger.exception("Benchmark evaluation failed")
        return 1
    
    finally:
        runner.close()


if __name__ == "__main__":
    sys.exit(main())
