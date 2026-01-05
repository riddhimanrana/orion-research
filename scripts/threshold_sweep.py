#!/usr/bin/env python3
"""
Phase 2 Threshold Sweep Experiment

Tests different similarity thresholds and evaluates the trade-off between:
- Fragmentation reduction (higher is better - fewer duplicate objects)
- Semantic correctness (verified by Gemini - should not over-merge)

Usage:
    python scripts/threshold_sweep.py --episode <episode> --thresholds 0.65,0.70,0.75,0.80,0.85
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@dataclass
class SweepResult:
    threshold: float
    tracks_before: int
    objects_after: int
    fragmentation_reduction: float
    avg_intra_similarity: float
    avg_inter_similarity: float
    separation_score: float
    elapsed_seconds: float
    gemini_correct_merges: Optional[int] = None
    gemini_incorrect_merges: Optional[int] = None
    gemini_accuracy: Optional[float] = None


def run_embed_with_threshold(episode: str, threshold: float, device: str = "cuda") -> Optional[SweepResult]:
    """Run Phase 2 embedding with a specific threshold."""
    console.print(f"\n[cyan]Running embed with threshold={threshold}...[/cyan]")
    
    # Run orion embed command
    cmd = [
        "python", "-m", "orion.cli.main", "embed",
        "--episode", episode,
        "--similarity", str(threshold),
        "--device", device,
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            console.print(f"[red]Error running embed: {result.stderr}[/red]")
            return None
        
        # Load stats
        stats_path = Path("results") / episode / "embed_stats.json"
        if not stats_path.exists():
            console.print(f"[red]Stats not found at {stats_path}[/red]")
            return None
        
        with open(stats_path) as f:
            stats = json.load(f)
        
        return SweepResult(
            threshold=threshold,
            tracks_before=stats['tracks_before'],
            objects_after=stats['objects_after'],
            fragmentation_reduction=stats['fragmentation_reduction'],
            avg_intra_similarity=stats['avg_intra_similarity'],
            avg_inter_similarity=stats['avg_inter_similarity'],
            separation_score=stats['avg_intra_similarity'] - stats['avg_inter_similarity'],
            elapsed_seconds=stats['elapsed_seconds'],
        )
    
    except subprocess.TimeoutExpired:
        console.print(f"[red]Timeout running embed[/red]")
        return None
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return None


def run_gemini_validation(episode: str, sample_size: int = 5) -> Optional[dict]:
    """Run Gemini validation for the current results."""
    console.print(f"\n[cyan]Running Gemini validation (sample_size={sample_size})...[/cyan]")
    
    cmd = [
        "python", "scripts/validate_phase2_gemini.py",
        "--episode", episode,
        "--sample-size", str(sample_size),
        "--skip-merge-validation",  # Only run scene analysis for sweep (faster)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            console.print(f"[yellow]Gemini validation failed (may be rate-limited)[/yellow]")
            return None
        
        # Load validation results
        validation_path = Path("results") / episode / "phase2_validation.json"
        if validation_path.exists():
            with open(validation_path) as f:
                return json.load(f)
        
        return None
    
    except Exception as e:
        console.print(f"[yellow]Gemini validation error: {e}[/yellow]")
        return None


def main():
    parser = argparse.ArgumentParser(description="Threshold sweep experiment")
    parser.add_argument("--episode", required=True, help="Episode name")
    parser.add_argument("--thresholds", default="0.65,0.70,0.75,0.80,0.85", 
                        help="Comma-separated thresholds to test")
    parser.add_argument("--device", default="cuda", help="Device (cuda/mps/cpu)")
    parser.add_argument("--with-gemini", action="store_true", 
                        help="Also run Gemini validation for each threshold")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()
    
    thresholds = [float(t) for t in args.thresholds.split(",")]
    
    console.print(Panel(f"[bold]Threshold Sweep: {args.episode}[/bold]"))
    console.print(f"Testing thresholds: {thresholds}")
    
    results = []
    
    for threshold in thresholds:
        console.print(f"\n{'='*60}")
        console.print(f"[bold]Testing threshold: {threshold}[/bold]")
        console.print('='*60)
        
        result = run_embed_with_threshold(args.episode, threshold, args.device)
        
        if result:
            # Optionally run Gemini validation
            if args.with_gemini:
                validation = run_gemini_validation(args.episode)
                if validation and 'metrics' in validation:
                    result.gemini_correct_merges = validation['metrics'].get('correct_merges')
                    result.gemini_incorrect_merges = validation['metrics'].get('incorrect_merges')
                    result.gemini_accuracy = validation['metrics'].get('merge_accuracy')
            
            results.append(result)
            
            console.print(f"  Objects: {result.objects_after} (from {result.tracks_before} tracks)")
            console.print(f"  Fragmentation reduction: {result.fragmentation_reduction:.1%}")
            console.print(f"  Separation score: {result.separation_score:.3f}")
    
    # Summary table
    console.print("\n")
    table = Table(title=f"Threshold Sweep Results: {args.episode}")
    table.add_column("Threshold", style="cyan")
    table.add_column("Objects", style="white")
    table.add_column("Frag. Reduction", style="green")
    table.add_column("Intra-Sim", style="yellow")
    table.add_column("Inter-Sim", style="yellow")
    table.add_column("Separation", style="magenta")
    table.add_column("Time (s)", style="white")
    
    if args.with_gemini:
        table.add_column("Gemini Accuracy", style="green")
    
    for r in results:
        row = [
            f"{r.threshold:.2f}",
            str(r.objects_after),
            f"{r.fragmentation_reduction:.1%}",
            f"{r.avg_intra_similarity:.3f}",
            f"{r.avg_inter_similarity:.3f}",
            f"{r.separation_score:.3f}",
            f"{r.elapsed_seconds:.1f}",
        ]
        
        if args.with_gemini:
            if r.gemini_accuracy is not None:
                row.append(f"{r.gemini_accuracy:.0%}")
            else:
                row.append("N/A")
        
        table.add_row(*row)
    
    console.print(table)
    
    # Find optimal threshold
    if results:
        # Optimal = best separation score (intra - inter) with reasonable fragmentation reduction
        best_by_separation = max(results, key=lambda r: r.separation_score)
        best_by_reduction = max(results, key=lambda r: r.fragmentation_reduction)
        
        console.print(f"\n[bold]Recommendations:[/bold]")
        console.print(f"  Best separation score: threshold={best_by_separation.threshold} (separation={best_by_separation.separation_score:.3f})")
        console.print(f"  Best fragmentation reduction: threshold={best_by_reduction.threshold} (reduction={best_by_reduction.fragmentation_reduction:.1%})")
        
        if args.with_gemini:
            results_with_gemini = [r for r in results if r.gemini_accuracy is not None]
            if results_with_gemini:
                best_by_gemini = max(results_with_gemini, key=lambda r: r.gemini_accuracy or 0)
                console.print(f"  Best Gemini accuracy: threshold={best_by_gemini.threshold} (accuracy={best_by_gemini.gemini_accuracy:.0%})")
    
    # Save results
    output_path = args.output or Path("results") / args.episode / "threshold_sweep.json"
    output_data = {
        "episode": args.episode,
        "timestamp": datetime.now().isoformat(),
        "thresholds_tested": thresholds,
        "results": [
            {
                "threshold": r.threshold,
                "tracks_before": r.tracks_before,
                "objects_after": r.objects_after,
                "fragmentation_reduction": r.fragmentation_reduction,
                "avg_intra_similarity": r.avg_intra_similarity,
                "avg_inter_similarity": r.avg_inter_similarity,
                "separation_score": r.separation_score,
                "elapsed_seconds": r.elapsed_seconds,
                "gemini_correct_merges": r.gemini_correct_merges,
                "gemini_incorrect_merges": r.gemini_incorrect_merges,
                "gemini_accuracy": r.gemini_accuracy,
            }
            for r in results
        ]
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    console.print(f"\n[green]Results saved to: {output_path}[/green]")


if __name__ == "__main__":
    main()
