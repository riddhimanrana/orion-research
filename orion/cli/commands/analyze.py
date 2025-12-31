"""Analyze command - process video with minimal Orion perception pipeline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from ...settings import OrionSettings

console = Console()


def handle_analyze(args: argparse.Namespace, settings: OrionSettings) -> None:
    """Handle the analyze command - run perception pipeline and optionally export to graph."""
    from ...perception import PerceptionEngine
    from ...perception.config import get_fast_config, get_balanced_config, get_accurate_config

    # Determine processing mode
    if args.fast:
        config = get_fast_config()
        mode_name = "fast"
    elif args.accurate:
        config = get_accurate_config()
        mode_name = "accurate"
    else:
        config = get_balanced_config()
        mode_name = "balanced"

    # Display analysis parameters
    params_table = Table(box=box.ROUNDED, show_header=False, border_style="cyan")
    params_table.add_column("Parameter", style="cyan bold", width=20)
    params_table.add_column("Value", style="yellow", width=60)

    params_table.add_row("Video", str(args.video))
    params_table.add_row("Mode", mode_name)
    params_table.add_row("Output", args.output)
    
    console.print("\n")
    console.print(Panel(params_table, title="[bold]Perception Analysis[/bold]", border_style="cyan"))
    console.print("\n")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run perception pipeline
    try:
        start_time = time.time()
        console.print("[bold cyan]Starting perception pipeline...[/bold cyan]\n")
        
        engine = PerceptionEngine(config=config)
        result = engine.process_video(
            str(args.video),
            save_visualizations=True,
            output_dir=str(output_dir)
        )
        
        elapsed = time.time() - start_time
        
        # Display results
        results_table = Table(box=box.ROUNDED, show_header=True, border_style="green")
        results_table.add_column("Metric", style="cyan bold")
        results_table.add_column("Value", style="yellow")
        
        results_table.add_row("Unique Entities", str(result.unique_entities))
        results_table.add_row("Total Observations", str(result.total_detections))
        results_table.add_row("Frames Processed", str(result.total_frames))
        results_table.add_row("Duration", f"{result.duration_seconds:.2f}s")
        results_table.add_row("Processing Time", f"{elapsed:.2f}s")
        results_table.add_row("Speed", f"{result.total_frames / elapsed:.1f} fps")
        
        if result.metrics:
            timings = result.metrics.get("timings", {})
            if timings:
                results_table.add_row("─" * 20, "─" * 20)
                results_table.add_row("Detection", f"{timings.get('detection_seconds', 0):.2f}s")
                results_table.add_row("Embedding", f"{timings.get('embedding_seconds', 0):.2f}s")
                results_table.add_row("Clustering", f"{timings.get('clustering_seconds', 0):.2f}s")
                results_table.add_row("Re-ID", f"{timings.get('reid_seconds', 0):.2f}s")
                results_table.add_row("Description", f"{timings.get('description_seconds', 0):.2f}s")
            
            reid_metrics = result.metrics.get("reid")
            if reid_metrics:
                results_table.add_row("─" * 20, "─" * 20)
                results_table.add_row("Re-ID Merges", str(reid_metrics.get("merges_total", 0)))
                results_table.add_row("Re-ID Reduction", str(reid_metrics.get("reduction", 0)))
        
        console.print("\n")
        console.print(Panel(results_table, title="[bold green]✓ Perception Complete[/bold green]", border_style="green"))
        console.print("\n")
        
        # Show entities summary
        console.print("[cyan]Detected Entities:[/cyan]")
        class_counts = {}
        for entity in result.entities:
            cls = entity.object_class.value if hasattr(entity.object_class, 'value') else str(entity.object_class)
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        for cls, count in sorted(class_counts.items()):
            console.print(f"  • {cls}: {count}")
        
        console.print(f"\n[green]✓ Results saved to: {output_dir}[/green]")
        
        # Export to graph if requested
        if not getattr(args, "skip_graph", False):
            try:
                from ...graph.builder import GraphBuilder
                
                neo4j_uri = args.neo4j_uri or settings.neo4j_uri
                neo4j_user = args.neo4j_user or settings.neo4j_user
                neo4j_password = args.neo4j_password or settings.get_neo4j_password()
                
                console.print("\n[cyan]Building knowledge graph...[/cyan]")
                builder = GraphBuilder(
                    neo4j_uri=neo4j_uri,
                    neo4j_user=neo4j_user,
                    neo4j_password=neo4j_password
                )
                
                if not getattr(args, "keep_db", False):
                    builder.clear_database()
                
                builder.build_from_perception(result)
                console.print("[green]✓ Graph built successfully[/green]\n")
                
            except ImportError as e:
                console.print(f"[yellow]⚠ Graph export skipped: {e}[/yellow]\n")
            except Exception as e:
                console.print(f"[red]✗ Graph export failed: {e}[/red]\n")

    except FileNotFoundError as e:
        console.print(f"[red]✗ Video file not found: {e}[/red]")
    except Exception as e:
        console.print(f"[red]✗ Pipeline execution failed: {e}[/red]")
        if getattr(args, "verbose", False):
            import traceback
            traceback.print_exc()


            console.print(f"[dim]{traceback.format_exc()}[/dim]")
