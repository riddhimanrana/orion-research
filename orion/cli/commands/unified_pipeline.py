"""Handle unified perception pipeline command (run)."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.box import ROUNDED
from rich.table import Table

from ...settings import OrionSettings
from ...perception.unified_pipeline import UnifiedPipeline

console = Console()


def handle_unified_pipeline(args: argparse.Namespace, settings: OrionSettings) -> None:
    """Run the unified 9-modality perception pipeline (Phases 1-5)."""
    video_path = args.video
    
    # Check video exists
    if not Path(video_path).exists():
        console.print(f"\n[red]❌ Error: Video not found: {video_path}[/red]")
        return
    
    try:
        console.print("\n" + "="*80)
        console.print("[bold cyan]🎯 ORION UNIFIED 9-MODALITY PERCEPTION PIPELINE[/bold cyan]")
        console.print("[bold]Phases 1-5: UnifiedFrame → Visualization → Scale → Tracking → Re-ID[/bold]")
        console.print("="*80 + "\n")
        
        # Initialize pipeline
        pipeline = UnifiedPipeline(
            video_path=video_path,
            max_frames=args.max_frames,
            use_rerun=not args.no_rerun
        )
        
        # Run pipeline
        results = pipeline.run(benchmark=args.benchmark)
        
        # Print summary
        summary_table = Table(
            title="[bold cyan]Pipeline Execution Summary[/bold cyan]",
            box=ROUNDED,
            expand=False
        )
        summary_table.add_column("Metric", style="cyan", no_wrap=True)
        summary_table.add_column("Value", style="green", justify="right")
        
        summary_table.add_row("Input Video", str(video_path))
        summary_table.add_row("Frames Processed", f"{results['frames_processed']}")
        summary_table.add_row("Raw Detections (Phase 1)", f"{results['total_detections']}")
        summary_table.add_row("Tracked Objects (Phase 4)", f"{results['tracked_objects']}")
        summary_table.add_row("Unified Objects (Phase 5)", f"{results['unified_objects']}")
        summary_table.add_row("Total Reduction Factor", f"{results['reduction_factor']:.1f}x")
        summary_table.add_row("Processing Time", f"{results['elapsed_time']:.1f}s")
        summary_table.add_row("FPS", f"{results['fps']:.1f}")
        
        console.print("\n", summary_table)
        console.print("\n" + "="*80)
        console.print("[green]✓ Pipeline execution complete[/green]\n")
        
        # Show timing breakdown if requested
        if args.benchmark and 'phase_times' in results:
            timing_table = Table(
                title="[bold yellow]Phase Timing Breakdown[/bold yellow]",
                box=ROUNDED,
                expand=False
            )
            timing_table.add_column("Phase", style="yellow", no_wrap=True)
            timing_table.add_column("Time (ms)", style="magenta", justify="right")
            
            for phase_name, phase_time in results['phase_times'].items():
                timing_table.add_row(phase_name, f"{phase_time:.1f}")
            
            console.print("\n", timing_table, "\n")
        
    except Exception as e:
        console.print(f"\n[red]❌ Pipeline error: {e}[/red]")
        import traceback
        traceback.print_exc()
