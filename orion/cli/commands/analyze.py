"""Analyze command - process video with the Orion pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..display import display_pipeline_results
from ..utils import prepare_runtime

if TYPE_CHECKING:
    from ...settings import OrionSettings

console = Console()


def _start_standard_qa(args, settings, neo4j_uri, neo4j_user, neo4j_password, console):
    """Start standard Q&A session (fallback when spatial intelligence not available)"""
    try:
        from ...video_qa import VideoQASystem

        qa = VideoQASystem(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            llm_model=getattr(args, "qa_model", None) or settings.qa_model,
        )
        qa.start_interactive_session()
    except ImportError:
        console.print("[red]✗ Q&A not available. Install: pip install ollama[/red]")
    except Exception as e:
        console.print(f"[red]✗ Q&A session failed: {e}[/red]")


def handle_analyze(args: argparse.Namespace, settings: OrionSettings) -> None:
    """Handle the analyze command - process video with the new modular pipeline."""
    # Import the new VideoPipeline
    from ...pipeline import VideoPipeline

    # Prepare runtime
    try:
        backend, _ = prepare_runtime(args.runtime or settings.runtime_backend)
    except Exception:
        return

    # Get Neo4j credentials
    neo4j_uri = args.neo4j_uri or settings.neo4j_uri
    neo4j_user = args.neo4j_user or settings.neo4j_user
    neo4j_password = args.neo4j_password or settings.get_neo4j_password()

    # Determine processing mode
    config = "balanced"
    if args.fast:
        config = "fast"
    elif args.accurate:
        config = "accurate"

    # Display analysis parameters
    params_table = Table(box=box.ROUNDED, show_header=False, border_style="cyan")
    params_table.add_column("Parameter", style="cyan bold", width=20)
    params_table.add_column("Value", style="yellow", width=60)

    params_table.add_row("Video", str(args.video))
    params_table.add_row("Mode", config)
    params_table.add_row("Runtime", backend)
    params_table.add_row("Output", args.output)

    # Show stage configuration
    skip_perception = getattr(args, "skip_perception", False)
    skip_semantic = getattr(args, "skip_semantic", False)
    skip_graph = getattr(args, "skip_graph", False)

    if skip_perception:
        params_table.add_row("Perception Stage", "[red]Skipped[/red]")
    if skip_semantic:
        params_table.add_row("Semantic Stage", "[red]Skipped[/red]")
    if skip_graph:
        params_table.add_row("Graph Stage", "[red]Skipped[/red]")

    if getattr(args, "keep_db", False):
        params_table.add_row("Database", "[yellow]Keeping existing data[/yellow]")
    
    # Show spatial memory if enabled
    if getattr(args, "use_spatial_memory", False):
        params_table.add_row("Spatial Memory", f"[green]✓ Enabled[/green] ({getattr(args, 'memory_dir', 'memory/spatial_intelligence')})")
    if getattr(args, "export_memgraph", False):
        params_table.add_row("Memgraph Export", "[green]✓ Enabled[/green]")

    # Show inspection mode if enabled
    if hasattr(args, "inspect") and args.inspect:
        params_table.add_row("Inspection Mode", f"[yellow]{args.inspect}[/yellow]")

    console.print("\n")
    console.print(Panel(params_table, title="[bold]Analysis Configuration[/bold]", border_style="cyan"))
    console.print("\n")

    # Create pipeline configuration
    config_dict = {
        "video_path": str(args.video),
        "output_dir": args.output,
        "neo4j": {
            "uri": neo4j_uri,
            "user": neo4j_user,
            "password": neo4j_password,
            "clear_db": not getattr(args, "keep_db", False),
        },
        "perception": {
            "mode": config,
            "runtime": backend,
        },
        "semantic": {
            "mode": config,
        },
    }

    # Initialize pipeline
    try:
        with VideoPipeline.from_config(config_dict) as pipeline:
            # Set inspection mode if requested
            if hasattr(args, "inspect") and args.inspect:
                pipeline.set_inspection_mode(args.inspect)

            # Run the pipeline
            console.print("[bold cyan]Starting pipeline execution...[/bold cyan]\n")

            # Skip stages based on arguments
            if skip_perception:
                console.print("[yellow]⊘ Skipping perception stage[/yellow]")
            if skip_semantic:
                console.print("[yellow]⊘ Skipping semantic stage[/yellow]")
            if skip_graph:
                console.print("[yellow]⊘ Skipping graph stage[/yellow]")

            # Execute pipeline
            results = pipeline.run(
                skip_perception=skip_perception,
                skip_semantic=skip_semantic,
                skip_graph=skip_graph,
            )

            # Display results
            if not getattr(args, "verbose", False):
                display_pipeline_results(results)

            # Start interactive mode if requested
            if getattr(args, "interactive", False) and results.get("success"):
                console.print("\n[bold cyan]════════════════════════════════════════════════[/bold cyan]")
                
                # Check if spatial memory/Memgraph export is requested
                use_spatial = getattr(args, "use_spatial_memory", False)
                use_memgraph = getattr(args, "export_memgraph", False)
                
                if use_spatial or use_memgraph:
                    console.print("[bold cyan]   Starting Spatial Intelligence Assistant[/bold cyan]")
                    console.print("[bold cyan]════════════════════════════════════════════════[/bold cyan]\n")
                    console.print("[yellow]NOTE: Spatial memory requires processing with 'orion research slam'[/yellow]")
                    console.print("[yellow]      The 'analyze' command uses Neo4j graph backend.[/yellow]\n")
                    
                    import subprocess
                    import sys
                    
                    assistant_script = Path(__file__).parent.parent.parent.parent / "scripts" / "spatial_intelligence_assistant.py"
                    
                    if assistant_script.exists():
                        console.print("[cyan]Checking for existing spatial memory...[/cyan]")
                        
                        memory_dir = Path(getattr(args, "memory_dir", "memory/spatial_intelligence"))
                        if memory_dir.exists() and (memory_dir / "entities.json").exists():
                            console.print(f"[green]✓ Found existing spatial memory at {memory_dir}[/green]\n")
                            
                            # Start interactive assistant
                            subprocess.run([sys.executable, str(assistant_script), "--interactive"])
                        else:
                            console.print(f"[yellow]⚠ No spatial memory found at {memory_dir}[/yellow]")
                            console.print("[yellow]  Process video with: orion research slam --video X --use-spatial-memory[/yellow]\n")
                            
                            # Fall back to standard Q&A
                            console.print("[cyan]Starting standard Q&A mode instead...[/cyan]\n")
                            _start_standard_qa(args, settings, neo4j_uri, neo4j_user, neo4j_password, console)
                    else:
                        console.print("[yellow]⚠ Spatial intelligence assistant not found. Using standard Q&A...[/yellow]\n")
                        _start_standard_qa(args, settings, neo4j_uri, neo4j_user, neo4j_password, console)
                else:
                    console.print("[bold cyan]       Starting Interactive Q&A Mode[/bold cyan]")
                    console.print("[bold cyan]════════════════════════════════════════════════[/bold cyan]\n")
                    _start_standard_qa(args, settings, neo4j_uri, neo4j_user, neo4j_password, console)

    except FileNotFoundError as e:
        console.print(f"[red]✗ Video file not found: {e}[/red]")
    except Exception as e:
        console.print(f"[red]✗ Pipeline execution failed: {e}[/red]")
        if getattr(args, "verbose", False):
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
