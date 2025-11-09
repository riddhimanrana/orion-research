"Command-line interface for the Orion research toolkit."

from __future__ import annotations

import argparse
import os
import secrets
import string
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout

from .models import AssetManager
from .runtime import select_backend, set_active_backend
from .settings import OrionSettings, SettingsError

# We avoid importing the heavy pipeline at module import time.
if TYPE_CHECKING:  # pragma: no cover
    from .run_pipeline import run_pipeline as _RunPipeline

console = Console()


def _display_detailed_results(results: dict) -> None:
    """Display comprehensive pipeline results with detailed metrics"""
    
    # Header Panel
    status_emoji = "✓" if results.get("success") else "✗"
    status_color = "green" if results.get("success") else "red"
    
    header = Panel(
        f"[bold {status_color}]{status_emoji} Pipeline Execution Complete[/bold {status_color}]",
        style=status_color,
        expand=False
    )
    console.print("\n", header)
    
    # ═══════════════════════════════════════════════════════════
    # PART 1: PERCEPTION STAGE
    # ═══════════════════════════════════════════════════════════
    if not results.get("part1", {}).get("skipped"):
        part1 = results.get("part1", {})
        perception_table = Table(
            title="[bold cyan]Part 1: Visual Perception[/bold cyan]",
            box=box.DOUBLE_EDGE,
            show_header=True,
            header_style="bold magenta",
            border_style="cyan",
            expand=True
        )
        perception_table.add_column("Metric", style="cyan", no_wrap=True, width=30)
        perception_table.add_column("Value", style="green", justify="right", width=20)
        perception_table.add_column("Details", style="yellow", width=50)
        
        # Video info
        perception_table.add_row(
            "Video Duration",
            f"{part1.get('duration_seconds', 0):.1f}s",
            f"{part1.get('total_frames', 0):,} frames @ {part1.get('fps', 0):.1f} FPS"
        )
        
        # Detection metrics
        num_objects = part1.get('num_objects', 0)
        num_unique = part1.get('unique_classes', 0)
        perception_table.add_row(
            "Objects Detected",
            f"{num_objects:,}",
            f"{num_unique} unique classes"
        )
        
        # Processing speed
        proc_time = part1.get('processing_time_seconds', 0)
        if proc_time > 0:
            fps = part1.get('total_frames', 0) / proc_time
            perception_table.add_row(
                "Processing Speed",
                f"{fps:.1f} FPS",
                f"Completed in {proc_time:.1f}s"
            )
        
        console.print("\n", perception_table)
    
    # ═══════════════════════════════════════════════════════════
    # PART 2: SEMANTIC UPLIFT
    # ═══════════════════════════════════════════════════════════
    if not results.get("part2", {}).get("skipped"):
        part2 = results.get("part2", {})
        
        # --- Main Semantic Metrics ---
        semantic_table = Table(
            title="[bold cyan]Part 2: Semantic Uplift & Knowledge Graph[/bold cyan]",
            box=box.DOUBLE_EDGE,
            show_header=True,
            header_style="bold magenta",
            border_style="cyan",
            expand=True
        )
        semantic_table.add_column("Metric", style="cyan", no_wrap=True, width=30)
        semantic_table.add_column("Value", style="green", justify="right", width=20)
        semantic_table.add_column("Details", style="yellow", width=50)
        
        # Entities
        num_entities = part2.get('num_entities', 0)
        num_descriptions = part2.get('num_descriptions', num_entities)  # Default: all entities described
        num_unique_classes = part2.get('num_unique_classes', 0)
        
        entity_detail = f"{num_descriptions} with descriptions"
        if num_unique_classes > 0:
            entity_detail += f", {num_unique_classes} classes"
        
        semantic_table.add_row(
            "Entities Created",
            f"{num_entities:,}",
            entity_detail
        )
        
        # Spatial co-location zones (regions where entities co-occur)
        num_zones = part2.get('num_spatial_zones', 0)
        if num_zones > 0:
            zone_detail = f"Regions where entities co-occur spatially & temporally"
        else:
            zone_detail = "No co-location zones detected (need 2+ entities near each other)"
        semantic_table.add_row(
            "Co-Location Zones",
            f"{num_zones:,}",
            zone_detail
        )
        
        # Class corrections (from perception phase, not LLM)
        num_corrections = part1.get('corrections_applied', 0)
        num_positioned = part2.get('num_entities_positioned', 0)
        correction_pct = (num_corrections / num_entities * 100) if num_entities > 0 else 0
        semantic_table.add_row(
            "Class Corrections",
            f"{num_corrections:,}",
            f"{correction_pct:.1f}% entities corrected (rule-based)"
        )
        
        # Entity positioning (entities with spatial zone classification)
        if num_positioned > 0:
            position_pct = (num_positioned / num_entities * 100) if num_entities > 0 else 0
            semantic_table.add_row(
                "Spatial Positioning",
                f"{num_positioned:,}",
                f"{position_pct:.1f}% entities with position tags"
            )
        
        # State changes
        num_states = part2.get('num_state_changes', 0)
        semantic_table.add_row(
            "State Changes",
            f"{num_states:,}",
            "Temporal state transitions detected"
        )
        
        # Causal links
        num_causal = part2.get('num_causal_links', 0)
        semantic_table.add_row(
            "Causal Links",
            f"{num_causal:,}",
            "Event causality relationships"
        )
        
        console.print("\n", semantic_table)
        
        # --- LLM Processing Details ---
        llm_table = Table(
            title="[bold yellow]LLM Processing Pipeline[/bold yellow]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold yellow",
            border_style="yellow"
        )
        llm_table.add_column("Stage", style="cyan", width=25)
        llm_table.add_column("Count", style="green", justify="right", width=15)
        llm_table.add_column("Notes", style="dim", width=60)
        
        windows_total = part2.get('llm_windows_total', 0)
        windows_composed = part2.get('llm_windows_composed', 0)
        windows_skipped = part2.get('llm_windows_skipped', 0)
        llm_batches = part2.get('llm_batches', 0)
        llm_calls = part2.get('llm_calls', 0)
        llm_latency = part2.get('llm_latency_seconds', 0)
        
        llm_table.add_row(
            "Total Windows",
            f"{windows_total:,}",
            "Temporal analysis windows"
        )
        llm_table.add_row(
            "Windows Processed",
            f"{windows_composed:,}",
            f"Skipped: {windows_skipped} (low activity/duplicates)"
        )
        llm_table.add_row(
            "Batch Operations",
            f"{llm_batches:,}",
            "Batched for efficiency"
        )
        llm_table.add_row(
            "LLM API Calls",
            f"{llm_calls:,}",
            f"Total latency: {llm_latency:.2f}s"
        )
        
        # Show async queue metrics if available
        if part2.get('async_queue_size'):
            llm_table.add_row(
                "Async Queue Processed",
                f"{part2.get('async_queue_size', 0):,}",
                f"Description tasks: {part2.get('async_descriptions_generated', 0)}"
            )
        
        console.print("\n", llm_table)
        
        # --- Graph Database Stats ---
        graph_stats = part2.get('graph_stats', {})
        if graph_stats:
            graph_table = Table(
                title="[bold green]Neo4j Knowledge Graph[/bold green]",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold green",
                border_style="green"
            )
            graph_table.add_column("Node/Relationship Type", style="cyan", width=30)
            graph_table.add_column("Count", style="green", justify="right", width=15)
            
            # Sort and display
            for key, value in sorted(graph_stats.items()):
                label = key.replace("_", " ").title()
                graph_table.add_row(label, f"{value:,}")
            
            console.print("\n", graph_table)
    
    # ═══════════════════════════════════════════════════════════
    # SUMMARY FOOTER
    # ═══════════════════════════════════════════════════════════
    summary_panel = Panel(
        f"[bold]Output:[/bold] {results.get('output_file', 'N/A')}\n"
        f"[bold]Runtime:[/bold] {results.get('runtime', 'N/A')}\n"
        f"[bold]Neo4j:[/bold] {results.get('neo4j_status', 'Unknown')}",
        title="[bold]Pipeline Artifacts[/bold]",
        style="blue",
        expand=False
    )
    console.print("\n", summary_panel)


def _import_video_qa() -> Any:
    """Import the unified video QA system when available."""
    try:
        from .video_qa import VideoQASystem  # type: ignore

        return VideoQASystem
    except Exception:  # pragma: no cover
        return None


def _prepare_runtime(requested: Optional[str]) -> Tuple[str, AssetManager]:
    """Prepare runtime with enhanced progress display"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("[cyan]Selecting runtime backend...", total=3)
        
        backend = select_backend(requested)
        set_active_backend(backend)
        progress.update(task, advance=1, description=f"[cyan]Selected: {backend}")
        
        manager = AssetManager()
        progress.update(task, advance=1, description=f"[yellow]Checking assets...")

        if manager.assets_ready(backend):
            progress.update(task, advance=1, description=f"[green]✓ Runtime '{backend}' ready", completed=3)
            time.sleep(0.3)
            return backend, manager

        progress.update(task, description=f"[yellow]Downloading models for '{backend}'...")
        try:
            manager.ensure_runtime_assets(backend)
            progress.update(task, advance=1, description=f"[green]✓ Models synchronized", completed=3)
        except Exception as exc:  # noqa: BLE001
            progress.stop()
            console.print(f"[red]✗ Failed to prepare models: {exc}[/red]")
            raise

        time.sleep(0.3)
        return backend, manager


def _prepare_runtime_for_backend(backend: str) -> AssetManager:
    """Prepare runtime assets for a specific backend with enhanced progress"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task(f"[cyan]Preparing {backend} runtime...", total=2)
        
        set_active_backend(backend)
        manager = AssetManager()
        progress.update(task, advance=1)

        if manager.assets_ready(backend):
            progress.update(task, advance=1, description=f"[green]✓ Runtime '{backend}' ready", completed=2)
            time.sleep(0.3)
            return manager

        progress.update(task, description=f"[yellow]Downloading {backend} models...")
        try:
            manager.ensure_runtime_assets(backend)
            progress.update(task, advance=1, description=f"[green]✓ Models synchronized", completed=2)
        except Exception as exc:  # noqa: BLE001
            progress.stop()
            console.print(f"[red]✗ Failed to prepare models: {exc}[/red]")
            raise

        time.sleep(0.3)
        return manager


def _handle_neo4j_command(args: argparse.Namespace) -> None:
    """Handle Neo4j service management commands with enhanced feedback"""
    action = getattr(args, "neo4j_action", None)
    if action is None:
        console.print("[red]No Neo4j action provided. Use 'orion services neo4j --help'.[/red]")
        return

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            if action == "start":
                task = progress.add_task("[cyan]Checking Neo4j container...", total=None)
                
                check_result = subprocess.run(
                    ['docker', 'ps', '-a', '--filter', 'name=orion-neo4j'],
                    capture_output=True, text=True
                )

                if 'orion-neo4j' in check_result.stdout:
                    status_result = subprocess.run(
                        ['docker', 'ps', '--filter', 'name=orion-neo4j'],
                        capture_output=True, text=True
                    )

                    if 'orion-neo4j' in status_result.stdout:
                        progress.update(task, description="[yellow]Neo4j already running")
                        progress.stop()
                        console.print("[green]✓ Neo4j container already running[/green]")
                    else:
                        progress.update(task, description="[cyan]Starting Neo4j...")
                        subprocess.run(['docker', 'start', 'orion-neo4j'], check=True,
                                     capture_output=True, text=True)
                        progress.stop()
                        console.print("[green]✓ Neo4j container started successfully[/green]")
                        console.print("[dim]   Access at: http://localhost:7474[/dim]")
                else:
                    progress.stop()
                    console.print("[red]✗ Neo4j container not found[/red]")
                    console.print("[yellow]   Run 'orion init' first to create the container[/yellow]")

            elif action == "stop":
                task = progress.add_task("[cyan]Stopping Neo4j...", total=None)
                subprocess.run(['docker', 'stop', 'orion-neo4j'], check=True,
                             capture_output=True, text=True)
                progress.stop()
                console.print("[green]✓ Neo4j container stopped[/green]")

            elif action == "status":
                task = progress.add_task("[cyan]Checking status...", total=None)
                result = subprocess.run(['docker', 'ps', '--filter', 'name=orion-neo4j'],
                                      capture_output=True, text=True)
                progress.stop()
                
                status_table = Table(box=box.ROUNDED, show_header=False)
                status_table.add_column("Item", style="cyan", width=20)
                status_table.add_column("Value", style="green", width=50)
                
                if 'orion-neo4j' in result.stdout:
                    status_table.add_row("Status", "[green]● Running[/green]")
                    status_table.add_row("Browser UI", "http://localhost:7474")
                    status_table.add_row("Bolt Port", "bolt://localhost:7687")
                    console.print(status_table)
                else:
                    status_table.add_row("Status", "[red]○ Stopped[/red]")
                    console.print(status_table)
                    console.print("[yellow]Run 'orion services neo4j start' to start[/yellow]")

            elif action == "restart":
                task = progress.add_task("[cyan]Restarting Neo4j...", total=None)
                subprocess.run(['docker', 'restart', 'orion-neo4j'], check=True,
                             capture_output=True, text=True)
                progress.stop()
                console.print("[green]✓ Neo4j container restarted[/green]")

    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Docker command failed: {e.stderr}[/red]")
    except FileNotFoundError:
        console.print("[red]✗ Docker not found. Please install Docker first.[/red]")


def _handle_ollama_command(args: argparse.Namespace) -> None:
    """Handle Ollama service management commands with enhanced feedback"""
    action = getattr(args, "ollama_action", None)
    if action is None:
        console.print("[red]No Ollama action provided. Use 'orion services ollama --help'.[/red]")
        return

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            if action == "start":
                task = progress.add_task("[cyan]Checking Ollama...", total=None)
                check_result = subprocess.run(
                    ['docker', 'ps', '-a', '--filter', 'name=orion-ollama'],
                    capture_output=True, text=True
                )

                if 'orion-ollama' in check_result.stdout:
                    status_result = subprocess.run(
                        ['docker', 'ps', '--filter', 'name=orion-ollama'],
                        capture_output=True, text=True
                    )

                    if 'orion-ollama' in status_result.stdout:
                        progress.update(task, description="[yellow]Ollama already running")
                        progress.stop()
                        console.print("[green]✓ Ollama container already running[/green]")
                    else:
                        progress.update(task, description="[cyan]Starting Ollama...")
                        subprocess.run(['docker', 'start', 'orion-ollama'], check=True,
                                     capture_output=True, text=True)
                        progress.stop()
                        console.print("[green]✓ Ollama container started successfully[/green]")
                        console.print("[dim]   API available at: http://localhost:11434[/dim]")
                else:
                    progress.stop()
                    console.print("[red]✗ Ollama container not found[/red]")
                    console.print("[yellow]   Run 'orion init' first to set up Ollama[/yellow]")

            elif action == "stop":
                task = progress.add_task("[cyan]Stopping Ollama...", total=None)
                subprocess.run(['docker', 'stop', 'orion-ollama'], check=True,
                             capture_output=True, text=True)
                progress.stop()
                console.print("[green]✓ Ollama container stopped[/green]")

            elif action == "status":
                task = progress.add_task("[cyan]Checking status...", total=None)
                result = subprocess.run(['docker', 'ps', '--filter', 'name=orion-ollama'],
                                      capture_output=True, text=True)
                progress.stop()
                
                status_table = Table(box=box.ROUNDED, show_header=False)
                status_table.add_column("Item", style="cyan", width=20)
                status_table.add_column("Value", style="green", width=50)
                
                if 'orion-ollama' in result.stdout:
                    status_table.add_row("Status", "[green]● Running[/green]")
                    status_table.add_row("API Endpoint", "http://localhost:11434")
                    status_table.add_row("Model", "gemma3:4b")
                    console.print(status_table)
                else:
                    status_table.add_row("Status", "[red]○ Stopped[/red]")
                    console.print(status_table)
                    console.print("[yellow]Run 'orion services ollama start' to start[/yellow]")

            elif action == "restart":
                task = progress.add_task("[cyan]Restarting Ollama...", total=None)
                subprocess.run(['docker', 'restart', 'orion-ollama'], check=True,
                             capture_output=True, text=True)
                progress.stop()
                console.print("[green]✓ Ollama container restarted[/green]")

    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Docker command failed: {e.stderr}[/red]")
    except FileNotFoundError:
        console.print("[red]✗ Docker not found. Please install Docker first.[/red]")


def _handle_config_command(args: argparse.Namespace, settings: OrionSettings) -> int:
    subcommand = getattr(args, "config_command", None)
    if subcommand is None:
        console.print(
            "[red]No configuration subcommand provided. Use 'orion config --help'.[/red]"
        )
        return 1

    if subcommand == "show":
        table = Table(title="Orion Configuration", box=box.ROUNDED)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        for key, value in settings.iter_display_items():
            table.add_row(key, value)
        console.print(table)
        return 0

    if subcommand == "set":
        try:
            settings.set_value(args.key, args.value)
            settings.save()
        except SettingsError as exc:
            console.print(f"[red]{exc}[/red]")
            return 1
        console.print(f"[green]Updated {args.key}.[/green]")
        return 0

    if subcommand == "reset":
        defaults = OrionSettings()
        defaults.save()
        console.print("[green]Configuration reset to defaults.[/green]")
        return 0

    if subcommand == "path":
        console.print(str(OrionSettings.config_path()))
        return 0

    console.print(f"[red]Unknown configuration subcommand '{subcommand}'.[/red]")
    return 1


def print_banner() -> None:
    """Print the Orion banner before executing commands."""
    console.print("\n")
    with console.status("[#9D66FF]Loading Orion CLI...", spinner="dots"):
        time.sleep(0.8)

    console.print()

    outer_ring = "#8645F7"
    inner_fill = "#8645F7"
    lines_shade = "#9D66FF"
    sphere_fill = "#DFE8FF"
    sphere_highlight = "#FDFEFF"
    text_white = "white"

    logo_art = [
        f"   [bold {outer_ring}]██████[/]",
        f" [bold {outer_ring}]██[{inner_fill}]▓▓▓▓▓▓[{outer_ring}]██[/]",
        f"[bold {outer_ring}]██[{inner_fill}]▓[{lines_shade}]▒[{sphere_highlight}]██[{sphere_fill}]▓▓[{lines_shade}]▒[{inner_fill}]▓[{outer_ring}]██[/]",
        f"[bold {outer_ring}]██[{inner_fill}]▓[{lines_shade}]▒[{sphere_fill}]▓▓[{sphere_fill}]▓▓[{lines_shade}]▒[{inner_fill}]▓[{outer_ring}]██[/]",
        f" [bold {outer_ring}]██[{inner_fill}]▓▓▓▓▓▓[{outer_ring}]██[/]",
        f"   [bold {outer_ring}]██████[/]",
    ]

    text_lines = [
        " ██████╗ ██████╗ ██╗ ██████╗ ███╗   ██╗",
        "██╔═══██╗██╔══██╗██║██╔═══██╗████╗  ██║",
        "██║   ██║██████╔╝██║██║   ██║██╔██╗ ██║",
        "██║   ██║██╔══██╗██║██║   ██║██║╚██╗██║",
        "╚██████╔╝██║  ██║██║╚██████╔╝██║ ╚████║",
        " ╚═════╝ ╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝",
    ]

    logo = Group(*(Text.from_markup(line) for line in logo_art))
    orion_text = Text("\n".join(text_lines), style=f"bold {text_white}")

    grid = Table.grid(expand=False, padding=(0, 4))
    grid.add_column()
    grid.add_column()
    grid.add_row(logo, orion_text)

    subtitle_table = Table.grid(expand=False, padding=(0, 4))
    subtitle_table.add_column(justify="center", width=50)
    subtitle_table.add_row(Text("CLI Version 0.1.0", style="dim"))
    subtitle_table.add_row(
        Text("Local Visual Intelligence at Speed", style=f"italic {outer_ring}")
    )

    render_group = Group(grid, Text(""), subtitle_table)

    console.print(render_group)
    console.print()


def show_models() -> None:
    """Display detailed model information"""
    table = Table(
        title="[bold cyan]Orion Model Architecture[/bold cyan]",
        box=box.DOUBLE_EDGE,
        show_header=True,
        header_style="bold magenta",
        border_style="cyan",
        expand=True
    )
    table.add_column("Model", style="cyan bold", no_wrap=True, width=20)
    table.add_column("Purpose", style="yellow", width=25)
    table.add_column("Size", style="green", justify="right", width=12)
    table.add_column("Performance", style="magenta", width=20)
    table.add_column("Details", style="dim", width=40)

    table.add_row(
        "YOLO11x",
        "Object Detection",
        "50MB",
        "⚡⚡⚡ Very Fast",
        "Real-time detection, 80+ classes"
    )
    table.add_row(
        "FastVLM-0.5B",
        "Visual Description",
        "~600MB",
        "⚡⚡ Fast",
        "Generates semantic descriptions"
    )
    table.add_row(
        "ResNet50",
        "Visual Embeddings",
        "~100MB",
        "⚡⚡⚡ Very Fast",
        "Appearance fingerprinting (512-dim)"
    )
    table.add_row(
        "CLIP (OpenAI)",
        "Text Embeddings",
        "~512MB",
        "⚡⚡ Fast",
        "Semantic text understanding (512-dim)"
    )
    table.add_row(
        "Gemma3:4b",
        "Q&A & Reasoning",
        "~1.6GB",
        "⚡ Medium",
        "Natural language interface via Ollama"
    )

    console.print("\n", table)
    
    # Add architecture diagram
    arch_panel = Panel(
        "[bold]Pipeline Flow:[/bold]\n"
        "1. YOLO11x → Detect objects in video frames\n"
        "2. ResNet50 → Generate visual embeddings\n"
        "3. FastVLM → Describe entities semantically\n"
        "4. CLIP → Embed descriptions for similarity\n"
        "5. Gemma3:4b → Answer questions about the scene",
        title="[bold cyan]Architecture Overview[/bold cyan]",
        border_style="cyan",
        expand=False
    )
    console.print("\n", arch_panel, "\n")


def show_modes() -> None:
    """Display detailed processing mode information"""
    table = Table(
        title="[bold cyan]Processing Modes[/bold cyan]",
        box=box.DOUBLE_EDGE,
        show_header=True,
        header_style="bold magenta",
        border_style="cyan",
        expand=True
    )
    table.add_column("Mode", style="cyan bold", width=15)
    table.add_column("Detection FPS", style="yellow", justify="center", width=15)
    table.add_column("Description Freq", style="green", justify="center", width=18)
    table.add_column("Use Case", style="magenta", width=35)
    table.add_column("Trade-offs", style="dim", width=40)

    table.add_row(
        "fast",
        "3 FPS",
        "Every 10th frame",
        "Long videos, quick overview",
        "Lower detail, faster processing"
    )
    table.add_row(
        "balanced",
        "5 FPS",
        "Every 5th frame",
        "General use ⭐ (recommended)",
        "Good balance of speed & accuracy"
    )
    table.add_row(
        "accurate",
        "10 FPS",
        "Every 2nd frame",
        "Short clips, high detail",
        "Best quality, slower processing"
    )

    console.print("\n", table)
    
    # Add recommendation panel
    rec_panel = Panel(
        "[bold]Recommendations:[/bold]\n"
        "• [yellow]fast[/yellow]: 30+ min videos, exploratory analysis\n"
        "• [green]balanced[/green]: Most use cases, 5-30 min videos\n"
        "• [magenta]accurate[/magenta]: <5 min clips, critical analysis",
        title="[bold cyan]Mode Selection Guide[/bold cyan]",
        border_style="cyan",
        expand=False
    )
    console.print("\n", rec_panel, "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orion",
        description="Orion Video Analysis Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  orion analyze video.mp4                     # Run full pipeline
  orion analyze video.mp4 -i                  # With interactive Q&A
  orion analyze video.mp4 --fast              # Fast mode
  orion qa                                    # Q&A only mode
  orion models                                # Show model info

For more help: orion <command> --help
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a video")
    analyze_parser.add_argument("video", help="Path to video file")
    analyze_parser.add_argument("--fast", action="store_true", help="Use fast mode")
    analyze_parser.add_argument(
        "--accurate", action="store_true", help="Use accurate mode"
    )
    analyze_parser.add_argument(
        "-i", "--interactive", action="store_true", help="Start Q&A after processing"
    )
    analyze_parser.add_argument(
        "--skip-perception", action="store_true", help="Skip visual perception (Part 1)"
    )
    analyze_parser.add_argument(
        "--skip-graph", action="store_true", help="Skip knowledge graph (Part 2)"
    )
    analyze_parser.add_argument(
        "--keep-db", action="store_true", help="Keep existing Neo4j data"
    )
    
    # 3D Perception options
    analyze_parser.add_argument(
        "--enable-3d", action="store_true", 
        help="Enable 3D perception (depth estimation, 3D coordinates, occlusion detection)"
    )
    analyze_parser.add_argument(
        "--depth-model", choices=["midas", "zoe"], default="midas",
        help="Depth estimation model (midas=fast, zoe=accurate, default: midas)"
    )
    analyze_parser.add_argument(
        "--enable-hands", action="store_true",
        help="Enable hand tracking with MediaPipe (requires --enable-3d)"
    )
    analyze_parser.add_argument(
        "--enable-occlusion", action="store_true",
        help="Enable occlusion detection (requires --enable-3d)"
    )
    
    analyze_parser.add_argument(
        "-o", "--output", default="data/testing", help="Output directory"
    )
    analyze_parser.add_argument(
        "--neo4j-uri", help="Neo4j connection URI (defaults to config)"
    )
    analyze_parser.add_argument(
        "--neo4j-user", help="Neo4j username (defaults to config)"
    )
    analyze_parser.add_argument(
        "--neo4j-password", help="Neo4j password (defaults to config)"
    )
    analyze_parser.add_argument(
        "--qa-model", help="Ollama model for interactive Q&A (defaults to config)"
    )
    analyze_parser.add_argument(
        "--embedding-backend",
        choices=["auto", "ollama", "sentence-transformer"],
        help="Embedding backend to use for semantic search",
    )
    analyze_parser.add_argument(
        "--embedding-model",
        help="Embedding model identifier (Ollama or sentence-transformer, defaults to config)",
    )
    analyze_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )
    analyze_parser.add_argument(
        "--runtime",
        help="Select runtime backend (auto or torch; defaults to config)",
    )

    qa_parser = subparsers.add_parser("qa", help="Q&A mode only")
    qa_parser.add_argument("--model", help="Ollama model to use (defaults to config)")
    qa_parser.add_argument(
        "--neo4j-uri", help="Neo4j connection URI (defaults to config)"
    )
    qa_parser.add_argument("--neo4j-user", help="Neo4j username (defaults to config)")
    qa_parser.add_argument(
        "--neo4j-password", help="Neo4j password (defaults to config)"
    )
    qa_parser.add_argument(
        "--embedding-backend",
        choices=["auto", "ollama", "sentence-transformer"],
        help="Embedding backend to use for semantic search",
    )
    qa_parser.add_argument(
        "--embedding-model",
        help="Embedding model identifier (defaults to config)",
    )
    qa_parser.add_argument(
        "--runtime",
        help="Select runtime backend (auto or torch; defaults to config)",
    )

    subparsers.add_parser("models", help="Show model information")
    subparsers.add_parser("modes", help="Show processing modes")

    # Service management commands
    services_parser = subparsers.add_parser("services", help="Manage Orion services (Neo4j, Ollama)")
    services_subparsers = services_parser.add_subparsers(dest="service_command", help="Service actions")

    # Neo4j management
    neo4j_parser = services_subparsers.add_parser("neo4j", help="Manage Neo4j service")
    neo4j_subparsers = neo4j_parser.add_subparsers(dest="neo4j_action", help="Neo4j actions")
    neo4j_subparsers.add_parser("start", help="Start Neo4j container")
    neo4j_subparsers.add_parser("stop", help="Stop Neo4j container")
    neo4j_subparsers.add_parser("status", help="Check Neo4j container status")
    neo4j_subparsers.add_parser("restart", help="Restart Neo4j container")

    # Ollama management
    ollama_parser = services_subparsers.add_parser("ollama", help="Manage Ollama service")
    ollama_subparsers = ollama_parser.add_subparsers(dest="ollama_action", help="Ollama actions")
    ollama_subparsers.add_parser("start", help="Start Ollama container")
    ollama_subparsers.add_parser("stop", help="Stop Ollama container")
    ollama_subparsers.add_parser("status", help="Check Ollama container status")
    ollama_subparsers.add_parser("restart", help="Restart Ollama container")
    
    # Enhanced status command
    status_parser = subparsers.add_parser(
        "status",
        help="Comprehensive system and service status check"
    )
    
    index_parser = subparsers.add_parser("index", help="Create vector indexes and backfill embeddings")
    
    # Benchmark evaluation command
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Evaluate Orion against standard benchmarks (Action Genome, VSGR, PVSG)"
    )
    benchmark_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["action-genome", "vsgr", "pvsg"],
        help="Benchmark dataset to evaluate on",
    )
    benchmark_parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to benchmark dataset directory",
    )
    benchmark_parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for evaluation results (default: results/)",
    )
    benchmark_parser.add_argument(
        "--video-ids",
        nargs="+",
        help="Specific video IDs to evaluate (optional)",
    )
    benchmark_parser.add_argument(
        "--max-videos",
        type=int,
        help="Maximum number of videos to evaluate (optional)",
    )
    benchmark_parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for entity matching (default: 0.5)",
    )
    benchmark_parser.add_argument(
        "--tiou-threshold",
        type=float,
        default=0.3,
        help="Temporal IoU threshold for event matching (default: 0.3)",
    )
    index_parser.add_argument(
        "--embedding-backend",
        choices=["auto", "ollama", "sentence-transformer"],
        help="Embedding backend to use for semantic indexing",
    )
    index_parser.add_argument(
        "--embedding-model", help="Embedding model identifier (defaults to config)")
    init_parser = subparsers.add_parser(
        "init", help="Initialize Orion (download models, setup environment)"
    )
    init_parser.add_argument(
        "--runtime",
        help="Select runtime backend to prepare (auto or torch; defaults to config)",
    )

    config_parser = subparsers.add_parser(
        "config", help="Inspect or update Orion configuration"
    )
    config_subparsers = config_parser.add_subparsers(
        dest="config_command", help="Configuration actions"
    )
    config_subparsers.required = True

    config_subparsers.add_parser(
        "show", help="Display the current configuration values"
    )
    config_subparsers.add_parser("path", help="Print the configuration file path")
    config_subparsers.add_parser("reset", help="Reset configuration to defaults")

    config_set_parser = config_subparsers.add_parser(
        "set", help="Update a configuration value"
    )
    config_set_parser.add_argument("key", help="Configuration key (e.g., neo4j.uri)")
    config_set_parser.add_argument("value", help="New value")

    return parser


def _run_command(command, description):
    """Runs a shell command and returns its output."""
    try:
        console.print(f"\n>> {description}...")
        process = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        return process.stdout.strip()
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print(f"[red]Stderr: {e.stderr}[/red]")
        return None

def _is_docker_daemon_running():
    """Checks if the Docker daemon is running."""
    console.print("[dim]Checking Docker daemon status...[/dim]")
    return _run_command("docker info", "Checking Docker status") is not None

def _is_container_running(container_name):
    """Checks if a Docker container is running."""
    output = _run_command(
        f"docker ps -q -f name={container_name}",
        f"Checking for container: {container_name}",
    )
    return output is not None and output != ""

def _prompt_user(question):
    """Prompts the user for a yes/no answer."""
    from rich.prompt import Confirm
    return Confirm.ask(question, default=True)

def _setup_neo4j():
    """Sets up the Neo4j container."""
    NEO4J_CONTAINER_NAME = "orion-neo4j"
    NEO4J_IMAGE = "neo4j:5"
    if _is_container_running(NEO4J_CONTAINER_NAME):
        console.print(f"[green]✓ Neo4j container '{NEO4J_CONTAINER_NAME}' is already running.[/green]")
        return

    if _prompt_user(
        f"Neo4j container '{NEO4J_CONTAINER_NAME}' not found. Would you like to create and start it?"
    ):
        _run_command(f"docker pull {NEO4J_IMAGE}", f"Pulling Neo4j image: {NEO4J_IMAGE}")
        _run_command(
            f"docker run -d --name {NEO4J_CONTAINER_NAME} "
            f"-p 7474:7474 -p 7687:7687 "
            f"-e NEO4J_AUTH=neo4j/orion123 "
            f"{NEO4J_IMAGE}",
            "Starting Neo4j container",
        )
        console.print("[green]✓ Neo4j container started.[/green]")

def _setup_ollama():
    """Checks for local Ollama installation and pulls models."""
    OLLAMA_MODELS = ["gemma3:4b"]

    # Check if ollama is installed
    if _run_command("ollama --version", "Checking for local Ollama installation") is None:
        console.print("[red]✗ Ollama is not installed.[/red]")
        console.print("Please install it from https://ollama.com and then run this command again.")
        sys.exit(1)

    console.print("[green]✓ Ollama is installed.[/green]")

    # Pull models
    for model in OLLAMA_MODELS:
        console.print(f"\n-- Checking for Ollama model: {model} --")
        _run_command(
            f"ollama pull {model}",
            f"Pulling Ollama model: {model}",
        )

def _handle_init_command(args: argparse.Namespace):
    """Handles the init command."""
    console.print("\n[bold cyan]🚀 Orion Initialization[/bold cyan]\n")

    # Pre-flight check: Verify all required services are available
    console.print("[bold]Step 1: Pre-flight Check - Verifying Prerequisites[/bold]\n")
    from .auto_config import AutoConfiguration
    config = AutoConfiguration()
    results = config.detect_all_services()
    
    neo4j_ok, neo4j_msg = results["neo4j"]
    ollama_ok, ollama_msg, _ = results["ollama"]
    docker_ok, docker_msg = results["docker"]
    
    status_table = Table(title="Service Status", box=box.ROUNDED)
    status_table.add_column("Service", style="cyan")
    status_table.add_column("Status", style="green")
    status_table.add_column("Details", style="yellow")
    
    status_table.add_row(
        "Neo4j",
        "[green]✓ Running[/green]" if neo4j_ok else "[red]✗ Not available[/red]",
        neo4j_msg
    )
    status_table.add_row(
        "Ollama",
        "[green]✓ Running[/green]" if ollama_ok else "[red]✗ Not available[/red]",
        ollama_msg
    )
    status_table.add_row(
        "Docker",
        "[green]✓ Available[/green]" if docker_ok else "[yellow]⚠ Not available[/yellow]",
        docker_msg
    )
    
    console.print(status_table)

    # Check for sentence-transformers
    try:
        import sentence_transformers
    except ImportError:
        console.print("[red]✗ sentence-transformers library not found.[/red]")
        console.print("Please install it by running: pip install sentence-transformers==2.2.2")
        sys.exit(1)

    if not docker_ok:
        console.print("\n[red]✗ Docker is not installed.[/red]")
        console.print("Please install Docker Desktop and then run this command again.")
        sys.exit(1)

    if not _is_docker_daemon_running():
        console.print("\n[red]✗ Docker daemon is not running.[/red]")
        console.print("Please start Docker Desktop and then run this command again.")
        sys.exit(1)

    console.print("\n--- Setting up Neo4j ---")
    _setup_neo4j()

    console.print("\n--- Setting up Ollama ---")
    _setup_ollama()
    
    console.print("\n[bold]Step 2: Detecting hardware and selecting runtime...[/bold]\n")

    # Auto-detect the best runtime for this system (ignore CLI args for auto-detection)
    try:
        from .runtime import select_backend
        backend: str = select_backend(None)  # Force auto-detection
        console.print(f"[green]Selected runtime: {backend}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to select runtime: {e}[/red]")
        sys.exit(1)

    # Download models and configure environment
    console.print("[bold]Preparing runtime and downloading models...[/bold]\n")
    try:
        manager = _prepare_runtime_for_backend(backend)
    except Exception:
        sys.exit(1)

    console.print(f"[green]Runtime '{backend}' assets are ready.[/green]\n")

    # Print setup summary with correct model paths
    summary_table = Table(title="Model Assets", box=box.ROUNDED)
    summary_table.add_column("Component", style="cyan", no_wrap=True)
    summary_table.add_column("Status", style="green", justify="center")
    summary_table.add_column("Details", style="magenta")

    # Show YOLO11x
    yolo_path = manager.get_asset_path("yolo11x") if "yolo11x" in manager._manifest else "Not found"
    summary_table.add_row("YOLO11x", "✓" if Path(yolo_path).exists() else "✗", str(yolo_path))

    # Show FastVLM (MLX or Torch)
    fastvlm_asset_name = "fastvlm-0.5b-mlx" if backend == "mlx" else "fastvlm-0.5b"
    fastvlm_path = manager.get_asset_path(fastvlm_asset_name) if fastvlm_asset_name in manager._manifest else "Not found"
    summary_table.add_row("FastVLM-0.5B", "✓" if Path(fastvlm_path).exists() else "✗", str(fastvlm_path))

    # Show Gemma3:4b and CLIP
    summary_table.add_row("gemma3:4b", "✓", "Available in Ollama")
    summary_table.add_row("CLIP (OpenAI)", "✓", "Available via sentence-transformers")

    console.print(summary_table)
    
    console.print("\n[green]✓ Orion initialization complete.[/green]")

def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)

    try:
        with console.status(
            "[dim]Loading Orion configuration...[/dim]", spinner="dots"
        ):
            settings = OrionSettings.load()
    except SettingsError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    if args.command == "config":
        status = _handle_config_command(args, settings)
        if status != 0:
            sys.exit(status)
        return

    if args.command:
        print_banner()

    if args.command == "analyze":
        from .pipeline import VideoPipeline

        VideoQASystem = _import_video_qa()

        try:
            backend, _ = _prepare_runtime(args.runtime or settings.runtime_backend)
        except Exception:
            return

        neo4j_uri = args.neo4j_uri or settings.neo4j_uri
        neo4j_user = args.neo4j_user or settings.neo4j_user
        neo4j_password = args.neo4j_password or settings.get_neo4j_password()
        qa_model = args.qa_model or settings.qa_model

        # Display analysis parameters
        params_table = Table(box=box.ROUNDED, show_header=False, border_style="cyan")
        params_table.add_column("Parameter", style="cyan bold", width=20)
        params_table.add_column("Value", style="yellow", width=60)
        
        config = "balanced"
        if args.fast:
            config = "fast"
        elif args.accurate:
            config = "accurate"
            
        params_table.add_row("Video", args.video)
        params_table.add_row("Mode", config)
        params_table.add_row("Runtime", backend)
        params_table.add_row("Output", args.output)
        if args.skip_perception:
            params_table.add_row("Perception", "[red]Skipped[/red]")
        if args.skip_graph:
            params_table.add_row("Graph Build", "[red]Skipped[/red]")
        if args.keep_db:
            params_table.add_row("Database", "[yellow]Keeping existing data[/yellow]")
        if args.enable_3d:
            depth_model_str = args.depth_model.upper()
            features = []
            if args.enable_hands:
                features.append("hands")
            if args.enable_occlusion:
                features.append("occlusion")
            features_str = ", ".join(features) if features else "depth only"
            params_table.add_row("3D Perception", f"[green]Enabled[/green] ({depth_model_str}, {features_str})")
        
        console.print("\n")
        console.print(Panel(params_table, title="[bold]Analysis Configuration[/bold]", border_style="cyan"))
        console.print("\n")

        # Create config for new VideoPipeline
        pipeline_config = {
            "video_path": args.video,
            "perception": {
                "mode": config,
                "enable_3d": args.enable_3d,
                "depth_model": args.depth_model if args.enable_3d else None,
                "enable_hands": args.enable_hands if args.enable_3d else False,
                "enable_occlusion": args.enable_occlusion if args.enable_3d else False,
            },
            "semantic": {"mode": config},
            "neo4j": {
                "uri": neo4j_uri,
                "user": neo4j_user,
                "password": neo4j_password,
                "clear_db": not args.keep_db,
            }
        }
        
        # Create and run pipeline
        pipeline = VideoPipeline.from_config(pipeline_config)
        results = pipeline.run(
            skip_perception=args.skip_perception,
            skip_semantic=False,
            skip_graph=args.skip_graph
        )

        # Display comprehensive results
        if not args.verbose:
            _display_detailed_results(results)

        if args.interactive and results.get("success"):
            console.print("\n[bold cyan]════════════════════════════════════════════════[/bold cyan]")
            console.print("[bold cyan]       Starting Interactive Q&A Mode[/bold cyan]")
            console.print("[bold cyan]════════════════════════════════════════════════[/bold cyan]\n")
            if VideoQASystem is None:
                console.print(
                    "[red]✗ Q&A not available. Install: pip install ollama[/red]" 
                )
            else:
                qa = VideoQASystem(
                    neo4j_uri=neo4j_uri,
                    neo4j_user=neo4j_user,
                    neo4j_password=neo4j_password,
                    llm_model=qa_model,
                )
                qa.start_interactive_session()

    elif args.command == "qa":
        console.print("\n[bold cyan]Starting Q&A mode...[/bold cyan]\n")
        VideoQASystem = _import_video_qa()
        if VideoQASystem is None:
            console.print("[red]Q&A not available. Install: pip install ollama")
        else:
            try:
                backend, _ = _prepare_runtime(args.runtime or settings.runtime_backend)
            except Exception:
                return
            console.print(f"[dim]Using runtime backend: {backend}[/dim]\n")
            qa = VideoQASystem(
                neo4j_uri=args.neo4j_uri or settings.neo4j_uri,
                neo4j_user=args.neo4j_user or settings.neo4j_user,
                neo4j_password=args.neo4j_password or settings.get_neo4j_password(),
                llm_model=args.model or settings.qa_model,
            )
            qa.start_interactive_session()

    elif args.command == "models":
        show_models()

    elif args.command == "modes":
        show_modes()

    elif args.command == "status":
        from .auto_config import status_command

        # Show comprehensive status including services and database
        status_command(args)

    elif args.command == "services":
        if args.service_command == "neo4j":
            _handle_neo4j_command(args)
        elif args.service_command == "ollama":
            _handle_ollama_command(args)
        else:
            console.print("[red]No service command provided. Use 'orion services --help'.[/red]")
            return

    elif args.command == "index":
        from .vector_indexing import backfill_embeddings
        try:
            backend, _ = _prepare_runtime(args.runtime or settings.runtime_backend)
        except Exception:
            return
        console.print("\n[bold cyan]Creating vector indexes and backfilling embeddings...[/bold cyan]\n")
        embedding_backend = args.embedding_backend or settings.embedding_backend
        prefer_ollama = embedding_backend in ("auto", "ollama")
        model_id = args.embedding_model or settings.embedding_model
        try:
            e_total, e_done, s_total, s_done = backfill_embeddings(
                settings.neo4j_uri,
                settings.neo4j_user,
                settings.get_neo4j_password(),
                prefer_ollama=prefer_ollama,
                backend=embedding_backend,
                model_name=model_id,
            )
            console.print(
                f"[green]Entities:[/green] {e_done}/{e_total} embedded | [green]Scenes:[/green] {s_done}/{s_total} embedded"
            )
        except Exception as exc:
            console.print(f"[red]Indexing failed: {exc}[/red]")
            return

    elif args.command == "benchmark":
        from pathlib import Path as PathLib
        from .evaluation.benchmark_runner import BenchmarkRunner
        
        console.print("\n[bold cyan]Running benchmark evaluation...[/bold cyan]\n")
        
        runner = BenchmarkRunner(
            dataset_name=args.dataset,
            data_dir=PathLib(args.data_dir),
            output_dir=PathLib(args.output_dir),
            neo4j_uri=settings.neo4j_uri,
            neo4j_user=settings.neo4j_user,
            neo4j_password=settings.get_neo4j_password(),
            iou_threshold=args.iou_threshold,
            tiou_threshold=args.tiou_threshold,
        )
        
        try:
            results = runner.run(
                video_ids=args.video_ids,
                max_videos=args.max_videos,
            )
            console.print("\n[bold green]✓ Benchmark evaluation complete![/bold green]\n")
        except Exception as e:
            console.print(f"\n[bold red]✗ Benchmark evaluation failed:[/bold red] {e}\n")
            import traceback
            traceback.print_exc()
        finally:
            runner.close()

    elif args.command == "init":
        _handle_init_command(args)

    else:
        print_banner()
        parser.print_help()


if __name__ == "__main__":
    main()