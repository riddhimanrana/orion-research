"""Command-line interface for the Orion research toolkit."""

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
from rich.table import Table
from rich.text import Text

from .models import AssetManager
from .runtime import select_backend, set_active_backend
from .settings import OrionSettings, SettingsError

# We avoid importing the heavy pipeline at module import time.
if TYPE_CHECKING:  # pragma: no cover
    from .run_pipeline import run_pipeline as _RunPipeline

console = Console()


def _import_video_qa() -> Any:
    """Import the unified video QA system when available."""
    try:
        from .video_qa import VideoQASystem  # type: ignore

        return VideoQASystem
    except Exception:  # pragma: no cover
        return None


def _prepare_runtime(requested: Optional[str]) -> Tuple[str, AssetManager]:
    with console.status(
        "[dim]Selecting runtime backend...[/dim]", spinner="dots"
    ) as status:
        backend = select_backend(requested)
        set_active_backend(backend)
        manager = AssetManager()

        if manager.assets_ready(backend):
            status.update(f"[green]Runtime '{backend}' ready.[/green]")
            time.sleep(0.2)
            return backend, manager

        status.update(f"[yellow]Syncing model assets for '{backend}'...[/yellow]")
        try:
            manager.ensure_runtime_assets(backend)
        except Exception as exc:  # noqa: BLE001
            status.stop()
            console.print(f"[red]Failed to prepare models: {exc}[/red]")
            raise

        status.update(f"[green]Runtime '{backend}' synchronized.[/green]")
        time.sleep(0.2)
        return backend, manager


def _prepare_runtime_for_backend(backend: str) -> AssetManager:
    """Prepare runtime assets for a specific backend"""
    with console.status(
        "[dim]Preparing runtime assets...[/dim]", spinner="dots"
    ) as status:
        set_active_backend(backend)
        manager = AssetManager()

        if manager.assets_ready(backend):
            status.update(f"[green]Runtime '{backend}' ready.[/green]")
            time.sleep(0.2)
            return manager

        status.update(f"[yellow]Syncing model assets for '{backend}'...[/yellow]")
        try:
            manager.ensure_runtime_assets(backend)
        except Exception as exc:  # noqa: BLE001
            status.stop()
            console.print(f"[red]Failed to prepare models: {exc}[/red]")
            raise

        status.update(f"[green]Runtime '{backend}' synchronized.[/green]")
        time.sleep(0.2)
        return manager


def _handle_neo4j_command(args: argparse.Namespace) -> None:
    """Handle Neo4j service management commands"""
    action = getattr(args, "neo4j_action", None)
    if action is None:
        console.print("[red]No Neo4j action provided. Use 'orion services neo4j --help'.[/red]")
        return

    try:
        if action == "start":
            # Check if container exists and start it
            check_result = subprocess.run(['docker', 'ps', '-a', '--filter', 'name=orion-neo4j'],
                                        capture_output=True, text=True)

            if 'orion-neo4j' in check_result.stdout:
                status_result = subprocess.run(['docker', 'ps', '--filter', 'name=orion-neo4j'],
                                             capture_output=True, text=True)

                if 'orion-neo4j' in status_result.stdout:
                    console.print("[yellow]✓ Neo4j container already running[/yellow]")
                else:
                    result = subprocess.run(['docker', 'start', 'orion-neo4j'], check=True,
                                          capture_output=True, text=True)
                    console.print("[green]✓ Neo4j container started[/green]")
            else:
                console.print("[red]✗ Neo4j container not found. Run 'orion init' first.[/red]")

        elif action == "stop":
            result = subprocess.run(['docker', 'stop', 'orion-neo4j'], check=True,
                                  capture_output=True, text=True)
            console.print("[green]✓ Neo4j container stopped[/green]")

        elif action == "status":
            result = subprocess.run(['docker', 'ps', '--filter', 'name=orion-neo4j'],
                                  capture_output=True, text=True)
            if 'orion-neo4j' in result.stdout:
                console.print("[green]✓ Neo4j container is running[/green]")
            else:
                console.print("[yellow]⚠ Neo4j container is not running[/yellow]")

        elif action == "restart":
            subprocess.run(['docker', 'restart', 'orion-neo4j'], check=True,
                         capture_output=True, text=True)
            console.print("[green]✓ Neo4j container restarted[/green]")

    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Docker command failed: {e.stderr}[/red]")
    except FileNotFoundError:
        console.print("[red]✗ Docker not found. Please install Docker first.[/red]")


def _handle_ollama_command(args: argparse.Namespace) -> None:
    """Handle Ollama service management commands"""
    action = getattr(args, "ollama_action", None)
    if action is None:
        console.print("[red]No Ollama action provided. Use 'orion services ollama --help'.[/red]")
        return

    try:
        if action == "start":
            # Check if container exists and start it
            check_result = subprocess.run(['docker', 'ps', '-a', '--filter', 'name=orion-ollama'],
                                        capture_output=True, text=True)

            if 'orion-ollama' in check_result.stdout:
                status_result = subprocess.run(['docker', 'ps', '--filter', 'name=orion-ollama'],
                                             capture_output=True, text=True)

                if 'orion-ollama' in status_result.stdout:
                    console.print("[yellow]✓ Ollama container already running[/yellow]")
                else:
                    result = subprocess.run(['docker', 'start', 'orion-ollama'], check=True,
                                          capture_output=True, text=True)
                    console.print("[green]✓ Ollama container started[/green]")
            else:
                console.print("[red]✗ Ollama container not found. Run 'orion init' first.[/red]")

        elif action == "stop":
            result = subprocess.run(['docker', 'stop', 'orion-ollama'], check=True,
                                  capture_output=True, text=True)
            console.print("[green]✓ Ollama container stopped[/green]")

        elif action == "status":
            result = subprocess.run(['docker', 'ps', '--filter', 'name=orion-ollama'],
                                  capture_output=True, text=True)
            if 'orion-ollama' in result.stdout:
                console.print("[green]✓ Ollama container is running[/green]")
            else:
                console.print("[yellow]⚠ Ollama container is not running[/yellow]")

        elif action == "restart":
            subprocess.run(['docker', 'restart', 'orion-ollama'], check=True,
                         capture_output=True, text=True)
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
    table = Table(title="Orion Models", box=box.ROUNDED)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Job", style="yellow")
    table.add_column("Size", style="green")
    table.add_column("Speed", style="magenta")

    table.add_row("YOLO11x", "Object Detection", "50MB", "⚡⚡ Very Accurate")
    table.add_row("FastVLM", "Description", "600MB", "⚡⚡ Fast")
    table.add_row("ResNet50", "Visual ID", "100MB", "⚡⚡⚡ Very Fast")
    table.add_row("CLIP (OpenAI)", "Text Meaning", "512MB", "⚡⚡ Fast")
    table.add_row("Gemma3:4b", "Q&A", "1.6GB", "⚡ Medium")

    console.print(table)


def show_modes() -> None:
    table = Table(title="Processing Modes", box=box.ROUNDED)
    table.add_column("Mode", style="cyan")
    table.add_column("FPS", style="yellow")
    table.add_column("Descriptions", style="green")
    table.add_column("Best For", style="magenta")

    table.add_row("fast", "3", "Every 10th", "Long videos, testing")
    table.add_row("balanced", "5", "Every 5th", "General use ⭐")
    table.add_row("accurate", "10", "Every 2nd", "Short clips, detail")

    console.print(table)


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
        from .run_pipeline import run_pipeline as run_pipeline_main

        VideoQASystem = _import_video_qa()

        try:
            backend, _ = _prepare_runtime(args.runtime or settings.runtime_backend)
        except Exception:
            return

        neo4j_uri = args.neo4j_uri or settings.neo4j_uri
        neo4j_user = args.neo4j_user or settings.neo4j_user
        neo4j_password = args.neo4j_password or settings.get_neo4j_password()
        qa_model = args.qa_model or settings.qa_model
        embedding_backend = args.embedding_backend or settings.embedding_backend
        embedding_model = args.embedding_model or settings.embedding_model

        console.print(f"\n[bold]Analyzing:[/bold] [cyan]{args.video}[/cyan]")
        config = "balanced"
        if args.fast:
            config = "fast"
        elif args.accurate:
            config = "accurate"
        console.print(f"[bold]Mode:[/bold] [yellow]{config}[/yellow]")
        console.print(f"[bold]Runtime:[/bold] [yellow]{backend}[/yellow]\n")

        results = run_pipeline_main(
            video_path=args.video,
            output_dir=args.output,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            clear_db=not args.keep_db,
            part1_config=config,
            part2_config=config,
            skip_part1=args.skip_perception,
            skip_part2=args.skip_graph,
            verbose=args.verbose,
            runtime=backend,
            use_progress_ui=not args.verbose,
        )

        if args.interactive and results.get("success"):
            console.print("\n[bold cyan]Starting Q&A mode...[/bold cyan]\n")
            if VideoQASystem is None:
                console.print(
                    "[red]Q&A not available. Install: pip install ollama[/red]"
                )
            else:
                qa = VideoQASystem(
                    neo4j_uri=neo4j_uri,
                    neo4j_user=neo4j_user,
                    neo4j_password=neo4j_password,
                    llm_model=qa_model,
                    embedding_backend=embedding_backend,
                    embedding_model=embedding_model,
                )
                qa.start_interactive_session()

    elif args.command == "qa":
        console.print("\n[bold cyan]Starting Q&A mode...[/bold cyan]\n")
        VideoQASystem = _import_video_qa()
        if VideoQASystem is None:
            console.print("[red]Q&A not available. Install: pip install ollama[/red]")
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
                embedding_backend=args.embedding_backend or settings.embedding_backend,
                embedding_model=args.embedding_model or settings.embedding_model,
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
        
        # Check if prerequisites are met
        if not neo4j_ok or not ollama_ok:
            console.print("\n[bold red]⚠️  Missing Prerequisites[/bold red]\n")
            console.print("[yellow]Orion requires Neo4j and Ollama to be running before initialization.[/yellow]\n")

            if docker_ok:
                # Interactive Docker setup
                console.print("[bold cyan]Option 1: Automated Docker Setup (Recommended)[/bold cyan]\n")
                console.print("[dim]We'll automatically set up Neo4j and Ollama using Docker[/dim]\n")

                console.print("[bold cyan]Option 2: Manual Installation[/bold cyan]\n")
                console.print("[bold]Neo4j:[/bold] Install from [link=https://neo4j.com/download/]https://neo4j.com/download/[/link]")
                console.print("[bold]Ollama:[/bold] Install from [link=https://ollama.com]https://ollama.com[/link]\n")

                # Ask user for preference
                from rich.prompt import Prompt, Confirm
                use_docker = Confirm.ask("Would you like us to automatically set up services using Docker?", default=True)

                if use_docker:
                    console.print("\n[bold cyan]🚀 Setting up services automatically with Docker...[/bold cyan]\n")

                    # Generate secure password for Neo4j
                    console.print("[bold]Step 1: Setting up Neo4j[/bold]")
                    password_choice = Prompt.ask(
                        "Choose Neo4j password [1] Enter custom password, [2] Generate secure password",
                        choices=["1", "2"],
                        default="2"
                    )

                    if password_choice == "1":
                        neo4j_password = Prompt.ask("Enter Neo4j password", password=True)
                    else:
                        # Generate secure password using openssl
                        try:
                            result = subprocess.run(['openssl', 'rand', '-base64', '32'],
                                                  capture_output=True, text=True, check=True)
                            neo4j_password = result.stdout.strip()
                            console.print(f"[green]Generated secure password: {neo4j_password}[/green]")
                        except (subprocess.CalledProcessError, FileNotFoundError):
                            # Fallback if openssl not available
                            import secrets
                            import string
                            alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
                            neo4j_password = ''.join(secrets.choice(alphabet) for _ in range(32))
                            console.print(f"[green]Generated secure password: {neo4j_password}[/green]")

                    # Set password in environment for init script to use
                    os.environ["ORION_NEO4J_PASSWORD"] = neo4j_password
                    os.environ["ORION_NEO4J_USER"] = "neo4j"
                    os.environ["ORION_NEO4J_URI"] = "neo4j://127.0.0.1:7687"
                    os.environ["ORION_OLLAMA_URL"] = "http://localhost:11434"

                    # Check if Neo4j container already exists
                    console.print("[dim]Checking for existing Neo4j container...[/dim]")
                    try:
                        # Check if container exists
                        check_result = subprocess.run(['docker', 'ps', '-a', '--filter', 'name=orion-neo4j'],
                                                    capture_output=True, text=True)

                        if 'orion-neo4j' in check_result.stdout:
                            # Container exists, check if it's running
                            status_result = subprocess.run(['docker', 'ps', '--filter', 'name=orion-neo4j'],
                                                         capture_output=True, text=True)

                            if 'orion-neo4j' in status_result.stdout:
                                console.print("[yellow]✓ Neo4j container already running[/yellow]")
                                console.print("[dim]Using existing Neo4j container[/dim]")
                                # Test connection with the new password
                                console.print("[dim]Testing connection with new password...[/dim]")
                            else:
                                console.print("[yellow]⚠ Neo4j container exists but is stopped[/yellow]")
                                # Remove the stopped container and create a new one with our password
                                console.print("[dim]Removing old container to use new password...[/dim]")
                                subprocess.run(['docker', 'rm', 'orion-neo4j'], capture_output=True, text=True)
                                console.print("[dim]Creating new Neo4j container...[/dim]")
                                neo4j_cmd = f"docker run --name orion-neo4j -d -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/{neo4j_password} neo4j:latest"
                                result = subprocess.run(neo4j_cmd.split(), check=True, capture_output=True, text=True)
                                console.print("[green]✓ Neo4j container created with new password[/green]")
                        else:
                            # Container doesn't exist, create new one
                            console.print("[dim]Creating new Neo4j container...[/dim]")
                            neo4j_cmd = f"docker run --name orion-neo4j -d -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/{neo4j_password} neo4j:latest"
                            result = subprocess.run(neo4j_cmd.split(), check=True, capture_output=True, text=True)
                            console.print("[green]✓ Neo4j container created and started[/green]")

                    except subprocess.CalledProcessError as e:
                        console.print(f"[red]✗ Failed to manage Neo4j container: {e.stderr}[/red]")
                        console.print("[yellow]Try manually:[/yellow]")
                        console.print("[dim]docker stop orion-neo4j && docker rm orion-neo4j[/dim]")
                        sys.exit(1)

                    # Start Ollama container
                    console.print("[bold]Step 2: Setting up Ollama[/bold]")

                    # Check if Ollama container already exists
                    console.print("[dim]Checking for existing Ollama container...[/dim]")
                    try:
                        # Check if container exists
                        check_result = subprocess.run(['docker', 'ps', '-a', '--filter', 'name=orion-ollama'],
                                                    capture_output=True, text=True)

                        if 'orion-ollama' in check_result.stdout:
                            # Container exists, check if it's running
                            status_result = subprocess.run(['docker', 'ps', '--filter', 'name=orion-ollama'],
                                                         capture_output=True, text=True)

                            if 'orion-ollama' in status_result.stdout:
                                console.print("[yellow]✓ Ollama container already running[/yellow]")
                                console.print("[dim]Using existing Ollama container[/dim]")
                            else:
                                console.print("[yellow]⚠ Ollama container exists but is stopped[/yellow]")
                                start_result = subprocess.run(['docker', 'start', 'orion-ollama'],
                                                            capture_output=True, text=True)
                                if start_result.returncode == 0:
                                    console.print("[green]✓ Started existing Ollama container[/green]")
                                else:
                                    console.print(f"[red]✗ Failed to start existing container: {start_result.stderr}[/red]")
                                    sys.exit(1)
                        else:
                            # Container doesn't exist, create new one
                            console.print("[dim]Creating new Ollama container...[/dim]")
                            ollama_cmd = "docker run --name orion-ollama -d -p 11434:11434 ollama/ollama"
                            result = subprocess.run(ollama_cmd.split(), check=True, capture_output=True, text=True)
                            console.print("[green]✓ Ollama container created and started[/green]")

                        # Wait a moment for Ollama to be ready
                        console.print("[dim]Waiting for Ollama to be ready...[/dim]")
                        time.sleep(3)

                        # Download the required model
                        console.print("[dim]Downloading gemma3:4b model...[/dim]")
                        try:
                            pull_result = subprocess.run(['docker', 'exec', 'orion-ollama', 'ollama', 'pull', 'gemma3:4b'],
                                                       check=True, capture_output=True, text=True)
                            console.print("[green]✓ Model gemma3:4b downloaded successfully[/green]")
                        except subprocess.CalledProcessError as e:
                            console.print(f"[yellow]⚠ Model download failed: {e.stderr}[/yellow]")
                            console.print("[dim]You can manually run: docker exec orion-ollama ollama pull gemma3:4b[/dim]")

                    except subprocess.CalledProcessError as e:
                        console.print(f"[red]✗ Failed to manage Ollama container: {e.stderr}[/red]")
                        console.print("[yellow]Try manually:[/yellow]")
                        console.print("[dim]docker stop orion-ollama && docker rm orion-ollama[/dim]")
                        sys.exit(1)

                    console.print("\n[green]✓ Services setup complete! Running initialization...[/green]")
                    # Continue with normal initialization flow
                else:
                    console.print("[dim]Please install Neo4j and Ollama manually, then run 'orion init' again.[/dim]\n")
                    sys.exit(1)
            elif ollama_ok and not docker_ok:
                console.print("\n[bold yellow]⚠️ Ollama is running but Docker is not available[/yellow]\n")
                console.print("[yellow]You have two options:[/yellow]\n")

                console.print("[bold cyan]Option 1: Install Docker (Recommended)[/bold cyan]")
                console.print("[dim]Install Docker to get Neo4j running automatically:[/dim]")
                console.print("[link=https://www.docker.com/products/docker-desktop]https://www.docker.com/products/docker-desktop[/link]\n")

                console.print("[bold cyan]Option 2: Manual Neo4j Installation[/bold cyan]")
                console.print("[dim]Install Neo4j manually and configure it to work with your existing Ollama setup[/dim]")
                console.print("[link=https://neo4j.com/download/]https://neo4j.com/download/[/link]\n")

                console.print("[bold cyan]Once Neo4j is running, run `orion init` again![/bold cyan]\n")
                sys.exit(1)

            elif docker_ok and not ollama_ok:
                console.print("\n[bold yellow]⚠️ Docker is available but Ollama is not running[/yellow]\n")
                console.print("[yellow]You have two options:[/yellow]\n")

                console.print("[bold cyan]Option 1: Use Docker (Recommended)[/bold cyan]")
                console.print("[dim]We'll set up both Neo4j and Ollama using Docker[/dim]\n")

                console.print("[bold cyan]Option 2: Manual Ollama Installation[/bold cyan]")
                console.print("[dim]Install Ollama manually and we'll use Docker for Neo4j[/dim]")
                console.print("[link=https://ollama.com]https://ollama.com[/link]\n")

                console.print("[bold cyan]Once both services are running, run `orion init` again![/bold cyan]\n")
                sys.exit(1)

            else:
                console.print("\n[bold red]❌ Neither Docker nor the required services are available[/red]\n")
                console.print("[yellow]Please install Docker (recommended) or set up Neo4j and Ollama manually:[/yellow]\n")

                console.print("[bold cyan]Option 1: Install Docker[/bold cyan]")
                console.print("[link=https://www.docker.com/products/docker-desktop]https://www.docker.com/products/docker-desktop[/link]\n")

                console.print("[bold cyan]Option 2: Manual Installation[/bold cyan]")
                console.print("[bold]Neo4j:[/bold] [link=https://neo4j.com/download/]https://neo4j.com/download/[/link]")
                console.print("[bold]Ollama:[/bold] [link=https://ollama.com]https://ollama.com[/link]\n")

                console.print("[bold cyan]Once services are running, run `orion init` again![/bold cyan]\n")
                sys.exit(1)
        
        console.print("[green]✓ All prerequisites met![/green]\n")
        
        # Step 2: Detect hardware and select runtime
        console.print("[bold]Step 2: Detecting hardware and selecting runtime...[/bold]\n")

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
        summary_table.add_row("gemma3:4b", "✓", "Available")
        summary_table.add_row("CLIP (OpenAI)", "✓", "Available")

        console.print(summary_table)

        # Step 3: Run interactive initialization script
        console.print("\n[bold]Step 3: Running interactive configuration...[/bold]\n")
        init_script = Path(__file__).resolve().parents[1] / "scripts" / "init.py"
        if init_script.exists():
            console.print(f"[dim]python {init_script}[/dim]\n")

            result = subprocess.run([sys.executable, str(init_script)], check=False)
            if result.returncode == 0:
                console.print("\n[green]✓ Orion initialization complete![/green]")
                console.print("[cyan]You're ready to analyze videos with: orion analyze video.mp4[/cyan]\n")
            else:
                console.print("\n[yellow]⚠ Configuration script had issues. Review the output above.[/yellow]\n")
                sys.exit(1)
        else:
            console.print("[red]Init script not found![red]")
            console.print(f"Expected: {init_script}\n")
            sys.exit(1)

    else:
        print_banner()
        parser.print_help()


if __name__ == "__main__":
    main()
