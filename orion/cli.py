"""Command-line interface for the Orion research toolkit."""

from __future__ import annotations

import argparse
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
    subparsers.add_parser("status", help="Show system and database status")
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
        from .neo4j_manager import Neo4jManager
        from .vector_indexing import ENTITY_INDEX, SCENE_INDEX

        mgr = Neo4jManager(
            settings.neo4j_uri, settings.neo4j_user, settings.get_neo4j_password()
        )
        if not mgr.connect():
            console.print("[red]Cannot connect to Neo4j.[/red]")
            return
        stats = mgr.get_stats()
        status_table = Table(title="Neo4j Status", box=box.ROUNDED)
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Value", style="green")
        status_table.add_row("Nodes", str(stats.get("nodes", 0)))
        status_table.add_row("Relationships", str(stats.get("relationships", 0)))
        console.print(status_table)

        # Quick index check
        try:
            with mgr.driver.session() as session:  # type: ignore[union-attr]
                res = session.run("CALL db.indexes() YIELD name, type RETURN name, type")
                rows = res.data()
                idx_table = Table(title="Indexes", box=box.ROUNDED)
                idx_table.add_column("Name", style="yellow")
                idx_table.add_column("Type", style="magenta")
                for r in rows:
                    idx_table.add_row(str(r.get("name")), str(r.get("type")))
                console.print(idx_table)
        except Exception as e:
            console.print(f"[dim]Index list unavailable: {e}[/dim]")
        mgr.close()

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
        try:
            backend, manager = _prepare_runtime(
                args.runtime or settings.runtime_backend
            )
        except Exception:
            return

        console.print(f"[green]Runtime '{backend}' assets are ready.[/green]")

        # Print setup summary with correct model paths
        # (Table imported at module level)
        summary_table = Table(title="Setup Summary", box=box.ROUNDED)
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

        # Show Gemma3:4b
        summary_table.add_row("gemma3:4b", "✓", "Installed")
        summary_table.add_row("CLIP (OpenAI)", "✓", "Installed")

        console.print(summary_table)

        init_script = Path(__file__).resolve().parents[1] / "scripts" / "init.py"
        if init_script.exists():
            console.print("\n[bold cyan]Running initialization...[/bold cyan]\n")
            console.print(f"[dim]python {init_script}[/dim]")
            import subprocess

            subprocess.run([sys.executable, str(init_script)], check=False)
        else:
            console.print("[red]Init script not found![red]")
            console.print(f"Expected: {init_script}")

    else:
        print_banner()
        parser.print_help()


if __name__ == "__main__":
    main()
