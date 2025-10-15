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

from .models import ModelManager
from .runtime import select_backend, set_active_backend
from .settings import OrionSettings, SettingsError

if TYPE_CHECKING:  # pragma: no cover
    from .run_pipeline import run_pipeline

console = Console()


def _import_video_qa() -> Any:
    """Import the optional video QA module when available."""
    try:
        from .video_qa import VideoQASystem  # type: ignore

        return VideoQASystem
    except Exception:  # pragma: no cover
        return None


def _prepare_runtime(requested: Optional[str]) -> Tuple[str, ModelManager]:
    with console.status(
        "[dim]Selecting runtime backend...[/dim]", spinner="dots"
    ) as status:
        backend = select_backend(requested)
        set_active_backend(backend)
        manager = ModelManager()

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

    table.add_row("YOLO11m", "Object Detection", "20MB", "⚡⚡⚡ Very Fast")
    table.add_row("FastVLM", "Description", "600MB", "⚡⚡ Fast")
    table.add_row("ResNet50", "Visual ID", "100MB", "⚡⚡⚡ Very Fast")
    table.add_row("EmbeddingGemma", "Text Meaning", "622MB", "⚡⚡ Fast")
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
        from .run_pipeline import run_pipeline  # Lazy import to speed up CLI startup

        VideoQASystem = _import_video_qa()

        try:
            backend, _ = _prepare_runtime(args.runtime or settings.runtime_backend)
        except Exception:
            return

        neo4j_uri = args.neo4j_uri or settings.neo4j_uri
        neo4j_user = args.neo4j_user or settings.neo4j_user
        neo4j_password = args.neo4j_password or settings.neo4j_password
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

        results = run_pipeline(
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
                    model=qa_model,
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
                neo4j_password=args.neo4j_password or settings.neo4j_password,
                model=args.model or settings.qa_model,
                embedding_backend=args.embedding_backend or settings.embedding_backend,
                embedding_model=args.embedding_model or settings.embedding_model,
            )
            qa.start_interactive_session()

    elif args.command == "models":
        show_models()

    elif args.command == "modes":
        show_modes()

    elif args.command == "init":
        try:
            backend, manager = _prepare_runtime(
                args.runtime or settings.runtime_backend
            )
        except Exception:
            return

        console.print(f"[green]Runtime '{backend}' assets are ready.[/green]")

        init_script = Path(__file__).resolve().parents[2] / "scripts" / "init.py"
        if init_script.exists():
            console.print("\n[bold cyan]Running initialization...[/bold cyan]\n")
            console.print(f"[dim]python {init_script}[/dim]")
            import subprocess

            subprocess.run([sys.executable, str(init_script)], check=False)
        else:
            console.print("[red]Init script not found![/red]")
            console.print(f"Expected: {init_script}")

    else:
        print_banner()
        parser.print_help()


if __name__ == "__main__":
    main()
