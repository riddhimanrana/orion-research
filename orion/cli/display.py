"""Display and UI utilities for the Orion CLI."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from typing import Any

console = Console()


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
    """Display detailed model information."""
    table = Table(
        title="[bold cyan]Orion Model Architecture[/bold cyan]",
        box=box.DOUBLE_EDGE,
        show_header=True,
        header_style="bold magenta",
        border_style="cyan",
        expand=True,
    )
    table.add_column("Model", style="cyan bold", no_wrap=True, width=20)
    table.add_column("Purpose", style="yellow", width=25)
    table.add_column("Size", style="green", justify="right", width=12)
    table.add_column("Performance", style="magenta", width=20)
    table.add_column("Details", style="dim", width=40)

    table.add_row(
        "YOLO11x", "Object Detection", "50MB", "⚡⚡⚡ Very Fast", "Real-time detection, 80+ classes"
    )
    table.add_row(
        "FastVLM-0.5B",
        "Visual Description",
        "~600MB",
        "⚡⚡ Fast",
        "Generates semantic descriptions",
    )
    table.add_row(
        "ResNet50",
        "Visual Embeddings",
        "~100MB",
        "⚡⚡⚡ Very Fast",
        "Appearance fingerprinting (512-dim)",
    )
    table.add_row(
        "CLIP (OpenAI)",
        "Text Embeddings",
        "~512MB",
        "⚡⚡ Fast",
        "Semantic text understanding (512-dim)",
    )
    table.add_row(
        "Gemma3:4b",
        "Q&A & Reasoning",
        "~1.6GB",
        "⚡ Medium",
        "Natural language interface via Ollama",
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
        expand=False,
    )
    console.print("\n", arch_panel, "\n")


def show_modes() -> None:
    """Display detailed processing mode information."""
    table = Table(
        title="[bold cyan]Processing Modes[/bold cyan]",
        box=box.DOUBLE_EDGE,
        show_header=True,
        header_style="bold magenta",
        border_style="cyan",
        expand=True,
    )
    table.add_column("Mode", style="cyan bold", width=15)
    table.add_column("Detection FPS", style="yellow", justify="center", width=15)
    table.add_column("Description Freq", style="green", justify="center", width=18)
    table.add_column("Use Case", style="magenta", width=35)
    table.add_column("Trade-offs", style="dim", width=40)

    table.add_row(
        "fast", "3 FPS", "Every 10th frame", "Long videos, quick overview", "Lower detail, faster processing"
    )
    table.add_row(
        "balanced",
        "5 FPS",
        "Every 5th frame",
        "General purpose",
        "Good balance of speed and accuracy",
    )
    table.add_row(
        "accurate",
        "10 FPS",
        "Every 3rd frame",
        "Detailed analysis",
        "High accuracy, slower processing",
    )

    console.print("\n", table)

    config_panel = Panel(
        "[bold]Configuration Tips:[/bold]\n"
        "• Use --fast for videos longer than 5 minutes\n"
        "• Use --accurate for short clips with important details\n"
        "• Default is 'balanced' mode for most use cases",
        title="[bold cyan]Mode Selection Guide[/bold cyan]",
        border_style="cyan",
        expand=False,
    )
    console.print("\n", config_panel, "\n")


def display_pipeline_results(results: dict[str, Any]) -> None:
    """Display comprehensive pipeline results with detailed metrics."""
    # Header Panel
    status_emoji = "✓" if results.get("success") else "✗"
    status_color = "green" if results.get("success") else "red"

    header = Panel(
        f"[bold {status_color}]{status_emoji} Pipeline Execution Complete[/bold {status_color}]",
        style=status_color,
        expand=False,
    )
    console.print("\n", header)

    # ═══════════════════════════════════════════════════════════
    # PERCEPTION STAGE
    # ═══════════════════════════════════════════════════════════
    if not results.get("perception", {}).get("skipped"):
        perc = results.get("perception", {})
        perception_table = Table(
            title="[bold cyan]Visual Perception Stage[/bold cyan]",
            box=box.DOUBLE_EDGE,
            show_header=True,
            header_style="bold magenta",
            border_style="cyan",
            expand=True,
        )
        perception_table.add_column("Metric", style="cyan", no_wrap=True, width=30)
        perception_table.add_column("Value", style="green", justify="right", width=20)
        perception_table.add_column("Details", style="yellow", width=50)

        # Video info
        perception_table.add_row(
            "Video Duration",
            f"{perc.get('duration_seconds', 0):.1f}s",
            f"{perc.get('total_frames', 0):,} frames @ {perc.get('fps', 0):.1f} FPS",
        )

        # Detection metrics
        num_entities = perc.get("num_entities", 0)
        num_unique = perc.get("unique_classes", 0)
        perception_table.add_row("Entities Detected", f"{num_entities:,}", f"{num_unique} unique classes")

        # Processing speed
        proc_time = perc.get("processing_time_seconds", 0)
        if proc_time > 0:
            fps = perc.get("total_frames", 0) / proc_time
            perception_table.add_row("Processing Speed", f"{fps:.1f} FPS", f"Completed in {proc_time:.1f}s")

        console.print("\n", perception_table)

    # ═══════════════════════════════════════════════════════════
    # SEMANTIC STAGE
    # ═══════════════════════════════════════════════════════════
    if not results.get("semantic", {}).get("skipped"):
        sem = results.get("semantic", {})

        semantic_table = Table(
            title="[bold cyan]Semantic Understanding Stage[/bold cyan]",
            box=box.DOUBLE_EDGE,
            show_header=True,
            header_style="bold magenta",
            border_style="cyan",
            expand=True,
        )
        semantic_table.add_column("Metric", style="cyan", no_wrap=True, width=30)
        semantic_table.add_column("Value", style="green", justify="right", width=20)
        semantic_table.add_column("Details", style="yellow", width=50)

        # Entities with descriptions
        num_entities = sem.get("num_entities", 0)
        num_descriptions = sem.get("num_described", num_entities)
        semantic_table.add_row(
            "Entities Processed", f"{num_entities:,}", f"{num_descriptions} with descriptions"
        )

        # Spatial zones
        num_zones = sem.get("num_spatial_zones", 0)
        semantic_table.add_row(
            "Spatial Zones",
            f"{num_zones:,}",
            "Co-location regions detected" if num_zones > 0 else "No zones detected",
        )

        # Events
        num_events = sem.get("num_events", 0)
        semantic_table.add_row("Events Detected", f"{num_events:,}", "State changes and transitions")

        # Causal links
        num_causal = sem.get("num_causal_links", 0)
        semantic_table.add_row("Causal Links", f"{num_causal:,}", "Event causality relationships")

        console.print("\n", semantic_table)

    # ═══════════════════════════════════════════════════════════
    # GRAPH STAGE
    # ═══════════════════════════════════════════════════════════
    if not results.get("graph", {}).get("skipped"):
        graph = results.get("graph", {})
        graph_stats = graph.get("stats", {})

        if graph_stats:
            graph_table = Table(
                title="[bold green]Knowledge Graph Built[/bold green]",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold green",
                border_style="green",
            )
            graph_table.add_column("Node/Relationship Type", style="cyan", width=30)
            graph_table.add_column("Count", style="green", justify="right", width=15)

            for key, value in sorted(graph_stats.items()):
                label = key.replace("_", " ").title()
                graph_table.add_row(label, f"{value:,}")

            console.print("\n", graph_table)

    # ═══════════════════════════════════════════════════════════
    # SUMMARY FOOTER
    # ═══════════════════════════════════════════════════════════
    summary_panel = Panel(
        f"[bold]Output:[/bold] {results.get('output_file', 'N/A')}\n"
        f"[bold]Runtime:[/bold] {results.get('total_time', 'N/A')}\n"
        f"[bold]Neo4j:[/bold] {results.get('neo4j_status', 'Available')}",
        title="[bold]Pipeline Summary[/bold]",
        style="blue",
        expand=False,
    )
    console.print("\n", summary_panel)
