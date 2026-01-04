"""QA command - interactive question answering mode."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from rich.console import Console

from ..utils import prepare_runtime

if TYPE_CHECKING:
    from ...settings import OrionSettings

console = Console()


def handle_qa(args: argparse.Namespace, settings: OrionSettings) -> None:
    """Handle the qa command - start interactive Q&A session."""
    console.print("\n[bold cyan]Starting Q&A mode...[/bold cyan]\n")

    try:
        from ...video_qa import VideoQASystem
    except ImportError:
        console.print("[red]Q&A not available. Install: pip install ollama[/red]")
        return

    try:
        backend, _ = prepare_runtime(args.runtime or settings.runtime_backend)
    except Exception:
        return

    console.print(f"[dim]Using runtime backend: {backend}[/dim]\n")

    qa = VideoQASystem(
        memgraph_host=args.memgraph_host or settings.memgraph_host,
        memgraph_port=args.memgraph_port or settings.memgraph_port,
        memgraph_user=args.memgraph_user or settings.memgraph_user,
        memgraph_password=args.memgraph_password or settings.get_memgraph_password(),
        llm_model=getattr(args, "model", None) or settings.qa_model,
    )
    qa.start_interactive_session()
