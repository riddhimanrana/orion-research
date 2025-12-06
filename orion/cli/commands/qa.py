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
        neo4j_uri=args.neo4j_uri or settings.neo4j_uri,
        neo4j_user=args.neo4j_user or settings.neo4j_user,
        neo4j_password=args.neo4j_password or settings.get_neo4j_password(),
        llm_model=getattr(args, "model", None) or settings.qa_model,
        results_dir=getattr(args, "results_dir", None),
        context_frames=getattr(args, "context_frames", 200),
        max_objects=getattr(args, "max_objects", 20),
        max_relations=getattr(args, "max_relations", 10),
        extra_entities_path=getattr(args, "entities_json", None),
    )
    qa.start_interactive_session()
