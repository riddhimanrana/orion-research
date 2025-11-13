"""CLI command handlers."""

from .analyze import handle_analyze
from .config import handle_config
from .init import handle_init
from .qa import handle_qa
from .research import handle_research
from .services import handle_neo4j, handle_ollama

# Stub for unified_pipeline to avoid breaking CLI
def handle_unified_pipeline(args, settings):
    from rich.console import Console
    console = Console()
    console.print("[yellow]⚠ Unified pipeline command deprecated. Use 'orion analyze' instead.[/yellow]")

__all__ = [
    "handle_analyze",
    "handle_config",
    "handle_init",
    "handle_qa",
    "handle_research",
    "handle_neo4j",
    "handle_ollama",
    "handle_unified_pipeline",
]

