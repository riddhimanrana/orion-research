"""Main CLI entry point for Orion."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from .commands import (
    handle_analyze,
    handle_config,
    handle_init,
    handle_neo4j,
    handle_ollama,
    handle_qa,
)
from .display import print_banner, show_models, show_modes
from ..settings import OrionSettings, SettingsError

console = Console()


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all commands and options."""
    parser = argparse.ArgumentParser(
        prog="orion",
        description="Orion Video Analysis Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  orion analyze video.mp4                     # Run full pipeline
  orion analyze video.mp4 -i                  # With interactive Q&A
  orion analyze video.mp4 --fast              # Fast mode
  orion analyze video.mp4 --skip-semantic     # Skip semantic stage
  orion analyze video.mp4 --inspect=perception  # Inspect after perception stage
  orion qa                                    # Q&A only mode
  orion models                                # Show model info
  orion services neo4j start                  # Start Neo4j

For more help: orion <command> --help
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ═══════════════════════════════════════════════════════════
    # ANALYZE COMMAND
    # ═══════════════════════════════════════════════════════════
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a video")
    analyze_parser.add_argument("video", help="Path to video file")
    analyze_parser.add_argument("--fast", action="store_true", help="Use fast mode")
    analyze_parser.add_argument("--accurate", action="store_true", help="Use accurate mode")
    analyze_parser.add_argument("-i", "--interactive", action="store_true", help="Start Q&A after processing")

    # Stage control
    analyze_parser.add_argument("--skip-perception", action="store_true", help="Skip perception stage")
    analyze_parser.add_argument("--skip-semantic", action="store_true", help="Skip semantic stage")
    analyze_parser.add_argument("--skip-graph", action="store_true", help="Skip graph building stage")

    # Inspection mode
    analyze_parser.add_argument(
        "--inspect",
        choices=["perception", "semantic", "graph"],
        help="Stop after stage and return intermediate results",
    )

    # Database and output
    analyze_parser.add_argument("--keep-db", action="store_true", help="Keep existing Neo4j data")
    analyze_parser.add_argument("-o", "--output", default="data/testing", help="Output directory")

    # Neo4j configuration
    analyze_parser.add_argument("--neo4j-uri", help="Neo4j connection URI (defaults to config)")
    analyze_parser.add_argument("--neo4j-user", help="Neo4j username (defaults to config)")
    analyze_parser.add_argument("--neo4j-password", help="Neo4j password (defaults to config)")

    # Models and runtime
    analyze_parser.add_argument("--qa-model", help="Ollama model for interactive Q&A (defaults to config)")
    analyze_parser.add_argument("--runtime", help="Select runtime backend (auto or torch; defaults to config)")

    # Display
    analyze_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # ═══════════════════════════════════════════════════════════
    # QA COMMAND
    # ═══════════════════════════════════════════════════════════
    qa_parser = subparsers.add_parser("qa", help="Q&A mode only")
    qa_parser.add_argument("--model", help="Ollama model to use (defaults to config)")
    qa_parser.add_argument("--neo4j-uri", help="Neo4j connection URI (defaults to config)")
    qa_parser.add_argument("--neo4j-user", help="Neo4j username (defaults to config)")
    qa_parser.add_argument("--neo4j-password", help="Neo4j password (defaults to config)")
    qa_parser.add_argument("--runtime", help="Select runtime backend (auto or torch; defaults to config)")

    # ═══════════════════════════════════════════════════════════
    # INFO COMMANDS
    # ═══════════════════════════════════════════════════════════
    subparsers.add_parser("models", help="Show model information")
    subparsers.add_parser("modes", help="Show processing modes")

    # ═══════════════════════════════════════════════════════════
    # SERVICE COMMANDS
    # ═══════════════════════════════════════════════════════════
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

    # ═══════════════════════════════════════════════════════════
    # STATUS COMMAND
    # ═══════════════════════════════════════════════════════════
    subparsers.add_parser("status", help="Comprehensive system and service status check")

    # ═══════════════════════════════════════════════════════════
    # INIT COMMAND
    # ═══════════════════════════════════════════════════════════
    init_parser = subparsers.add_parser(
        "init", 
        help="Full system setup (config, services, models)"
    )
    init_parser.add_argument(
        "--runtime",
        help="Select runtime backend to prepare (auto or torch; defaults to config)",
    )

    # ═══════════════════════════════════════════════════════════
    # CONFIG COMMANDS
    # ═══════════════════════════════════════════════════════════
    config_parser = subparsers.add_parser("config", help="Inspect or update Orion configuration")
    config_subparsers = config_parser.add_subparsers(dest="config_command", help="Configuration actions")
    config_subparsers.required = True

    config_subparsers.add_parser("show", help="Display the current configuration values")
    config_subparsers.add_parser("path", help="Print the configuration file path")
    config_subparsers.add_parser("reset", help="Reset configuration to defaults")
    config_subparsers.add_parser("credentials", help="Show Neo4j credentials")

    config_set_parser = config_subparsers.add_parser("set", help="Update a configuration value")
    config_set_parser.add_argument("key", help="Configuration key (e.g., neo4j.uri)")
    config_set_parser.add_argument("value", help="New value")

    return parser


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Load settings
    try:
        settings = OrionSettings.load()
    except SettingsError:
        # If config doesn't exist and command is init, allow it
        if args.command == "init":
            settings = OrionSettings()
        else:
            console.print("[yellow]⚠ Config not found. Run 'orion init' to initialize.[/yellow]")
            sys.exit(1)

    # Handle config commands first (don't need banner)
    if args.command == "config":
        status = handle_config(args, settings)
        if status != 0:
            sys.exit(status)
        return

    # Show banner for all other commands
    if args.command:
        print_banner()

    # Route to command handlers
    if args.command == "analyze":
        handle_analyze(args, settings)

    elif args.command == "qa":
        handle_qa(args, settings)

    elif args.command == "models":
        show_models()

    elif args.command == "modes":
        show_modes()

    elif args.command == "services":
        if args.service_command == "neo4j":
            handle_neo4j(args)
        elif args.service_command == "ollama":
            handle_ollama(args)
        else:
            console.print("[red]No service command provided. Use 'orion services --help'.[/red]")

    elif args.command == "status":
        from ..managers.auto_config import status_command

        status_command(args)

    elif args.command == "init":
        handle_init(args)

    else:
        print_banner()
        parser.print_help()


if __name__ == "__main__":
    main()
