"""Main CLI entry point for Orion."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from .commands import (
    handle_analyze,
    handle_config,
    handle_detect,
    handle_init,
    handle_memgraph,
    handle_ollama,
    handle_qa,
    handle_research,
    handle_unified_pipeline,
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
  orion services memgraph start               # Start Memgraph

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
    analyze_parser.add_argument("--keep-db", action="store_true", help="Keep existing Memgraph data")
    analyze_parser.add_argument("-o", "--output", default="data/testing", help="Output directory")

    # Memgraph configuration
    analyze_parser.add_argument("--memgraph-host", help="Memgraph host (defaults to config)")
    analyze_parser.add_argument("--memgraph-port", help="Memgraph port (defaults to config)")
    analyze_parser.add_argument("--memgraph-user", help="Memgraph username (defaults to config)")
    analyze_parser.add_argument("--memgraph-password", help="Memgraph password (defaults to config)")

    # Models and runtime
    analyze_parser.add_argument("--qa-model", help="Ollama model for interactive Q&A (defaults to config)")
    analyze_parser.add_argument("--runtime", help="Select runtime backend (auto or torch; defaults to config)")

    # Display
    analyze_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    # Spatial Memory (Experimental)
    analyze_parser.add_argument(
        "--use-spatial-memory",
        action="store_true",
        help="Enable persistent spatial intelligence (remembers everything across sessions)"
    )
    analyze_parser.add_argument(
        "--memory-dir",
        type=str,
        default="memory/spatial_intelligence",
        help="Directory for persistent spatial memory storage"
    )
    analyze_parser.add_argument(
        "--export-memgraph",
        action="store_true",
        help="Export to Memgraph for real-time spatial queries"
    )

    # ═══════════════════════════════════════════════════════════
    # QA COMMAND
    # ═══════════════════════════════════════════════════════════
    qa_parser = subparsers.add_parser("qa", help="Q&A mode only")
    qa_parser.add_argument("--model", help="Ollama model to use (defaults to config)")
    qa_parser.add_argument("--memgraph-host", help="Memgraph host (defaults to config)")
    qa_parser.add_argument("--memgraph-port", help="Memgraph port (defaults to config)")
    qa_parser.add_argument("--memgraph-user", help="Memgraph username (defaults to config)")
    qa_parser.add_argument("--memgraph-password", help="Memgraph password (defaults to config)")
    qa_parser.add_argument("--runtime", help="Select runtime backend (auto or torch; defaults to config)")

    # ═══════════════════════════════════════════════════════════
    # DETECT COMMAND (Phase 1: YOLO-World Detection + Tracking)
    # ═══════════════════════════════════════════════════════════
    detect_parser = subparsers.add_parser(
        "detect", 
        help="Run Phase 1: YOLO-World detection + tracking"
    )
    detect_parser.add_argument("--video", "-v", required=True, help="Path to video file")
    detect_parser.add_argument("--episode", "-e", required=True, help="Episode name/ID for output")
    detect_parser.add_argument("--output-dir", "-o", help="Output directory (default: results/<episode>)")
    detect_parser.add_argument(
        "--detector", 
        default="yolov8x-worldv2",
        choices=["yolov8s-worldv2", "yolov8m-worldv2", "yolov8l-worldv2", "yolov8x-worldv2"],
        help="YOLO-World model variant (default: yolov8x-worldv2 for best quality)"
    )
    detect_parser.add_argument("--fps", type=float, default=5.0, help="Target FPS for sampling (default: 5)")
    detect_parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence threshold (default: 0.25)")
    detect_parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"], help="Device to run on")
    detect_parser.add_argument("--classes", nargs="+", help="Custom class prompts (overrides default vocabulary)")

    # ═══════════════════════════════════════════════════════════
    # INFO COMMANDS
    # ═══════════════════════════════════════════════════════════
    subparsers.add_parser("models", help="Show model information")
    subparsers.add_parser("modes", help="Show processing modes")

    # ═══════════════════════════════════════════════════════════
    # UNIFIED PERCEPTION PIPELINE - Phases 1-5
    # ═══════════════════════════════════════════════════════════
    run_parser = subparsers.add_parser(
        "run",
        help="Run unified 9-modality perception pipeline (Phases 1-5)"
    )
    run_parser.add_argument(
        "--video",
        type=str,
        default="data/examples/video_short.mp4",
        help="Path to input video file (default: data/examples/video_short.mp4)"
    )
    run_parser.add_argument(
        "--max-frames",
        type=int,
        default=60,
        help="Maximum frames to process (default: 60)"
    )
    run_parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Show detailed timing breakdown for each phase"
    )
    run_parser.add_argument(
        "--no-rerun",
        action="store_true",
        help="Disable Rerun visualization logging"
    )
    run_parser.add_argument(
        "--runtime",
        help="Select runtime backend (auto or torch; defaults to config)"
    )

    # ═══════════════════════════════════════════════════════════
    # SERVICE COMMANDS
    # ═══════════════════════════════════════════════════════════
    services_parser = subparsers.add_parser("services", help="Manage Orion services (Memgraph, Ollama)")
    services_subparsers = services_parser.add_subparsers(dest="service_command", help="Service actions")

    # Memgraph management
    memgraph_parser = services_subparsers.add_parser("memgraph", help="Manage Memgraph service")
    memgraph_subparsers = memgraph_parser.add_subparsers(dest="memgraph_action", help="Memgraph actions")
    memgraph_subparsers.add_parser("start", help="Start Memgraph container")
    memgraph_subparsers.add_parser("stop", help="Stop Memgraph container")
    memgraph_subparsers.add_parser("status", help="Check Memgraph container status")
    memgraph_subparsers.add_parser("restart", help="Restart Memgraph container")

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
    # RESEARCH COMMAND - Advanced SLAM visualization
    # ═══════════════════════════════════════════════════════════
    research_parser = subparsers.add_parser(
        "research",
        help="Research mode: SLAM, depth, tracking, zones (with 3D visualization)"
    )
    research_subparsers = research_parser.add_subparsers(
        dest="research_mode",
        help="Research mode to run"
    )
    research_subparsers.required = True
    
    # SLAM mode
    slam_parser = research_subparsers.add_parser(
        "slam",
        help="Run complete SLAM pipeline with advanced visualization"
    )
    slam_parser.add_argument("--video", type=str, required=True, help="Path to video file")
    slam_parser.add_argument(
        "--viz", 
        choices=["rerun", "opencv", "none"], 
        default="rerun",
        help="Visualization mode (rerun=3D browser, opencv=windows, none=headless)"
    )
    slam_parser.add_argument("--max-frames", type=int, help="Limit number of frames to process")
    slam_parser.add_argument(
        "--skip", 
        type=int, 
        default=15, 
        help="Frame skip interval (default: 15 = ~2fps for 30fps video)"
    )
    slam_parser.add_argument(
        "--zone-mode",
        choices=["dense", "sparse"],
        default="dense",
        help="Spatial zone clustering mode"
    )
    slam_parser.add_argument(
        "--no-adaptive",
        action="store_true",
        help="Disable adaptive frame skip"
    )
    slam_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with additional logging"
    )
    slam_parser.add_argument(
        "--yolo-model",
        choices=["yolo11n", "yolo11s", "yolo11m", "yolo11x"],
        default="yolo11n",
        help="YOLO model variant (n=fastest/real-time, m=balanced, x=accurate)"
    )
    slam_parser.add_argument(
        "--export-memgraph",
        action="store_true",
        help="Export to Memgraph for real-time queries (FastVLM captions on-demand)"
    )
    slam_parser.add_argument(
        "--use-spatial-memory",
        action="store_true",
        help="Enable persistent spatial intelligence (remembers everything across sessions)"
    )
    slam_parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Start interactive spatial intelligence assistant after processing"
    )
    slam_parser.add_argument(
        "--memory-dir",
        type=str,
        default="memory/spatial_intelligence",
        help="Directory for persistent spatial memory (default: memory/spatial_intelligence)"
    )
    
    # Depth mode (placeholder)
    depth_parser = research_subparsers.add_parser(
        "depth",
        help="Test depth estimation on video frames"
    )
    depth_parser.add_argument("--video", type=str, required=True, help="Path to video file")
    depth_parser.add_argument("--model", choices=["midas", "zoe"], default="midas")
    depth_parser.add_argument("--viz", choices=["rerun", "opencv"], default="rerun")
    
    # Tracking mode (placeholder)
    tracking_parser = research_subparsers.add_parser(
        "tracking",
        help="Test 3D entity tracking with Re-ID"
    )
    tracking_parser.add_argument("--video", type=str, required=True, help="Path to video file")
    tracking_parser.add_argument("--viz", choices=["rerun", "opencv"], default="rerun")
    
    # Zones mode (placeholder)
    zones_parser = research_subparsers.add_parser(
        "zones",
        help="Test spatial zone detection and classification"
    )
    zones_parser.add_argument("--video", type=str, required=True, help="Path to video file")
    zones_parser.add_argument("--mode", choices=["dense", "sparse"], default="dense")
    zones_parser.add_argument("--viz", choices=["rerun", "opencv"], default="rerun")

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
    init_parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset configuration and reconfigure password (ignores existing config)",
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
    config_subparsers.add_parser("credentials", help="Show Memgraph credentials")
    config_subparsers.add_parser("reset-password", help="Reset Memgraph password (interactive)")

    config_set_parser = config_subparsers.add_parser("set", help="Update a configuration value")
    config_set_parser.add_argument("key", help="Configuration key (e.g., memgraph.host)")
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

    elif args.command == "detect":
        handle_detect(args, settings)

    elif args.command == "qa":
        handle_qa(args, settings)

    elif args.command == "models":
        show_models()

    elif args.command == "modes":
        show_modes()

    elif args.command == "run":
        handle_unified_pipeline(args, settings)

    elif args.command == "services":
        if args.service_command == "memgraph":
            handle_memgraph(args)
        elif args.service_command == "ollama":
            handle_ollama(args)
        else:
            console.print("[red]No service command provided. Use 'orion services --help'.[/red]")

    elif args.command == "status":
        from ..managers.auto_config import status_command

        status_command(args)

    elif args.command == "research":
        handle_research(args, settings)

    elif args.command == "init":
        handle_init(args)

    else:
        print_banner()
        parser.print_help()


if __name__ == "__main__":
    main()
