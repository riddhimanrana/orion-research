"""
Orion v2 Main CLI Entry Point

Commands:
  orion init      - Initialize a new episode from video
  orion analyze   - Run full pipeline (detect → embed → filter → graph)
  orion detect    - Run detection + tracking only
  orion embed     - Run V-JEPA2 re-identification
  orion filter    - Run FastVLM semantic filtering
  orion graph     - Build scene graph and CIS
  orion export    - Export to Memgraph
  orion query     - Natural language query interface
  orion status    - Show episode status
  orion diagnose  - Diagnostic tools for debugging
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="orion",
        description="Orion v2: Memory-Centric Video Understanding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  orion init --episode room_scan --video /path/to/video.mp4
  orion analyze --episode room_scan --fps 5
  orion query "What objects are in the kitchen?"
  orion status --episode room_scan
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # === INIT ===
    init_parser = subparsers.add_parser("init", help="Initialize a new episode")
    init_parser.add_argument("--episode", "-e", required=True, help="Episode name/ID")
    init_parser.add_argument("--video", "-v", required=True, help="Path to video file")
    init_parser.add_argument("--fps", type=float, default=5.0, help="Target FPS for sampling")
    init_parser.add_argument("--output-dir", help="Output directory (default: results/<episode>)")
    
    # === ANALYZE ===
    analyze_parser = subparsers.add_parser("analyze", help="Run full analysis pipeline")
    analyze_parser.add_argument("--episode", "-e", required=True, help="Episode name/ID")
    analyze_parser.add_argument("--fps", type=float, default=5.0, help="Target FPS")
    analyze_parser.add_argument("--detector", choices=["yoloworld", "yolo11x", "yolo11m"], 
                                default="yoloworld", help="Detection backend")
    analyze_parser.add_argument("--embedder", choices=["vjepa2", "dino", "clip"], 
                                default="vjepa2", help="Re-ID embedder")
    analyze_parser.add_argument("--skip-filter", action="store_true", help="Skip FastVLM filtering")
    analyze_parser.add_argument("--skip-graph", action="store_true", help="Skip scene graph")
    analyze_parser.add_argument("--export-memgraph", action="store_true", help="Export to Memgraph")
    analyze_parser.add_argument("--device", default="cuda", help="Device (cuda/cpu/mps)")
    
    # === DETECT ===
    detect_parser = subparsers.add_parser("detect", help="Run detection + tracking only")
    detect_parser.add_argument("--episode", "-e", required=True, help="Episode name/ID")
    detect_parser.add_argument("--detector", choices=["yoloworld", "yolo11x"], default="yoloworld")
    detect_parser.add_argument("--classes", nargs="+", help="Custom class prompts for YOLO-World")
    detect_parser.add_argument("--confidence", type=float, default=0.25)
    detect_parser.add_argument("--device", default="cuda")
    
    # === EMBED ===
    embed_parser = subparsers.add_parser("embed", help="Run V-JEPA2 re-identification")
    embed_parser.add_argument("--episode", "-e", required=True, help="Episode name/ID")
    embed_parser.add_argument("--embedder", choices=["vjepa2", "dino", "clip"], default="vjepa2")
    embed_parser.add_argument("--mode", choices=["single", "video"], default="single",
                              help="single=best frame, video=multi-crop as video")
    embed_parser.add_argument("--threshold", type=float, default=0.75, 
                              help="Cosine similarity threshold for clustering")
    
    # === FILTER ===
    filter_parser = subparsers.add_parser("filter", help="Run FastVLM semantic filtering")
    filter_parser.add_argument("--episode", "-e", required=True, help="Episode name/ID")
    filter_parser.add_argument("--sentence-model", default="sentence-transformers/all-mpnet-base-v2")
    filter_parser.add_argument("--scene-trigger", choices=["none", "cosine"], default="cosine")
    filter_parser.add_argument("--scene-threshold", type=float, default=0.98)
    
    # === GRAPH ===
    graph_parser = subparsers.add_parser("graph", help="Build scene graph + CIS")
    graph_parser.add_argument("--episode", "-e", required=True, help="Episode name/ID")
    graph_parser.add_argument("--compute-cis", action="store_true", help="Compute Causal Influence Scores")
    
    # === EXPORT ===
    export_parser = subparsers.add_parser("export", help="Export to Memgraph")
    export_parser.add_argument("--episode", "-e", required=True, help="Episode name/ID")
    export_parser.add_argument("--memgraph-host", default="localhost")
    export_parser.add_argument("--memgraph-port", type=int, default=7687)
    export_parser.add_argument("--clear-existing", action="store_true", 
                               help="Clear existing data for this episode")
    
    # === QUERY ===
    query_parser = subparsers.add_parser("query", help="Natural language query")
    query_parser.add_argument("question", nargs="?", help="Question to ask")
    query_parser.add_argument("--episode", "-e", help="Episode to query (optional)")
    query_parser.add_argument("--model", default="qwen2.5:14b-instruct-q8_0", 
                              help="Ollama model to use")
    query_parser.add_argument("--interactive", "-i", action="store_true", 
                              help="Interactive query mode")
    
    # === STATUS ===
    status_parser = subparsers.add_parser("status", help="Show episode status")
    status_parser.add_argument("--episode", "-e", help="Episode name (or list all)")
    status_parser.add_argument("--verbose", "-v", action="store_true")
    
    # === DIAGNOSE ===
    diagnose_parser = subparsers.add_parser("diagnose", help="Diagnostic tools")
    diagnose_subparsers = diagnose_parser.add_subparsers(dest="diagnose_type")
    
    # diagnose reid
    reid_diag = diagnose_subparsers.add_parser("reid", help="Diagnose Re-ID issues")
    reid_diag.add_argument("--episode", "-e", required=True)
    reid_diag.add_argument("--track-id", type=int, help="Specific track to analyze")
    reid_diag.add_argument("--auto-pair", action="store_true", help="Auto-find worst pair")
    
    # diagnose detection
    det_diag = diagnose_subparsers.add_parser("detection", help="Diagnose detection issues")
    det_diag.add_argument("--episode", "-e", required=True)
    det_diag.add_argument("--frame", type=int, help="Specific frame to analyze")
    
    # === OVERLAY ===
    overlay_parser = subparsers.add_parser("overlay", help="Render video overlay")
    overlay_parser.add_argument("--episode", "-e", required=True, help="Episode name/ID")
    overlay_parser.add_argument("--output", "-o", help="Output video path")
    overlay_parser.add_argument("--style", choices=["v1", "v2", "v3"], default="v3",
                                help="Overlay style (v1=simple, v2=with conf, v3=pseudo-3D)")
    
    # Parse args
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Route to appropriate handler
    if args.command == "init":
        from orion.cli.v2.init_cmd import run_init
        return run_init(args)
    
    elif args.command == "analyze":
        from orion.cli.v2.analyze_cmd import run_analyze
        return run_analyze(args)
    
    elif args.command == "detect":
        from orion.cli.v2.detect_cmd import run_detect
        return run_detect(args)
    
    elif args.command == "embed":
        from orion.cli.v2.embed_cmd import run_embed
        return run_embed(args)
    
    elif args.command == "filter":
        from orion.cli.v2.filter_cmd import run_filter
        return run_filter(args)
    
    elif args.command == "graph":
        from orion.cli.v2.graph_cmd import run_graph
        return run_graph(args)
    
    elif args.command == "export":
        from orion.cli.v2.export_cmd import run_export
        return run_export(args)
    
    elif args.command == "query":
        from orion.cli.v2.query_cmd import run_query
        return run_query(args)
    
    elif args.command == "status":
        from orion.cli.v2.status_cmd import run_status
        return run_status(args)
    
    elif args.command == "diagnose":
        from orion.cli.v2.diagnose_cmd import run_diagnose
        return run_diagnose(args)
    
    elif args.command == "overlay":
        from orion.cli.v2.overlay_cmd import run_overlay
        return run_overlay(args)
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
