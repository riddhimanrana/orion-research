"""
Orion Video Analysis Pipeline
==============================

Complete video understanding pipeline with:
- Visual perception & object detection
- Semantic knowledge graph construction  
- Interactive Q&A with local LLM

Author: Orion Research Team
"""

import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich import print as rprint

# Import pipeline components
from .perception_engine import run_perception_engine
from .semantic_uplift import run_semantic_uplift
from .neo4j_manager import clear_neo4j_for_new_run
from .models import ModelManager
try:
    from .runtime import BackendName, get_active_backend, select_backend, set_active_backend
except ImportError:  # pragma: no cover
    BackendName = str  # type: ignore[assignment,misc]
    from .runtime import get_active_backend, select_backend, set_active_backend

VideoQASystem: Any

try:
    from .video_qa import VideoQASystem as _VideoQASystem
    VideoQASystem = _VideoQASystem
    QA_AVAILABLE = True
except ImportError:  # pragma: no cover
    VideoQASystem = None
    QA_AVAILABLE = False

# Configurations
apply_part1_config: Any
apply_part2_config: Any

try:
    from .perception_config import apply_config as apply_part1_config
    from .semantic_config import apply_config as apply_part2_config
    CONFIGS_AVAILABLE = True
except ImportError:  # pragma: no cover
    apply_part1_config = None
    apply_part2_config = None
    CONFIGS_AVAILABLE = False

console = Console()


def setup_logging(verbose: bool = False):
    """Setup clean logging"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure logging (console only, no file logging)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Silence noisy libraries
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('coremltools').setLevel(logging.WARNING)


def print_banner():
    """Print startup banner"""
    banner = """
[bold cyan]╔═══════════════════════════════════════════════════════════════╗
║                  ORION VIDEO ANALYSIS PIPELINE                ║
║                    From Moments to Memory                     ║
╚═══════════════════════════════════════════════════════════════╝[/bold cyan]
    """
    console.print(banner)


def _resolve_runtime(preferred: Optional[str]) -> Tuple[BackendName, ModelManager]:
    normalized: Optional[str] = preferred.lower() if preferred else None
    if normalized == "auto":
        normalized = None
    if normalized is not None and normalized not in {"mlx", "torch"}:
        raise ValueError(f"Unsupported runtime preference '{preferred}'.")

    active_backend = get_active_backend()
    if normalized in {"mlx", "torch"}:
        backend = select_backend(normalized)
    elif active_backend is not None:
        backend = active_backend
    else:
        backend = select_backend(None)

    set_active_backend(backend)
    manager = ModelManager()
    if not manager.assets_ready(backend):
        console.print(f"[yellow]Preparing model assets for runtime '{backend}'...[/yellow]")
        manager.ensure_runtime_assets(backend)
    return backend, manager


def run_pipeline(
    video_path: str,
    output_dir: str = "data/testing",
    neo4j_uri: str = "neo4j://127.0.0.1:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "orion123",
    clear_db: bool = True,
    part1_config: str = "balanced",
    part2_config: str = "balanced",
    skip_part1: bool = False,
    skip_part2: bool = False,
    verbose: bool = False,
    runtime: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run complete video analysis pipeline
    
    Args:
        video_path: Path to input video
        output_dir: Directory for output files
        neo4j_uri: Neo4j database URI
    neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        clear_db: Whether to clear Neo4j before running
        part1_config: Configuration for perception engine (fast/balanced/accurate)
        part2_config: Configuration for semantic uplift (fast/balanced/accurate)
        skip_part1: Skip perception engine (use existing results)
        skip_part2: Skip semantic uplift
        verbose: Enable debug logging
        
    Returns:
        Results dictionary with statistics and file paths
    """
    setup_logging(verbose)
    
    results = {
        'video_path': video_path,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'part1': {},
        'part2': {},
        'success': False,
        'errors': []
    }
    
    try:
        backend, _ = _resolve_runtime(runtime)
        results['runtime'] = backend
        console.print(f"[dim]Runtime backend: {backend}[/dim]")

        # Clear Neo4j if requested
        if clear_db and not skip_part2:
            with console.status("[yellow]Clearing Neo4j database...[/yellow]"):
                if clear_neo4j_for_new_run(neo4j_uri, neo4j_user, neo4j_password):
                    console.print("✓ [green]Neo4j cleared[/green]")
                else:
                    console.print("⚠ [yellow]Could not clear Neo4j (may not be running)[/yellow]")
        
        # PART 1: Perception Engine
        if not skip_part1:
            console.print("\n" + "="*65)
            console.print("[bold cyan]PART 1: VISUAL PERCEPTION ENGINE[/bold cyan]")
            console.print("="*65)
            
            # Apply configuration
            if CONFIGS_AVAILABLE:
                if part1_config == 'fast':
                    from .perception_config import FAST_CONFIG
                    apply_part1_config(FAST_CONFIG)
                elif part1_config == 'accurate':
                    from .perception_config import ACCURATE_CONFIG
                    apply_part1_config(ACCURATE_CONFIG)
                else:  # balanced
                    from .perception_config import BALANCED_CONFIG
                    apply_part1_config(BALANCED_CONFIG)
                console.print(f"[dim]Using {part1_config} mode[/dim]\n")
            
            start_time = time.time()
            
            perception_output = run_perception_engine(video_path)
            
            duration = time.time() - start_time
            
            results['part1'] = {
                'success': True,
                'duration_seconds': duration,
                'num_objects': len(perception_output),
                'output_file': perception_output
            }
            
            # Show stats
            table = Table(title="Perception Results", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Objects Detected", str(len(perception_output)))
            table.add_row("Processing Time", f"{duration:.1f}s")
            table.add_row("Output File", str(perception_output))
            
            console.print(table)
        
        else:
            console.print("[yellow]Skipping Part 1 (using existing perception data)[/yellow]")
            perception_output = None
        
        # PART 2: Semantic Uplift
        if not skip_part2:
            console.print("\n" + "="*65)
            console.print("[bold cyan]PART 2: SEMANTIC KNOWLEDGE GRAPH[/bold cyan]")
            console.print("="*65)
            
            # Apply configuration
            if CONFIGS_AVAILABLE:
                if part2_config == 'fast':
                    from .semantic_config import FAST_CONFIG as SEM_FAST
                    apply_part2_config(SEM_FAST)
                elif part2_config == 'accurate':
                    from .semantic_config import ACCURATE_CONFIG as SEM_ACCURATE
                    apply_part2_config(SEM_ACCURATE)
                else:  # balanced
                    from .semantic_config import BALANCED_CONFIG as SEM_BALANCED
                    apply_part2_config(SEM_BALANCED)
                console.print(f"[dim]Using {part2_config} mode[/dim]\n")
            
            start_time = time.time()
            
            # Get perception log path
            if perception_output:
                perception_log_path = perception_output
            else:
                # Find most recent
                logs = sorted(Path(output_dir).glob("perception_log_*.json"))
                if not logs:
                    raise ValueError("No perception logs found. Run Part 1 first!")
                perception_log_path = str(logs[-1])
            
            uplift_results = run_semantic_uplift(  # type: ignore[arg-type]
                perception_log_path,
                neo4j_uri=neo4j_uri,
                neo4j_password=neo4j_password
            )
            
            duration = time.time() - start_time
            
            results['part2'] = {
                'success': uplift_results.get('success', False),
                'duration_seconds': duration,
                **uplift_results
            }
            
            # Show stats
            table = Table(title="Semantic Uplift Results", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Entities Tracked", str(uplift_results.get('num_entities', 0)))
            table.add_row("Scenes Segmented", str(uplift_results.get('num_scenes', 0)))
            table.add_row("Locations Inferred", str(uplift_results.get('num_locations', 0)))
            table.add_row("Scene Similarity Links", str(uplift_results.get('num_scene_similarity_links', 0)))
            table.add_row("State Changes", str(uplift_results.get('num_state_changes', 0)))
            table.add_row("Cypher Queries", str(uplift_results.get('num_queries', 0)))
            table.add_row("Processing Time", f"{duration:.1f}s")
            
            console.print(table)
        
        results['success'] = True
        
        # Save results
        output_file = Path(output_dir) / f"pipeline_results_{Path(video_path).stem}_{results['timestamp']}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        console.print(f"\n✓ [bold green]Pipeline complete![/bold green]")
        console.print(f"[dim]Results saved to: {output_file}[/dim]")
        
        return results
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user[/yellow]")
        results['success'] = False
        results['errors'].append('User interrupt')
        return results
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Pipeline failed: {e}[/bold red]")
        results['success'] = False
        results['errors'].append(str(e))
        import traceback
        logging.error(traceback.format_exc())
        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Orion Video Analysis Pipeline - Extract knowledge from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_pipeline.py video.mp4
  
  # Run with custom config
  python run_pipeline.py video.mp4 --config accurate
  
  # Skip perception (use existing results)
  python run_pipeline.py video.mp4 --skip-part1
  
  # Run then start Q&A session
  python run_pipeline.py video.mp4 --interactive
        """
    )
    
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output-dir', default='data/testing', help='Output directory')
    parser.add_argument('--config', choices=['fast', 'balanced', 'accurate'], 
                       default='balanced', help='Processing configuration')
    parser.add_argument('--neo4j-uri', default='neo4j://127.0.0.1:7687', 
                       help='Neo4j URI')
    parser.add_argument('--neo4j-password', default='orion123', 
                       help='Neo4j password')
    parser.add_argument('--no-clear-db', action='store_true',
                       help='Do not clear Neo4j before running')
    parser.add_argument('--skip-part1', action='store_true',
                       help='Skip perception engine')
    parser.add_argument('--skip-part2', action='store_true',
                       help='Skip semantic uplift')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Start interactive Q&A after pipeline')
    parser.add_argument('--qa-only', action='store_true',
                       help='Skip pipeline, only run Q&A')
    parser.add_argument('--qa-model', default='gemma3:4b',
                       help='Ollama model for Q&A')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Q&A only mode
    if args.qa_only:
        if not QA_AVAILABLE:
            console.print("[red]Error: Q&A system not available (install ollama package)[/red]")
            console.print("[dim]Run: pip install ollama[/dim]")
            return
        console.print("\n[bold cyan]Starting Q&A session...[/bold cyan]\n")
        qa = VideoQASystem(
            neo4j_password=args.neo4j_password,
            model=args.qa_model
        )
        qa.start_interactive_session()
        return
    
    # Run pipeline
    results = run_pipeline(
        video_path=args.video_path,
        output_dir=args.output_dir,
        neo4j_uri=args.neo4j_uri,
        neo4j_password=args.neo4j_password,
        clear_db=not args.no_clear_db,
        part1_config=args.config,
        part2_config=args.config,
        skip_part1=args.skip_part1,
        skip_part2=args.skip_part2,
        verbose=args.verbose
    )
    
    # Interactive Q&A if requested
    if args.interactive and results['success']:
        if not QA_AVAILABLE:
            console.print("\n[yellow]Warning: Q&A system not available[/yellow]")
            console.print("[dim]Install with: pip install ollama[/dim]")
            return
        console.print("\n[bold cyan]Starting interactive Q&A session...[/bold cyan]")
        console.print("[dim]You can now ask questions about the video![/dim]\n")
        
        qa = VideoQASystem(
            neo4j_password=args.neo4j_password,
            model=args.qa_model
        )
        qa.start_interactive_session()


if __name__ == "__main__":
    main()
