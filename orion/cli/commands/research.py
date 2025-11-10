"""Research mode command handlers."""

import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

from ...settings import OrionSettings

console = Console()


def handle_research(args, settings: OrionSettings) -> None:
    """Handle research mode commands (SLAM, depth, tracking, zones)"""
    
    if args.research_mode == "slam":
        # Run complete SLAM pipeline
        console.print("\n[bold cyan]🗺️  Starting SLAM Research Mode[/bold cyan]\n")
        
        # Check if video file exists
        video_path = Path(args.video)
        if not video_path.exists():
            console.print(f"[red]✗ Video file not found: {args.video}[/red]")
            sys.exit(1)
        
        params_table = Table(box=box.ROUNDED, show_header=False, border_style="cyan")
        params_table.add_column("Parameter", style="cyan bold", width=20)
        params_table.add_column("Value", style="yellow", width=60)
        
        params_table.add_row("Video", str(video_path.absolute()))
        params_table.add_row("Visualization", args.viz.upper())
        params_table.add_row("YOLO Model", args.yolo_model.upper())
        params_table.add_row("Frame Skip", str(args.skip))
        params_table.add_row("Zone Mode", args.zone_mode)
        
        if args.max_frames:
            params_table.add_row("Max Frames", str(args.max_frames))
        if args.no_adaptive:
            params_table.add_row("Adaptive Skip", "[red]Disabled[/red]")
        else:
            params_table.add_row("Adaptive Skip", "[green]Enabled[/green]")
        
        # FastVLM is always on-demand for real-time performance
        params_table.add_row("FastVLM", "[yellow]Query-time only (real-time mode)[/yellow]")
        
        if args.export_memgraph:
            params_table.add_row("Memgraph Export", "[green]✓ Enabled[/green]")
        if args.use_spatial_memory:
            params_table.add_row("Spatial Memory", f"[green]✓ Enabled[/green] ({args.memory_dir})")
        if args.interactive:
            params_table.add_row("Interactive Mode", "[green]✓ Spatial Intelligence Assistant[/green]")
        if args.debug:
            params_table.add_row("Debug Mode", "[yellow]Enabled[/yellow]")
        
        console.print(params_table)
        console.print("\n")
        
        # Build command to run SLAM script
        script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "run_slam_complete.py"
        
        if not script_path.exists():
            console.print(f"[red]✗ SLAM script not found: {script_path}[/red]")
            console.print("[yellow]Expected location: scripts/run_slam_complete.py[/yellow]")
            sys.exit(1)
        
        cmd = [
            sys.executable,
            str(script_path),
            "--video", str(video_path.absolute()),
            "--skip", str(args.skip),
            "--zone-mode", args.zone_mode,
            "--yolo-model", args.yolo_model,
        ]
        
        if args.viz == "rerun":
            cmd.append("--rerun")
        
        if args.max_frames:
            cmd.extend(["--max-frames", str(args.max_frames)])
        
        if args.no_adaptive:
            cmd.append("--no-adaptive")
        
        if args.export_memgraph:
            cmd.append("--export-memgraph")
        
        if args.use_spatial_memory:
            cmd.extend(["--use-spatial-memory", "--memory-dir", args.memory_dir])
        
        try:
            subprocess.run(cmd, check=True)
            
            # Post-processing: Sync to spatial memory and start interactive mode
            if args.interactive:
                console.print("\n[bold cyan]════════════════════════════════════════════════[/bold cyan]")
                console.print("[bold cyan]   Spatial Intelligence Assistant[/bold cyan]")
                console.print("[bold cyan]════════════════════════════════════════════════[/bold cyan]\n")
                
                # Check which script to use
                if args.use_spatial_memory or args.export_memgraph:
                    # Use the new spatial intelligence assistant
                    assistant_script = Path(__file__).parent.parent.parent.parent / "scripts" / "spatial_intelligence_assistant.py"
                    
                    if assistant_script.exists():
                        # First sync from Memgraph if enabled
                        if args.export_memgraph:
                            console.print("[cyan]🔄 Syncing latest data to persistent memory...[/cyan]\n")
                            sync_cmd = [sys.executable, str(assistant_script), "--sync"]
                            subprocess.run(sync_cmd)
                            console.print()
                        
                        # Start interactive mode
                        query_cmd = [sys.executable, str(assistant_script), "--interactive"]
                        subprocess.run(query_cmd)
                    else:
                        console.print(f"[red]✗ Spatial intelligence assistant not found: {assistant_script}[/red]")
                        console.print("[yellow]Falling back to basic query mode...[/yellow]\n")
                        
                        # Fallback to basic query mode
                        query_script = Path(__file__).parent.parent.parent.parent / "scripts" / "query_memgraph.py"
                        if query_script.exists() and args.export_memgraph:
                            query_cmd = [sys.executable, str(query_script), "--interactive"]
                            subprocess.run(query_cmd)
                else:
                    console.print("[yellow]⚠️  Interactive mode works best with --export-memgraph or --use-spatial-memory[/yellow]")
                    console.print("[dim]Example:[/dim]")
                    console.print("[dim]  orion research slam --video X --export-memgraph --use-spatial-memory -i[/dim]\n")
                
        except subprocess.CalledProcessError as e:
            console.print(f"[red]✗ SLAM pipeline failed with exit code {e.returncode}[/red]")
            sys.exit(e.returncode)
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
            sys.exit(0)
    
    elif args.research_mode == "depth":
        console.print("\n[bold cyan]📊 Depth Estimation Test Mode[/bold cyan]\n")
        console.print("[yellow]⚠ Coming soon: Isolated depth estimation testing[/yellow]\n")
        console.print("This mode will allow you to:")
        console.print("  • Test different depth models (MiDaS, ZoeDepth)")
        console.print("  • Visualize depth maps in 2D and 3D")
        console.print("  • Compare depth estimation quality")
        console.print("  • Benchmark inference speed\n")
    
    elif args.research_mode == "tracking":
        console.print("\n[bold cyan]👁️  Entity Tracking Test Mode[/bold cyan]\n")
        console.print("[yellow]⚠ Coming soon: 3D tracking with Re-ID testing[/yellow]\n")
        console.print("This mode will allow you to:")
        console.print("  • Test entity tracking across frames")
        console.print("  • Visualize Re-ID embeddings")
        console.print("  • Benchmark tracking accuracy")
        console.print("  • Debug trajectory estimation\n")
    
    elif args.research_mode == "zones":
        console.print("\n[bold cyan]🗂️  Spatial Zones Test Mode[/bold cyan]\n")
        console.print("[yellow]⚠ Coming soon: Zone detection and classification testing[/yellow]\n")
        console.print("This mode will allow you to:")
        console.print("  • Test zone detection algorithms")
        console.print("  • Visualize spatial clustering")
        console.print("  • Adjust HDBSCAN parameters")
        console.print("  • Export zone definitions\n")
    
    else:
        console.print("[red]No research mode specified. Use 'orion research --help'[/red]")
