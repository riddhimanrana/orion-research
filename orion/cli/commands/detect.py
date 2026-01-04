"""
orion detect - Phase 1 Detection + Tracking Command

Run YOLO-World open-vocabulary detection with built-in tracking.
Produces tracks.jsonl with per-frame detections and track IDs.

Usage:
    orion detect --video data/examples/test.mp4 --episode phase1_test
    orion detect --video data/examples/test.mp4 --episode phase1_test --detector yolov8x-worldv2
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class DetectionStats:
    """Statistics from detection run."""
    total_frames: int = 0
    processed_frames: int = 0
    total_detections: int = 0
    unique_tracks: int = 0
    elapsed_seconds: float = 0.0
    fps: float = 0.0


def handle_detect(args, settings) -> int:
    """
    Run Phase 1: YOLO-World detection + tracking pipeline.
    
    Outputs:
        - results/<episode>/tracks.jsonl: Per-frame track observations
        - results/<episode>/episode_meta.json: Episode metadata
        - results/<episode>/detection_stats.json: Run statistics
    """
    import cv2
    
    # Validate inputs
    video_path = Path(args.video)
    if not video_path.exists():
        console.print(f"[red]✗ Video not found: {video_path}[/red]")
        return 1
    
    episode = args.episode
    output_dir = Path(args.output_dir) if hasattr(args, 'output_dir') and args.output_dir else Path("results") / episode
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = getattr(args, 'device', 'cuda')
    fps = getattr(args, 'fps', 5.0)
    confidence = getattr(args, 'confidence', 0.25)
    detector_model = getattr(args, 'detector', 'yolov8x-worldv2')
    classes = getattr(args, 'classes', None)
    
    console.print(f"""
[bold blue]╔══════════════════════════════════════════════════════════════════╗[/bold blue]
[bold blue]║[/bold blue]        [bold white]ORION V2 - PHASE 1: DETECTION + TRACKING[/bold white]            [bold blue]║[/bold blue]
[bold blue]╠══════════════════════════════════════════════════════════════════╣[/bold blue]
[bold blue]║[/bold blue]  Episode:   [cyan]{episode:<52}[/cyan] [bold blue]║[/bold blue]
[bold blue]║[/bold blue]  Video:     [cyan]{video_path.name:<52}[/cyan] [bold blue]║[/bold blue]
[bold blue]║[/bold blue]  Detector:  [cyan]{detector_model:<52}[/cyan] [bold blue]║[/bold blue]
[bold blue]║[/bold blue]  Device:    [cyan]{device:<52}[/cyan] [bold blue]║[/bold blue]
[bold blue]║[/bold blue]  FPS:       [cyan]{fps:<52}[/cyan] [bold blue]║[/bold blue]
[bold blue]╚══════════════════════════════════════════════════════════════════╝[/bold blue]
""")
    
    # Get video metadata
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        console.print(f"[red]✗ Cannot open video: {video_path}[/red]")
        return 1
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    cap.release()
    
    console.print(f"  Video: [cyan]{video_fps:.1f}[/cyan] FPS, [cyan]{total_frames}[/cyan] frames, [cyan]{duration:.1f}s[/cyan] duration")
    console.print(f"  Resolution: [cyan]{width}x{height}[/cyan]")
    
    # Calculate sample interval
    sample_interval = max(1, int(video_fps / fps))
    expected_samples = total_frames // sample_interval
    console.print(f"  Sampling every [cyan]{sample_interval}[/cyan] frames → ~[cyan]{expected_samples}[/cyan] samples\n")
    
    # Save episode metadata
    episode_meta = {
        "episode_id": episode,
        "video_path": str(video_path.absolute()),
        "video": {
            "fps": video_fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_seconds": duration
        },
        "config": {
            "detector": detector_model,
            "device": device,
            "target_fps": fps,
            "sample_interval": sample_interval,
            "confidence_threshold": confidence
        },
        "status": {
            "detected": False,
            "embedded": False,
            "filtered": False,
            "graphed": False,
            "exported": False
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    meta_path = output_dir / "episode_meta.json"
    with open(meta_path, "w") as f:
        json.dump(episode_meta, f, indent=2)
    console.print(f"  ✓ Episode metadata → [green]{meta_path}[/green]\n")
    
    # Initialize YOLO-World detector
    console.print("[bold]Loading YOLO-World detector...[/bold]")
    
    try:
        from orion.backends.yoloworld_backend import YOLOWorldDetector, YOLOWorldConfig, DEFAULT_CLASSES
        
        config = YOLOWorldConfig(
            model=detector_model.replace(".pt", ""),  # Remove .pt if present
            device=device,
            confidence=confidence,
            classes=classes if classes else DEFAULT_CLASSES,
        )
        detector = YOLOWorldDetector(config)
        
        # Force load to show class count
        detector._ensure_loaded()
        console.print(f"  ✓ YOLO-World loaded: [cyan]{len(config.classes)}[/cyan] classes\n")
        
    except Exception as e:
        console.print(f"[red]✗ Failed to load detector: {e}[/red]")
        logger.exception("Detector load failed")
        return 1
    
    # Run detection + tracking
    console.print("[bold]Running detection + tracking...[/bold]\n")
    
    start_time = time.time()
    tracks_data = []
    unique_track_ids = set()
    detection_count = 0
    frame_count = 0
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Processing frames", total=expected_samples)
            
            # Use YOLO-World's built-in tracking
            for frame_id, tracks in detector.track(str(video_path)):
                # Sample frames at target FPS
                if frame_id % sample_interval != 0:
                    continue
                
                timestamp = frame_id / video_fps
                
                for track in tracks:
                    track_id = track.get("track_id", -1)
                    if track_id == -1:
                        continue  # Skip untracked detections
                    
                    unique_track_ids.add(track_id)
                    detection_count += 1
                    
                    tracks_data.append({
                        "frame_id": frame_id,
                        "timestamp": round(timestamp, 3),
                        "track_id": track_id,
                        "bbox": [round(x, 1) for x in track["bbox"]],
                        "confidence": round(track["confidence"], 4),
                        "label": track["label"],
                        "class_id": track["class_id"]
                    })
                
                frame_count += 1
                progress.update(task, advance=1)
    
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]✗ Detection error: {e}[/red]")
        logger.exception("Detection failed")
        return 1
    
    elapsed = time.time() - start_time
    
    # Save tracks
    tracks_path = output_dir / "tracks.jsonl"
    with open(tracks_path, "w") as f:
        for track in tracks_data:
            f.write(json.dumps(track) + "\n")
    
    # Compute stats
    stats = DetectionStats(
        total_frames=total_frames,
        processed_frames=frame_count,
        total_detections=detection_count,
        unique_tracks=len(unique_track_ids),
        elapsed_seconds=round(elapsed, 2),
        fps=round(frame_count / elapsed, 2) if elapsed > 0 else 0
    )
    
    # Save stats
    stats_path = output_dir / "detection_stats.json"
    with open(stats_path, "w") as f:
        json.dump(asdict(stats), f, indent=2)
    
    # Update episode status
    episode_meta["status"]["detected"] = True
    with open(meta_path, "w") as f:
        json.dump(episode_meta, f, indent=2)
    
    # Print summary
    console.print(f"""
[bold green]═══════════════════════════════════════════════════════════════════[/bold green]
[bold green]  ✓ PHASE 1 COMPLETE[/bold green]
[bold green]═══════════════════════════════════════════════════════════════════[/bold green]

  [bold]Results:[/bold]
    • Frames processed:  [cyan]{stats.processed_frames}[/cyan]
    • Total detections:  [cyan]{stats.total_detections}[/cyan]
    • Unique tracks:     [cyan]{stats.unique_tracks}[/cyan]
    • Processing time:   [cyan]{stats.elapsed_seconds:.1f}s[/cyan]
    • Processing FPS:    [cyan]{stats.fps:.1f}[/cyan]

  [bold]Output files:[/bold]
    • [green]{tracks_path}[/green]
    • [green]{stats_path}[/green]
    • [green]{meta_path}[/green]

  [bold]Next step:[/bold]
    [dim]orion embed --episode {episode}[/dim]
""")
    
    # Print track distribution by class
    if tracks_data:
        from collections import Counter
        label_counts = Counter(t["label"] for t in tracks_data)
        top_labels = label_counts.most_common(10)
        
        console.print("  [bold]Top detected classes:[/bold]")
        for label, count in top_labels:
            console.print(f"    • {label}: [cyan]{count}[/cyan]")
    
    return 0
