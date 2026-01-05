#!/usr/bin/env python3
"""
Phase 1 Improved Detection - Run with improved vocabulary
Tests the new 151-class vocabulary vs baseline 55-class vocabulary.

Usage:
    python scripts/run_phase1_improved.py --episode phase1_improved_v1 --fps 5
"""

import argparse
import cv2
import json
import sys
import time
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Optional

# Add orion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Rich for pretty output
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

console = Console()


@dataclass
class DetectionResult:
    frame_idx: int
    label: str
    confidence: float
    bbox: list  # [x1, y1, x2, y2]
    track_id: Optional[int] = None


def run_detection(
    video_path: str,
    output_dir: Path,
    fps: float = 5.0,
    confidence: float = 0.25,
    device: str = "cuda"
):
    """Run YOLO-World detection with improved vocabulary."""
    
    from orion.backends.yoloworld_backend import YOLOWorldDetector, YOLOWorldConfig, DEFAULT_CLASSES
    
    console.print(Panel(
        f"[bold cyan]Phase 1 Detection - Improved Vocabulary[/bold cyan]\n\n"
        f"Video: {video_path}\n"
        f"Output: {output_dir}\n"
        f"FPS: {fps}\n"
        f"Confidence: {confidence}\n"
        f"Classes: {len(DEFAULT_CLASSES)}\n"
        f"Device: {device}",
        title="Configuration"
    ))
    
    # Initialize detector
    console.print("\n[yellow]Loading YOLO-World detector...[/yellow]")
    config = YOLOWorldConfig(device=device, confidence=confidence)
    detector = YOLOWorldDetector(config)
    console.print(f"[green]✓ Loaded with {len(DEFAULT_CLASSES)} classes[/green]")
    
    # Show vocabulary sample
    console.print("\n[dim]Vocabulary sample:[/dim]")
    for i in range(0, min(30, len(DEFAULT_CLASSES)), 10):
        console.print(f"  {DEFAULT_CLASSES[i:i+10]}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        console.print(f"[red]Error: Cannot open video {video_path}[/red]")
        return None
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    
    # Calculate sampling
    sample_interval = max(1, int(video_fps / fps))
    expected_samples = total_frames // sample_interval
    
    console.print(f"\n[cyan]Video Info:[/cyan]")
    console.print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    console.print(f"  Duration: {duration:.1f}s ({total_frames} frames @ {video_fps:.1f} FPS)")
    console.print(f"  Sampling: every {sample_interval} frames → ~{expected_samples} samples")
    
    # Run detection
    all_detections = []
    frame_idx = 0
    sampled = 0
    start_time = time.time()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("{task.completed}/{task.total} frames"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Detecting...", total=expected_samples)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_interval == 0:
                # Run detection
                results = detector.detect(frame)
                
                frame_detections = 0
                for r in results:
                    if hasattr(r, 'boxes') and r.boxes is not None:
                        for i, box in enumerate(r.boxes):
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            label = r.names.get(cls_id, str(cls_id))
                            
                            all_detections.append({
                                'frame': frame_idx,
                                'label': label,
                                'confidence': round(conf, 4),
                                'bbox': [round(float(x1), 1), round(float(y1), 1), 
                                        round(float(x2), 1), round(float(y2), 1)]
                            })
                            frame_detections += 1
                
                sampled += 1
                progress.update(task, advance=1, description=f"Frame {frame_idx}: {frame_detections} dets")
            
            frame_idx += 1
    
    cap.release()
    elapsed = time.time() - start_time
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    detections_file = output_dir / "detections.json"
    with open(detections_file, 'w') as f:
        json.dump({
            'metadata': {
                'video': video_path,
                'fps': fps,
                'confidence': confidence,
                'vocab_size': len(DEFAULT_CLASSES),
                'total_frames': total_frames,
                'sampled_frames': sampled,
                'processing_time': elapsed
            },
            'detections': all_detections
        }, f, indent=2)
    
    # Statistics
    console.print(f"\n[green]✓ Detection complete![/green]")
    console.print(f"  Time: {elapsed:.1f}s ({sampled/elapsed:.1f} frames/sec)")
    console.print(f"  Total detections: {len(all_detections)}")
    console.print(f"  Avg per frame: {len(all_detections)/sampled:.1f}")
    
    # Class distribution
    labels = Counter(d['label'] for d in all_detections)
    
    table = Table(title=f"Top 20 Detected Classes ({len(labels)} unique)")
    table.add_column("Class", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_column("Avg Conf", style="yellow", justify="right")
    
    for label, count in labels.most_common(20):
        avg_conf = sum(d['confidence'] for d in all_detections if d['label'] == label) / count
        table.add_row(label, str(count), f"{avg_conf:.2f}")
    
    console.print(table)
    
    # Save summary
    summary_file = output_dir / "detection_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'total_detections': len(all_detections),
            'sampled_frames': sampled,
            'unique_classes': len(labels),
            'avg_per_frame': len(all_detections) / sampled,
            'processing_time': elapsed,
            'class_distribution': dict(labels.most_common())
        }, f, indent=2)
    
    console.print(f"\n[dim]Results saved to {output_dir}[/dim]")
    
    return all_detections


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Improved Detection")
    parser.add_argument("--video", "-v", default="data/examples/test.mp4",
                       help="Path to video file")
    parser.add_argument("--episode", "-e", default="phase1_improved_v1",
                       help="Episode name for output")
    parser.add_argument("--fps", type=float, default=5.0,
                       help="Target FPS for sampling")
    parser.add_argument("--confidence", type=float, default=0.25,
                       help="Detection confidence threshold")
    parser.add_argument("--device", default="cuda",
                       help="Device (cuda/cpu/mps)")
    
    args = parser.parse_args()
    
    video_path = args.video
    output_dir = Path("results") / args.episode
    
    run_detection(
        video_path=video_path,
        output_dir=output_dir,
        fps=args.fps,
        confidence=args.confidence,
        device=args.device
    )


if __name__ == "__main__":
    main()
