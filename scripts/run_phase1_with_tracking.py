#!/usr/bin/env python3
"""
Phase 1 Detection + Tracking with YOLO-World (v5 vocabulary)

This script runs the full Phase 1 pipeline using YOLO-World open-vocabulary detector
with our optimized 124-class vocabulary, and includes ByteTrack for tracking.

Output: tracks.jsonl compatible with Phase 2 Re-ID
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add orion to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Phase 1: YOLO-World Detection + Tracking")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--episode", required=True, help="Episode ID for results")
    parser.add_argument("--fps", type=float, default=5.0, help="Target FPS (default: 5)")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default="auto", help="Device (auto/cuda/cpu)")
    args = parser.parse_args()

    results_dir = Path("results") / args.episode
    results_dir.mkdir(parents=True, exist_ok=True)

    # Configuration display
    console.print(Panel(
        f"[bold]Phase 1 Detection + Tracking[/bold]\n\n"
        f"Video: {args.video}\n"
        f"Output: {results_dir}\n"
        f"FPS: {args.fps}\n"
        f"Confidence: {args.confidence}",
        title="Configuration"
    ))

    # Load YOLO-World detector
    console.print("\n[bold]Loading YOLO-World detector...[/bold]")
    
    from orion.backends.yoloworld_backend import YOLOWorldDetector, YOLOWorldConfig, DEFAULT_CLASSES
    
    # Determine device
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    config = YOLOWorldConfig(
        model="yolov8x-worldv2",
        classes=DEFAULT_CLASSES.copy(),
        confidence=args.confidence,
        device=device,
    )
    detector = YOLOWorldDetector(config)
    
    console.print(f"✓ Loaded with [cyan]{len(config.classes)}[/cyan] classes on [yellow]{device}[/yellow]")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        console.print(f"[red]Error: Cannot open video {args.video}[/red]")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    # Calculate frame sampling
    frame_interval = max(1, int(video_fps / args.fps))
    expected_samples = total_frames // frame_interval

    console.print(f"\n[bold]Video Info:[/bold]")
    console.print(f"  Resolution: {width}x{height}")
    console.print(f"  Duration: {duration:.1f}s ({total_frames} frames @ {video_fps:.1f} FPS)")
    console.print(f"  Sampling: every {frame_interval} frames → ~{expected_samples} samples")

    # Initialize simple IoU tracker
    console.print("\n[bold]Initializing tracker...[/bold]")
    
    class SimpleTracker:
        """Simple IoU-based tracker for object tracking."""
        def __init__(self, max_age=30, iou_threshold=0.3):
            self.max_age = max_age
            self.iou_threshold = iou_threshold
            self.tracks = {}
            self.next_id = 1
            self.track_ages = {}
        
        def update(self, detections):
            """Update tracks with new detections."""
            results = []
            matched = set()
            
            # Calculate IoU between existing tracks and new detections
            for det in detections:
                best_iou = 0
                best_track_id = None
                det_box = det['bbox']
                det_label = det['label']
                
                for track_id, track in self.tracks.items():
                    if track['label'] != det_label:
                        continue
                    iou = self._calc_iou(det_box, track['bbox'])
                    if iou > best_iou and iou > self.iou_threshold:
                        best_iou = iou
                        best_track_id = track_id
                
                if best_track_id is not None:
                    # Update existing track
                    self.tracks[best_track_id]['bbox'] = det_box
                    self.tracks[best_track_id]['confidence'] = det['confidence']
                    self.track_ages[best_track_id] = 0
                    matched.add(best_track_id)
                    results.append({**det, 'track_id': best_track_id})
                else:
                    # Create new track
                    track_id = self.next_id
                    self.next_id += 1
                    self.tracks[track_id] = {
                        'bbox': det_box,
                        'label': det_label,
                        'confidence': det['confidence']
                    }
                    self.track_ages[track_id] = 0
                    matched.add(track_id)
                    results.append({**det, 'track_id': track_id})
            
            # Age unmatched tracks
            to_remove = []
            for track_id in self.tracks:
                if track_id not in matched:
                    self.track_ages[track_id] += 1
                    if self.track_ages[track_id] > self.max_age:
                        to_remove.append(track_id)
            
            for track_id in to_remove:
                del self.tracks[track_id]
                del self.track_ages[track_id]
            
            return results
        
        def _calc_iou(self, box1, box2):
            """Calculate IoU between two boxes [x1, y1, x2, y2]."""
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - inter
            
            return inter / union if union > 0 else 0
    
    tracker = SimpleTracker(max_age=30, iou_threshold=0.3)
    console.print("✓ Simple IoU tracker initialized")

    # Process video
    console.print("\n[bold]Running detection + tracking...[/bold]")
    
    all_tracks = []
    class_counts = defaultdict(int)
    frame_idx = 0
    sampled_count = 0
    start_time = time.time()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Processing frames...", total=total_frames)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # Detect objects
                detections = detector.detect(frame)
                
                # Track objects
                tracked = tracker.update(detections)
                
                # Store tracks
                for t in tracked:
                    track_record = {
                        'frame_id': frame_idx,
                        'track_id': t['track_id'],
                        'label': t['label'],
                        'confidence': t['confidence'],
                        'bbox': t['bbox'],
                    }
                    all_tracks.append(track_record)
                    class_counts[t['label']] += 1
                
                sampled_count += 1
                
                # Update progress with stats
                if sampled_count % 50 == 0:
                    elapsed = time.time() - start_time
                    fps = sampled_count / elapsed if elapsed > 0 else 0
                    progress.update(task, description=f"Frame {sampled_count}/{expected_samples} - {len(all_tracks)} tracks - {fps:.1f} fps")
            
            frame_idx += 1
            progress.update(task, advance=1)
    
    cap.release()
    elapsed = time.time() - start_time
    
    # Get unique tracks
    unique_tracks = len(set(t['track_id'] for t in all_tracks))
    
    console.print(f"\n✓ Detection + Tracking complete!")
    console.print(f"  Time: {elapsed:.1f}s ({sampled_count/elapsed:.1f} frames/sec)")
    console.print(f"  Track observations: {len(all_tracks)}")
    console.print(f"  Unique tracks: {unique_tracks}")

    # Save tracks.jsonl
    tracks_path = results_dir / "tracks.jsonl"
    with open(tracks_path, 'w') as f:
        for t in all_tracks:
            f.write(json.dumps(t) + '\n')
    
    console.print(f"\n[green]✓ Saved tracks.jsonl ({len(all_tracks)} records)[/green]")

    # Save metadata
    metadata = {
        "video": args.video,
        "fps": args.fps,
        "confidence": args.confidence,
        "vocab_size": len(config.classes),
        "total_frames": total_frames,
        "sampled_frames": sampled_count,
        "processing_time": elapsed,
        "statistics": {
            "frames_processed": sampled_count,
            "unique_tracks": unique_tracks,
            "total_observations": len(all_tracks),
        }
    }
    
    meta_path = results_dir / "run_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Display class distribution
    sorted_classes = sorted(class_counts.items(), key=lambda x: -x[1])[:20]
    
    table = Table(title=f"Top 20 Detected Classes ({len(class_counts)} unique)")
    table.add_column("Class", style="cyan")
    table.add_column("Count", justify="right")
    
    for label, count in sorted_classes:
        table.add_row(label, str(count))
    
    console.print(table)
    console.print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
