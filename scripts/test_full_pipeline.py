#!/usr/bin/env python3
"""
Full Phase 1 + Phase 2 Pipeline Test

Tests both vocabulary approaches end-to-end:
1. Fixed Vocab (v5) - 124 classes
2. Prompt Vocab (DetectionConfig) - 104 classes

Measures:
- Detection count
- Tracking quality
- Re-ID reduction
- Final object count
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

import cv2
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


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
                self.tracks[best_track_id]['bbox'] = det_box
                self.tracks[best_track_id]['confidence'] = det['confidence']
                self.track_ages[best_track_id] = 0
                matched.add(best_track_id)
                results.append({**det, 'track_id': best_track_id})
            else:
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
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0


def run_pipeline(video_path: str, episode: str, vocab_type: str, fps: float = 5.0, confidence: float = 0.25):
    """Run full Phase 1 + Phase 2 pipeline."""
    import torch
    from ultralytics import YOLOWorld
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = Path("results") / episode
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load detector based on vocab type
    if vocab_type == "fixed":
        from orion.backends.yoloworld_backend import DEFAULT_CLASSES
        classes = DEFAULT_CLASSES.copy()
        console.print(f"[cyan]Using Fixed Vocab (v5): {len(classes)} classes[/cyan]")
    else:  # prompt
        from orion.perception.config import DetectionConfig
        dc = DetectionConfig(backend="yoloworld")
        classes = dc.yoloworld_categories()
        console.print(f"[cyan]Using Prompt Vocab: {len(classes)} classes[/cyan]")
    
    # Initialize model
    model = YOLOWorld("yolov8x-worldv2.pt")
    model.to(device)
    model.set_classes(classes)
    
    # Phase 1: Detection + Tracking
    console.print("\n[bold]Phase 1: Detection + Tracking[/bold]")
    start_time = time.time()
    
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps))
    
    tracker = SimpleTracker(max_age=30, iou_threshold=0.3)
    all_tracks = []
    class_counts = defaultdict(int)
    frame_idx = 0
    sampled_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            # Detect
            results = model.predict(source=frame, conf=confidence, device=device, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                for i in range(len(boxes)):
                    det = {
                        'bbox': boxes.xyxy[i].tolist(),
                        'confidence': float(boxes.conf[i]),
                        'label': result.names[int(boxes.cls[i].item())]
                    }
                    detections.append(det)
            
            # Track
            tracked = tracker.update(detections)
            
            for t in tracked:
                all_tracks.append({
                    'frame_id': frame_idx,
                    'track_id': t['track_id'],
                    'label': t['label'],
                    'confidence': t['confidence'],
                    'bbox': t['bbox'],
                })
                class_counts[t['label']] += 1
            
            sampled_count += 1
            if sampled_count % 50 == 0:
                console.print(f"  Frame {sampled_count}... {len(all_tracks)} observations")
        
        frame_idx += 1
    
    cap.release()
    phase1_time = time.time() - start_time
    unique_tracks = len(set(t['track_id'] for t in all_tracks))
    
    console.print(f"  ✓ Phase 1 complete: {len(all_tracks)} observations, {unique_tracks} tracks in {phase1_time:.1f}s")
    
    # Save tracks
    tracks_path = results_dir / "tracks.jsonl"
    with open(tracks_path, 'w') as f:
        for t in all_tracks:
            f.write(json.dumps(t) + '\n')
    
    # Save metadata
    metadata = {
        "video": video_path,
        "vocab_type": vocab_type,
        "vocab_size": len(classes),
        "fps": fps,
        "phase1_time": phase1_time,
        "observations": len(all_tracks),
        "unique_tracks": unique_tracks,
    }
    with open(results_dir / "run_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Phase 2: Re-ID
    console.print("\n[bold]Phase 2: Re-ID[/bold]")
    start_time = time.time()
    
    try:
        from orion.perception.reid.matcher import build_memory_from_tracks
        
        memory_path = build_memory_from_tracks(
            episode_id=episode,
            video_path=Path(video_path),
            tracks_path=tracks_path,
            results_dir=results_dir,
            cosine_threshold=0.70,  # Optimized threshold
            max_crops_per_track=5,
        )
        
        with open(memory_path) as f:
            memory = json.load(f)
        
        final_objects = len(memory.get("objects", []))
        phase2_time = time.time() - start_time
        
        console.print(f"  ✓ Phase 2 complete: {unique_tracks} → {final_objects} objects in {phase2_time:.1f}s")
        
        reduction = (1 - final_objects / unique_tracks) * 100 if unique_tracks > 0 else 0
        
    except Exception as e:
        console.print(f"  [red]Phase 2 failed: {e}[/red]")
        final_objects = unique_tracks
        phase2_time = 0
        reduction = 0
    
    return {
        "episode": episode,
        "vocab_type": vocab_type,
        "vocab_size": len(classes),
        "observations": len(all_tracks),
        "unique_tracks": unique_tracks,
        "final_objects": final_objects,
        "reduction": reduction,
        "phase1_time": phase1_time,
        "phase2_time": phase2_time,
        "top_classes": dict(sorted(class_counts.items(), key=lambda x: -x[1])[:10]),
    }


def main():
    parser = argparse.ArgumentParser(description="Full Phase 1+2 Pipeline Test")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--fps", type=float, default=5.0, help="Target FPS")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    console.print(Panel(
        f"[bold]Full Pipeline Comparison[/bold]\n\n"
        f"Video: {args.video}\n"
        f"FPS: {args.fps}",
        title="Configuration"
    ))

    results = []

    # Test 1: Fixed Vocab (v5)
    console.print("\n" + "="*60)
    console.print("[bold magenta]TEST 1: Fixed Vocabulary (v5)[/bold magenta]")
    console.print("="*60)
    r1 = run_pipeline(args.video, "pipeline_test_fixed", "fixed", args.fps, args.confidence)
    results.append(r1)

    # Test 2: Prompt Vocab
    console.print("\n" + "="*60)
    console.print("[bold magenta]TEST 2: Prompt Vocabulary[/bold magenta]")
    console.print("="*60)
    r2 = run_pipeline(args.video, "pipeline_test_prompt", "prompt", args.fps, args.confidence)
    results.append(r2)

    # Summary
    console.print("\n" + "="*60)
    console.print("[bold green]SUMMARY[/bold green]")
    console.print("="*60)

    table = Table(title="Full Pipeline Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Fixed (v5)", justify="right")
    table.add_column("Prompt", justify="right")
    table.add_column("Winner", justify="center")

    metrics = [
        ("Vocabulary Size", "vocab_size", "lower"),
        ("Observations", "observations", "higher"),
        ("Unique Tracks", "unique_tracks", "higher"),
        ("Final Objects", "final_objects", "compare"),
        ("Re-ID Reduction", "reduction", "higher"),
        ("Phase 1 Time", "phase1_time", "lower"),
        ("Phase 2 Time", "phase2_time", "lower"),
    ]

    for name, key, comparison in metrics:
        v1 = r1.get(key, 0)
        v2 = r2.get(key, 0)
        
        if key in ["phase1_time", "phase2_time"]:
            s1, s2 = f"{v1:.1f}s", f"{v2:.1f}s"
        elif key == "reduction":
            s1, s2 = f"{v1:.1f}%", f"{v2:.1f}%"
        else:
            s1, s2 = str(v1), str(v2)
        
        if comparison == "higher":
            winner = "Fixed" if v1 > v2 else "Prompt" if v2 > v1 else "Tie"
        elif comparison == "lower":
            winner = "Fixed" if v1 < v2 else "Prompt" if v2 < v1 else "Tie"
        else:
            winner = "-"
        
        table.add_row(name, s1, s2, winner)

    console.print(table)

    # Save results
    output_path = Path("results/pipeline_comparison.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[green]Results saved to {output_path}[/green]")


if __name__ == "__main__":
    main()
