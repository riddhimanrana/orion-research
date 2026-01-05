#!/usr/bin/env python3
"""
Compare YOLO-World detection approaches:
1. Fixed vocabulary (v5) - Our optimized 124 classes
2. Open vocabulary (no set_classes) - Default YOLO-World behavior
3. Prompt-based - Using DetectionConfig.yoloworld_prompt

Goal: Understand trade-offs between approaches
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


def test_fixed_vocab(video_path: str, num_frames: int = 10, confidence: float = 0.25):
    """Test with our optimized v5 vocabulary (124 classes)."""
    from orion.backends.yoloworld_backend import YOLOWorldDetector, YOLOWorldConfig, DEFAULT_CLASSES
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = YOLOWorldConfig(
        model="yolov8x-worldv2",
        classes=DEFAULT_CLASSES.copy(),
        confidence=confidence,
        device=device,
    )
    detector = YOLOWorldDetector(config)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)
    
    all_detections = []
    class_counts = defaultdict(int)
    
    for i in range(num_frames):
        frame_idx = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        detections = detector.detect(frame)
        for det in detections:
            all_detections.append(det)
            class_counts[det['label']] += 1
    
    cap.release()
    return {
        "approach": "Fixed Vocab (v5)",
        "classes": len(DEFAULT_CLASSES),
        "total_detections": len(all_detections),
        "unique_classes": len(class_counts),
        "avg_per_frame": len(all_detections) / num_frames,
        "top_classes": dict(sorted(class_counts.items(), key=lambda x: -x[1])[:15]),
    }


def test_open_vocab(video_path: str, num_frames: int = 10, confidence: float = 0.25):
    """Test with open vocabulary (no set_classes - uses COCO as base)."""
    from ultralytics import YOLOWorld
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model WITHOUT calling set_classes
    model = YOLOWorld("yolov8x-worldv2.pt")
    model.to(device)
    # Note: NOT calling set_classes - using default vocabulary
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)
    
    all_detections = []
    class_counts = defaultdict(int)
    
    for i in range(num_frames):
        frame_idx = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.predict(
            source=frame,
            conf=confidence,
            device=device,
            verbose=False
        )
        
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                label = result.names[class_id]
                all_detections.append({"label": label})
                class_counts[label] += 1
    
    cap.release()
    return {
        "approach": "Open Vocab (COCO base)",
        "classes": "80 (COCO)",
        "total_detections": len(all_detections),
        "unique_classes": len(class_counts),
        "avg_per_frame": len(all_detections) / num_frames,
        "top_classes": dict(sorted(class_counts.items(), key=lambda x: -x[1])[:15]),
    }


def test_prompt_vocab(video_path: str, num_frames: int = 10, confidence: float = 0.25):
    """Test with DetectionConfig prompt-based vocabulary."""
    from orion.perception.config import DetectionConfig
    from ultralytics import YOLOWorld
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get prompt-based classes from DetectionConfig
    det_config = DetectionConfig(backend="yoloworld")
    prompt_classes = det_config.yoloworld_categories()
    
    # Load and configure model
    model = YOLOWorld("yolov8x-worldv2.pt")
    model.to(device)
    model.set_classes(prompt_classes)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)
    
    all_detections = []
    class_counts = defaultdict(int)
    
    for i in range(num_frames):
        frame_idx = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.predict(
            source=frame,
            conf=confidence,
            device=device,
            verbose=False
        )
        
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                label = result.names[class_id]
                all_detections.append({"label": label})
                class_counts[label] += 1
    
    cap.release()
    return {
        "approach": "Prompt Vocab (DetectionConfig)",
        "classes": len(prompt_classes),
        "total_detections": len(all_detections),
        "unique_classes": len(class_counts),
        "avg_per_frame": len(all_detections) / num_frames,
        "top_classes": dict(sorted(class_counts.items(), key=lambda x: -x[1])[:15]),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare YOLO-World vocabulary approaches")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to sample")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    console.print(Panel(
        f"[bold]YOLO-World Vocabulary Comparison[/bold]\n\n"
        f"Video: {args.video}\n"
        f"Frames: {args.frames}\n"
        f"Confidence: {args.confidence}",
        title="Configuration"
    ))

    results = []

    # Test 1: Fixed vocabulary (v5)
    console.print("\n[bold cyan]Testing Fixed Vocabulary (v5)...[/bold cyan]")
    start = time.time()
    r1 = test_fixed_vocab(args.video, args.frames, args.confidence)
    r1["time"] = time.time() - start
    results.append(r1)
    console.print(f"  ✓ {r1['total_detections']} detections, {r1['unique_classes']} classes in {r1['time']:.1f}s")

    # Test 2: Open vocabulary (no set_classes)
    console.print("\n[bold cyan]Testing Open Vocabulary (COCO base)...[/bold cyan]")
    start = time.time()
    r2 = test_open_vocab(args.video, args.frames, args.confidence)
    r2["time"] = time.time() - start
    results.append(r2)
    console.print(f"  ✓ {r2['total_detections']} detections, {r2['unique_classes']} classes in {r2['time']:.1f}s")

    # Test 3: Prompt-based vocabulary
    console.print("\n[bold cyan]Testing Prompt Vocabulary (DetectionConfig)...[/bold cyan]")
    start = time.time()
    r3 = test_prompt_vocab(args.video, args.frames, args.confidence)
    r3["time"] = time.time() - start
    results.append(r3)
    console.print(f"  ✓ {r3['total_detections']} detections, {r3['unique_classes']} classes in {r3['time']:.1f}s")

    # Summary table
    console.print("\n")
    table = Table(title="Vocabulary Comparison Results")
    table.add_column("Approach", style="cyan")
    table.add_column("Classes", justify="right")
    table.add_column("Detections", justify="right")
    table.add_column("Unique", justify="right")
    table.add_column("Avg/Frame", justify="right")
    table.add_column("Time", justify="right")

    for r in results:
        table.add_row(
            r["approach"],
            str(r["classes"]),
            str(r["total_detections"]),
            str(r["unique_classes"]),
            f"{r['avg_per_frame']:.1f}",
            f"{r['time']:.1f}s"
        )

    console.print(table)

    # Print top classes for each
    console.print("\n[bold]Top Classes by Approach:[/bold]")
    for r in results:
        console.print(f"\n[cyan]{r['approach']}:[/cyan]")
        for i, (label, count) in enumerate(list(r["top_classes"].items())[:10]):
            console.print(f"  {i+1}. {label}: {count}")

    # Save results
    output_path = Path("results/vocab_comparison.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[green]Results saved to {output_path}[/green]")


if __name__ == "__main__":
    main()
