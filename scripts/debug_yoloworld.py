#!/usr/bin/env python3
"""Debug YOLO-World detection to find why 0 detections."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from rich.console import Console
console = Console()

def main():
    # Load video
    video_path = "data/examples/test.mp4"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    console.print(f"[cyan]Frame shape: {frame.shape}[/cyan]")
    
    # Test 1: Load YOLO-World directly via ultralytics
    console.print("\n[yellow]Test 1: Direct Ultralytics YOLO-World[/yellow]")
    from ultralytics import YOLOWorld
    
    model = YOLOWorld("yolov8x-worldv2.pt")
    console.print(f"  Model loaded on CPU")
    
    # Try with just a few classes first
    simple_classes = ["person", "chair", "table", "couch"]
    console.print(f"  Setting {len(simple_classes)} simple classes...")
    model.set_classes(simple_classes)
    console.print(f"  Classes set!")
    
    # Move to GPU
    model.to("cuda")
    console.print(f"  Moved to CUDA")
    
    # Detect
    results = model.predict(
        source=frame,
        conf=0.25,
        iou=0.45,
        device="cuda",
        verbose=False
    )
    
    console.print(f"  Results: {len(results)}")
    for r in results:
        if r.boxes is not None:
            console.print(f"  Boxes: {len(r.boxes)}")
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = simple_classes[cls_id] if cls_id < len(simple_classes) else f"class_{cls_id}"
                console.print(f"    - {label}: {conf:.2f}")
        else:
            console.print(f"  Boxes: None")
    
    # Test 2: Now test with our backend
    console.print("\n[yellow]Test 2: Our YOLOWorldDetector backend[/yellow]")
    from orion.backends.yoloworld_backend import YOLOWorldDetector, YOLOWorldConfig, DEFAULT_CLASSES
    
    console.print(f"  DEFAULT_CLASSES: {len(DEFAULT_CLASSES)} classes")
    console.print(f"  First 10: {DEFAULT_CLASSES[:10]}")
    
    config = YOLOWorldConfig(device="cuda", confidence=0.25)
    detector = YOLOWorldDetector(config)
    
    console.print(f"  Running detect...")
    detections = detector.detect(frame)
    console.print(f"  Detections: {len(detections)}")
    
    for det in detections[:10]:
        console.print(f"    - {det['label']}: {det['confidence']:.2f}")
    
    console.print("\n[green]Debug complete![/green]")

if __name__ == "__main__":
    main()
