#!/usr/bin/env python3
"""
YOLO-World Detection Diagnostic Script

Tests different YOLO-World configurations to understand detection behavior:
1. Open vocabulary (COCO 80 classes - default)
2. Custom vocabulary with different sizes
3. Background class trick (adding "" to vocab)
4. Different model sizes (S, M, L, X)

Based on Ultralytics YOLO-World documentation insights.
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class DetectionResult:
    """Single detection result."""
    label: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)


@dataclass 
class FrameResult:
    """Results for a single frame."""
    frame_id: int
    config_name: str
    detections: list[DetectionResult] = field(default_factory=list)
    inference_time_ms: float = 0.0


def load_frame(video_path: str, frame_id: int, rotate: bool = True) -> np.ndarray:
    """Load a specific frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Failed to read frame {frame_id}")
    
    # Check if portrait video (width > height after rotation)
    if rotate and frame.shape[0] > frame.shape[1]:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return frame


def get_video_info(video_path: str) -> dict:
    """Get video metadata."""
    cap = cv2.VideoCapture(video_path)
    info = {
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def test_yoloworld_config(
    model,
    frame: np.ndarray,
    vocab: Optional[list[str]] = None,
    conf_threshold: float = 0.15,
) -> tuple[list[DetectionResult], float]:
    """
    Test YOLO-World with specific configuration.
    
    Args:
        model: YOLO model instance
        frame: Image frame
        vocab: Custom vocabulary (None = use default COCO)
        conf_threshold: Confidence threshold
        
    Returns:
        List of detections and inference time in ms
    """
    if vocab is not None:
        model.set_classes(vocab)
    
    start = time.perf_counter()
    results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
    inference_time = (time.perf_counter() - start) * 1000
    
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()
        
        # Get label based on vocab or model names
        if vocab is not None:
            label = vocab[cls_id] if cls_id < len(vocab) else f"class_{cls_id}"
        else:
            label = results.names[cls_id]
        
        detections.append(DetectionResult(
            label=label,
            confidence=conf,
            bbox=tuple(xyxy)
        ))
    
    return detections, inference_time


def run_diagnostics(
    video_path: str,
    frame_ids: list[int],
    model_sizes: list[str] = ['m'],
    conf_threshold: float = 0.15,
    output_path: Optional[str] = None,
):
    """
    Run comprehensive YOLO-World diagnostics.
    
    Tests:
    1. Open vocab (COCO 80 - default)
    2. Office-focused vocab (10 classes)
    3. Office vocab + background class
    4. Minimal vocab (5 classes)
    """
    from ultralytics import YOLO
    
    # Define vocabulary configurations
    vocab_configs = {
        'coco_default': None,  # Use default COCO 80 classes
        
        'office_10': [
            'person', 'chair', 'desk', 'monitor', 'laptop',
            'keyboard', 'mouse', 'bottle', 'cup', 'book'
        ],
        
        'office_10_bg': [
            'person', 'chair', 'desk', 'monitor', 'laptop',
            'keyboard', 'mouse', 'bottle', 'cup', 'book',
            ''  # Background class trick from docs
        ],
        
        'office_5': ['person', 'chair', 'desk', 'monitor', 'laptop'],
        
        'gemini_detected': [
            # Objects Gemini reported seeing in test.mp4
            'monitor', 'desk', 'chair', 'keyboard', 'laptop',
            'picture frame', 'water bottle', 'mouse', 'person'
        ],
        
        'gemini_detected_bg': [
            'monitor', 'desk', 'chair', 'keyboard', 'laptop',
            'picture frame', 'water bottle', 'mouse', 'person',
            ''  # Background class
        ],
    }
    
    # Get video info
    video_info = get_video_info(video_path)
    print(f"\n{'='*70}")
    print(f"YOLO-World Detection Diagnostics")
    print(f"{'='*70}")
    print(f"Video: {video_path}")
    print(f"  Frames: {video_info['total_frames']}, FPS: {video_info['fps']:.1f}")
    print(f"  Resolution: {video_info['width']}x{video_info['height']}")
    print(f"Testing frames: {frame_ids}")
    print(f"Model sizes: {model_sizes}")
    print(f"Confidence threshold: {conf_threshold}")
    
    all_results = []
    
    for model_size in model_sizes:
        model_name = f"yolov8{model_size}-worldv2.pt"
        print(f"\n{'='*70}")
        print(f"Loading {model_name}...")
        
        for frame_id in frame_ids:
            print(f"\n--- Frame {frame_id} ---")
            frame = load_frame(video_path, frame_id)
            
            for config_name, vocab in vocab_configs.items():
                # Need to reload model for each vocab change to avoid device issues
                model = YOLO(model_name)
                
                try:
                    detections, inference_time = test_yoloworld_config(
                        model, frame, vocab, conf_threshold
                    )
                    
                    result = FrameResult(
                        frame_id=frame_id,
                        config_name=f"{model_size}_{config_name}",
                        detections=detections,
                        inference_time_ms=inference_time
                    )
                    all_results.append(result)
                    
                    # Print summary
                    det_summary = [f"{d.label}:{d.confidence:.2f}" for d in detections[:5]]
                    vocab_size = len(vocab) if vocab else 80
                    print(f"  {config_name} ({vocab_size} classes): {det_summary if det_summary else 'No detections'}")
                    
                except Exception as e:
                    print(f"  {config_name}: ERROR - {e}")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    # Group by config
    config_stats = {}
    for result in all_results:
        config = result.config_name
        if config not in config_stats:
            config_stats[config] = {'total_detections': 0, 'frames': 0, 'inference_ms': []}
        config_stats[config]['total_detections'] += len(result.detections)
        config_stats[config]['frames'] += 1
        config_stats[config]['inference_ms'].append(result.inference_time_ms)
    
    print(f"\n{'Config':<30} {'Detections':>12} {'Avg/Frame':>10} {'Avg Time':>10}")
    print("-" * 70)
    for config, stats in sorted(config_stats.items()):
        avg_det = stats['total_detections'] / stats['frames']
        avg_time = np.mean(stats['inference_ms'])
        print(f"{config:<30} {stats['total_detections']:>12} {avg_det:>10.1f} {avg_time:>9.1f}ms")
    
    # Save results
    if output_path:
        output_data = {
            'video': video_path,
            'video_info': video_info,
            'frame_ids': frame_ids,
            'conf_threshold': conf_threshold,
            'results': [
                {
                    'frame_id': r.frame_id,
                    'config': r.config_name,
                    'detections': [
                        {'label': d.label, 'conf': d.confidence, 'bbox': list(d.bbox)}
                        for d in r.detections
                    ],
                    'inference_ms': r.inference_time_ms
                }
                for r in all_results
            ]
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="YOLO-World Detection Diagnostics")
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--frames', type=str, default='10,50,100,200',
                        help='Comma-separated frame IDs to test')
    parser.add_argument('--models', type=str, default='m,l',
                        help='Model sizes to test (s,m,l,x)')
    parser.add_argument('--conf', type=float, default=0.15,
                        help='Confidence threshold')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path for results')
    
    args = parser.parse_args()
    
    frame_ids = [int(f.strip()) for f in args.frames.split(',')]
    model_sizes = [s.strip() for s in args.models.split(',')]
    
    run_diagnostics(
        video_path=args.video,
        frame_ids=frame_ids,
        model_sizes=model_sizes,
        conf_threshold=args.conf,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
