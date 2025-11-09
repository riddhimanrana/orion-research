"""
Performance Profiling Script for Orion Pipeline

Measures per-component timing to identify bottlenecks and guide optimization.

Usage:
    python scripts/profile_performance.py --video data/examples/video.mp4 --frames 50

Author: Orion Research Team
Date: November 2025
"""

import argparse
import cv2
import numpy as np
import time
from pathlib import Path
import sys
from typing import Dict, List
from dataclasses import dataclass, field
from contextlib import contextmanager

# Add orion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.managers.model_manager import ModelManager
from orion.perception.perception_3d import Perception3DEngine
from orion.slam.slam_engine import SLAMEngine, SLAMConfig


@dataclass
class PerformanceMetrics:
    """Stores timing metrics for each component"""
    component: str
    times_ms: List[float] = field(default_factory=list)
    
    def add(self, time_ms: float):
        self.times_ms.append(time_ms)
    
    def mean(self) -> float:
        return np.mean(self.times_ms) if self.times_ms else 0.0
    
    def std(self) -> float:
        return np.std(self.times_ms) if self.times_ms else 0.0
    
    def total(self) -> float:
        return sum(self.times_ms)


class PerformanceProfiler:
    """Profile Orion pipeline performance"""
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
    
    @contextmanager
    def timer(self, component: str):
        """Context manager for timing a component"""
        if component not in self.metrics:
            self.metrics[component] = PerformanceMetrics(component)
        
        start = time.time()
        yield
        elapsed_ms = (time.time() - start) * 1000
        self.metrics[component].add(elapsed_ms)
    
    def print_report(self):
        """Print performance report"""
        print("\n" + "="*80)
        print("PERFORMANCE PROFILE")
        print("="*80)
        
        # Calculate total time
        total_time = sum(m.total() for m in self.metrics.values())
        
        # Sort by total time
        sorted_metrics = sorted(
            self.metrics.values(),
            key=lambda m: m.total(),
            reverse=True
        )
        
        # Print header
        print(f"\n{'Component':<30} {'Mean (ms)':<12} {'Std (ms)':<12} {'% Total':<10} {'Calls':<8}")
        print("-" * 80)
        
        # Print each component
        for metric in sorted_metrics:
            mean = metric.mean()
            std = metric.std()
            pct = (metric.total() / total_time * 100) if total_time > 0 else 0
            calls = len(metric.times_ms)
            
            print(f"{metric.component:<30} {mean:>10.1f}   {std:>10.1f}   {pct:>8.1f}%   {calls:>6}")
        
        print("-" * 80)
        
        # Summary
        num_frames = len(self.metrics.get("Total Frame", PerformanceMetrics("")).times_ms)
        if num_frames > 0:
            avg_fps = 1000.0 / self.metrics["Total Frame"].mean() if "Total Frame" in self.metrics else 0
            print(f"\nFrames processed: {num_frames}")
            print(f"Average time per frame: {self.metrics['Total Frame'].mean():.1f}ms")
            print(f"Average FPS: {avg_fps:.2f}")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Profile Orion pipeline performance")
    parser.add_argument("--video", type=str, default="data/examples/video.mp4",
                       help="Path to input video")
    parser.add_argument("--frames", type=int, default=50,
                       help="Number of frames to profile")
    parser.add_argument("--use-slam", action="store_true",
                       help="Enable SLAM tracking")
    parser.add_argument("--use-depth", action="store_true",
                       help="Enable depth estimation")
    parser.add_argument("--depth-model", type=str, default="midas",
                       choices=["midas", "zoe"],
                       help="Depth model to use")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("ORION PERFORMANCE PROFILER")
    print("="*80)
    print(f"Video: {args.video}")
    print(f"Frames to process: {args.frames}")
    print(f"SLAM enabled: {args.use_slam}")
    print(f"Depth enabled: {args.use_depth}")
    if args.use_depth:
        print(f"Depth model: {args.depth_model}")
    print("="*80)
    
    # Initialize profiler
    profiler = PerformanceProfiler()
    
    # Load video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return 1
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    
    # Initialize models
    print("\nInitializing models...")
    
    with profiler.timer("Model Loading"):
        model_manager = ModelManager.get_instance()
        
        with profiler.timer("  YOLO Loading"):
            yolo = model_manager.yolo
        
        with profiler.timer("  CLIP Loading"):
            clip = model_manager.clip
        
        # Initialize depth estimator if requested
        depth_engine = None
        if args.use_depth:
            with profiler.timer("  Depth Model Loading"):
                depth_engine = Perception3DEngine(
                    enable_depth=True,
                    enable_hands=False,
                    enable_occlusion=False,
                    depth_model=args.depth_model
                )
        
        # Initialize SLAM if requested
        slam_engine = None
        if args.use_slam:
            with profiler.timer("  SLAM Initialization"):
                slam_config = SLAMConfig(
                    method="opencv",
                    num_features=3000,
                    match_ratio_test=0.85,
                    min_matches=8
                )
                slam_engine = SLAMEngine(config=slam_config)
    
    print("✓ Models loaded\n")
    
    # Process frames
    frame_count = 0
    
    print(f"Processing {args.frames} frames...")
    
    try:
        while frame_count < args.frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / fps
            
            with profiler.timer("Total Frame"):
                # YOLO detection
                with profiler.timer("  YOLO Detection"):
                    results = yolo(
                        frame,
                        conf=0.6,
                        verbose=False
                    )
                    
                    # Extract detections
                    detections = []
                    for result in results:
                        boxes = result.boxes
                        for i in range(len(boxes)):
                            bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                            conf = float(boxes.conf[i])
                            cls = int(boxes.cls[i])
                            class_name = result.names[cls]
                            
                            detections.append({
                                'bbox': bbox,
                                'class': class_name,
                                'confidence': conf
                            })
                
                # Depth estimation
                depth_map = None
                if depth_engine:
                    with profiler.timer("  Depth Estimation"):
                        yolo_dets = [{
                            'entity_id': f'det_{frame_count}_{i}',
                            'class': d['class'],
                            'confidence': d['confidence'],
                            'bbox': d['bbox']
                        } for i, d in enumerate(detections)]
                        
                        perception_3d = depth_engine.process_frame(
                            frame, yolo_dets, frame_count, timestamp
                        )
                        depth_map = perception_3d.depth_map
                
                # CLIP embedding (sample on a few crops)
                if detections:
                    with profiler.timer("  CLIP Embedding"):
                        # Take first 3 crops
                        for det in detections[:3]:
                            x1, y1, x2, y2 = det['bbox']
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0:
                                # Encode image
                                _ = clip.encode_image(crop)
                
                # SLAM tracking
                if slam_engine:
                    with profiler.timer("  SLAM Tracking"):
                        pose = slam_engine.process_frame(
                            frame, timestamp, frame_count
                        )
            
            frame_count += 1
            
            # Progress
            if frame_count % 10 == 0:
                avg_time = profiler.metrics["Total Frame"].mean()
                current_fps = 1000.0 / avg_time if avg_time > 0 else 0
                print(f"  Frame {frame_count}/{args.frames} | "
                      f"Avg time: {avg_time:.1f}ms | FPS: {current_fps:.2f}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        cap.release()
    
    # Print final report
    profiler.print_report()
    
    # Recommendations
    print("\n" + "="*80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*80)
    
    if "  Depth Estimation" in profiler.metrics:
        depth_time = profiler.metrics["  Depth Estimation"].mean()
        depth_pct = (profiler.metrics["  Depth Estimation"].total() / 
                     profiler.metrics["Total Frame"].total() * 100)
        
        if depth_pct > 40:
            print(f"\n🔴 CRITICAL: Depth estimation is {depth_pct:.1f}% of compute time")
            print(f"   Average: {depth_time:.1f}ms per frame")
            print("\n   Recommendations:")
            print("   1. Switch to FastDepth or MiDaS_small (3-4x faster)")
            print("   2. Reduce depth resolution (540x960 instead of 1080x1920)")
            print("   3. Use conditional depth (skip on good SLAM frames)")
    
    if "  YOLO Detection" in profiler.metrics:
        yolo_time = profiler.metrics["  YOLO Detection"].mean()
        yolo_pct = (profiler.metrics["  YOLO Detection"].total() / 
                   profiler.metrics["Total Frame"].total() * 100)
        
        if yolo_pct > 20:
            print(f"\n🟡 HIGH PRIORITY: YOLO is {yolo_pct:.1f}% of compute time")
            print(f"   Average: {yolo_time:.1f}ms per frame")
            print("\n   Recommendations:")
            print("   1. Switch to YOLO11s or YOLO11n (3-4x faster)")
            print("   2. Reduce input resolution")
            print("   3. Use TensorRT export (if NVIDIA GPU available)")
    
    if "  SLAM Tracking" in profiler.metrics:
        slam_time = profiler.metrics["  SLAM Tracking"].mean()
        slam_pct = (profiler.metrics["  SLAM Tracking"].total() / 
                   profiler.metrics["Total Frame"].total() * 100)
        
        if slam_pct > 15:
            print(f"\n🟡 MEDIUM PRIORITY: SLAM is {slam_pct:.1f}% of compute time")
            print(f"   Average: {slam_time:.1f}ms per frame")
            print("\n   Recommendations:")
            print("   1. Adaptive feature count (fewer when tracking is good)")
            print("   2. GPU-accelerated feature matching")
            print("   3. Skip SLAM every N frames (interpolate between)")
    
    # Overall FPS assessment
    if "Total Frame" in profiler.metrics:
        avg_time = profiler.metrics["Total Frame"].mean()
        avg_fps = 1000.0 / avg_time
        
        print(f"\n{'='*80}")
        print(f"TARGET: 1.0-1.5 FPS (667-1000ms per frame)")
        print(f"CURRENT: {avg_fps:.2f} FPS ({avg_time:.0f}ms per frame)")
        
        if avg_fps >= 1.0:
            print(f"✅ MEETS TARGET! System is fast enough for real-time use.")
        elif avg_fps >= 0.7:
            print(f"🟡 CLOSE! Need {1000/avg_fps - avg_time:.0f}ms improvement to reach 1.0 FPS")
            print(f"   Focus on top 2 bottlenecks above.")
        else:
            print(f"🔴 OPTIMIZATION NEEDED! {1000/avg_fps - avg_time:.0f}ms too slow.")
            print(f"   Must address all critical bottlenecks.")
    
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
