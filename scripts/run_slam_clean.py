#!/usr/bin/env python3
"""
Orion SLAM Dashboard - Clean OpenCV UI
=======================================

Simple, clean visualization with proper aspect ratio handling.

Key improvements:
1. Proper aspect ratio (no distortion for vertical videos)
2. Smaller frame skip (every 3 frames) for better SLAM tracking  
3. Clean bounding boxes and labels
4. Simple metrics panel
5. Lightweight (no heavy imports)

Usage:
    python scripts/run_slam_clean.py --video data/examples/video.mp4
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import colorsys
from typing import Dict, List, Optional, Tuple, Any
import time
from collections import deque
import psutil
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.slam.slam_engine import OpenCVSLAM, SLAMConfig
from orion.slam.semantic_slam import SemanticSLAM
from orion.managers.model_manager import ModelManager
from orion.perception.depth import DepthEstimator


class CleanSLAMDashboard:
    """Clean, simple SLAM dashboard with proper aspect ratio"""
    
    # Modern color palette
    BG_DARK = (25, 25, 30)
    BG_PANEL = (40, 40, 50)
    ACCENT = (100, 200, 255)
    TEXT = (220, 220, 230)
    SUCCESS = (100, 255, 150)
    WARNING = (255, 200, 100)
    ERROR = (255, 100, 100)
    
    def __init__(self, video_path: str, skip_frames: int = 3):
        self.video_path = video_path
        self.skip_frames = skip_frames
        
        print("="*70)
        print(" ORION SLAM - CLEAN DASHBOARD")
        print("="*70)
        
        # Video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")
        
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nVideo: {self.frame_width}x{self.frame_height} @ {self.video_fps:.1f} FPS")
        print(f"Skip: Every {skip_frames} frames → ~{self.video_fps/skip_frames:.1f} FPS processing")
        print(f"Total: {self.total_frames} frames\n")
        
        # Models
        print("[1/3] Loading YOLO...")
        self.model_manager = ModelManager.get_instance()
        self.yolo_model = self.model_manager.yolo
        
        print("[2/3] Loading depth estimator...")
        self.depth_estimator = DepthEstimator(model_name="midas", device="mps")
        
        print("[3/3] Initializing SLAM...")
        slam_config = SLAMConfig(
            num_features=2000,
            match_ratio_test=0.75,  # More lenient
            min_matches=6  # Lower threshold
        )
        base_slam = OpenCVSLAM(config=slam_config)
        self.slam = SemanticSLAM(
            base_slam=base_slam,
            use_landmarks=True,
            landmark_weight=0.4
        )
        
        # State
        self.entities = {}
        self.next_entity_id = 0
        self.entity_colors = {}
        self.frame_count = 0
        self.processed_frames = 0
        self.fps_history = deque(maxlen=30)
        self.paused = False
        
        # Calculate display size (maintain aspect ratio, max 1200px height)
        max_display_height = 900
        if self.frame_height > max_display_height:
            scale = max_display_height / self.frame_height
            self.display_width = int(self.frame_width * scale)
            self.display_height = max_display_height
        else:
            self.display_width = self.frame_width
            self.display_height = self.frame_height
        
        print(f"\n✓ Ready! Display: {self.display_width}x{self.display_height}")
        print("="*70)
        print("Controls: Space=Pause | Q=Quit | S=Screenshot")
        print("="*70 + "\n")
    
    def _get_entity_color(self, entity_id: int) -> Tuple[int, int, int]:
        """Unique color per entity"""
        if entity_id not in self.entity_colors:
            hue = (entity_id * 137.508) % 360
            rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.9, 1.0)
            self.entity_colors[entity_id] = (
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
        return self.entity_colors[entity_id]
    
    def _assign_entity_id(self, bbox: np.ndarray, class_name: str) -> int:
        """Simple spatial matching"""
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        
        for eid, entity in self.entities.items():
            if entity['class'] == class_name:
                ex, ey = entity['centroid']
                dist = np.sqrt((cx - ex)**2 + (cy - ey)**2)
                if dist < 100:
                    return eid
        
        new_id = self.next_entity_id
        self.next_entity_id += 1
        return new_id
    
    def _draw_entity(self, frame: np.ndarray, entity: Dict):
        """Draw single entity with clean styling"""
        bbox = entity['bbox']
        x1, y1, x2, y2 = bbox
        color = self._get_entity_color(entity['id'])
        
        # Bounding box (2px, clean)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label
        label = f"ID{entity['id']}: {entity['class']} ({entity['distance']:.1f}m)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thick = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
        
        # Label background
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        
        # Label text
        cv2.putText(frame, label, (x1 + 2, y1 - 4), font, font_scale, 
                   (255, 255, 255), font_thick, cv2.LINE_AA)
        
        # Centroid
        cx, cy = entity['centroid']
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 3, color, -1)
    
    def _create_metrics_panel(self, width: int, height: int, metrics: Dict) -> np.ndarray:
        """Create clean metrics sidebar"""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = self.BG_PANEL
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 30
        line_height = 25
        
        def draw_text(text, color=self.TEXT, bold=False):
            nonlocal y
            thickness = 2 if bold else 1
            scale = 0.6 if bold else 0.5
            cv2.putText(panel, text, (15, y), font, scale, color, thickness, cv2.LINE_AA)
            y += line_height
        
        # Header
        draw_text("METRICS", self.ACCENT, True)
        y += 10
        
        # System
        draw_text("System", self.ACCENT)
        draw_text(f"  CPU: {metrics.get('cpu', 0):.1f}%")
        draw_text(f"  Mem: {metrics.get('mem_gb', 0):.1f} GB")
        draw_text(f"  FPS: {metrics.get('fps', 0):.2f}")
        y += 10
        
        # SLAM
        status = metrics.get('slam_status', 'UNKNOWN')
        status_color = self.SUCCESS if status == 'TRACKING' else self.ERROR
        draw_text("SLAM", self.ACCENT)
        draw_text(f"  {status}", status_color)
        draw_text(f"  Poses: {metrics.get('poses', 0)}")
        draw_text(f"  Rescues: {metrics.get('rescues', 0)}")
        y += 10
        
        # Entities
        draw_text("Entities", self.ACCENT)
        draw_text(f"  Active: {metrics.get('entities', 0)}")
        draw_text(f"  Total: {metrics.get('total_entities', 0)}")
        y += 10
        
        # Progress
        progress = metrics.get('progress', 0)
        draw_text(f"Progress: {progress:.1f}%", self.WARNING)
        draw_text(f"Frame: {metrics.get('frame', 0)}/{metrics.get('total', 0)}")
        
        return panel
    
    def run(self):
        """Main loop"""
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Skip frames
            if self.frame_count % self.skip_frames != 0:
                continue
            
            if self.paused:
                key = cv2.waitKey(100) & 0xFF
                if key == ord(' '):
                    self.paused = False
                elif key == ord('q'):
                    break
                continue
            
            frame_start = time.time()
            
            # === Processing ===
            
            # YOLO
            yolo_results = self.yolo_model(frame, conf=0.35, verbose=False)[0]
            
            # Depth
            depth_map, _ = self.depth_estimator.estimate(frame)
            
            # SLAM
            object_detections = []
            for det in yolo_results.boxes.data:
                if det[4] > 0.35:
                    object_detections.append({
                        'bbox': det[:4].cpu().numpy(),
                        'class': yolo_results.names[int(det[5])],
                        'confidence': float(det[4])
                    })
            
            slam_pose = self.slam.track(frame, time.time(), self.frame_count, object_detections)
            stats = self.slam.get_statistics()
            
            # Build entities
            current_entities = []
            for det in yolo_results.boxes.data:
                if det[4] < 0.35:
                    continue
                
                class_id = int(det[5])
                class_name = yolo_results.names[class_id]
                bbox = det[:4].cpu().numpy().astype(int)
                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                distance_mm = depth_map[cy, cx] if depth_map is not None else 0
                distance_m = distance_mm / 1000.0
                
                entity_id = self._assign_entity_id(bbox, class_name)
                
                entity = {
                    'id': entity_id,
                    'class': class_name,
                    'bbox': bbox,
                    'centroid': (cx, cy),
                    'distance': distance_m
                }
                
                current_entities.append(entity)
                self.entities[entity_id] = entity
            
            # === Visualization ===
            
            vis = frame.copy()
            
            # Draw entities
            for entity in current_entities:
                self._draw_entity(vis, entity)
            
            # SLAM status overlay
            status = "TRACKING" if slam_pose is not None else "LOST"
            status_color = self.SUCCESS if slam_pose is not None else self.ERROR
            cv2.putText(vis, f"SLAM: {status}", (15, vis.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
            
            # Metrics
            frame_time = time.time() - frame_start
            self.processed_frames += 1
            self.fps_history.append(1.0 / frame_time if frame_time > 0 else 0)
            
            metrics = {
                'cpu': psutil.cpu_percent(interval=0),
                'mem_gb': psutil.virtual_memory().used / (1024**3),
                'fps': np.mean(self.fps_history) if self.fps_history else 0,
                'slam_status': status,
                'poses': stats.get('total_poses', 0),
                'rescues': stats.get('landmark_only', 0),
                'entities': len(current_entities),
                'total_entities': self.next_entity_id,
                'progress': (self.frame_count / self.total_frames) * 100,
                'frame': self.frame_count,
                'total': self.total_frames
            }
            
            # Resize for display (maintain aspect ratio)
            if vis.shape[1] != self.display_width or vis.shape[0] != self.display_height:
                vis = cv2.resize(vis, (self.display_width, self.display_height))
            
            # Create metrics panel
            metrics_panel = self._create_metrics_panel(300, self.display_height, metrics)
            
            # Combine
            combined = np.hstack([vis, metrics_panel])
            
            # Show
            cv2.imshow('Orion SLAM Dashboard', combined)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = True
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"slam_screenshot_{timestamp}.png"
                cv2.imwrite(filename, combined)
                print(f"Screenshot: {filename}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Final stats
        final_stats = self.slam.get_statistics()
        print("\n" + "="*70)
        print("SESSION COMPLETE")
        print("="*70)
        print(f"Total frames: {self.frame_count}")
        print(f"Processed: {self.processed_frames}")
        print(f"Avg FPS: {np.mean(self.fps_history):.2f}")
        print(f"Entities: {self.next_entity_id}")
        print(f"SLAM poses: {final_stats.get('total_poses', 0)}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Clean SLAM Dashboard')
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--skip', type=int, default=3, 
                       help='Process every Nth frame (default: 3)')
    
    args = parser.parse_args()
    
    dashboard = CleanSLAMDashboard(args.video, skip_frames=args.skip)
    dashboard.run()


if __name__ == '__main__':
    main()
