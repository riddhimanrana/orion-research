#!/usr/bin/env python3
"""
Orion SLAM Dashboard - Phase 1 Visual Style
============================================

Clean SLAM visualization using the exact same high-quality rendering
from Phase 1 tests:
- TURBO depth colormap
- 3D axes at entity centroids
- Arrowed lines for motion indicators
- Side-by-side RGB + Depth view
- Clean labels and annotations

Frame Skip: Every 3 frames for better SLAM tracking

Usage:
    python scripts/run_slam_phase1_style.py --video data/examples/video.mp4
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import colorsys
from typing import Dict, List, Optional, Tuple
import time
from collections import deque

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.slam.slam_engine import OpenCVSLAM, SLAMConfig
from orion.slam.semantic_slam import SemanticSLAM
from orion.managers.model_manager import ModelManager
from orion.perception.depth import DepthEstimator


def create_depth_colormap(depth_map: np.ndarray) -> np.ndarray:
    """Convert depth map to colored visualization (Phase 1 style)."""
    # Normalize to 0-255
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
    
    # Apply TURBO colormap for better depth perception
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
    
    return depth_colored


def draw_3d_axes(frame: np.ndarray, origin_2d: Tuple[int, int], scale: float = 40):
    """Draw 3D coordinate axes at a point (Phase 1 style)."""
    ox, oy = int(origin_2d[0]), int(origin_2d[1])
    
    # X-axis (red) - right
    cv2.arrowedLine(frame, (ox, oy), (ox + int(scale), oy), 
                    (0, 0, 255), 2, tipLength=0.3)
    cv2.putText(frame, "X", (ox + int(scale) + 5, oy), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Y-axis (green) - down
    cv2.arrowedLine(frame, (ox, oy), (ox, oy + int(scale)), 
                    (0, 255, 0), 2, tipLength=0.3)
    cv2.putText(frame, "Y", (ox, oy + int(scale) + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Z-axis (blue) - depth (represented diagonally)
    cv2.arrowedLine(frame, (ox, oy), (ox - int(scale*0.7), oy - int(scale*0.7)), 
                    (255, 0, 0), 2, tipLength=0.3)
    cv2.putText(frame, "Z", (ox - int(scale*0.7) - 20, oy - int(scale*0.7)), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


class SLAMVisualizer:
    """SLAM Dashboard with Phase 1 visual quality"""
    
    def __init__(self, video_path: str, skip_frames: int = 3):
        self.video_path = video_path
        self.skip_frames = skip_frames
        
        print("="*80)
        print("ORION SLAM - Phase 1 Visual Style")
        print("="*80)
        
        # Video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")
        
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nVideo: {self.frame_width}x{self.frame_height} @ {self.video_fps:.1f} FPS")
        print(f"Skip: Every {skip_frames} frames → ~{self.video_fps/skip_frames:.1f} FPS")
        print(f"Total: {self.total_frames} frames ({self.total_frames/self.video_fps:.1f}s)\n")
        
        # Models
        print("Loading models...")
        self.model_manager = ModelManager.get_instance()
        self.yolo_model = self.model_manager.yolo
        self.depth_estimator = DepthEstimator(model_name="midas", device="mps")
        
        # SLAM (relaxed parameters)
        slam_config = SLAMConfig(
            num_features=2000,
            match_ratio_test=0.75,
            min_matches=6
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
        self.prev_entities = {}
        
        print("\n✓ Ready!")
        print("="*80)
        print("Controls: Space=Pause | Q=Quit | A=Toggle Axes")
        print("="*80 + "\n")
        
        self.show_axes = True
    
    def _get_entity_color(self, entity_id: int) -> Tuple[int, int, int]:
        """Phase 1 style: Bright, distinct colors"""
        if entity_id not in self.entity_colors:
            # Use Phase 1 approach: green for visible
            hue = (entity_id * 137.508) % 360
            rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.9, 1.0)
            self.entity_colors[entity_id] = (
                int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)  # BGR
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
    
    def _visualize_entities(self, frame: np.ndarray, entities: List[Dict], 
                           show_axes: bool = True) -> np.ndarray:
        """Phase 1 style entity visualization"""
        vis = frame.copy()
        
        for entity in entities:
            bbox = entity['bbox']
            x1, y1, x2, y2 = bbox
            color = self._get_entity_color(entity['id'])
            
            # Draw bbox (2px for crisp lines)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Class label
            label = f"{entity['class']} {entity['confidence']:.2f}"
            cv2.putText(vis, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Centroid marker
            cx, cy = entity['centroid']
            cv2.circle(vis, (cx, cy), 4, color, -1)
            
            # 3D coordinates (Phase 1 style)
            if 'distance' in entity:
                X, Y, Z = 0, 0, entity['distance'] * 1000  # Convert to mm
                coord_text = f"3D: ({X:.0f}, {Y:.0f}, {Z:.0f})mm"
                cv2.putText(vis, coord_text, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Distance in meters
                depth_text = f"Distance: {entity['distance']:.2f}m"
                cv2.putText(vis, depth_text, (x1, y2 + 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Motion indicator (Phase 1 style arrows)
                if entity['id'] in self.prev_entities:
                    prev_dist = self.prev_entities[entity['id']].get('distance', entity['distance'])
                    dist_change = entity['distance'] - prev_dist
                    
                    if abs(dist_change) > 0.05:  # 5cm threshold
                        if dist_change < 0:
                            motion_text = "↗ APPROACHING"
                            motion_color = (0, 255, 255)  # Yellow
                        else:
                            motion_text = "↘ RECEDING"
                            motion_color = (255, 0, 255)  # Magenta
                        cv2.putText(vis, motion_text, (x1, y2 + 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, motion_color, 1)
            
            # 3D axes at centroid (Phase 1 style)
            if show_axes:
                draw_3d_axes(vis, (cx, cy), scale=30)
        
        return vis
    
    def run(self):
        """Main processing loop"""
        
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
                    'confidence': float(det[4]),
                    'distance': distance_m
                }
                
                current_entities.append(entity)
                self.entities[entity_id] = entity
            
            # === Visualization (Phase 1 Style) ===
            
            # Annotated frame
            vis_frame = self._visualize_entities(frame, current_entities, self.show_axes)
            
            # SLAM status overlay
            status = "TRACKING" if slam_pose is not None else "LOST"
            status_color = (0, 255, 0) if slam_pose is not None else (0, 0, 255)
            cv2.putText(vis_frame, f"SLAM: {status}", (15, vis_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Depth colormap (Phase 1 TURBO style)
            if depth_map is not None:
                depth_colored = create_depth_colormap(depth_map)
                
                # Resize to match frame
                if depth_colored.shape[:2] != vis_frame.shape[:2]:
                    depth_colored = cv2.resize(depth_colored, 
                                              (vis_frame.shape[1], vis_frame.shape[0]))
                
                # Side-by-side (Phase 1 layout)
                combined = np.hstack([vis_frame, depth_colored])
                
                # Labels (Phase 1 style)
                cv2.putText(combined, "RGB + 3D Annotations", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(combined, "Depth Map (Turbo)", 
                           (vis_frame.shape[1] + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Frame info (bottom left of depth)
                info_x = vis_frame.shape[1] + 10
                info_y = combined.shape[0] - 60
                cv2.putText(combined, f"Frame: {self.frame_count}/{self.total_frames}", 
                           (info_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(combined, f"Entities: {len(current_entities)}", 
                           (info_x, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(combined, f"SLAM Poses: {stats.get('total_poses', 0)}", 
                           (info_x, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                combined = vis_frame
            
            # Update metrics
            frame_time = time.time() - frame_start
            self.processed_frames += 1
            self.fps_history.append(1.0 / frame_time if frame_time > 0 else 0)
            
            # Store for motion detection
            self.prev_entities = {e['id']: e for e in current_entities}
            
            # Show
            cv2.imshow('Orion SLAM - Phase 1 Style', combined)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = True
            elif key == ord('a'):
                self.show_axes = not self.show_axes
                print(f"3D Axes: {'ON' if self.show_axes else 'OFF'}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Final stats
        final_stats = self.slam.get_statistics()
        print("\n" + "="*80)
        print("SESSION COMPLETE")
        print("="*80)
        print(f"Frames: {self.frame_count}/{self.total_frames}")
        print(f"Processed: {self.processed_frames}")
        print(f"Avg FPS: {np.mean(self.fps_history):.2f}")
        print(f"Entities tracked: {self.next_entity_id}")
        print(f"SLAM poses: {final_stats.get('total_poses', 0)}")
        print(f"Semantic rescues: {final_stats.get('landmark_only', 0)}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='SLAM with Phase 1 Visuals')
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--skip', type=int, default=3)
    
    args = parser.parse_args()
    
    viz = SLAMVisualizer(args.video, skip_frames=args.skip)
    viz.run()


if __name__ == '__main__':
    main()
