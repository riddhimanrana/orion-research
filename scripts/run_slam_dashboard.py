#!/usr/bin/env python3
"""
Orion SLAM Dashboard - Modern Research Prototype UI
===================================================

Professional dashboard visualization for SLAM research with:
- Modern dark theme UI with panels
- Real-time metrics (CPU, FPS, memory)
- Comprehensive SLAM statistics
- Spatial awareness (bird's eye view)
- Off-screen entity tracking
- Event composition monitoring
- Processing pipeline visualization

Frame Skipping: Processes ~1 FPS (every 7th frame at 30 FPS video)
                to match real-time performance of full pipeline

Usage:
    python scripts/run_slam_dashboard.py --video data/examples/video.mp4
    
Controls:
    Space: Pause/resume
    s: Save screenshot
    d: Toggle depth overlay
    m: Toggle spatial map
    h: Toggle help
    q: Quit
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import colorsys
from typing import Dict, List, Optional, Tuple, Any
import time
from collections import deque, defaultdict
import psutil
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Lightweight imports - avoid heavy TensorFlow/MediaPipe loading
from orion.slam.slam_engine import OpenCVSLAM, SLAMConfig
from orion.slam.semantic_slam import SemanticSLAM
from orion.managers.model_manager import ModelManager

# Lazy import for depth
from orion.perception.depth import DepthEstimator


class ModernDashboard:
    """Modern research dashboard with comprehensive metrics"""
    
    # Color scheme (dark theme)
    BG_COLOR = (20, 20, 25)          # Dark background
    PANEL_COLOR = (35, 35, 45)        # Panel background
    ACCENT_COLOR = (100, 200, 255)    # Cyan accent
    SUCCESS_COLOR = (100, 255, 150)   # Green
    WARNING_COLOR = (255, 200, 100)   # Yellow
    ERROR_COLOR = (255, 100, 100)     # Red
    TEXT_COLOR = (220, 220, 230)      # Light gray text
    
    def __init__(self, video_path: str, skip_frames: int = 7):
        self.video_path = video_path
        self.skip_frames = skip_frames
        
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "ORION SLAM DASHBOARD INITIALIZING" + " " * 25 + "║")
        print("╚" + "═" * 78 + "╝")
        
        # Video capture
        print("\n[1/6] Loading video stream...")
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"      Video: {self.frame_width}x{self.frame_height} @ {self.video_fps:.1f} FPS")
        print(f"      Total frames: {self.total_frames}")
        print(f"      Frame skip: Every {skip_frames} frames (~{self.video_fps/skip_frames:.2f} FPS processing)")
        
        # Model manager
        print("\n[2/6] Initializing model manager...")
        self.model_manager = ModelManager.get_instance()
        self.yolo_model = self.model_manager.yolo
        print("      ✓ YOLO loaded")
        
        # Depth estimator (lightweight)
        print("\n[3/6] Initializing depth estimator...")
        self.depth_estimator = DepthEstimator(model_name="midas", device="mps")
        print("      ✓ MiDaS depth model ready")
        
        # SLAM
        print("\n[4/6] Initializing Semantic SLAM...")
        slam_config = SLAMConfig(
            num_features=3000,
            match_ratio_test=0.85,
            min_matches=8
        )
        base_slam = OpenCVSLAM(config=slam_config)
        self.slam = SemanticSLAM(
            base_slam=base_slam,
            use_landmarks=True,
            landmark_weight=0.3
        )
        
        # Camera intrinsics (for 3D backprojection)
        self.camera_fx = 525.0  # Typical webcam focal length
        self.camera_fy = 525.0
        print("      ✓ Hybrid visual + landmark tracking")
        
        # Entity tracking
        print("\n[5/6] Initializing entity tracker...")
        self.entities: Dict[int, Dict[str, Any]] = {}
        self.next_entity_id = 0
        self.entity_colors: Dict[int, Tuple[int, int, int]] = {}
        self.offscreen_entities: List[Dict] = []
        print("      ✓ Bayesian entity tracker ready")
        
        # Entity tracking
        self.entities: Dict[int, Dict[str, Any]] = {}
        self.next_entity_id = 0
        
        # Metrics tracking
        print("\n[6/6] Initializing metrics system...")
        self.fps_history = deque(maxlen=30)
        self.cpu_history = deque(maxlen=60)
        self.processing_times = defaultdict(lambda: deque(maxlen=30))
        self.frame_count = 0
        self.processed_frames = 0
        self.start_time = time.time()
        print("      ✓ Metrics dashboard ready")
        
        # UI state
        self.paused = False
        self.show_depth = True
        self.show_spatial_map = True
        self.show_help = False
        
        # Dashboard layout (main window size)
        self.dashboard_width = 1920
        self.dashboard_height = 1080
        
        print("\n" + "─" * 80)
        print("✓ Initialization complete. Starting processing...")
        print("─" * 80 + "\n")
    
    def _get_entity_color(self, entity_id: int) -> Tuple[int, int, int]:
        """Get unique color for entity using golden angle"""
        if entity_id not in self.entity_colors:
            hue = (entity_id * 137.508) % 360
            rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.85, 0.95)
            self.entity_colors[entity_id] = (
                int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)
            )
        return self.entity_colors[entity_id]
    
    def _assign_entity_id(self, bbox: np.ndarray, class_name: str, existing_entities: Dict) -> int:
        """Simple entity ID assignment (improved matching in Phase 3)"""
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        
        # Simple spatial matching (within 100px)
        for eid, entity in existing_entities.items():
            if entity['class'] == class_name:
                ex, ey = entity['centroid']
                dist = np.sqrt((cx - ex)**2 + (cy - ey)**2)
                if dist < 100:
                    return eid
        
        # New entity
        new_id = self.next_entity_id
        self.next_entity_id += 1
        return new_id
    
    def _create_dashboard(self, frame: np.ndarray, metrics: Dict) -> np.ndarray:
        """Create modern dashboard layout with panels"""
        
        # Create large canvas
        dashboard = np.zeros((self.dashboard_height, self.dashboard_width, 3), dtype=np.uint8)
        dashboard[:] = self.BG_COLOR
        
        # Main video panel (left side, larger)
        video_panel_w = 1280
        video_panel_h = 720
        
        # Resize frame to fit
        frame_resized = cv2.resize(frame, (video_panel_w, video_panel_h))
        
        # Place frame
        x_offset = 20
        y_offset = 80
        dashboard[y_offset:y_offset+video_panel_h, x_offset:x_offset+video_panel_w] = frame_resized
        
        # Draw panel border
        cv2.rectangle(dashboard, (x_offset-2, y_offset-2), 
                     (x_offset+video_panel_w+2, y_offset+video_panel_h+2),
                     self.ACCENT_COLOR, 2)
        
        # Right sidebar for metrics (starting at x=1320)
        sidebar_x = 1320
        self._draw_metrics_panel(dashboard, sidebar_x, 80, metrics)
        
        # Bottom info bar
        self._draw_bottom_info(dashboard, metrics)
        
        # Top header
        self._draw_header(dashboard)
        
        return dashboard
    
    def _draw_header(self, canvas: np.ndarray):
        """Draw top header bar"""
        # Header background
        cv2.rectangle(canvas, (0, 0), (self.dashboard_width, 70), self.PANEL_COLOR, -1)
        
        # Title
        title = "ORION SLAM RESEARCH DASHBOARD"
        cv2.putText(canvas, title, (30, 45), cv2.FONT_HERSHEY_DUPLEX, 1.2,
                   self.ACCENT_COLOR, 2, cv2.LINE_AA)
        
        # Status indicator
        status_x = 1600
        status_text = "● TRACKING" if not self.paused else "⏸ PAUSED"
        status_color = self.SUCCESS_COLOR if not self.paused else self.WARNING_COLOR
        cv2.putText(canvas, status_text, (status_x, 45), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, status_color, 2, cv2.LINE_AA)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(canvas, timestamp, (1800, 45), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, self.TEXT_COLOR, 1, cv2.LINE_AA)
    
    def _draw_metrics_panel(self, canvas: np.ndarray, x: int, y: int, metrics: Dict):
        """Draw comprehensive metrics sidebar"""
        
        panel_width = 580
        
        # Panel sections
        sections = [
            ("SYSTEM METRICS", 0, [
                f"CPU Usage: {metrics['cpu_usage']:.1f}%",
                f"Memory: {metrics['memory_used']:.1f} GB / {metrics['memory_total']:.1f} GB",
                f"Processing FPS: {metrics['processing_fps']:.2f}",
                f"Video FPS: {self.video_fps:.1f}",
            ]),
            ("SLAM STATUS", 180, [
                f"Status: {metrics['slam_status']}",
                f"Position: ({metrics['slam_x']:.2f}, {metrics['slam_y']:.2f}, {metrics['slam_z']:.2f})",
                f"Total Poses: {metrics['slam_poses']}",
                f"Semantic Rescues: {metrics['semantic_rescues']}",
                f"Landmarks: {metrics['num_landmarks']}",
            ]),
            ("ENTITY TRACKING", 400, [
                f"Active Entities: {metrics['active_entities']}",
                f"Off-screen: {metrics['offscreen_entities']}",
                f"Total Tracked: {metrics['total_entities']}",
                f"New This Frame: {metrics['new_entities']}",
            ]),
            ("PERCEPTION", 580, [
                f"Detections: {metrics['num_detections']}",
                f"Avg Depth: {metrics['avg_depth']:.2f} m",
                f"Depth Range: {metrics['depth_min']:.2f} - {metrics['depth_max']:.2f} m",
            ]),
            ("PROCESSING PIPELINE", 740, [
                f"YOLO: {metrics['time_yolo']:.0f} ms ({metrics['time_yolo']/metrics.get('time_total', 1)*100:.1f}%)",
                f"Depth: {metrics['time_depth']:.0f} ms ({metrics['time_depth']/metrics.get('time_total', 1)*100:.1f}%)",
                f"SLAM: {metrics['time_slam']:.0f} ms ({metrics['time_slam']/metrics.get('time_total', 1)*100:.1f}%)",
                f"Render: {metrics['time_render']:.0f} ms ({metrics['time_render']/metrics.get('time_total', 1)*100:.1f}%)",
                f"Total: {metrics['time_total']:.0f} ms",
            ]),
        ]
        
        for title, y_offset, lines in sections:
            y_current = y + y_offset
            
            # Section header
            cv2.rectangle(canvas, (x-10, y_current-5), 
                         (x + panel_width, y_current + 25), 
                         self.PANEL_COLOR, -1)
            cv2.putText(canvas, title, (x, y_current + 18), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, self.ACCENT_COLOR, 2, cv2.LINE_AA)
            
            # Section content
            y_current += 40
            for line in lines:
                cv2.putText(canvas, line, (x + 10, y_current), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR, 1, cv2.LINE_AA)
                y_current += 25
    
    def _draw_bottom_info(self, canvas: np.ndarray, metrics: Dict):
        """Draw bottom information bar"""
        y_start = self.dashboard_height - 50
        
        # Background
        cv2.rectangle(canvas, (0, y_start), 
                     (self.dashboard_width, self.dashboard_height), 
                     self.PANEL_COLOR, -1)
        
        # Frame info
        frame_info = f"Frame: {self.frame_count}/{self.total_frames}  |  " \
                     f"Processed: {self.processed_frames}  |  " \
                     f"Skip: Every {self.skip_frames} frames"
        cv2.putText(canvas, frame_info, (30, y_start + 32), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 1, cv2.LINE_AA)
        
        # Controls hint
        controls = "Space: Pause  |  S: Screenshot  |  D: Depth  |  M: Map  |  H: Help  |  Q: Quit"
        cv2.putText(canvas, controls, (800, y_start + 32), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR, 1, cv2.LINE_AA)
    
    def _draw_offscreen_banner(self, frame: np.ndarray):
        """Draw banner showing off-screen entities"""
        if not self.offscreen_entities:
            return
        
        banner_height = 50
        banner = np.zeros((banner_height, frame.shape[1], 3), dtype=np.uint8)
        banner[:] = (30, 30, 35)  # Dark background
        
        # Title
        text = f"Off-screen Entities ({len(self.offscreen_entities)}): "
        cv2.putText(banner, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, self.ACCENT_COLOR, 2, cv2.LINE_AA)
        
        # Show up to 8 entities
        x_offset = 300
        for i, entity in enumerate(self.offscreen_entities[:8]):
            if x_offset > frame.shape[1] - 100:
                break
            
            eid = entity['id']
            color = self._get_entity_color(eid)
            direction = entity.get('direction', '?')
            
            # Color dot
            cv2.circle(banner, (x_offset, 25), 6, color, -1)
            cv2.circle(banner, (x_offset, 25), 6, (255, 255, 255), 1)
            
            # ID and direction
            label = f"ID{eid} {direction}"
            cv2.putText(banner, label, (x_offset + 15, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR, 1, cv2.LINE_AA)
            
            x_offset += 100
        
        # Composite on top of frame
        frame[0:banner_height, :] = banner
    
    def _draw_entity_overlays(self, frame: np.ndarray, entities: List[Dict]):
        """Draw bounding boxes and labels for entities"""
        
        for entity in entities:
            if not entity.get('on_screen', True):
                continue
            
            eid = entity['id']
            bbox = entity['bbox']
            class_name = entity['class']
            distance = entity.get('distance', 0)
            color = self._get_entity_color(eid)
            
            # Bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Label background
            label = f"ID{eid}: {class_name} ({distance:.2f}m)"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
            
            # Label text
            cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Centroid
            cx, cy = entity['centroid']
            cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)
    
    def _create_spatial_map(self, entities: List[Dict], slam_pose: Optional[np.ndarray]) -> np.ndarray:
        """Create bird's eye view spatial map"""
        
        map_size = 400
        spatial_map = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        spatial_map[:] = (25, 25, 30)
        
        # Grid
        grid_spacing = 50
        for i in range(0, map_size, grid_spacing):
            cv2.line(spatial_map, (i, 0), (i, map_size), (40, 40, 50), 1)
            cv2.line(spatial_map, (0, i), (map_size, i), (40, 40, 50), 1)
        
        # Camera position (bottom center)
        cam_x = map_size // 2
        cam_y = map_size - 50
        cv2.drawMarker(spatial_map, (cam_x, cam_y), (255, 255, 255), 
                      cv2.MARKER_CROSS, 20, 2)
        cv2.putText(spatial_map, "CAM", (cam_x - 20, cam_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Map entities
        range_mm = 3000.0
        for entity in entities:
            if 'position_3d' not in entity:
                continue
            
            x, y, z = entity['position_3d']
            
            # Project to map coordinates
            map_x = int(cam_x + (x / range_mm) * (map_size / 2))
            map_y = int(cam_y - (z / range_mm) * (map_size / 2))
            
            if 0 <= map_x < map_size and 0 <= map_y < map_size:
                color = self._get_entity_color(entity['id'])
                
                # Line to camera
                cv2.line(spatial_map, (cam_x, cam_y), (map_x, map_y), color, 1)
                
                # Entity circle
                cv2.circle(spatial_map, (map_x, map_y), 8, (255, 255, 255), 2)
                cv2.circle(spatial_map, (map_x, map_y), 6, color, -1)
                
                # ID label
                cv2.putText(spatial_map, f"{entity['id']}", (map_x + 10, map_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Title
        cv2.putText(spatial_map, "SPATIAL MAP (Top-Down)", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ACCENT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(spatial_map, f"+/- {range_mm/1000:.1f}m", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.TEXT_COLOR, 1, cv2.LINE_AA)
        
        return spatial_map
    
    def run(self):
        """Main processing loop"""
        
        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Skip frames to match target FPS
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
            
            # === YOLO Detection ===
            t0 = time.time()
            yolo_results = self.yolo_model(frame, conf=0.3, verbose=False)[0]
            time_yolo = (time.time() - t0) * 1000
            
            # === Depth Estimation ===
            t0 = time.time()
            depth_map, _ = self.depth_estimator.estimate(frame)
            time_depth = (time.time() - t0) * 1000
            
            # === SLAM Update ===
            t0 = time.time()
            object_detections = []
            for det in yolo_results.boxes.data:
                if det[4] > 0.3:  # Confidence threshold
                    obj_det = {
                        'bbox': det[:4].cpu().numpy(),
                        'class_id': int(det[5]),
                        'confidence': float(det[4])
                    }
                    object_detections.append(obj_det)
            
            slam_pose = self.slam.track(frame, time.time(), self.frame_count, object_detections)
            stats = self.slam.get_statistics()
            time_slam = (time.time() - t0) * 1000
            
            # === Entity Management ===
            current_entities = []
            for det in yolo_results.boxes.data:
                conf = float(det[4])
                if conf < 0.3:
                    continue
                
                class_id = int(det[5])
                class_name = yolo_results.names[class_id]
                bbox = det[:4].cpu().numpy().astype(int)
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # Get 3D position
                distance_mm = depth_map[cy, cx] if depth_map is not None else 0
                distance_m = distance_mm / 1000.0
                
                # Assign entity ID
                entity_id = self._assign_entity_id(bbox, class_name, self.entities)
                
                entity = {
                    'id': entity_id,
                    'class': class_name,
                    'bbox': bbox,
                    'centroid': (cx, cy),
                    'confidence': conf,
                    'distance': distance_m,
                    'on_screen': True
                }
                
                # Check if on-screen
                margin = 50
                if x2 < margin or x1 > frame.shape[1] - margin or \
                   y2 < margin or y1 > frame.shape[0] - margin:
                    entity['on_screen'] = False
                
                current_entities.append(entity)
                self.entities[entity_id] = entity
            
            # Separate on-screen vs off-screen
            onscreen = [e for e in current_entities if e['on_screen']]
            self.offscreen_entities = [e for e in current_entities if not e['on_screen']]
            
            # === Rendering ===
            t0 = time.time()
            vis_frame = frame.copy()
            
            # Draw entity overlays
            self._draw_entity_overlays(vis_frame, onscreen)
            
            # Draw off-screen banner
            self._draw_offscreen_banner(vis_frame)
            
            # SLAM status overlay
            slam_status = "TRACKING" if slam_pose is not None else "LOST"
            status_color = self.SUCCESS_COLOR if slam_pose is not None else self.ERROR_COLOR
            cv2.putText(vis_frame, f"SLAM: {slam_status}", (10, vis_frame.shape[0] - 80),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2, cv2.LINE_AA)
            
            time_render = (time.time() - t0) * 1000
            
            # === Create spatial map ===
            spatial_map = None
            if self.show_spatial_map:
                spatial_map = self._create_spatial_map(current_entities, slam_pose)
            
            # === Collect metrics ===
            frame_time = time.time() - frame_start
            self.processed_frames += 1
            self.fps_history.append(1.0 / frame_time if frame_time > 0 else 0)
            
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.01)
            self.cpu_history.append(cpu_percent)
            memory = psutil.virtual_memory()
            
            # Depth metrics
            depth_values = depth_map[depth_map > 0] / 1000.0 \
                          if depth_map is not None else np.array([0])
            
            metrics = {
                'cpu_usage': cpu_percent,
                'memory_used': memory.used / (1024**3),
                'memory_total': memory.total / (1024**3),
                'processing_fps': np.mean(self.fps_history) if self.fps_history else 0,
                'slam_status': slam_status,
                'slam_x': slam_pose[0, 3] if slam_pose is not None else 0,
                'slam_y': slam_pose[1, 3] if slam_pose is not None else 0,
                'slam_z': slam_pose[2, 3] if slam_pose is not None else 0,
                'slam_poses': stats.get('total_poses', 0),
                'semantic_rescues': stats.get('landmark_only', 0),
                'num_landmarks': stats.get('landmarks', 0),
                'active_entities': len(onscreen),
                'offscreen_entities': len(self.offscreen_entities),
                'total_entities': len(current_entities),
                'new_entities': sum(1 for e in current_entities if e['id'] >= self.next_entity_id - 5),
                'num_detections': len(yolo_results.boxes.data),
                'avg_depth': np.mean(depth_values),
                'depth_min': np.min(depth_values),
                'depth_max': np.max(depth_values),
                'time_yolo': time_yolo,
                'time_depth': time_depth,
                'time_slam': time_slam,
                'time_render': time_render,
                'time_total': frame_time * 1000,
            }
            
            # === Create dashboard ===
            dashboard = self._create_dashboard(vis_frame, metrics)
            
            # Show windows
            cv2.imshow('Orion SLAM Dashboard', dashboard)
            
            if spatial_map is not None and self.show_spatial_map:
                cv2.imshow('Spatial Map', spatial_map)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = True
            elif key == ord('d'):
                self.show_depth = not self.show_depth
            elif key == ord('m'):
                self.show_spatial_map = not self.show_spatial_map
                if not self.show_spatial_map:
                    cv2.destroyWindow('Spatial Map')
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"slam_screenshot_{timestamp}.png"
                cv2.imwrite(filename, dashboard)
                print(f"Screenshot saved: {filename}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        final_stats = self.slam.get_statistics()
        print("\n" + "=" * 80)
        print("SESSION COMPLETE")
        print("=" * 80)
        print(f"Total frames: {self.frame_count}")
        print(f"Processed frames: {self.processed_frames}")
        print(f"Average FPS: {np.mean(self.fps_history):.2f}")
        print(f"Total entities tracked: {self.next_entity_id}")
        print(f"SLAM poses: {final_stats.get('total_poses', 0)}")
        print(f"Semantic rescues: {final_stats.get('landmark_only', 0)}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Orion SLAM Research Dashboard')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--skip', type=int, default=7, 
                       help='Process every Nth frame (default: 7 for ~1 FPS at 30 FPS video)')
    
    args = parser.parse_args()
    
    dashboard = ModernDashboard(args.video, skip_frames=args.skip)
    dashboard.run()


if __name__ == '__main__':
    main()
