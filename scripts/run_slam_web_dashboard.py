#!/usr/bin/env python3
"""
Orion SLAM Web Dashboard - Modern Browser-Based UI
===================================================

Professional web-based dashboard using Dash/Plotly for clean, responsive UI.

Features:
- Proper aspect ratio handling (vertical/horizontal)
- High-quality rendering
- Real-time metrics
- Clean, simple design
- Smaller frame skip for better SLAM tracking

Frame Skipping: Processes every 3rd frame (~10 FPS from 30 FPS video)
                More frequent for SLAM feature tracking

Usage:
    python scripts/run_slam_web_dashboard.py --video data/examples/video.mp4
    
    Then open: http://localhost:8050 in your browser
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
import base64
from io import BytesIO
from PIL import Image
import threading
import queue

# Dash imports
try:
    import dash
    from dash import dcc, html, Input, Output, State
    import plotly.graph_objs as go
    from dash.exceptions import PreventUpdate
except ImportError:
    print("ERROR: Dash not installed. Install with:")
    print("  pip install dash plotly pillow")
    sys.exit(1)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.slam.slam_engine import OpenCVSLAM, SLAMConfig
from orion.slam.semantic_slam import SemanticSLAM
from orion.managers.model_manager import ModelManager
from orion.perception.depth import DepthEstimator


class SLAMProcessor:
    """Background processor for SLAM pipeline"""
    
    def __init__(self, video_path: str, skip_frames: int = 3):
        self.video_path = video_path
        self.skip_frames = skip_frames
        
        # Video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {self.frame_width}x{self.frame_height} @ {self.video_fps:.1f} FPS")
        print(f"Processing every {skip_frames} frames (~{self.video_fps/skip_frames:.1f} FPS)")
        
        # Initialize models
        print("Loading models...")
        self.model_manager = ModelManager.get_instance()
        self.yolo_model = self.model_manager.yolo
        
        # Depth estimator
        self.depth_estimator = DepthEstimator(model_name="midas", device="mps")
        
        # SLAM with relaxed parameters for better tracking
        slam_config = SLAMConfig(
            num_features=2000,  # More features
            match_ratio_test=0.75,  # More lenient matching
            min_matches=6  # Lower threshold
        )
        base_slam = OpenCVSLAM(config=slam_config)
        self.slam = SemanticSLAM(
            base_slam=base_slam,
            use_landmarks=True,
            landmark_weight=0.4
        )
        
        # State
        self.current_frame = None
        self.current_vis = None
        self.entities = {}
        self.next_entity_id = 0
        self.entity_colors = {}
        self.metrics = {}
        self.frame_count = 0
        self.processed_frames = 0
        self.is_running = False
        self.fps_history = deque(maxlen=30)
        
        print("✓ Initialization complete\n")
    
    def _get_entity_color(self, entity_id: int) -> Tuple[int, int, int]:
        """Get unique color for entity"""
        if entity_id not in self.entity_colors:
            hue = (entity_id * 137.508) % 360
            rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.85, 0.95)
            self.entity_colors[entity_id] = (
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
        return self.entity_colors[entity_id]
    
    def _assign_entity_id(self, bbox: np.ndarray, class_name: str) -> int:
        """Simple spatial matching for entity ID"""
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        
        # Match within 80px
        for eid, entity in self.entities.items():
            if entity['class'] == class_name:
                ex, ey = entity['centroid']
                dist = np.sqrt((cx - ex)**2 + (cy - ey)**2)
                if dist < 80:
                    return eid
        
        # New entity
        new_id = self.next_entity_id
        self.next_entity_id += 1
        return new_id
    
    def _draw_visualizations(self, frame: np.ndarray, entities: List[Dict], 
                            slam_pose: Optional[np.ndarray]) -> np.ndarray:
        """Draw clean visualizations on frame"""
        vis = frame.copy()
        
        # Draw entities
        for entity in entities:
            if not entity.get('on_screen', True):
                continue
            
            eid = entity['id']
            bbox = entity['bbox']
            x1, y1, x2, y2 = bbox
            color = self._get_entity_color(eid)
            
            # Bounding box (2px for clean look)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Label with semi-transparent background
            label = f"ID{eid}: {entity['class']} ({entity['distance']:.1f}m)"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Background
            cv2.rectangle(vis, (x1, y1 - label_h - 10), 
                         (x1 + label_w + 6, y1), color, -1)
            
            # Text
            cv2.putText(vis, label, (x1 + 3, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Centroid
            cx, cy = entity['centroid']
            cv2.circle(vis, (cx, cy), 4, color, -1)
            cv2.circle(vis, (cx, cy), 4, (255, 255, 255), 1)
        
        # SLAM status (bottom left)
        status = "TRACKING" if slam_pose is not None else "LOST"
        status_color = (0, 255, 0) if slam_pose is not None else (0, 0, 255)
        cv2.putText(vis, f"SLAM: {status}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
        
        return vis
    
    def process_frame(self) -> bool:
        """Process next frame, returns False when done"""
        ret, frame = self.cap.read()
        if not ret:
            return False
        
        self.frame_count += 1
        
        # Skip frames
        if self.frame_count % self.skip_frames != 0:
            return True
        
        frame_start = time.time()
        
        # YOLO detection
        yolo_results = self.yolo_model(frame, conf=0.35, verbose=False)[0]
        
        # Depth estimation
        depth_map, _ = self.depth_estimator.estimate(frame)
        
        # SLAM update
        object_detections = []
        for det in yolo_results.boxes.data:
            if det[4] > 0.35:
                obj_det = {
                    'bbox': det[:4].cpu().numpy(),
                    'class': yolo_results.names[int(det[5])],
                    'confidence': float(det[4])
                }
                object_detections.append(obj_det)
        
        slam_pose = self.slam.track(frame, time.time(), self.frame_count, object_detections)
        stats = self.slam.get_statistics()
        
        # Build entities
        current_entities = []
        for det in yolo_results.boxes.data:
            conf = float(det[4])
            if conf < 0.35:
                continue
            
            class_id = int(det[5])
            class_name = yolo_results.names[class_id]
            bbox = det[:4].cpu().numpy().astype(int)
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Get depth
            distance_mm = depth_map[cy, cx] if depth_map is not None else 0
            distance_m = distance_mm / 1000.0
            
            # Assign ID
            entity_id = self._assign_entity_id(bbox, class_name)
            
            entity = {
                'id': entity_id,
                'class': class_name,
                'bbox': bbox,
                'centroid': (cx, cy),
                'confidence': conf,
                'distance': distance_m,
                'on_screen': True
            }
            
            current_entities.append(entity)
            self.entities[entity_id] = entity
        
        # Create visualization
        vis_frame = self._draw_visualizations(frame, current_entities, slam_pose)
        
        # Update state
        self.current_frame = frame
        self.current_vis = vis_frame
        self.processed_frames += 1
        
        # Metrics
        frame_time = time.time() - frame_start
        self.fps_history.append(1.0 / frame_time if frame_time > 0 else 0)
        
        cpu_percent = psutil.cpu_percent(interval=0)
        memory = psutil.virtual_memory()
        
        self.metrics = {
            'frame_count': self.frame_count,
            'processed_frames': self.processed_frames,
            'total_frames': self.total_frames,
            'progress_pct': (self.frame_count / self.total_frames) * 100,
            'processing_fps': np.mean(self.fps_history) if self.fps_history else 0,
            'cpu_usage': cpu_percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'slam_status': "TRACKING" if slam_pose is not None else "LOST",
            'slam_poses': stats.get('total_poses', 0),
            'semantic_rescues': stats.get('landmark_only', 0),
            'active_entities': len(current_entities),
            'total_entities': self.next_entity_id,
        }
        
        return True
    
    def cleanup(self):
        """Release resources"""
        self.cap.release()


# Global processor instance
processor = None


def create_app():
    """Create Dash application"""
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("Orion SLAM Dashboard", style={
                'textAlign': 'center',
                'color': '#00d9ff',
                'margin': '20px 0',
                'fontFamily': 'system-ui, -apple-system, sans-serif'
            }),
        ], style={'backgroundColor': '#1a1a1f', 'padding': '10px'}),
        
        # Main content
        html.Div([
            # Left side - Video
            html.Div([
                html.Div([
                    html.Img(id='video-feed', style={
                        'width': '100%',
                        'height': 'auto',
                        'borderRadius': '8px'
                    })
                ], style={
                    'backgroundColor': '#2a2a35',
                    'padding': '20px',
                    'borderRadius': '8px',
                    'marginBottom': '20px'
                }),
                
                # Progress bar
                html.Div([
                    html.Progress(id='progress-bar', value='0', max='100', style={
                        'width': '100%',
                        'height': '30px'
                    }),
                    html.P(id='progress-text', style={
                        'textAlign': 'center',
                        'color': '#ffffff',
                        'margin': '10px 0 0 0'
                    })
                ])
            ], style={'flex': '2', 'marginRight': '20px'}),
            
            # Right side - Metrics
            html.Div([
                # System metrics
                html.Div([
                    html.H3("System Metrics", style={'color': '#00d9ff', 'marginBottom': '15px'}),
                    html.Div(id='system-metrics')
                ], style={
                    'backgroundColor': '#2a2a35',
                    'padding': '20px',
                    'borderRadius': '8px',
                    'marginBottom': '20px'
                }),
                
                # SLAM metrics
                html.Div([
                    html.H3("SLAM Status", style={'color': '#00d9ff', 'marginBottom': '15px'}),
                    html.Div(id='slam-metrics')
                ], style={
                    'backgroundColor': '#2a2a35',
                    'padding': '20px',
                    'borderRadius': '8px',
                    'marginBottom': '20px'
                }),
                
                # Entity metrics
                html.Div([
                    html.H3("Entity Tracking", style={'color': '#00d9ff', 'marginBottom': '15px'}),
                    html.Div(id='entity-metrics')
                ], style={
                    'backgroundColor': '#2a2a35',
                    'padding': '20px',
                    'borderRadius': '8px'
                })
            ], style={'flex': '1'})
        ], style={
            'display': 'flex',
            'padding': '20px',
            'backgroundColor': '#1a1a1f',
            'minHeight': '100vh'
        }),
        
        # Update interval
        dcc.Interval(
            id='interval-component',
            interval=100,  # Update every 100ms
            n_intervals=0
        )
    ], style={'backgroundColor': '#1a1a1f', 'margin': 0, 'padding': 0})
    
    @app.callback(
        [Output('video-feed', 'src'),
         Output('progress-bar', 'value'),
         Output('progress-text', 'children'),
         Output('system-metrics', 'children'),
         Output('slam-metrics', 'children'),
         Output('entity-metrics', 'children')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_dashboard(n):
        global processor
        
        if processor is None or processor.current_vis is None:
            raise PreventUpdate
        
        # Convert frame to base64
        vis_rgb = cv2.cvtColor(processor.current_vis, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(vis_rgb)
        buffer = BytesIO()
        pil_img.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        img_src = f"data:image/jpeg;base64,{img_str}"
        
        # Progress
        progress = processor.metrics.get('progress_pct', 0)
        progress_text = f"Frame {processor.metrics.get('frame_count', 0)}/{processor.metrics.get('total_frames', 0)} " \
                       f"(Processed: {processor.metrics.get('processed_frames', 0)})"
        
        # System metrics
        system_metrics = html.Div([
            html.P(f"CPU: {processor.metrics.get('cpu_usage', 0):.1f}%", 
                  style={'color': '#ffffff', 'margin': '5px 0'}),
            html.P(f"Memory: {processor.metrics.get('memory_used_gb', 0):.1f} / "
                  f"{processor.metrics.get('memory_total_gb', 0):.1f} GB",
                  style={'color': '#ffffff', 'margin': '5px 0'}),
            html.P(f"Processing FPS: {processor.metrics.get('processing_fps', 0):.2f}",
                  style={'color': '#ffffff', 'margin': '5px 0'})
        ])
        
        # SLAM metrics
        slam_color = '#64ff64' if processor.metrics.get('slam_status') == 'TRACKING' else '#ff6464'
        slam_metrics = html.Div([
            html.P(f"Status: {processor.metrics.get('slam_status', 'UNKNOWN')}", 
                  style={'color': slam_color, 'margin': '5px 0', 'fontWeight': 'bold'}),
            html.P(f"Total Poses: {processor.metrics.get('slam_poses', 0)}",
                  style={'color': '#ffffff', 'margin': '5px 0'}),
            html.P(f"Semantic Rescues: {processor.metrics.get('semantic_rescues', 0)}",
                  style={'color': '#ffffff', 'margin': '5px 0'})
        ])
        
        # Entity metrics
        entity_metrics = html.Div([
            html.P(f"Active: {processor.metrics.get('active_entities', 0)}",
                  style={'color': '#ffffff', 'margin': '5px 0'}),
            html.P(f"Total Tracked: {processor.metrics.get('total_entities', 0)}",
                  style={'color': '#ffffff', 'margin': '5px 0'})
        ])
        
        return img_src, progress, progress_text, system_metrics, slam_metrics, entity_metrics
    
    return app


def process_video_background(processor_instance):
    """Background thread to process video"""
    while processor_instance.is_running:
        if not processor_instance.process_frame():
            processor_instance.is_running = False
            break


def main():
    parser = argparse.ArgumentParser(description='Orion SLAM Web Dashboard')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--skip', type=int, default=3, 
                       help='Process every Nth frame (default: 3 for better SLAM)')
    parser.add_argument('--port', type=int, default=8050, help='Web server port')
    
    args = parser.parse_args()
    
    # Initialize processor
    global processor
    processor = SLAMProcessor(args.video, skip_frames=args.skip)
    
    # Start background processing
    processor.is_running = True
    process_thread = threading.Thread(target=process_video_background, args=(processor,))
    process_thread.daemon = True
    process_thread.start()
    
    # Create and run app
    app = create_app()
    
    print(f"\n{'='*80}")
    print(f"Dashboard running at: http://localhost:{args.port}")
    print(f"{'='*80}\n")
    
    try:
        app.run_server(debug=False, port=args.port, host='0.0.0.0')
    except KeyboardInterrupt:
        print("\nShutting down...")
        processor.is_running = False
        processor.cleanup()


if __name__ == '__main__':
    main()
