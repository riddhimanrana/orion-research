#!/usr/bin/env python3
"""
Orion Complete SLAM System
===========================

Integrates ALL features:
- Phase 1: High-quality visualization (3D axes, depth heatmap, motion indicators)
- Phase 2: Entity tracking with Re-ID (permanent IDs, off-screen banner)
- Phase 3: Spatial zones + Scene classification + Enhanced CIS
- Phase 4: Visual SLAM with landmark tracking + Semantic rescues
- Visualization Improvements: Spatial map, direction hints, clean UI

Frame Skip: Every 3 frames for optimal SLAM tracking

Usage:
    python scripts/run_slam_complete.py --video data/examples/video.mp4 --max-frames 150

Controls:
    Space: Pause/Resume
    Q: Quit
    A: Toggle 3D axes
    Z: Toggle zone overlays
    S: Toggle spatial map

Author: Orion Research Team
Date: November 2025
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
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.slam.slam_engine import OpenCVSLAM, SLAMConfig
from orion.slam.semantic_slam import SemanticSLAM
from orion.managers.model_manager import ModelManager
from orion.perception.depth import DepthEstimator
from orion.perception.tracking import EntityTracker3D, TrackingConfig
from orion.perception.types import CameraIntrinsics
from orion.perception.camera_intrinsics import backproject_bbox
from orion.semantic.zone_manager import ZoneManager
from orion.semantic.scene_classifier import SceneClassifier
from orion.semantic.cis_scorer_3d import CausalInfluenceScorer3D


# ============================================================================
# Phase 1 Style: Depth Colormap + 3D Axes
# ============================================================================

def create_depth_colormap(depth_map: np.ndarray) -> np.ndarray:
    """Convert depth map to TURBO colormap (Phase 1 style)."""
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
    return depth_colored


def draw_3d_axes(frame: np.ndarray, origin_2d: Tuple[int, int], scale: float = 40):
    """Draw 3D coordinate axes at a point (Phase 1 style)."""
    ox, oy = int(origin_2d[0]), int(origin_2d[1])
    
    # X-axis (red) - right
    cv2.arrowedLine(frame, (ox, oy), (ox + int(scale), oy), 
                    (0, 0, 255), 2, tipLength=0.3)
    cv2.putText(frame, "X", (ox + int(scale) + 5, oy), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Y-axis (green) - down
    cv2.arrowedLine(frame, (ox, oy), (ox, oy + int(scale)), 
                    (0, 255, 0), 2, tipLength=0.3)
    cv2.putText(frame, "Y", (ox, oy + int(scale) + 12), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Z-axis (blue) - depth
    cv2.arrowedLine(frame, (ox, oy), (ox - int(scale*0.7), oy - int(scale*0.7)), 
                    (255, 0, 0), 2, tipLength=0.3)
    cv2.putText(frame, "Z", (ox - int(scale*0.7) - 15, oy - int(scale*0.7)), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)


# ============================================================================
# Phase 2: Off-Screen Banner + Direction Hints
# ============================================================================

def get_direction_hint(bbox: List[float], frame_width: int, frame_height: int) -> str:
    """Get direction arrow for off-screen entity."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    # Determine primary direction
    left = cx < 0
    right = cx > frame_width
    up = cy < 0
    down = cy > frame_height
    
    if left and up:
        return "↖"
    elif right and up:
        return "↗"
    elif left and down:
        return "↙"
    elif right and down:
        return "↘"
    elif left:
        return "←"
    elif right:
        return "→"
    elif up:
        return "↑"
    elif down:
        return "↓"
    else:
        return ""


def is_bbox_onscreen(bbox: List[float], frame_width: int, frame_height: int, margin: int = 50) -> bool:
    """Check if bbox is visible on screen."""
    x1, y1, x2, y2 = bbox
    return not (x2 < -margin or x1 > frame_width + margin or 
                y2 < -margin or y1 > frame_height + margin)


# ============================================================================
# Phase 3: Spatial Map with Zones
# ============================================================================

def create_spatial_map(
    tracks: List,
    zones: Dict,
    map_size: Tuple[int, int] = (500, 500),
    range_mm: float = 3000.0,
    show_zones: bool = True,
    selected_entity_id: Optional[int] = None,
    zoom: float = 1.0,
    pan_x: float = 0,
    pan_y: float = 0
) -> np.ndarray:
    """Create top-down bird's-eye spatial map (Phase 2/3 style) with interactive features."""
    map_h, map_w = map_size
    spatial_map = np.zeros((map_h, map_w, 3), dtype=np.uint8)
    
    # Apply zoom and pan
    effective_range = range_mm / zoom
    
    # Grid
    grid_spacing = int(50 * zoom)
    if grid_spacing > 5:  # Don't draw grid if too dense
        for i in range(0, map_w, grid_spacing):
            cv2.line(spatial_map, (i, 0), (i, map_h), (40, 40, 40), 1)
        for i in range(0, map_h, grid_spacing):
            cv2.line(spatial_map, (0, i), (map_w, i), (40, 40, 40), 1)
    
    # Camera at bottom center (adjusted for pan)
    camera_x = int(map_w // 2 + pan_x)
    camera_y = int(map_h - 30 + pan_y)
    cv2.drawMarker(spatial_map, (camera_x, camera_y), (255, 255, 255), 
                   cv2.MARKER_CROSS, 20, 2)
    cv2.putText(spatial_map, "Camera", (camera_x - 30, camera_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw zones first (if enabled)
    if show_zones and zones:
        for zone_id, zone in zones.items():
            x_mm, y_mm, z_mm = zone.centroid_3d_mm
            
            map_x = int(camera_x + (x_mm / effective_range) * (map_w / 2))
            map_y = int(camera_y - (z_mm / effective_range) * (map_h - 50))
            
            map_x = np.clip(map_x, 10, map_w - 10)
            map_y = np.clip(map_y, 10, map_h - 40)
            
            # Zone circle (larger, semi-transparent)
            cv2.circle(spatial_map, (map_x, map_y), 25, zone.color, 2)
            cv2.circle(spatial_map, (map_x, map_y), 3, zone.color, -1)
            
            # Zone label
            zone_num = zone_id.split('_')[-1]
            label = f"Z{zone_num}"
            cv2.putText(spatial_map, label, (map_x + 30, map_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw entities
    for track in tracks:
        if track.centroid_3d_mm is None:
            continue
        
        x_mm, y_mm, z_mm = track.centroid_3d_mm
        
        map_x = int(camera_x + (x_mm / effective_range) * (map_w / 2))
        map_y = int(camera_y - (z_mm / effective_range) * (map_h - 50))
        
        if map_x < 0 or map_x >= map_w or map_y < 0 or map_y >= map_h:
            continue
        
        # Line from camera to entity
        cv2.line(spatial_map, (camera_x, camera_y), (map_x, map_y), 
                (100, 100, 100), 1)
        
        # Entity circle
        color = track.color if hasattr(track, 'color') else (0, 255, 0)
        
        # Highlight selected entity
        if selected_entity_id is not None and track.entity_id == selected_entity_id:
            cv2.circle(spatial_map, (map_x, map_y), 15, (0, 255, 255), 2)  # Yellow ring
            cv2.putText(spatial_map, "SELECTED", (map_x + 20, map_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        cv2.circle(spatial_map, (map_x, map_y), 6, color, -1)
        cv2.circle(spatial_map, (map_x, map_y), 8, color, 1)
        
        # Entity ID
        cv2.putText(spatial_map, f"ID{track.entity_id}", (map_x + 10, map_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Title
    cv2.putText(spatial_map, "Interactive Spatial Map", (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Range and zoom indicators
    cv2.putText(spatial_map, f"Range: ±{effective_range/1000:.1f}m | Zoom: {zoom:.1f}x", 
               (10, map_h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Instructions
    cv2.putText(spatial_map, "L-Click:Select | Scroll:Zoom", 
               (10, map_h - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    
    return spatial_map


# ============================================================================
# Main SLAM System
# ============================================================================

class CompleteSLAMSystem:
    """Complete SLAM system integrating all phases"""
    
    def __init__(self, video_path: str, skip_frames: int = 3, zone_mode: str = "dense", adaptive_skip: bool = True):
        self.video_path = video_path
        self.skip_frames = skip_frames
        self.zone_mode = zone_mode
        self.adaptive_skip = adaptive_skip
        self.min_skip = 2
        self.max_skip = 8
        self.slam_failure_count = 0
        self.slam_success_count = 0
        
        print("="*80)
        print("ORION COMPLETE SLAM SYSTEM")
        print("="*80)
        
        # Video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")
        
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n📹 Video: {self.frame_width}x{self.frame_height} @ {self.video_fps:.1f} FPS")
        print(f"⏭️  Skip: Every {skip_frames} frames → ~{self.video_fps/skip_frames:.1f} FPS")
        print(f"📊 Total: {self.total_frames} frames ({self.total_frames/self.video_fps:.1f}s)\n")
        
        # Models
        print("🔧 Loading models...")
        self.model_manager = ModelManager.get_instance()
        self.yolo_model = self.model_manager.yolo
        self.clip_model = self.model_manager.clip
        self.depth_estimator = DepthEstimator(model_name="midas", device="mps")
        print("  ✓ YOLO + CLIP + Depth loaded")
        
        # YOLO classes
        self.yolo_classes = list(self.yolo_model.names.values()) if hasattr(self.yolo_model, 'names') else []
        
        # SLAM (Phase 4)
        print("\n🗺️  Initializing SLAM...")
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
        print("  ✓ Visual SLAM with semantic landmarks")
        
        # Tracking (Phase 2)
        print("\n👁️  Initializing entity tracking...")
        tracking_config = TrackingConfig(
            max_distance_pixels=150.0,
            max_distance_3d_mm=1500.0,
            ttl_frames=30,
            reid_window_frames=90,
        )
        self.tracker = EntityTracker3D(config=tracking_config, yolo_classes=self.yolo_classes)
        print("  ✓ 3D tracker with Re-ID")
        
        # Scene classification (Phase 3)
        print("\n🏠 Initializing scene classification...")
        self.scene_classifier = SceneClassifier(clip_model=self.clip_model)
        print("  ✓ Scene classifier")
        
        # Zone manager (Phase 3)
        print("\n🗂️  Initializing spatial zones...")
        self.zone_manager = ZoneManager(
            mode=zone_mode,
            min_cluster_size=8,
            min_samples=3,
            merge_distance_mm=2500.0,
        )
        print(f"  ✓ Zone manager ({zone_mode} mode)")
        
        # CIS scorer (Phase 3)
        print("\n🧮 Initializing CIS scorer...")
        self.cis_scorer = CausalInfluenceScorer3D()
        print("  ✓ Causal Influence Scorer 3D")
        
        # Camera intrinsics
        self.camera_intrinsics = CameraIntrinsics.auto_estimate(
            self.frame_width, self.frame_height
        )
        
        # State
        self.frame_count = 0
        self.processed_frames = 0
        self.fps_history = deque(maxlen=30)
        self.paused = False
        self.prev_frame = None
        self.prev_entities = {}
        self.current_scene_type = "unknown"
        
        # UI toggles
        self.show_axes = True
        self.show_zones = True
        self.show_spatial_map = True
        
        # Interactive map state
        self.selected_entity_id = None
        self.spatial_map_pan_x = 0
        self.spatial_map_pan_y = 0
        self.spatial_map_zoom = 1.0
        self.mouse_dragging = False
        self.drag_start = (0, 0)
        self.current_tracks = []  # Store tracks for mouse callback
        
        print("\n✅ System ready!")
        print("="*80)
        print("Controls: Space=Pause | Q=Quit | A=Toggle Axes | Z=Toggle Zones | S=Toggle Spatial Map")
        print("Spatial Map: Left-Click=Select Entity | Right-Click=Zone Info | Scroll=Zoom")
        print("="*80 + "\n")
    
    def _on_spatial_map_mouse(self, event, x, y, flags, param):
        """Mouse callback for interactive spatial map"""
        map_size = (500, 500)
        map_h, map_w = map_size
        camera_x = map_w // 2 + int(self.spatial_map_pan_x)
        camera_y = map_h - 30 + int(self.spatial_map_pan_y)
        effective_range = 3000.0 / self.spatial_map_zoom
        
        # Left click: Select entity
        if event == cv2.EVENT_LBUTTONDOWN:
            # Find entity near click
            min_dist = float('inf')
            selected = None
            
            for track in self.current_tracks:
                if track.centroid_3d_mm is None:
                    continue
                
                x_mm, y_mm, z_mm = track.centroid_3d_mm
                map_x = int(camera_x + (x_mm / effective_range) * (map_w / 2))
                map_y = int(camera_y - (z_mm / effective_range) * (map_h - 50))
                
                # Check if click is near entity
                dist = np.sqrt((x - map_x)**2 + (y - map_y)**2)
                if dist < 15 and dist < min_dist:  # Within 15 pixels
                    min_dist = dist
                    selected = track
            
            if selected:
                self.selected_entity_id = selected.entity_id
                print(f"\n{'='*60}")
                print(f"📍 SELECTED ENTITY: ID{selected.entity_id}")
                print(f"{'='*60}")
                print(f"  Class: {selected.most_likely_class}")
                if selected.centroid_3d_mm is not None:
                    x, y, z = selected.centroid_3d_mm
                    print(f"  World Position: ({x:.0f}, {y:.0f}, {z:.0f}) mm")
                    print(f"  Distance: {z/1000:.2f} m from camera")
                if hasattr(selected, 'zone_id') and selected.zone_id:
                    print(f"  Zone: {selected.zone_id}")
                print(f"  Tracking confidence: {selected.bayesian_score:.2f}")
                print(f"  Frames tracked: {selected.total_observations}")
                print(f"{'='*60}\n")
            else:
                self.selected_entity_id = None
        
        # Right click: Show zone info
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Find zone near click
            for zone_id, zone in self.zone_manager.zones.items():
                x_mm, y_mm, z_mm = zone.centroid_3d_mm
                map_x = int(camera_x + (x_mm / effective_range) * (map_w / 2))
                map_y = int(camera_y - (z_mm / effective_range) * (map_h - 50))
                
                dist = np.sqrt((x - map_x)**2 + (y - map_y)**2)
                if dist < 25:  # Within 25 pixels
                    print(f"\n{'='*60}")
                    print(f"🗂️  ZONE INFO: {zone_id}")
                    print(f"{'='*60}")
                    print(f"  Label: {zone.label}")
                    print(f"  Centroid: {zone.centroid_3d_mm}")
                    print(f"  Entity count: {len(zone.entity_ids)}")
                    print(f"  Entities: {zone.entity_ids}")
                    print(f"  Last updated: Frame {zone.last_updated_frame}")
                    print(f"{'='*60}\n")
                    break
        
        # Mouse wheel: Zoom
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.spatial_map_zoom *= 1.2
            else:
                self.spatial_map_zoom /= 1.2
            self.spatial_map_zoom = np.clip(self.spatial_map_zoom, 0.5, 5.0)
            print(f"Zoom: {self.spatial_map_zoom:.1f}x")
    
    def _visualize_entity(self, frame: np.ndarray, track, show_axes: bool = True) -> np.ndarray:
        """Phase 1 + Phase 2 style entity visualization"""
        vis = frame.copy()
        
        bbox = track.bbox
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = track.color if hasattr(track, 'color') else (0, 255, 0)
        
        # Check if recently re-identified (orange highlight for 15 frames)
        reid_highlight = False
        if hasattr(track, 'last_reid_frame') and track.last_reid_frame is not None:
            frames_since_reid = self.frame_count - track.last_reid_frame
            if frames_since_reid <= 15:
                color = (0, 165, 255)  # Orange
                reid_highlight = True
        
        # Draw bbox (2px)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        
        # Class label
        class_name = track.most_likely_class or "unknown"
        conf = track.confidence if hasattr(track, 'confidence') else 0.0
        label = f"ID{track.entity_id}: {class_name}"
        if reid_highlight:
            label += " [RE-ID]"
        
        cv2.putText(vis, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Centroid
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(vis, (cx, cy), 4, color, -1)
        
        # 3D info (Phase 1 style)
        if track.centroid_3d_mm is not None:
            x_mm, y_mm, z_mm = track.centroid_3d_mm
            
            # 3D coordinates
            coord_text = f"3D: ({x_mm:.0f}, {y_mm:.0f}, {z_mm:.0f})mm"
            cv2.putText(vis, coord_text, (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Distance
            distance_m = z_mm / 1000.0
            dist_text = f"Distance: {distance_m:.2f}m"
            cv2.putText(vis, dist_text, (x1, y2 + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Motion indicators (Phase 1 style)
            if track.entity_id in self.prev_entities:
                prev_track = self.prev_entities[track.entity_id]
                if prev_track.centroid_3d_mm is not None:
                    prev_z = prev_track.centroid_3d_mm[2]
                    z_change = z_mm - prev_z
                    
                    if abs(z_change) > 50:  # 5cm threshold
                        if z_change < 0:
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
    
    def _draw_offscreen_banner(self, frame: np.ndarray, offscreen_tracks: List) -> np.ndarray:
        """Phase 2 style: Off-screen tracks banner"""
        if not offscreen_tracks:
            return frame
        
        vis = frame.copy()
        h, w = frame.shape[:2]
        
        # Banner background
        banner_height = 35
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
        
        # Banner text
        max_show = 8
        shown_tracks = offscreen_tracks[:max_show]
        
        banner_text = f"Off-screen Tracks ({len(offscreen_tracks)}): "
        for track in shown_tracks:
            direction = get_direction_hint(track.bbox, self.frame_width, self.frame_height)
            banner_text += f"ID{track.entity_id}{direction} "
        
        if len(offscreen_tracks) > max_show:
            banner_text += f"... +{len(offscreen_tracks) - max_show} more"
        
        cv2.putText(vis, banner_text, (10, 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        
        return vis
    
    def run(self, max_frames: Optional[int] = None):
        """Main processing loop"""
        
        zone_update_interval = int(self.video_fps)  # Update zones every 1 second
        scene_update_interval = int(self.video_fps * 30)  # Classify scene every 30 seconds
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Skip frames
            if self.frame_count % self.skip_frames != 0:
                continue
            
            if max_frames and self.processed_frames >= max_frames:
                print(f"\n✓ Reached max frames: {max_frames}")
                break
            
            if self.paused:
                key = cv2.waitKey(100) & 0xFF
                if key == ord(' '):
                    self.paused = False
                elif key == ord('q'):
                    break
                continue
            
            frame_start = time.time()
            timestamp = self.frame_count / self.video_fps
            
            # === Scene Classification ===
            if self.frame_count % scene_update_interval == 0:
                scene_result = self.scene_classifier.classify_detailed(frame, self.prev_frame)
                self.current_scene_type = scene_result.scene_type.value
                print(f"[Frame {self.frame_count}] Scene: {self.current_scene_type} "
                      f"(conf: {scene_result.confidence:.2f}, indoor: {scene_result.is_indoor})")
            
            # === YOLO Detection ===
            yolo_results = self.yolo_model(frame, conf=0.35, verbose=False)[0]
            
            # === Depth Estimation ===
            depth_map, _ = self.depth_estimator.estimate(frame)
            
            # === Build Detections with 3D + Embeddings ===
            detections = []
            object_detections = []  # For SLAM
            
            for det in yolo_results.boxes.data:
                if det[4] < 0.35:
                    continue
                
                class_id = int(det[5])
                class_name = self.yolo_classes[class_id] if class_id < len(self.yolo_classes) else "unknown"
                bbox = det[:4].cpu().numpy()
                x1, y1, x2, y2 = bbox
                conf = float(det[4])
                
                # Centroid
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                centroid_2d = np.array([cx, cy])
                
                # 3D backprojection
                bbox_3d_info = backproject_bbox(bbox, depth_map, self.camera_intrinsics)
                centroid_3d_mm = np.array(bbox_3d_info['centroid_3d'])
                
                # CLIP embedding
                x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
                x2_int, y2_int = min(self.frame_width, int(x2)), min(self.frame_height, int(y2))
                crop = frame[y1_int:y2_int, x1_int:x2_int]
                
                embedding = None
                if crop.size > 0 and crop.shape[0] > 0 and crop.shape[1] > 0:
                    crop_resized = cv2.resize(crop, (224, 224))
                    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                    embedding = self.clip_model.encode_image(crop_rgb, normalize=True)
                
                detection = {
                    'bbox': bbox,
                    'class_name': class_name,
                    'confidence': conf,
                    'centroid_2d': centroid_2d,
                    'centroid_3d_mm': centroid_3d_mm,
                    'appearance_embedding': embedding,
                }
                
                detections.append(detection)
                
                # For SLAM
                object_detections.append({
                    'bbox': bbox,
                    'class': class_name,
                    'confidence': conf
                })
            
            # === SLAM Tracking (Phase 4) ===
            slam_pose = self.slam.track(frame, timestamp, self.frame_count, object_detections, depth_map)
            slam_stats = self.slam.get_statistics()
            
            # Adaptive skip adjustment based on SLAM performance
            if self.adaptive_skip and self.processed_frames > 20:
                if slam_pose is None:
                    self.slam_failure_count += 1
                    self.slam_success_count = 0
                    
                    # SLAM failing - reduce skip for better tracking
                    if self.slam_failure_count >= 5 and self.skip_frames > self.min_skip:
                        self.skip_frames -= 1
                        print(f"⚠️  SLAM struggling → reducing skip to {self.skip_frames}")
                        self.slam_failure_count = 0
                else:
                    self.slam_success_count += 1
                    self.slam_failure_count = 0
                    
                    # SLAM stable - can increase skip for efficiency
                    if self.slam_success_count >= 25 and self.skip_frames < self.max_skip:
                        self.skip_frames += 1
                        print(f"✓ SLAM stable → increasing skip to {self.skip_frames}")
                        self.slam_success_count = 0
            
            # === Entity Tracking (Phase 2) ===
            tracks = self.tracker.track_frame(detections, self.frame_count, timestamp)
            
            # Assign colors to tracks
            for track in tracks:
                if not hasattr(track, 'color'):
                    hue = (track.entity_id * 137.508) % 360
                    rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.9, 1.0)
                    track.color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            
            # === Zone Management (Phase 3) ===
            for track in tracks:
                if track.centroid_3d_mm is not None:
                    # Transform to SLAM world frame if available
                    world_pos = track.centroid_3d_mm
                    if slam_pose is not None and self.processed_frames > 0:
                        # Transform camera-relative to world coordinates
                        try:
                            point_h = np.append(track.centroid_3d_mm, 1.0)
                            world_h = slam_pose @ point_h
                            world_pos = world_h[:3]
                        except:
                            world_pos = track.centroid_3d_mm
                    
                    self.zone_manager.add_observation(
                        entity_id=str(track.entity_id),
                        timestamp=timestamp,
                        centroid_3d_mm=world_pos,  # Use world coordinates
                        embedding=track.appearance_embedding,
                        class_label=track.most_likely_class,
                        frame_idx=self.frame_count
                    )
            
            if self.frame_count % zone_update_interval == 0:
                self.zone_manager.update_zones(timestamp, frame)
                zone_stats = self.zone_manager.get_zone_statistics()
                if zone_stats['total_zones'] > 0:
                    print(f"[Frame {self.frame_count}] Zones: {zone_stats['total_zones']} "
                          f"({zone_stats['zone_types']})")
            
            # === Visualization ===
            
            # Separate on-screen vs off-screen tracks
            onscreen_tracks = []
            offscreen_tracks = []
            
            for track in tracks:
                if is_bbox_onscreen(track.bbox, self.frame_width, self.frame_height):
                    onscreen_tracks.append(track)
                else:
                    offscreen_tracks.append(track)
            
            # Draw on-screen entities (Phase 1 + Phase 2 style)
            vis_frame = frame.copy()
            for track in onscreen_tracks:
                vis_frame = self._visualize_entity(vis_frame, track, self.show_axes)
            
            # Off-screen banner (Phase 2)
            vis_frame = self._draw_offscreen_banner(vis_frame, offscreen_tracks)
            
            # SLAM status overlay (Phase 4)
            status = "TRACKING" if slam_pose is not None else "LOST"
            status_color = (0, 255, 0) if slam_pose is not None else (0, 0, 255)
            cv2.putText(vis_frame, f"SLAM: {status}", (15, vis_frame.shape[0] - 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Frame info (bottom-left)
            info_y = vis_frame.shape[0] - 70
            cv2.putText(vis_frame, f"Frame: {self.frame_count}/{self.total_frames}", 
                       (15, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vis_frame, f"Tracks: {len(onscreen_tracks)} on-screen, {len(offscreen_tracks)} off-screen", 
                       (15, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vis_frame, f"Zones: {len(self.zone_manager.zones)} | SLAM Poses: {slam_stats.get('total_poses', 0)}", 
                       (15, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Scene type
            if self.current_scene_type != "unknown":
                cv2.putText(vis_frame, f"Scene: {self.current_scene_type}", 
                           (15, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 200), 1)
            
            # Depth colormap (Phase 1 TURBO style)
            depth_colored = None
            if depth_map is not None:
                depth_colored = create_depth_colormap(depth_map)
                
                if depth_colored.shape[:2] != vis_frame.shape[:2]:
                    depth_colored = cv2.resize(depth_colored, 
                                              (vis_frame.shape[1], vis_frame.shape[0]))
                
                # Side-by-side
                combined = np.hstack([vis_frame, depth_colored])
                
                # Labels
                cv2.putText(combined, "RGB + 3D Annotations + SLAM", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(combined, "Depth Map (Turbo)", 
                           (vis_frame.shape[1] + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                vis_frame = combined
            
            # Spatial map (Phase 2 + Phase 3 with zones)
            spatial_map = None
            if self.show_spatial_map:
                # Store tracks for mouse callback
                self.current_tracks = tracks
                
                spatial_map = create_spatial_map(
                    tracks, 
                    self.zone_manager.zones if self.show_zones else {},
                    map_size=(500, 500),
                    range_mm=3000.0,
                    show_zones=self.show_zones,
                    selected_entity_id=self.selected_entity_id,
                    zoom=self.spatial_map_zoom,
                    pan_x=self.spatial_map_pan_x,
                    pan_y=self.spatial_map_pan_y
                )
            
            # Update metrics
            frame_time = time.time() - frame_start
            self.processed_frames += 1
            self.fps_history.append(1.0 / frame_time if frame_time > 0 else 0)
            
            # Store for motion detection
            self.prev_entities = {track.entity_id: track for track in tracks}
            self.prev_frame = frame.copy()
            
            # Show
            cv2.imshow('Orion Complete SLAM System', vis_frame)
            
            if spatial_map is not None:
                cv2.imshow('Interactive Spatial Map', spatial_map)
                # Set mouse callback for interactive features
                cv2.setMouseCallback('Interactive Spatial Map', self._on_spatial_map_mouse)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = True
                print("\n⏸️  Paused. Press Space to resume...")
            elif key == ord('a'):
                self.show_axes = not self.show_axes
                print(f"3D Axes: {'ON' if self.show_axes else 'OFF'}")
            elif key == ord('z'):
                self.show_zones = not self.show_zones
                print(f"Zone Overlays: {'ON' if self.show_zones else 'OFF'}")
            elif key == ord('s'):
                self.show_spatial_map = not self.show_spatial_map
                print(f"Spatial Map: {'ON' if self.show_spatial_map else 'OFF'}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Final stats
        slam_stats = self.slam.get_statistics()
        tracker_stats = self.tracker.get_statistics()
        zone_stats = self.zone_manager.get_zone_statistics()
        
        print("\n" + "="*80)
        print("SESSION COMPLETE")
        print("="*80)
        print(f"📊 Frames: {self.frame_count}/{self.total_frames}")
        print(f"⚡ Processed: {self.processed_frames}")
        print(f"🎥 Avg FPS: {np.mean(self.fps_history):.2f}")
        print(f"\n👁️  Tracking (Phase 2):")
        print(f"  Total entities: {tracker_stats.get('total_entities_seen', 0)}")
        print(f"  Re-identifications: {tracker_stats.get('reidentifications', 0)}")
        print(f"\n🗺️  SLAM (Phase 4):")
        print(f"  Total poses: {slam_stats.get('total_poses', 0)}")
        print(f"  Semantic rescues: {slam_stats.get('landmark_only', 0)}")
        print(f"\n🗂️  Spatial Zones (Phase 3):")
        print(f"  Total zones: {zone_stats['total_zones']} (camera-relative viewpoints)")
        print(f"  Zone types: {zone_stats['zone_types']}")
        print(f"  Observations: {zone_stats['total_observations']}")
        print(f"\n  NOTE: Zones represent camera-relative viewpoints. Multiple zones may")
        print(f"        correspond to the same physical area viewed from different angles.")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Complete SLAM System (All Phases)')
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--skip', type=int, default=3, help='Frame skip (default: 3 for SLAM accuracy)')
    parser.add_argument('--max-frames', type=int, default=None, help='Max frames to process')
    parser.add_argument('--zone-mode', type=str, default='dense', choices=['dense', 'sparse'])
    parser.add_argument('--no-adaptive', action='store_true', help='Disable adaptive frame skip')
    
    args = parser.parse_args()
    
    system = CompleteSLAMSystem(args.video, skip_frames=args.skip, zone_mode=args.zone_mode, 
                                adaptive_skip=not args.no_adaptive)
    system.run(max_frames=args.max_frames)


if __name__ == '__main__':
    main()
