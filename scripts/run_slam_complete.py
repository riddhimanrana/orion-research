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
from orion.perception.scale_estimator import ScaleEstimator  # NEW
from orion.semantic.zone_manager import ZoneManager
from orion.semantic.scene_classifier import SceneClassifier
from orion.semantic.cis_scorer_3d import CausalInfluenceScorer3D
from orion.semantic.strategic_captioner import StrategicCaptioner  # NEW: Phase 2


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
    
    def __init__(self, video_path: str, skip_frames: int = 10, zone_mode: str = "dense", adaptive_skip: bool = True, use_rerun: bool = False):
        self.video_path = video_path
        self.skip_frames = skip_frames
        self.zone_mode = zone_mode
        self.adaptive_skip = adaptive_skip
        self.use_rerun = use_rerun
        self.min_skip = 5  # High activity: 6 fps
        self.max_skip = 60  # Low activity: 0.5 fps  
        self.slam_failure_count = 0
        self.slam_success_count = 0
        
        # Motion detection for adaptive skip
        self.prev_gray = None
        self.motion_history = []
        self.motion_window = 10
        
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
        # YOLO model will be set before loading via model_manager.yolo_model_name
        self.yolo_model = self.model_manager.yolo
        self.clip_model = self.model_manager.clip
        self.depth_estimator = DepthEstimator(model_name="midas", device="mps")
        
        # FastVLM for semantic enrichment (ON-DEMAND ONLY)
        # Don't caption during processing - too slow!
        # Instead: Store frames + bboxes, caption at query time
        self.fastvlm = None  # Will load on first use
        self.enable_fastvlm = False  # DISABLED for real-time processing (<66s)
        self.entity_captions = {}  # Cache: entity_id → caption
        self.zone_captions = {}  # Cache: zone_id → caption
        self.scene_captions = {}  # Cache: frame_key → caption
        self.seen_classes = set()  # Track which object classes we've seen
        self.captioned_classes = set()  # Track which classes we've actually captioned
        self.scene_type_history = []  # Track scene changes
        self.last_captioned_frame = -100  # Minimum gap between captions
        self.caption_budget = 15  # NEW: Increased from 8 to 15 for Phase 2

        # NEW: Strategic captioner (Phase 2)
        # VERY LOW thresholds for compatibility with fast mode (skip=40)
        # In fast mode, entities may only be detected 1-3 times even if visible for 1000+ frames
        self.strategic_captioner = StrategicCaptioner(
            caption_budget=15,
            min_confidence=0.1,  # Very low - accept almost anything detected
            min_tracking_frames=1  # Just 1 detection is enough in fast mode
        )
        self.enable_strategic_captioning = False  # Set via CLI or at end of processing        # Memgraph export (initialize BEFORE using it)
        self.export_to_memgraph = False  # Set via CLI flag
        self.memgraph_backend = None
        
        # Store frame crops for query-time captioning
        self.enable_crop_storage = self.export_to_memgraph  # Only if exporting
        self.frame_crop_cache = {}  # entity_id → {frame_idx: crop_path}
        self.crop_storage_dir = Path("debug_crops/query_cache")  # Storage location
        
        # Persistent spatial memory (set via CLI)
        self.spatial_memory = None  # Will be set in main() if enabled
        
        print("  ✓ YOLO + CLIP + Depth loaded")
        print(f"  ✓ FastVLM: Ready for post-processing (strategic captioning)")
        print(f"  ✓ Caption budget: {self.caption_budget} entities (est. ~{self.strategic_captioner.estimate_captioning_time(15):.0f}s)")
        
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
        
        # Scale estimator (Phase 5: Absolute scale recovery)
        print("\n📏 Initializing scale estimator...")
        self.scale_estimator = ScaleEstimator(
            min_estimates=10,
            confidence_threshold=0.7,
            outlier_threshold=2.0
        )
        self.absolute_scale = None  # Will be set once scale is locked
        print("  ✓ Scale estimator (monocular → metric)")
        
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
        
        # Rerun visualization (optional)
        self.rerun_logger = None
        if self.use_rerun:
            print("\n📊 Initializing Rerun.io visualization...")
            from orion.visualization.rerun_logger import RerunLogger, RerunConfig
            
            rerun_config = RerunConfig(
                log_video=True,
                log_depth=True,
                log_depth_3d=True,
                log_detections=True,
                log_entities=True,
                log_slam_trajectory=True,
                log_zones=True,
                log_zones_3d=True,
                log_metrics=True,
                log_annotations=True,
                downsample_depth=4,
                max_points_per_frame=5000,
                batch_logging=True
            )
            
            self.rerun_logger = RerunLogger(config=rerun_config)
            print("[Rerun] Initialized: orion-slam")
            print("  ✓ Rerun.io logger initialized with FULL features")
            print("  ✓ 3D point clouds, velocity vectors, room meshes enabled")
        
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
        if self.use_rerun:
            print("🎨 RERUN MODE: Visualization in browser (OpenCV windows disabled)")
            print("   Check your browser for interactive 3D view!")
        else:
            print("Controls: Space=Pause | Q=Quit | A=Toggle Axes | Z=Toggle Zones | S=Toggle Spatial Map")
            print("Spatial Map: Left-Click=Select Entity | Right-Click=Zone Info | Scroll=Zoom")
        print("="*80 + "\n")
    
    def _ensure_fastvlm_loaded(self):
        """Lazy-load FastVLM model on first use"""
        if self.fastvlm is None and self.enable_fastvlm:
            print("\n🧠 Loading FastVLM for semantic enrichment...")
            try:
                self.fastvlm = self.model_manager.fastvlm
                print("  ✓ FastVLM loaded (0.5B params, MLX-optimized)")
            except Exception as e:
                print(f"  ⚠️  FastVLM loading failed: {e}")
                print("  Continuing without semantic captions...")
                self.enable_fastvlm = False
    
    def _generate_entity_caption(self, frame: np.ndarray, bbox: List[float], entity_id: int) -> Optional[str]:
        """Generate rich semantic caption for an entity using FastVLM
        
        Only called periodically to avoid performance hit.
        Caches results for quick queries.
        """
        if not self.enable_fastvlm:
            return None
        
        # Check cache first
        if entity_id in self.entity_captions:
            return self.entity_captions[entity_id]
        
        # Ensure model loaded
        self._ensure_fastvlm_loaded()
        if self.fastvlm is None:
            return None
        
        try:
            # Crop entity from frame
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            crop = frame[y1:y2, x1:x2]
            
            # Convert BGR → RGB → PIL Image for FastVLM
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            from PIL import Image
            crop_pil = Image.fromarray(crop_rgb)
            
            # Generate caption with focused prompt
            prompt = "Describe this object in detail, including its color, appearance, and any distinctive features:"
            caption = self.fastvlm.generate_description(
                crop_pil,
                prompt,
                max_tokens=128,
                temperature=0.3
            )
            
            # Cache result
            self.entity_captions[entity_id] = caption
            
            return caption
            
        except Exception as e:
            print(f"  ⚠️  Caption generation failed for entity {entity_id}: {e}")
            return None
    
    def _generate_scene_caption(self, frame: np.ndarray) -> Optional[str]:
        """Generate scene-level caption using FastVLM
        
        Describes overall scene context, activities, and environment.
        """
        if not self.enable_fastvlm:
            return None
        
        # Ensure model loaded
        self._ensure_fastvlm_loaded()
        if self.fastvlm is None:
            return None
        
        try:
            # Convert BGR → RGB → PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image
            frame_pil = Image.fromarray(frame_rgb)
            
            # Generate scene caption
            prompt = "Describe this scene in detail: what room is this, what activities are happening, what objects are visible?"
            caption = self.fastvlm.generate_description(
                frame_pil,
                prompt,
                max_tokens=256,
                temperature=0.3
            )
            
            # Cache by frame number (simple cache)
            cache_key = f"frame_{self.frame_count}"
            self.scene_captions[cache_key] = caption
            
            return caption
            
        except Exception as e:
            print(f"  ⚠️  Scene caption generation failed: {e}")
            return None
    
    def _export_to_memgraph(self):
        """Export all video data to Memgraph graph database"""
        try:
            from orion.graph.memgraph_backend import MemgraphBackend
            
            # Connect to Memgraph
            print("  Connecting to Memgraph...")
            self.memgraph_backend = MemgraphBackend()
            
            # Clear previous data
            self.memgraph_backend.clear_all()
            
            # Export entity observations
            print("  Exporting entity observations...")
            for track in self.tracker.tracks.values():
                for obs in track.observations:
                    frame_idx = obs['frame_idx']
                    timestamp = frame_idx / self.video_fps
                    bbox = obs['bbox']
                    confidence = obs.get('confidence', 0.8)
                    zone_id = obs.get('zone_id')
                    
                    # Get caption if exists
                    caption = self.entity_captions.get(track.id)
                    
                    # Get crop path for query-time captioning
                    x1, y1, x2, y2 = map(int, bbox)
                    cache_key = f"{frame_idx}_{x1}_{y1}_{x2}_{y2}"
                    crop_path = None
                    if cache_key in self.frame_crop_cache:
                        crop_path = self.frame_crop_cache[cache_key]['path']
                    
                    self.memgraph_backend.add_entity_observation(
                        entity_id=track.id,
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        bbox=bbox,
                        class_name=track.most_likely_class,
                        confidence=confidence,
                        zone_id=zone_id,
                        caption=caption,
                        crop_path=crop_path  # NEW: For query-time captioning
                    )
            
            # Get statistics
            stats = self.memgraph_backend.get_statistics()
            print(f"  ✓ Exported {stats['entities']} entities")
            print(f"  ✓ Exported {stats['observations']} observations")
            print(f"  ✓ Created {stats['zones']} zones")
            print(f"\n  🔍 Query with: python scripts/query_memgraph.py")
            
        except ImportError:
            print("  ⚠️  Memgraph backend not available. Install with: pip install pymgclient")
        except Exception as e:
            print(f"  ⚠️  Memgraph export failed: {e}")
            print(f"     Make sure Memgraph is running: cd memgraph-platform && docker compose up -d")
    
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
                
                # Show FastVLM caption if available
                if selected.entity_id in self.entity_captions:
                    print(f"\n  🧠 Description: {self.entity_captions[selected.entity_id]}")
                
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
        
        # Add caption preview if available
        if track.entity_id in self.entity_captions:
            caption = self.entity_captions[track.entity_id]
            # Show first 30 chars of caption
            label += f" | {caption[:30]}..."
        
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
    
    def _store_frame_crop(self, crop, frame_idx: int, bbox, class_name: str):
        """Store frame crop for query-time captioning"""
        try:
            # Create storage directory
            self.crop_storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename: frame_bbox_class.jpg
            x1, y1, x2, y2 = map(int, bbox)
            crop_filename = f"frame_{frame_idx:06d}_bbox_{x1}_{y1}_{x2}_{y2}_{class_name}.jpg"
            crop_path = self.crop_storage_dir / crop_filename
            
            # Save crop
            cv2.imwrite(str(crop_path), crop)
            
            # Store in cache for later export
            cache_key = f"{frame_idx}_{x1}_{y1}_{x2}_{y2}"
            if cache_key not in self.frame_crop_cache:
                self.frame_crop_cache[cache_key] = {
                    'path': str(crop_path),
                    'frame_idx': frame_idx,
                    'bbox': bbox,
                    'class_name': class_name
                }
        except Exception as e:
            # Don't fail processing if crop storage fails
            pass
    
    def _generate_rich_caption(self, entity: Dict, track, reason: str) -> str:
        """
        Generate a rich, descriptive caption using all available entity metadata
        
        Args:
            entity: Entity info dict with class, confidence, frames_tracked, etc.
            track: Track object with full history
            reason: Selection reason from StrategicCaptioner
            
        Returns:
            Rich descriptive caption string
        """
        class_name = entity['class_name']
        frames = entity['frames_tracked']
        zone_id = entity['zone_id']
        confidence = entity['confidence']
        bbox = entity['bbox']
        centroid_3d = entity['centroid_3d']
        
        # Size estimation
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        size_desc = "large" if bbox_width * bbox_height > 0.15 else "medium" if bbox_width * bbox_height > 0.05 else "small"
        
        # Aspect ratio (wide/tall/square)
        aspect = bbox_width / bbox_height if bbox_height > 0 else 1.0
        shape_desc = "wide" if aspect > 1.5 else "tall" if aspect < 0.67 else "square"
        
        # Position description
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        
        # Horizontal position
        if x_center < 0.33:
            h_pos = "left"
        elif x_center > 0.67:
            h_pos = "right"
        else:
            h_pos = "center"
        
        # Vertical position
        if y_center < 0.33:
            v_pos = "upper"
        elif y_center > 0.67:
            v_pos = "lower"
        else:
            v_pos = "middle"
        
        # Movement analysis (if track available)
        movement_desc = ""
        if track and hasattr(track, 'movement_history') and len(track.movement_history) > 2:
            movements = track.movement_history[-10:]  # Last 10 positions
            x_variance = np.var([m[0] for m in movements])
            y_variance = np.var([m[1] for m in movements])
            total_variance = x_variance + y_variance
            
            if total_variance > 5000:  # Significant movement
                movement_desc = "moving"
            elif total_variance > 1000:
                movement_desc = "slightly moving"
            else:
                movement_desc = "stationary"
        else:
            movement_desc = "stationary"
        
        # 3D position (if available)
        position_3d = ""
        if centroid_3d is not None and len(centroid_3d) == 3:
            x, y, z = centroid_3d
            # Estimate distance from camera (z-axis in mm)
            dist_m = z / 1000.0  # Convert mm to meters
            if dist_m < 1.0:
                position_3d = f"very close to camera (~{dist_m:.1f}m)"
            elif dist_m < 3.0:
                position_3d = f"near camera (~{dist_m:.1f}m)"
            elif dist_m < 6.0:
                position_3d = f"moderate distance (~{dist_m:.1f}m)"
            else:
                position_3d = f"far from camera (~{dist_m:.1f}m)"
        
        # Class-specific enhancements
        class_contexts = {
            'person': ['standing', 'sitting', 'walking'],
            'chair': ['furniture', 'seating'],
            'table': ['furniture', 'surface'],
            'laptop': ['electronic', 'device'],
            'keyboard': ['electronic', 'input device'],
            'mouse': ['electronic', 'input device'],
            'tv': ['electronic', 'display'],
            'bottle': ['container', 'object'],
            'cup': ['container', 'drinkware'],
            'book': ['reading material', 'object']
        }
        context = class_contexts.get(class_name, ['object'])[0]
        
        # Build caption parts
        parts = []
        
        # Opening: "A [size] [shape] [class_name]"
        parts.append(f"A {size_desc} {class_name}")
        
        # Context hint
        if context and context not in ['object']:
            parts.append(f"({context})")
        
        # Position
        parts.append(f"positioned in the {v_pos} {h_pos} of the frame")
        
        # Movement
        if movement_desc:
            parts.append(f"{movement_desc}")
        
        # 3D position
        if position_3d:
            parts.append(f"{position_3d}")
        
        # Tracking duration
        parts.append(f"tracked across {frames} frames in spatial zone {zone_id}")
        
        # Confidence
        if confidence > 0.9:
            parts.append(f"with very high confidence ({confidence:.2f})")
        elif confidence > 0.7:
            parts.append(f"with high confidence ({confidence:.2f})")
        
        # Selection reason
        parts.append(f"Selected because: {reason}")
        
        # Combine into natural sentence
        caption = " ".join(parts)
        
        # Clean up multiple spaces
        caption = " ".join(caption.split())
        
        return caption
    
    def _perform_strategic_captioning(self):
        """
        Perform strategic captioning after video processing
        
        Uses StrategicCaptioner to select and caption the most important entities.
        This runs AFTER main processing to keep video processing fast.
        """
        if not self.enable_strategic_captioning:
            return
        
        print(f"\n🎨 Strategic Captioning (Phase 2)...")
        caption_start = time.time()
        
        # Prepare entity info for selection
        entity_tracks = []
        for entity_id, track in self.tracker.tracks.items():
            # Calculate bbox size (normalized)
            bbox = track.bbox
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            bbox_area = bbox_width * bbox_height
            frame_area = self.frame_width * self.frame_height
            bbox_size_normalized = bbox_area / frame_area if frame_area > 0 else 0
            
            # Use frame span instead of detection count for better compatibility with fast mode
            # Frame span = number of frames entity was alive (first_seen to last_seen)
            frames_alive = (track.last_frame_seen - track.first_frame_seen) if hasattr(track, 'first_frame_seen') else track.total_detections
            
            entity_info = {
                'entity_id': entity_id,
                'class_name': track.most_likely_class,
                'confidence': max(track.class_posterior.values()) if track.class_posterior else 0.5,
                'frames_tracked': frames_alive,  # Use frame span, not detection count
                'zone_id': getattr(track, 'zone_id', 0),
                'bbox_size': bbox_size_normalized,
                'bbox': bbox,
                'centroid_3d': track.centroid_3d_mm
            }
            entity_tracks.append(entity_info)
        
        # Also include disappeared tracks (for Re-ID entities)
        for entity_id, track in self.tracker.disappeared_tracks.items():
            bbox = track.bbox
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            bbox_area = bbox_width * bbox_height
            bbox_size_normalized = bbox_area / (self.frame_width * self.frame_height)
            
            frames_alive = (track.last_frame_seen - track.first_frame_seen) if hasattr(track, 'first_frame_seen') else track.total_detections
            
            entity_info = {
                'entity_id': entity_id,
                'class_name': track.most_likely_class,
                'confidence': max(track.class_posterior.values()) if track.class_posterior else 0.5,
                'frames_tracked': frames_alive,  # Use frame span, not detection count
                'zone_id': getattr(track, 'zone_id', 0),
                'bbox_size': bbox_size_normalized,
                'bbox': bbox,
                'centroid_3d': track.centroid_3d_mm
            }
            entity_tracks.append(entity_info)
        
        print(f"  Total entities: {len(entity_tracks)}")
        
        # Select entities to caption
        selected = self.strategic_captioner.select_entities_to_caption(
            tracks=entity_tracks,
            zones=None  # Could pass zone info here
        )
        
        print(f"  Selected: {len(selected)} entities for captioning")
        
        # DEBUG: Show why entities weren't selected
        if len(selected) == 0 and len(entity_tracks) > 0:
            print(f"  DEBUG: Sample entity stats (first 3):")
            for e in entity_tracks[:3]:
                print(f"    - Entity {e['entity_id']} ({e['class_name']}): confidence={e['confidence']:.2f}, frames={e['frames_tracked']}")
        
        if len(selected) == 0:
            print(f"  ⚠️  No entities selected (try lowering thresholds)")
            return
        
        # Load FastVLM model (lazy loading) - skip if model not available
        try:
            if self.fastvlm is None:
                print(f"  Loading FastVLM model...")
                model_manager = ModelManager.get_instance()
                self.fastvlm = model_manager.fastvlm  # Property, not method
                print(f"  ✓ FastVLM loaded")
        except Exception as e:
            print(f"  ⚠️  FastVLM not available ({type(e).__name__}), using template captions")
            self.fastvlm = None
        
        # Caption selected entities
        print(f"  Generating captions...")
        for entity_id, score, reason in selected:
            # Find entity info
            entity = next((e for e in entity_tracks if e['entity_id'] == entity_id), None)
            if not entity:
                continue
            
            # Get full track info for richer captions
            track = self.tracker.tracks.get(entity_id) or self.tracker.disappeared_tracks.get(entity_id)
            
            if self.fastvlm and track:
                # TODO: Use FastVLM with stored frame crops for vision-based captions
                caption = f"A {entity['class_name']} (vision-based caption placeholder)"
            else:
                # Generate rich template-based caption using all available metadata
                caption = self._generate_rich_caption(entity, track, reason)
            
            # Parse caption for attributes
            parsed = self.strategic_captioner.parse_caption(caption)
            
            # Format for memory
            formatted = self.strategic_captioner.format_caption_for_memory(
                entity_id, entity['class_name'], parsed
            )
            
            # Store in entity_captions cache
            self.entity_captions[entity_id] = formatted['caption']
            
            # If spatial memory exists, update it
            if self.spatial_memory and entity_id in self.spatial_memory.entities:
                # Append to captions list (not replace)
                if formatted['caption'] not in self.spatial_memory.entities[entity_id].captions:
                    self.spatial_memory.entities[entity_id].captions.append(formatted['caption'])
                    # Increment counter for statistics
                    self.spatial_memory.total_captions_generated += 1
                # Store attributes (as dict, not directly)
                if not hasattr(self.spatial_memory.entities[entity_id], 'semantic_attributes'):
                    self.spatial_memory.entities[entity_id].semantic_attributes = {}
                self.spatial_memory.entities[entity_id].semantic_attributes.update(formatted['attributes'])
        
        caption_time = time.time() - caption_start
        print(f"  ✓ Captioned {len(selected)} entities in {caption_time:.1f}s")
        print(f"  📊 Captions stored in spatial memory")
    
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
            
            # === Motion Detection for Adaptive Skip ===
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.prev_gray is not None:
                frame_diff = cv2.absdiff(gray, self.prev_gray)
                motion_score = np.mean(frame_diff)
                self.motion_history.append(motion_score)
                if len(self.motion_history) > self.motion_window:
                    self.motion_history.pop(0)
            self.prev_gray = gray.copy()
            
            # === Scene Classification ===
            scene_result = None
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
                
                # Store crop for query-time captioning (if enabled)
                if self.enable_crop_storage and crop.size > 0:
                    self._store_frame_crop(crop, self.frame_count, bbox, class_name)
                
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
                
                # === Scale Estimation (Phase 5: Absolute scale) ===
                # Feed known-size objects to scale estimator
                if not self.scale_estimator.is_locked():
                    # Extract depth ROI for this object
                    x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
                    x2_int, y2_int = min(self.frame_width, int(x2)), min(self.frame_height, int(y2))
                    depth_roi = depth_map[y1_int:y2_int, x1_int:x2_int]
                    
                    # Try to estimate scale from this object
                    estimate = self.scale_estimator.estimate_from_object(
                        bbox=(x1, y1, x2, y2),
                        depth_roi=depth_roi,
                        class_name=class_name,
                        frame_idx=self.frame_count
                    )
                    
                    if estimate:
                        self.scale_estimator.add_estimate(estimate)
                        
                        # Check if scale just locked
                        if self.scale_estimator.is_locked() and self.absolute_scale is None:
                            self.absolute_scale = self.scale_estimator.get_scale()
                            print(f"\n📐 ABSOLUTE SCALE LOCKED: {self.absolute_scale:.3f} m/unit")
                            print(f"   Real-world 3D coordinates now available!")
                
                # For SLAM
                object_detections.append({
                    'bbox': bbox,
                    'class': class_name,
                    'confidence': conf
                })
            
            # === SLAM Tracking (Phase 4) ===
            slam_pose = self.slam.track(frame, timestamp, self.frame_count, object_detections, depth_map)
            slam_stats = self.slam.get_statistics()
            
            # === ENHANCED Adaptive Skip (Motion + SLAM) ===
            if self.adaptive_skip and self.processed_frames > 20 and len(self.motion_history) >= 5:
                avg_motion = np.mean(self.motion_history)
                HIGH_MOTION = 15.0  # Fast movement
                LOW_MOTION = 5.0    # Static scene
                
                if slam_pose is None:
                    # SLAM failing - reduce skip aggressively
                    self.slam_failure_count += 1
                    self.slam_success_count = 0
                    if self.slam_failure_count >= 3 and self.skip_frames > self.min_skip:
                        self.skip_frames = max(self.min_skip, self.skip_frames - 2)
                        print(f"⚠️  SLAM failing → skip={self.skip_frames}")
                        self.slam_failure_count = 0
                
                elif avg_motion > HIGH_MOTION:
                    # High motion - reduce skip
                    if self.skip_frames > 4:
                        self.skip_frames = max(4, self.skip_frames - 1)
                        print(f"🏃 High motion ({avg_motion:.1f}) → skip={self.skip_frames} (~{30/self.skip_frames:.1f} fps)")
                    self.slam_success_count += 1
                    self.slam_failure_count = 0
                
                elif avg_motion < LOW_MOTION:
                    # Low motion - increase skip
                    self.slam_success_count += 1
                    self.slam_failure_count = 0
                    if self.slam_success_count >= 15 and self.skip_frames < self.max_skip:
                        self.skip_frames = min(self.max_skip, self.skip_frames + 3)
                        print(f"😴 Low motion ({avg_motion:.1f}) → skip={self.skip_frames} (~{30/self.skip_frames:.1f} fps)")
                        self.slam_success_count = 0
                
                else:
                    # Medium motion - gradual optimization
                    self.slam_success_count += 1
                    self.slam_failure_count = 0
                    if self.slam_success_count >= 25:
                        if self.skip_frames < 10:
                            self.skip_frames += 1
                        elif self.skip_frames > 15:
                            self.skip_frames -= 1
                        self.slam_success_count = 0
            
            # === Entity Tracking (Phase 2) ===
            tracks = self.tracker.track_frame(detections, self.frame_count, timestamp)
            
            # === FastVLM Semantic Enrichment (ULTRA-SELECTIVE TRIGGERING) ===
            # STRICT BUDGET: Max 8-10 captions total to stay under 66s
            # Only caption the MOST important semantic events
            if (self.enable_fastvlm and 
                len(self.entity_captions) < self.caption_budget and
                self.frame_count - self.last_captioned_frame >= 45):  # Min 45 frames gap (1.5s)
                
                self._ensure_fastvlm_loaded()
                caption_triggered = False
                
                # Priority 1: NEW SPATIAL ZONE (highest priority - context shift)
                new_zones = []
                for track in tracks:
                    if hasattr(track, 'zone_id') and track.zone_id is not None:
                        if track.zone_id not in self.zone_captions:
                            new_zones.append(track.zone_id)
                
                if new_zones and len(self.zone_captions) < 3:  # Max 3 zone captions
                    print(f"\n🗺️  New spatial zone {new_zones[0]}! Captioning context...")
                    scene_caption = self._generate_scene_caption(frame)
                    for zone_id in set(new_zones):
                        self.zone_captions[zone_id] = scene_caption
                    caption_triggered = True
                
                # Priority 2: FIRST occurrence of HIGH-VALUE object classes only
                # Focus on: person, laptop, phone, book, tv, monitor (user queries these)
                high_value_classes = {'person', 'laptop', 'cell phone', 'book', 'tv', 'monitor', 'keyboard', 'mouse'}
                
                if not caption_triggered:
                    for track in tracks:
                        if (track.most_likely_class in high_value_classes and 
                            track.most_likely_class not in self.captioned_classes):
                            
                            self.seen_classes.add(track.most_likely_class)
                            self.captioned_classes.add(track.most_likely_class)
                            
                            # Only caption if object is reasonably large (>3% of frame)
                            bbox_area = (track.bbox[2] - track.bbox[0]) * (track.bbox[3] - track.bbox[1])
                            frame_area = self.frame_width * self.frame_height
                            
                            if bbox_area / frame_area > 0.03:
                                print(f"\n🎯 HIGH-VALUE object: {track.most_likely_class}")
                                caption = self._generate_entity_caption(frame, track.bbox, track.entity_id)
                                if caption:
                                    print(f"  ✓ {caption[:60]}...")
                                caption_triggered = True
                                break  # Only ONE caption per trigger
                
                # Priority 3: Scene type change (only if haven't used budget)
                if (not caption_triggered and 
                    len(self.zone_captions) < 2 and 
                    scene_result):
                    
                    self.scene_type_history.append(scene_result.scene_type)
                    if len(self.scene_type_history) >= 2:
                        if self.scene_type_history[-1] != self.scene_type_history[-2]:
                            print(f"\n🎬 Major scene change: {self.scene_type_history[-2]} → {self.scene_type_history[-1]}")
                            scene_caption = self._generate_scene_caption(frame)
                            if scene_caption:
                                print(f"  ✓ {scene_caption[:60]}...")
                                # Store in zone captions for spatial context
                                current_zone = tracks[0].zone_id if tracks and hasattr(tracks[0], 'zone_id') else 0
                                self.zone_captions[current_zone] = scene_caption
                            caption_triggered = True
                
                if caption_triggered:
                    self.last_captioned_frame = self.frame_count
                    remaining = self.caption_budget - len(self.entity_captions)
                    print(f"  📊 Caption budget: {len(self.entity_captions)}/{self.caption_budget} used")
            
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
            
            # === Feed to Persistent Spatial Memory ===
            if self.spatial_memory:
                for track in tracks:
                    # Get caption if exists (from query-time FastVLM)
                    caption = self.entity_captions.get(track.entity_id)
                    
                    # Apply absolute scale if available
                    position_3d = track.centroid_3d_mm if hasattr(track, 'centroid_3d_mm') else None
                    if position_3d is not None and self.absolute_scale is not None:
                        # Convert from mm to meters, then apply scale
                        position_3d = (position_3d / 1000.0) * self.absolute_scale
                    
                    # Feed observation to memory system
                    self.spatial_memory.add_entity_observation(
                        entity_id=track.entity_id,
                        class_name=track.most_likely_class,
                        timestamp=timestamp,
                        position_3d=position_3d,
                        zone_id=track.zone_id if hasattr(track, 'zone_id') else None,
                        caption=caption,
                        confidence=track.confidence if hasattr(track, 'confidence') else 0.8
                    )
            
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
            
            # Get zones for logging
            zones = self.zone_manager.zones
            
            # === Rerun Logging ===
            if self.rerun_logger:
                avg_fps = np.mean(self.fps_history) if len(self.fps_history) > 0 else 0
                self.rerun_logger.log_frame(frame, self.frame_count)
                self.rerun_logger.log_depth(depth_map, self.frame_count, self.camera_intrinsics)
                self.rerun_logger.log_entities(tracks, self.frame_count, show_trajectories=True, show_velocities=True)
                if slam_pose is not None:
                    self.rerun_logger.log_slam_pose(slam_pose, self.frame_count, show_trajectory=True)
                if zones:
                    self.rerun_logger.log_zones(zones, self.frame_count, construct_meshes=True)
                self.rerun_logger.log_metrics(
                    frame_idx=self.frame_count,
                    fps=avg_fps,
                    num_entities=len(tracks),
                    num_zones=len(zones),
                    slam_poses=slam_stats.get('total_poses', 0),
                    loop_closures=slam_stats.get('loop_closures', 0)
                )
            
            # === OpenCV Display (only if not using Rerun) ===
            if not self.use_rerun:
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
            else:
                # Rerun mode - no pause/resume, runs continuously
                pass
        
        # Cleanup
        self.cap.release()
        if not self.use_rerun:
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
        
        # Scale estimation statistics
        scale_stats = self.scale_estimator.get_statistics()
        print(f"\n📏 Absolute Scale Recovery (Phase 5):")
        print(f"  Estimates collected: {scale_stats['total_estimates']}")
        print(f"  Scale locked: {'✓' if scale_stats['scale_locked'] else '✗'}")
        if scale_stats['committed_scale']:
            print(f"  Final scale: {scale_stats['committed_scale']:.3f} meters/unit")
            print(f"  Source objects: {', '.join(scale_stats['source_classes'])}")
        elif scale_stats['provisional_scale']:
            print(f"  Provisional scale: {scale_stats['provisional_scale']:.3f} meters/unit (not locked yet)")
        else:
            print(f"  ⚠️  No scale estimates (no known-size objects detected)")
        
        # Perform strategic captioning (Phase 2) - Post-processing
        if self.enable_strategic_captioning:
            self._perform_strategic_captioning()
        
        # Export to Memgraph if requested
        if self.export_to_memgraph:
            print(f"\n📊 Exporting to Memgraph...")
            self._export_to_memgraph()
        
        # Save persistent spatial memory
        if self.spatial_memory:
            print(f"\n💾 Saving persistent spatial memory...")
            self.spatial_memory.save()
            stats = self.spatial_memory.get_statistics()
            print(f"   ✓ Saved: {stats['total_entities']} entities with {stats['total_captions']} captions")
            print(f"   ✓ Total observations: {sum(e.observations_count for e in self.spatial_memory.entities.values())}")
            print(f"   ✓ Memory persisted to: {self.spatial_memory.memory_dir}")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Complete SLAM System (All Phases)')
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--skip', type=int, default=10, help='Frame skip (default: 10, ~3 FPS)')
    parser.add_argument('--max-frames', type=int, default=None, help='Max frames to process')
    parser.add_argument('--zone-mode', type=str, default='dense', choices=['dense', 'sparse'])
    parser.add_argument('--no-adaptive', action='store_true', help='Disable adaptive frame skip')
    parser.add_argument('--rerun', action='store_true', help='Use Rerun 3D visualization (disables OpenCV windows)')
    
    # Model selection
    parser.add_argument('--yolo-model', type=str, default='yolo11m', 
                        choices=['yolo11n', 'yolo11s', 'yolo11m', 'yolo11x'],
                        help='YOLO model variant (n=fastest, m=balanced, x=accurate)')
    
    # Performance presets
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: YOLO11n + skip=40 + no adaptive (real-time for 60s video)')
    
    # Graph database export
    parser.add_argument('--export-memgraph', action='store_true',
                        help='Export to Memgraph for real-time queries (FastVLM captions on-demand)')
    
    # Persistent spatial memory
    parser.add_argument('--use-spatial-memory', action='store_true',
                        help='Enable persistent spatial intelligence (remembers everything across sessions)')
    parser.add_argument('--memory-dir', type=str, default='memory/spatial_intelligence',
                        help='Directory for persistent spatial memory storage')
    
    # Strategic captioning (Phase 2)
    parser.add_argument('--enable-captions', action='store_true',
                        help='Enable strategic captioning of top 15 entities (post-processing, ~24s)')
    
    args = parser.parse_args()
    
    # Apply fast preset if requested
    if args.fast:
        print("⚡ FAST MODE enabled:")
        print("  - YOLO11n (fastest)")
        print("  - skip=40 (0.75 FPS sampling)")
        print("  - Adaptive skip disabled")
        print("  - Intelligent FastVLM (zone/novelty triggered)\n")
        args.yolo_model = 'yolo11n'
        args.skip = 40
        args.no_adaptive = True
    
    # Configure YOLO model before initialization
    model_manager = ModelManager.get_instance()
    model_manager.yolo_model_name = args.yolo_model
    
    system = CompleteSLAMSystem(
        args.video, 
        skip_frames=args.skip, 
        zone_mode=args.zone_mode,
        adaptive_skip=not args.no_adaptive,
        use_rerun=args.rerun
    )
    
    # Configure Memgraph export
    if args.export_memgraph:
        system.export_to_memgraph = True
        print("📊 Memgraph export enabled")
        print("   Data will be queryable with: python scripts/query_memgraph.py --interactive")
        print("   FastVLM captioning: ON-DEMAND at query time (keeps processing <66s)\n")
    
    # Configure persistent spatial memory
    if args.use_spatial_memory:
        try:
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from orion.graph.spatial_memory import SpatialMemorySystem
            
            system.spatial_memory = SpatialMemorySystem(memory_dir=Path(args.memory_dir))
            print("🧠 Persistent spatial memory enabled")
            print(f"   Memory dir: {args.memory_dir}")
            
            # Show existing memory stats if any
            stats = system.spatial_memory.get_statistics()
            if stats['total_entities'] > 0:
                print(f"   Loaded: {stats['total_entities']} entities, {stats['total_captions']} captions")
            else:
                print("   Starting fresh memory\n")
        except ImportError as e:
            print(f"⚠️  Could not load spatial memory system: {e}")
            system.spatial_memory = None
    else:
        system.spatial_memory = None
    
    # Configure strategic captioning (Phase 2)
    if args.enable_captions:
        system.enable_strategic_captioning = True
        print("🎨 Strategic captioning enabled")
        print(f"   Will caption top {system.caption_budget} entities after processing")
        print(f"   Estimated overhead: ~{system.strategic_captioner.estimate_captioning_time(system.caption_budget):.0f}s\n")
    
    system.run(max_frames=args.max_frames)


if __name__ == '__main__':
    main()
