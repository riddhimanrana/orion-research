#!/usr/bin/env python3
"""
Rerun Visualization for Super Accurate Mode
============================================

Comprehensive 3D visualization showing:
- Camera trajectory through space
- 3D bounding boxes in world coordinates
- Depth maps as point clouds
- Instance masks overlaid
- Scene graph relationships (lines between objects)
- Camera intrinsics/extrinsics
- Rich semantic labels
- CIS scores and reasoning

Author: Orion Research
Date: November 10, 2025
"""

import numpy as np
import rerun as rr
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import cv2


@dataclass
class CameraState:
    """Camera state for visualization"""
    position: np.ndarray  # (3,) world position
    rotation: np.ndarray  # (3, 3) rotation matrix
    intrinsics: Dict[str, float]  # fx, fy, cx, cy
    frame_idx: int


class RerunSuperAccurateVisualizer:
    """
    Comprehensive Rerun visualization for Super Accurate Mode
    
    Features:
    - 3D scene reconstruction with point clouds
    - Camera trajectory visualization
    - 3D bounding boxes in world space
    - Depth maps and masks
    - Scene graph relationships
    - Rich semantic labels with confidence
    - CIS reasoning annotations
    """
    
    def __init__(self, session_name: str = "super_accurate"):
        """Initialize Rerun session"""
        rr.init(session_name, spawn=True)
        
        # Cumulative state
        self.camera_positions = []
        self.world_objects = {}  # Track objects in world space
        
        # Color palette for objects
        self.colors = self._generate_colors(80)  # 80 COCO classes
        
        print(f"✓ Rerun visualization initialized: {session_name}")
    
    def _generate_colors(self, n: int) -> List[tuple]:
        """Generate distinct colors for classes"""
        colors = []
        for i in range(n):
            hue = (i * 137.5) % 360  # Golden angle
            sat = 0.7 + (i % 3) * 0.1
            val = 0.8 + (i % 2) * 0.1
            
            # HSV to RGB
            c = val * sat
            x = c * (1 - abs((hue / 60) % 2 - 1))
            m = val - c
            
            if hue < 60:
                r, g, b = c, x, 0
            elif hue < 120:
                r, g, b = x, c, 0
            elif hue < 180:
                r, g, b = 0, c, x
            elif hue < 240:
                r, g, b = 0, x, c
            elif hue < 300:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            colors.append((
                int((r + m) * 255),
                int((g + m) * 255),
                int((b + m) * 255)
            ))
        return colors
    
    def log_frame(self, 
                  frame: np.ndarray,
                  frame_idx: int,
                  detections: List[Dict],
                  depth_map: np.ndarray,
                  camera_state: CameraState,
                  scene_graph: List[Dict],
                  motion_desc: Optional[str] = None,
                  diagnostics: Optional[Dict] = None):
        """
        Log complete frame to Rerun
        
        Args:
            frame: RGB frame (H, W, 3)
            frame_idx: Frame index
            detections: List of detection dicts with bbox, class, score, depth, etc
            depth_map: Depth map (H, W) in meters
            camera_state: Camera position/rotation/intrinsics
            scene_graph: List of spatial relationships
            motion_desc: Camera motion description
            diagnostics: CIS scores, reasoning, etc.
        """
        time_ns = frame_idx
        
        # 1. Log RGB frame
        rr.set_time_sequence("frame", frame_idx)
        rr.log("camera/rgb", rr.Image(frame))
        
        # 2. Log depth map as image and point cloud
        self._log_depth(depth_map, frame, time_ns)
        
        # 3. Log camera pose
        self._log_camera_pose(camera_state, time_ns)
        
        # 4. Log 2D detections on image
        self._log_2d_detections(detections, frame.shape, time_ns)
        
        # 5. Log 3D bounding boxes in world space
        self._log_3d_boxes(detections, depth_map, camera_state, time_ns)
        
        # 6. Log scene graph relationships
        self._log_scene_graph(scene_graph, detections, time_ns)
        
        # 7. Log camera motion
        if motion_desc:
            rr.log("diagnostics/camera_motion", rr.TextLog(motion_desc))
        
        # 8. Log diagnostics (CIS, reasoning, etc.)
        if diagnostics:
            self._log_diagnostics(diagnostics, time_ns)
    
    def _log_depth(self, depth_map: np.ndarray, rgb_frame: np.ndarray, time_ns: int):
        """Log depth as image and point cloud"""
        # Depth image (normalized for visualization) - FIXED: handle NaN/Inf
        depth_safe = depth_map.copy()
        
        # Remove NaN and Inf values
        depth_safe = np.nan_to_num(depth_safe, nan=0.0, posinf=10.0, neginf=0.0)
        
        # Clip to realistic indoor range (0.25m to 10m)
        depth_safe = np.clip(depth_safe, 0.25, 10.0)
        
        # Check if depth has valid data
        if depth_safe.max() > depth_safe.min():
            # Normalize to [0, 1]
            depth_vis = (depth_safe - depth_safe.min()) / (depth_safe.max() - depth_safe.min() + 1e-8)
            depth_vis = (depth_vis * 255).astype(np.uint8)
            
            # Apply colormap (TURBO: blue=close, red=far)
            depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)  # Fix BGR->RGB
            
            rr.log("camera/depth", rr.Image(depth_colored))
            
            # Log depth statistics for debugging
            rr.log("diagnostics/depth_stats",
                   rr.TextLog(f"Depth: min={depth_safe.min():.2f}m, max={depth_safe.max():.2f}m, "
                             f"mean={depth_safe.mean():.2f}m, valid={((depth_safe > 0.25) & (depth_safe < 10.0)).sum() / depth_safe.size * 100:.1f}%"))
        else:
            # No valid depth - log warning
            rr.log("camera/depth", rr.Image(np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)))
            rr.log("diagnostics/depth_stats", rr.TextLog("⚠️ No valid depth data"))
        
        # Use cleaned depth for point cloud
        depth_map = depth_safe
        
        # Point cloud (sample for performance)
        h, w = depth_map.shape
        step = 4  # Sample every 4 pixels
        
        y_coords, x_coords = np.meshgrid(
            np.arange(0, h, step),
            np.arange(0, w, step),
            indexing='ij'
        )
        
        # Get depth and colors
        depths = depth_map[y_coords, x_coords]
        colors = rgb_frame[y_coords, x_coords]
        
        # Filter valid depths
        valid = (depths > 0.1) & (depths < 10.0)
        x_coords = x_coords[valid]
        y_coords = y_coords[valid]
        depths = depths[valid]
        colors = colors[valid]
        
        # Convert to 3D points (camera coordinates)
        # Assume simple pinhole model
        fx = fy = 500  # Approximate
        cx, cy = w / 2, h / 2
        
        X = (x_coords - cx) * depths / fx
        Y = (y_coords - cy) * depths / fy
        Z = depths
        
        points = np.stack([X, Y, Z], axis=-1)
        
        # Log point cloud
        rr.log("world/point_cloud",
               rr.Points3D(points, colors=colors, radii=0.01))
    
    def _log_camera_pose(self, camera_state: CameraState, time_ns: int):
        """Log camera pose in world space"""
        # Camera position
        pos = camera_state.position
        rot = camera_state.rotation
        
        # Store trajectory
        self.camera_positions.append(pos.copy())
        
        # Log camera transform
        # Rerun expects row-major 3x4 transform matrix
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = pos
        
        rr.log("world/camera",
               rr.Transform3D(translation=pos, mat3x3=rot))
        
        # Log camera trajectory
        if len(self.camera_positions) > 1:
            trajectory = np.array(self.camera_positions)
            rr.log("world/camera_trajectory",
                   rr.LineStrips3D([trajectory], colors=[(0, 255, 0)]))
        
        # Log camera frustum
        intrinsics = camera_state.intrinsics
        fx = intrinsics.get('fx', 500)
        fy = intrinsics.get('fy', 500)
        cx = intrinsics.get('cx', 320)
        cy = intrinsics.get('cy', 240)
        
        rr.log("world/camera/image",
               rr.Pinhole(
                   resolution=[640, 480],
                   focal_length=[fx, fy],
                   principal_point=[cx, cy]
               ))
        
        # Log intrinsics as text
        rr.log("diagnostics/camera_intrinsics",
               rr.TextLog(f"fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}"))
    
    def _log_2d_detections(self, detections: List[Dict], frame_shape: tuple, time_ns: int):
        """Log 2D bounding boxes on image"""
        if not detections:
            return
        
        boxes = []
        labels = []
        colors = []
        
        for det in detections:
            bbox = det['bbox']  # [x1, y1, x2, y2]
            class_name = det.get('rich_description', det['class'])
            score = det['score']
            depth = det.get('depth', 0.0)
            
            # Box in [x_min, y_min, x_max, y_max] format
            boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            labels.append(f"{class_name} ({score:.2f}) @ {depth:.2f}m")
            
            # Color by class
            class_id = hash(det['class']) % len(self.colors)
            colors.append(self.colors[class_id])
        
        # Log all boxes
        rr.log("camera/detections",
               rr.Boxes2D(
                   array=boxes,
                   array_format=rr.Box2DFormat.XYXY,
                   labels=labels,
                   colors=colors
               ))
    
    def _log_3d_boxes(self, 
                      detections: List[Dict],
                      depth_map: np.ndarray,
                      camera_state: CameraState,
                      time_ns: int):
        """Log 3D bounding boxes in world space"""
        if not detections:
            return
        
        intrinsics = camera_state.intrinsics
        fx = intrinsics.get('fx', 500)
        fy = intrinsics.get('fy', 500)
        cx = intrinsics.get('cx', 320)
        cy = intrinsics.get('cy', 240)
        
        for i, det in enumerate(detections):
            bbox = det['bbox']  # [x1, y1, x2, y2]
            depth = det.get('depth', 1.0)
            
            # Get 3D position (center of bbox at depth)
            x1, y1, x2, y2 = bbox
            cx_box = (x1 + x2) / 2
            cy_box = (y1 + y2) / 2
            
            # Backproject to 3D (camera coordinates)
            X = (cx_box - cx) * depth / fx
            Y = (cy_box - cy) * depth / fy
            Z = depth
            
            pos_cam = np.array([X, Y, Z])
            
            # Transform to world coordinates
            pos_world = camera_state.rotation @ pos_cam + camera_state.position
            
            # Estimate box size (assume height/width from 2D bbox)
            width = (x2 - x1) * depth / fx
            height = (y2 - y1) * depth / fy
            depth_size = min(width, height) * 0.5  # Rough estimate
            
            half_size = np.array([width/2, height/2, depth_size/2])
            
            # Log 3D box
            class_name = det.get('rich_description', det['class'])
            score = det['score']
            
            rr.log(f"world/objects/{class_name}_{i}",
                   rr.Boxes3D(
                       half_sizes=[half_size],
                       centers=[pos_world],
                       labels=[f"{class_name} ({score:.2f})"],
                       colors=[self.colors[hash(det['class']) % len(self.colors)]]
                   ))
    
    def _log_scene_graph(self, scene_graph: List[Dict], detections: List[Dict], time_ns: int):
        """Log scene graph relationships as lines"""
        if not scene_graph or not detections:
            return
        
        # Build position lookup
        positions = []
        for det in detections:
            bbox = det['bbox']
            positions.append([
                (bbox[0] + bbox[2]) / 2,
                (bbox[1] + bbox[3]) / 2
            ])
        
        # Log relationships
        for rel in scene_graph:
            subj_id = rel['subject']
            obj_id = rel['object']
            predicate = rel['predicate']
            confidence = rel['confidence']
            
            if subj_id < len(positions) and obj_id < len(positions):
                # Draw line in 2D view
                p1 = positions[subj_id]
                p2 = positions[obj_id]
                
                rr.log(f"camera/scene_graph/{subj_id}_{obj_id}",
                       rr.LineStrips2D(
                           [[p1, p2]],
                           labels=[f"{predicate} ({confidence:.2f})"],
                           colors=[(255, 255, 0)]
                       ))
    
    def _log_diagnostics(self, diagnostics: Dict, time_ns: int):
        """Log diagnostic information (CIS, reasoning, etc.)"""
        # CIS scores
        if 'cis_scores' in diagnostics:
            cis_text = "CIS Scores:\n"
            for obj_id, score in diagnostics['cis_scores'].items():
                cis_text += f"  Object {obj_id}: {score:.3f}\n"
            rr.log("diagnostics/cis_scores", rr.TextLog(cis_text))
        
        # Reasoning
        if 'reasoning' in diagnostics:
            rr.log("diagnostics/reasoning", 
                   rr.TextLog(diagnostics['reasoning']))
        
        # Performance stats
        if 'timing' in diagnostics:
            timing_text = "Frame Timing:\n"
            for component, time_ms in diagnostics['timing'].items():
                timing_text += f"  {component}: {time_ms:.1f}ms\n"
            rr.log("diagnostics/timing", rr.TextLog(timing_text))
        
        # Quality metrics
        if 'quality' in diagnostics:
            quality_text = "Quality Metrics:\n"
            for metric, value in diagnostics['quality'].items():
                quality_text += f"  {metric}: {value}\n"
            rr.log("diagnostics/quality", rr.TextLog(quality_text))
    
    def log_summary(self, summary: Dict):
        """Log final summary statistics"""
        summary_text = "=== PROCESSING SUMMARY ===\n"
        summary_text += f"Total frames: {summary.get('total_frames', 0)}\n"
        summary_text += f"Avg FPS: {summary.get('avg_fps', 0):.2f}\n"
        summary_text += f"Total objects detected: {summary.get('total_objects', 0)}\n"
        summary_text += f"Avg objects per frame: {summary.get('avg_objects_per_frame', 0):.1f}\n"
        
        if 'component_timing' in summary:
            summary_text += "\nComponent Timing (avg):\n"
            for comp, time_ms in summary['component_timing'].items():
                summary_text += f"  {comp}: {time_ms:.1f}ms\n"
        
        rr.log("summary", rr.TextLog(summary_text))
        print(summary_text)
