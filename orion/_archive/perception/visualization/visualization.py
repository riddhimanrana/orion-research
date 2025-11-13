"""
Phase 2: Comprehensive Visualization System for Tracking + 3D Perception

Provides rich visualizations including:
- Tracking overlays (bounding boxes, IDs, trajectories)
- Depth heatmaps (colorized depth visualization)
- Distance annotations (3D distances to camera)
- Class belief distributions (Bayesian posterior bars)
- Motion vectors (velocity arrows)
- Occlusion indicators
- Re-identification highlights

Author: Orion Research Team
Date: November 2025
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import colorsys

from orion.perception.tracking import BayesianEntityBelief


@dataclass
class VisualizationConfig:
    """Configuration for visualization rendering."""
    
    # Display options
    show_bboxes: bool = True
    show_ids: bool = True
    show_class_labels: bool = True
    show_distances: bool = True
    show_trajectories: bool = True
    show_velocities: bool = True
    show_belief_bars: bool = True
    show_depth_heatmap: bool = True
    show_occlusion: bool = True
    show_reid_highlights: bool = True
    show_offscreen_banner: bool = True  # NEW: Show banner for off-screen tracks
    show_spatial_map: bool = True  # NEW: Show separate spatial map window
    
    # Trajectory settings
    trajectory_length: int = 30  # frames
    trajectory_thickness: int = 2
    
    # Visual styling
    bbox_thickness: int = 2
    text_scale: float = 0.6
    text_thickness: int = 2
    
    # Color scheme
    active_track_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    disappeared_track_color: Tuple[int, int, int] = (128, 128, 128)  # Gray
    reidentified_color: Tuple[int, int, int] = (255, 165, 0)  # Orange
    offscreen_color: Tuple[int, int, int] = (100, 100, 255)  # Blue for off-screen
    
    # Heatmap
    depth_colormap: int = cv2.COLORMAP_TURBO
    depth_min_mm: float = 500.0
    depth_max_mm: float = 5000.0
    heatmap_alpha: float = 0.4  # Blend with original frame
    
    # Spatial map settings
    spatial_map_size: Tuple[int, int] = (400, 400)  # Width, height
    spatial_map_range_mm: float = 3000.0  # Show +/- 3 meters


class TrackingVisualizer:
    """
    Comprehensive visualization system for tracking and 3D perception.
    
    Renders multi-layered overlays on video frames showing tracking state,
    3D perception, motion, and belief distributions.
    """
    
    def __init__(self, config: VisualizationConfig = None):
        """
        Initialize visualizer.
        
        Args:
            config: Visualization configuration (uses defaults if None)
        """
        self.config = config or VisualizationConfig()
        
        # Trajectory history: {entity_id: deque of centroids}
        self.trajectories: Dict[int, deque] = {}
        
        # Color palette for entity IDs (HSV-based for distinctiveness)
        self.id_colors: Dict[int, Tuple[int, int, int]] = {}
        
        # Re-ID highlights (flash for a few frames)
        self.reid_highlights: Dict[int, int] = {}  # {entity_id: frames_remaining}
        
        # Frame dimensions (set on first frame)
        self.frame_width = 0
        self.frame_height = 0
    
    def visualize_frame(
        self,
        frame: np.ndarray,
        tracks: List[BayesianEntityBelief],
        depth_map: Optional[np.ndarray] = None,
        frame_number: int = 0
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create comprehensive visualization overlay on frame.
        
        Args:
            frame: Input video frame (H, W, 3)
            tracks: List of entity tracks to visualize
            depth_map: Optional depth map for heatmap overlay (H, W)
            frame_number: Current frame number
        
        Returns:
            Tuple of (main_visualization, spatial_map)
            - main_visualization: Frame with overlays
        Returns:
            Tuple of (main_visualization, spatial_map)
            - main_visualization: Frame with overlays
            - spatial_map: Separate top-down spatial map (or None if disabled)
        """
        # Store frame dimensions
        if self.frame_height == 0:
            self.frame_height, self.frame_width = frame.shape[:2]
        
        # Separate on-screen vs off-screen tracks
        onscreen_tracks, offscreen_tracks = self._separate_tracks(tracks)
        
        vis_frame = frame.copy()
        
        # Clean up trajectories for entities not in current tracks
        active_ids = {track.entity_id for track in tracks}
        stale_ids = set(self.trajectories.keys()) - active_ids
        for stale_id in stale_ids:
            del self.trajectories[stale_id]
        
        # Layer 1: Depth heatmap (if available and enabled)
        if self.config.show_depth_heatmap and depth_map is not None:
            vis_frame = self._overlay_depth_heatmap(vis_frame, depth_map)
        
        # Layer 2: Trajectories (before bboxes so they appear behind) - ONLY for on-screen
        if self.config.show_trajectories:
            vis_frame = self._draw_trajectories(vis_frame, onscreen_tracks)
        
        # Layer 3: Bounding boxes and entity info - ONLY for on-screen tracks
        for track in onscreen_tracks:
            # Get or assign color for this entity
            if track.entity_id not in self.id_colors:
                self.id_colors[track.entity_id] = self._get_unique_color(track.entity_id)
            
            color = self.id_colors[track.entity_id]
            
            # Check if recently re-identified (highlight in orange)
            is_reid = track.entity_id in self.reid_highlights
            if is_reid:
                color = self.config.reidentified_color
                self.reid_highlights[track.entity_id] -= 1
                if self.reid_highlights[track.entity_id] <= 0:
                    del self.reid_highlights[track.entity_id]
            
            # Draw bbox
            if self.config.show_bboxes:
                vis_frame = self._draw_bbox(vis_frame, track, color)
            
            # Draw ID label
            if self.config.show_ids:
                vis_frame = self._draw_id_label(vis_frame, track, color)
            
            # Draw class label
            if self.config.show_class_labels:
                vis_frame = self._draw_class_label(vis_frame, track, color)
            
            # Draw distance (if 3D available)
            if self.config.show_distances and track.centroid_3d_mm is not None:
                vis_frame = self._draw_distance(vis_frame, track, color)
            
            # Draw velocity vector
            if self.config.show_velocities:
                vis_frame = self._draw_velocity(vis_frame, track, color)
            
            # Draw belief distribution bars
            if self.config.show_belief_bars:
                vis_frame = self._draw_belief_bars(vis_frame, track, color)
            
            # Update trajectory history
            self._update_trajectory(track)
        
        # Layer 4: Off-screen tracks banner (top)
        if self.config.show_offscreen_banner and offscreen_tracks:
            vis_frame = self._draw_offscreen_banner(vis_frame, offscreen_tracks)
        
        # Layer 5: Frame info overlay (top-left corner)
        vis_frame = self._draw_frame_info(vis_frame, frame_number, len(onscreen_tracks), len(offscreen_tracks))
        
        # Create spatial map (separate window)
        spatial_map = None
        if self.config.show_spatial_map:
            spatial_map = self._create_spatial_map(tracks, depth_map)
        
        return vis_frame, spatial_map
    
    def _overlay_depth_heatmap(self, frame: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """Overlay colorized depth map on frame."""
        # Normalize depth to [0, 255]
        depth_normalized = np.clip(
            (depth_map - self.config.depth_min_mm) / (self.config.depth_max_mm - self.config.depth_min_mm),
            0, 1
        )
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        
        # Resize depth map to match frame size if needed
        if depth_uint8.shape[:2] != frame.shape[:2]:
            depth_uint8 = cv2.resize(depth_uint8, (frame.shape[1], frame.shape[0]))
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_uint8, self.config.depth_colormap)
        
        # Blend with original frame
        blended = cv2.addWeighted(frame, 1 - self.config.heatmap_alpha, 
                                 depth_colored, self.config.heatmap_alpha, 0)
        
        return blended
    
    def _draw_bbox(self, frame: np.ndarray, track: BayesianEntityBelief, 
                   color: Tuple[int, int, int]) -> np.ndarray:
        """Draw bounding box for entity."""
        x1, y1, x2, y2 = track.bbox.astype(int)
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.config.bbox_thickness)
        
        # Draw filled corner markers for better visibility
        corner_size = 10
        cv2.rectangle(frame, (x1, y1), (x1 + corner_size, y1 + corner_size), color, -1)
        cv2.rectangle(frame, (x2 - corner_size, y1), (x2, y1 + corner_size), color, -1)
        cv2.rectangle(frame, (x1, y2 - corner_size), (x1 + corner_size, y2), color, -1)
        cv2.rectangle(frame, (x2 - corner_size, y2 - corner_size), (x2, y2), color, -1)
        
        return frame
    
    def _draw_id_label(self, frame: np.ndarray, track: BayesianEntityBelief,
                      color: Tuple[int, int, int]) -> np.ndarray:
        """Draw entity ID above bbox."""
        x1, y1, _, _ = track.bbox.astype(int)
        
        # ID text
        id_text = f"ID:{track.entity_id}"
        
        # Re-ID indicator
        if track.reidentified_times > 0:
            id_text += f" (ReID:{track.reidentified_times})"
        
        # Text background for readability
        (text_w, text_h), _ = cv2.getTextSize(
            id_text, cv2.FONT_HERSHEY_SIMPLEX, 
            self.config.text_scale, self.config.text_thickness
        )
        
        # Draw filled rectangle background
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), 
                     color, -1)
        
        # Draw text
        cv2.putText(frame, id_text, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, self.config.text_scale,
                   (255, 255, 255), self.config.text_thickness)
        
        return frame
    
    def _draw_class_label(self, frame: np.ndarray, track: BayesianEntityBelief,
                         color: Tuple[int, int, int]) -> np.ndarray:
        """Draw most likely class label."""
        x1, y1, _, _ = track.bbox.astype(int)
        
        # Get top probability
        prob = track.class_posterior.get(track.most_likely_class, 0.0) * 100
        
        class_text = f"{track.most_likely_class}: {prob:.1f}%"
        
        # Position below ID label
        y_offset = y1 - 35
        
        # Text background
        (text_w, text_h), _ = cv2.getTextSize(
            class_text, cv2.FONT_HERSHEY_SIMPLEX,
            self.config.text_scale * 0.8, self.config.text_thickness - 1
        )
        
        cv2.rectangle(frame, (x1, y_offset - text_h - 5), 
                     (x1 + text_w + 10, y_offset), color, -1)
        
        cv2.putText(frame, class_text, (x1 + 5, y_offset - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, self.config.text_scale * 0.8,
                   (255, 255, 255), self.config.text_thickness - 1)
        
        return frame
    
    def _draw_distance(self, frame: np.ndarray, track: BayesianEntityBelief,
                      color: Tuple[int, int, int]) -> np.ndarray:
        """Draw 3D distance to camera."""
        if track.centroid_3d_mm is None:
            return frame
        
        # Calculate distance from camera (assuming camera at origin)
        distance_m = np.linalg.norm(track.centroid_3d_mm) / 1000.0
        
        # Position at bottom of bbox
        x1, _, _, y2 = track.bbox.astype(int)
        
        dist_text = f"{distance_m:.2f}m"
        
        # Text background
        (text_w, text_h), _ = cv2.getTextSize(
            dist_text, cv2.FONT_HERSHEY_SIMPLEX,
            self.config.text_scale, self.config.text_thickness
        )
        
        cv2.rectangle(frame, (x1, y2 + 5), (x1 + text_w + 10, y2 + text_h + 15),
                     color, -1)
        
        cv2.putText(frame, dist_text, (x1 + 5, y2 + text_h + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, self.config.text_scale,
                   (255, 255, 255), self.config.text_thickness)
        
        return frame
    
    def _draw_velocity(self, frame: np.ndarray, track: BayesianEntityBelief,
                      color: Tuple[int, int, int]) -> np.ndarray:
        """Draw velocity vector as arrow."""
        if np.linalg.norm(track.velocity_2d) < 1.0:
            return frame  # Too small to visualize
        
        # Start at centroid
        start_point = track.centroid_2d.astype(int)
        
        # End point scaled by velocity (scale for visibility)
        scale = 3.0
        end_point = (start_point + track.velocity_2d * scale).astype(int)
        
        # Draw arrow
        cv2.arrowedLine(frame, tuple(start_point), tuple(end_point),
                       color, self.config.trajectory_thickness, tipLength=0.3)
        
        return frame
    
    def _draw_belief_bars(self, frame: np.ndarray, track: BayesianEntityBelief,
                         color: Tuple[int, int, int]) -> np.ndarray:
        """Draw horizontal bars showing class belief distribution (top 3 classes)."""
        if not track.class_posterior:
            return frame
        
        # Get top 3 classes
        sorted_beliefs = sorted(track.class_posterior.items(), 
                              key=lambda x: x[1], reverse=True)[:3]
        
        # Position to right of bbox
        _, y1, x2, _ = track.bbox.astype(int)
        
        bar_width = 100
        bar_height = 15
        x_offset = x2 + 10
        y_offset = y1
        
        for i, (class_name, prob) in enumerate(sorted_beliefs):
            y_pos = y_offset + i * (bar_height + 5)
            
            # Background bar
            cv2.rectangle(frame, (x_offset, y_pos), 
                         (x_offset + bar_width, y_pos + bar_height),
                         (50, 50, 50), -1)
            
            # Filled bar (proportional to probability)
            filled_width = int(bar_width * prob)
            cv2.rectangle(frame, (x_offset, y_pos),
                         (x_offset + filled_width, y_pos + bar_height),
                         color, -1)
            
            # Label
            label = f"{class_name[:8]}: {prob*100:.0f}%"
            cv2.putText(frame, label, (x_offset + bar_width + 5, y_pos + bar_height - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def _draw_trajectories(self, frame: np.ndarray, 
                          tracks: List[BayesianEntityBelief]) -> np.ndarray:
        """Draw trajectory trails for all entities."""
        for track in tracks:
            if track.entity_id not in self.trajectories:
                continue
            
            trajectory = self.trajectories[track.entity_id]
            if len(trajectory) < 2:
                continue
            
            color = self.id_colors.get(track.entity_id, (255, 255, 255))
            
            # Draw lines connecting trajectory points
            points = np.array(list(trajectory), dtype=np.int32)
            
            # Gradient alpha for fade effect (older points more transparent)
            for i in range(len(points) - 1):
                alpha = i / len(points)  # 0 (old) to 1 (recent)
                thickness = max(1, int(self.config.trajectory_thickness * alpha))
                
                cv2.line(frame, tuple(points[i]), tuple(points[i + 1]),
                        color, thickness, cv2.LINE_AA)
        
        return frame
    
    def _update_trajectory(self, track: BayesianEntityBelief):
        """Add current position to trajectory history."""
        if track.entity_id not in self.trajectories:
            self.trajectories[track.entity_id] = deque(
                maxlen=self.config.trajectory_length
            )
        
        self.trajectories[track.entity_id].append(track.centroid_2d.copy())
    
    def _separate_tracks(self, tracks: List[BayesianEntityBelief]) -> Tuple[List[BayesianEntityBelief], List[BayesianEntityBelief]]:
        """
        Separate tracks into on-screen and off-screen based on bbox position.
        
        Args:
            tracks: All active tracks
        
        Returns:
            Tuple of (onscreen_tracks, offscreen_tracks)
        """
        onscreen = []
        offscreen = []
        
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            
            # Check if bbox is within frame bounds (with some margin)
            margin = 50
            is_onscreen = (
                x2 > -margin and x1 < self.frame_width + margin and
                y2 > -margin and y1 < self.frame_height + margin
            )
            
            if is_onscreen:
                onscreen.append(track)
            else:
                offscreen.append(track)
        
        return onscreen, offscreen
    
    def _draw_offscreen_banner(self, frame: np.ndarray, 
                               offscreen_tracks: List[BayesianEntityBelief]) -> np.ndarray:
        """
        Draw banner at top showing off-screen/hidden tracks.
        
        Args:
            frame: Frame to draw on
            offscreen_tracks: List of off-screen tracks
        
        Returns:
            Frame with banner overlay
        """
        if not offscreen_tracks:
            return frame
        
        # Banner background (semi-transparent black)
        banner_height = 50
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, banner_height), 
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        
        # Title
        title = f"Off-screen Tracks ({len(offscreen_tracks)}):"
        cv2.putText(frame, title, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # List entities with direction hints
        x_offset = 250
        for track in offscreen_tracks[:8]:  # Show max 8
            # Get direction hint
            cx, cy = track.centroid_2d
            direction = self._get_direction_hint(cx, cy)
            
            # Get color
            if track.entity_id not in self.id_colors:
                self.id_colors[track.entity_id] = self._get_unique_color(track.entity_id)
            color = self.id_colors[track.entity_id]
            
            # Draw colored dot
            cv2.circle(frame, (x_offset, 25), 8, color, -1)
            
            # Draw ID and direction
            text = f"ID{track.entity_id} {direction}"
            cv2.putText(frame, text, (x_offset + 15, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            x_offset += 120
            
            if x_offset > self.frame_width - 100:
                break
        
        return frame
    
    def _get_direction_hint(self, cx: float, cy: float) -> str:
        """Get direction hint for off-screen entity (e.g., '↑', '←', '↓→')."""
        directions = []
        
        # Vertical
        if cy < 0:
            directions.append("↑")
        elif cy > self.frame_height:
            directions.append("↓")
        
        # Horizontal
        if cx < 0:
            directions.append("←")
        elif cx > self.frame_width:
            directions.append("→")
        
        return "".join(directions) if directions else "?"
    
    def _create_spatial_map(self, tracks: List[BayesianEntityBelief], 
                           depth_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create top-down spatial map showing entity positions.
        Improved with accurate coordinate transformation and grid.
        
        Args:
            tracks: All tracks to show
            depth_map: Optional depth map (for context)
        
        Returns:
            Spatial map image (bird's eye view)
        """
        map_w, map_h = self.config.spatial_map_size
        spatial_map = np.zeros((map_h, map_w, 3), dtype=np.uint8)
        
        # Background color
        spatial_map[:] = (30, 30, 30)
        
        # Draw reference grid (1m intervals)
        range_mm = self.config.spatial_map_range_mm
        range_m = range_mm / 1000.0
        center_x, center_y = map_w // 2, map_h - 30
        pixels_per_meter = (map_h - 50) / range_m
        
        # Draw horizontal grid lines (depth, every 1m)
        for i in range(int(-range_m), int(range_m) + 1):
            if i == 0:
                continue
            y = int(center_y - i * pixels_per_meter)
            if 10 < y < map_h - 40:
                cv2.line(spatial_map, (10, y), (map_w - 10, y), 
                        (60, 60, 60), 1)
                cv2.putText(spatial_map, f"{i}m", (5, y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        
        # Draw vertical grid lines (left-right, every 1m)
        for i in range(int(-range_m), int(range_m) + 1):
            if i == 0:
                continue
            x = int(center_x + i * pixels_per_meter)
            if 10 < x < map_w - 10:
                cv2.line(spatial_map, (x, 10), (x, map_h - 40), 
                        (60, 60, 60), 1)
        
        # Draw axes (stronger lines at 0)
        cv2.line(spatial_map, (center_x, 10), (center_x, map_h - 40), 
                (80, 80, 80), 2)  # Z-axis (depth)
        cv2.line(spatial_map, (10, center_y), (map_w - 10, center_y), 
                (80, 80, 80), 2)  # X-axis (left-right)
        
        # Draw center crosshair (camera position)
        cv2.circle(spatial_map, (center_x, center_y), 10, (255, 255, 255), 2)
        cv2.line(spatial_map, (center_x - 15, center_y), (center_x + 15, center_y), (255, 255, 255), 2)
        cv2.line(spatial_map, (center_x, center_y - 15), (center_x, center_y + 15), (255, 255, 255), 2)
        cv2.putText(spatial_map, "Camera", (center_x - 30, center_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw entities with 3D positions
        for track in tracks:
            if track.centroid_3d_mm is None:
                continue
            
            x_mm, y_mm, z_mm = track.centroid_3d_mm
            
            # Convert mm to meters
            x_m = x_mm / 1000.0
            z_m = z_mm / 1000.0
            
            # Convert to map coordinates (top-down orthographic projection)
            # X (left-right) → map X: right is positive
            # Z (depth) → map Y: forward is positive, up on map
            map_x = int(center_x + x_m * pixels_per_meter)
            map_y = int(center_y - z_m * pixels_per_meter)
            
            # Only draw if within reasonable bounds
            if not (0 <= map_x < map_w and 0 <= map_y < map_h):
                continue
            
            # Get color
            if track.entity_id not in self.id_colors:
                self.id_colors[track.entity_id] = self._get_unique_color(track.entity_id)
            color = self.id_colors[track.entity_id]
            
            # Draw entity with size based on distance (farther = smaller)
            radius = max(6, int(12 * (1.0 - min(abs(z_m) / range_m, 0.8))))
            cv2.circle(spatial_map, (map_x, map_y), radius, color, -1)
            cv2.circle(spatial_map, (map_x, map_y), radius, (255, 255, 255), 1)
            
            # Draw ID
            cv2.putText(spatial_map, str(track.entity_id), 
                       (map_x - 6, map_y + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            # Draw line to camera (for depth indication)
            cv2.line(spatial_map, (map_x, map_y), (center_x, center_y), 
                    color, 1, cv2.LINE_AA)
        
        # Title and info
        cv2.putText(spatial_map, "Spatial Map (Top-Down)", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(spatial_map, f"Range: ±{range_m:.1f}m", (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.putText(spatial_map, "Grid: 1m", (map_w - 65, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        return spatial_map
    
    def _draw_frame_info(self, frame: np.ndarray, frame_number: int, 
                        num_onscreen: int, num_offscreen: int = 0) -> np.ndarray:
        """Draw frame info overlay (top-left corner)."""
        info_lines = [
            f"Frame: {frame_number}",
            f"On-screen: {num_onscreen}",
        ]
        
        if num_offscreen > 0:
            info_lines.append(f"Off-screen: {num_offscreen}")
        
        y_offset = 80  # Start below the off-screen banner
        for line in info_lines:
            # Background
            (text_w, text_h), _ = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (10, y_offset - text_h - 5),
                         (20 + text_w, y_offset + 5), (0, 0, 0), -1)
            
            # Text
            cv2.putText(frame, line, (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            y_offset += 30
        
        return frame
    
    def _get_unique_color(self, entity_id: int) -> Tuple[int, int, int]:
        """Generate unique color for entity ID using HSV color space."""
        # Use golden ratio for well-distributed hues
        golden_ratio = 0.618033988749895
        hue = (entity_id * golden_ratio) % 1.0
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
        
        # Convert to BGR (OpenCV format) and scale to [0, 255]
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        
        return bgr
    
    def mark_reidentification(self, entity_id: int, duration_frames: int = 15):
        """Mark entity as recently re-identified (for visual highlight)."""
        self.reid_highlights[entity_id] = duration_frames
    
    def draw_slam_trajectory(
        self,
        frame: np.ndarray,
        slam_trajectory: List[np.ndarray],
        current_pose: Optional[np.ndarray] = None,
        minimap: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Draw SLAM camera trajectory visualization
        
        Args:
            frame: Input frame to draw on
            slam_trajectory: List of camera positions (Nx3 array or list of 3D points)
            current_pose: Current camera pose (4x4 matrix)
            minimap: If True, also return separate mini-map
        
        Returns:
            Tuple of (frame_with_trajectory, minimap_or_None)
        """
        if not slam_trajectory or len(slam_trajectory) < 2:
            return frame, None
        
        # Draw trajectory overlay on frame
        frame_with_traj = self._draw_trajectory_overlay(frame, slam_trajectory, current_pose)
        
        # Create mini-map if requested
        minimap_viz = None
        if minimap:
            minimap_viz = self._create_trajectory_minimap(slam_trajectory, current_pose)
        
        return frame_with_traj, minimap_viz
    
    def _draw_trajectory_overlay(
        self,
        frame: np.ndarray,
        slam_trajectory: List[np.ndarray],
        current_pose: Optional[np.ndarray]
    ) -> np.ndarray:
        """Draw trajectory path overlay in corner of frame"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay in bottom-right corner
        overlay_size = 200
        overlay = np.zeros((overlay_size, overlay_size, 3), dtype=np.uint8)
        
        # Convert trajectory to array
        trajectory_array = np.array(slam_trajectory)
        
        if trajectory_array.ndim != 2 or trajectory_array.shape[1] < 3:
            return frame
        
        # Extract X and Z coordinates (top-down view: Y is up/down)
        x_coords = trajectory_array[:, 0]
        z_coords = trajectory_array[:, 2]
        
        # Normalize to overlay size
        x_min, x_max = x_coords.min(), x_coords.max()
        z_min, z_max = z_coords.min(), z_coords.max()
        
        x_range = max(x_max - x_min, 100)  # At least 100mm range
        z_range = max(z_max - z_min, 100)
        padding = 20
        
        # Map to overlay coordinates
        x_norm = ((x_coords - x_min) / x_range * (overlay_size - 2*padding) + padding).astype(int)
        z_norm = ((z_coords - z_min) / z_range * (overlay_size - 2*padding) + padding).astype(int)
        z_norm = overlay_size - z_norm  # Flip Z for image coordinates
        
        # Draw trajectory with color gradient (blue -> red over time)
        num_points = len(x_norm)
        for i in range(num_points - 1):
            # Color gradient
            t = i / max(num_points - 1, 1)
            color = (
                int(255 * t),       # R: 0 -> 255
                0,                  # G: constant
                int(255 * (1-t))    # B: 255 -> 0
            )
            
            pt1 = (x_norm[i], z_norm[i])
            pt2 = (x_norm[i+1], z_norm[i+1])
            
            cv2.line(overlay, pt1, pt2, color, 2, cv2.LINE_AA)
        
        # Draw current position (yellow circle)
        if num_points > 0:
            cv2.circle(overlay, (x_norm[-1], z_norm[-1]), 5, (0, 255, 255), -1)
        
        # Draw start point (green)
        cv2.circle(overlay, (x_norm[0], z_norm[0]), 5, (0, 255, 0), -1)
        
        # Add labels
        cv2.putText(overlay, "Camera Path", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, "Start", (5, overlay_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.3, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Blend overlay onto frame (bottom-right corner)
        if h > overlay_size + 20 and w > overlay_size + 20:
            roi = frame[h-overlay_size-10:h-10, w-overlay_size-10:w-10]
            blended = cv2.addWeighted(roi, 0.3, overlay, 0.7, 0)
            frame[h-overlay_size-10:h-10, w-overlay_size-10:w-10] = blended
        
        return frame
    
    def _create_trajectory_minimap(
        self,
        slam_trajectory: List[np.ndarray],
        current_pose: Optional[np.ndarray],
        size: Tuple[int, int] = (400, 400)
    ) -> np.ndarray:
        """Create separate mini-map visualization of trajectory"""
        w, h = size
        minimap = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Convert trajectory to array
        trajectory_array = np.array(slam_trajectory)
        
        if trajectory_array.ndim != 2 or trajectory_array.shape[1] < 3:
            return minimap
        
        # Extract X and Z coordinates (top-down view)
        x_coords = trajectory_array[:, 0]
        z_coords = trajectory_array[:, 2]
        
        # Normalize to map size
        x_min, x_max = x_coords.min(), x_coords.max()
        z_min, z_max = z_coords.min(), z_coords.max()
        
        x_range = max(x_max - x_min, 100)
        z_range = max(z_max - z_min, 100)
        padding = 40
        
        x_norm = ((x_coords - x_min) / x_range * (w - 2*padding) + padding).astype(int)
        z_norm = ((z_coords - z_min) / z_range * (h - 2*padding) + padding).astype(int)
        z_norm = h - z_norm  # Flip Z
        
        # Draw grid
        grid_spacing = 50
        for i in range(0, w, grid_spacing):
            cv2.line(minimap, (i, 0), (i, h), (40, 40, 40), 1)
        for i in range(0, h, grid_spacing):
            cv2.line(minimap, (0, i), (w, i), (40, 40, 40), 1)
        
        # Draw trajectory with gradient
        num_points = len(x_norm)
        for i in range(num_points - 1):
            t = i / max(num_points - 1, 1)
            color = (int(255 * t), 50, int(255 * (1-t)))
            
            pt1 = (x_norm[i], z_norm[i])
            pt2 = (x_norm[i+1], z_norm[i+1])
            cv2.line(minimap, pt1, pt2, color, 3, cv2.LINE_AA)
            
            # Draw waypoint markers every 10 points
            if i % 10 == 0:
                cv2.circle(minimap, pt1, 3, (255, 255, 255), -1)
        
        # Draw start and end
        cv2.circle(minimap, (x_norm[0], z_norm[0]), 8, (0, 255, 0), -1)
        cv2.circle(minimap, (x_norm[-1], z_norm[-1]), 8, (0, 255, 255), -1)
        
        # Add text annotations
        cv2.putText(minimap, "SLAM Trajectory (Top-Down)", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(minimap, f"{len(slam_trajectory)} poses", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Calculate distance traveled
        total_dist = 0
        for i in range(num_points - 1):
            dx = x_coords[i+1] - x_coords[i]
            dz = z_coords[i+1] - z_coords[i]
            total_dist += np.sqrt(dx**2 + dz**2)
        
        cv2.putText(minimap, f"Distance: {total_dist/1000:.1f}m", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Legend
        cv2.circle(minimap, (w - 30, 20), 5, (0, 255, 0), -1)
        cv2.putText(minimap, "Start", (w - 70, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.circle(minimap, (w - 30, 40), 5, (0, 255, 255), -1)
        cv2.putText(minimap, "Current", (w - 70, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return minimap


def create_side_by_side_visualization(
    original_frame: np.ndarray,
    tracking_frame: np.ndarray,
    depth_frame: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create side-by-side comparison visualization.
    
    Args:
        original_frame: Original video frame
        tracking_frame: Frame with tracking overlays
        depth_frame: Optional depth visualization
    
    Returns:
        Combined frame with all views side-by-side
    """
    if depth_frame is None:
        # Just original + tracking
        return np.hstack([original_frame, tracking_frame])
    else:
        # Original + tracking + depth
        return np.hstack([original_frame, tracking_frame, depth_frame])


def export_tracking_statistics(
    tracks: List[BayesianEntityBelief],
    frame_number: int,
    output_file: str
):
    """
    Export tracking statistics to JSON file for analysis.
    
    Args:
        tracks: Current active tracks
        frame_number: Current frame number
        output_file: Path to output JSON file
    """
    import json
    
    stats = {
        'frame': frame_number,
        'num_tracks': len(tracks),
        'tracks': []
    }
    
    for track in tracks:
        track_data = {
            'entity_id': track.entity_id,
            'most_likely_class': track.most_likely_class,
            'class_posterior': track.class_posterior,
            'centroid_2d': track.centroid_2d.tolist(),
            'centroid_3d_mm': track.centroid_3d_mm.tolist() if track.centroid_3d_mm is not None else None,
            'velocity_2d': track.velocity_2d.tolist(),
            'first_seen_frame': track.first_seen_frame,
            'total_detections': track.total_detections,
            'reidentified_times': track.reidentified_times
        }
        stats['tracks'].append(track_data)
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
