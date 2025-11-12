"""
Unified Rerun Visualizer for all 9 perception modalities.

Logs the complete UnifiedFrame to Rerun for real-time interactive 3D visualization:
- Point cloud (world frame)
- Camera pose and frustum
- 3D objects with tracking IDs
- Depth map and uncertainty
- Motion heatmap (optical flow)
- Change heatmap (depth changes)
- Attention heatmap (detection confidence)
- CLIP embeddings (semantic space)
- Re-ID features (appearance similarity)
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import rerun as rr
from orion.perception.unified_frame import UnifiedFrame, Object3D
from orion.perception.types import CameraIntrinsics


@dataclass
class VisualizationConfig:
    """Configuration for Rerun visualization"""
    # Enable/disable logging for each modality
    log_point_cloud: bool = True
    log_camera_frustum: bool = True
    log_objects_3d: bool = True
    log_depth_map: bool = True
    log_heatmaps: bool = True
    log_semantic_embeddings: bool = False  # Large, optional
    log_rgb_frame: bool = True
    
    # Visualization parameters
    point_cloud_radius: float = 0.01  # meters, for point size
    object_box_color: Tuple[int, int, int] = (0, 255, 0)  # RGB
    frustum_scale: float = 0.2  # camera frustum size
    frustum_color: Tuple[int, int, int] = (255, 0, 0)  # RGB
    
    # Recording/streaming
    recording_path: Optional[str] = None  # Save to .rrd file
    stream_to_viewer: bool = True  # Stream to connected viewer


class UnifiedRerunVisualizer:
    """
    Real-time visualization of all perception modalities using Rerun.
    
    Supports:
    1. Point cloud in world frame with colors
    2. Camera pose (3x3 grid, updated each frame)
    3. 3D bounding boxes for detected objects
    4. Depth maps with uncertainty visualization
    5. Heatmaps: motion, depth change, attention
    6. Camera intrinsics visualization
    7. Semantic clustering (if CLIP embeddings available)
    """
    
    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        app_id: str = "orion-unified-perception",
    ):
        """
        Initialize Rerun visualizer.
        
        Args:
            config: Visualization configuration
            app_id: Rerun app identifier
        """
        self.config = config or VisualizationConfig()
        self.app_id = app_id
        self.frame_count = 0
        
        # Initialize Rerun
        if self.config.stream_to_viewer:
            rr.init(self.app_id, spawn=False)
        
        if self.config.recording_path:
            rr.set_recording_id(self.config.recording_path)
    
    def log_frame(self, frame: UnifiedFrame) -> None:
        """
        Log a complete UnifiedFrame to Rerun.
        
        Args:
            frame: UnifiedFrame with all modalities
        """
        self.frame_count += 1
        timestamp_ns = int(frame.timestamp * 1e9)
        rr.set_time_seconds("timestamp", frame.timestamp)
        rr.set_time_sequence("frame", self.frame_count)
        
        # Log RGB frame
        if self.config.log_rgb_frame and frame.rgb_frame is not None:
            self._log_rgb_frame(frame)
        
        # Log camera pose and frustum
        if self.config.log_camera_frustum:
            self._log_camera_pose(frame)
            self._log_camera_frustum(frame)
        
        # Log point cloud
        if self.config.log_point_cloud and len(frame.point_cloud) > 0:
            self._log_point_cloud(frame)
        
        # Log depth map
        if self.config.log_depth_map and frame.depth_map is not None:
            self._log_depth_map(frame)
        
        # Log 3D objects
        if len(frame.objects_3d) > 0:
            self._log_objects_3d(frame)
        
        # Log heatmaps
        if self.config.log_heatmaps:
            self._log_heatmaps(frame)
        
        # Log semantic embeddings (optional, can be expensive)
        if self.config.log_semantic_embeddings and len(frame.clip_embeddings) > 0:
            self._log_semantic_embeddings(frame)
    
    def _log_rgb_frame(self, frame: UnifiedFrame) -> None:
        """Log RGB frame as 2D image"""
        rr.log(
            "rgb",
            rr.Image(frame.rgb_frame),
        )
    
    def _log_camera_pose(self, frame: UnifiedFrame) -> None:
        """Log camera pose as 3x3 grid of axes"""
        pose = frame.camera_pose  # (4x4) matrix
        
        # Extract rotation and translation
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # Log as transform3d
        rr.log(
            "world/camera",
            rr.Transform3D(
                mat3x3=R,
                translation=t,
            ),
        )
        
        # Log camera intrinsics as pinhole camera model
        if frame.camera_intrinsics is not None:
            K = frame.camera_intrinsics
            rr.log(
                "world/camera",
                rr.Pinhole(
                    resolution=[K.width, K.height],
                    focal_length=K.fx,
                ),
            )
    
    def _log_camera_frustum(self, frame: UnifiedFrame) -> None:
        """Log camera viewing frustum"""
        if frame.camera_intrinsics is None:
            return
        
        K = frame.camera_intrinsics
        pose = frame.camera_pose
        
        # Generate frustum corners (near and far planes)
        near = 0.1
        far = 5.0
        
        # Image dimensions
        h, w = K.height, K.width
        
        # Frustum corner positions in camera frame
        corners_cam = np.array([
            # Near plane
            [-w/(2*K.fx)*near, -h/(2*K.fy)*near, near],
            [w/(2*K.fx)*near, -h/(2*K.fy)*near, near],
            [w/(2*K.fx)*near, h/(2*K.fy)*near, near],
            [-w/(2*K.fx)*near, h/(2*K.fy)*near, near],
            # Far plane
            [-w/(2*K.fx)*far, -h/(2*K.fy)*far, far],
            [w/(2*K.fx)*far, -h/(2*K.fy)*far, far],
            [w/(2*K.fx)*far, h/(2*K.fy)*far, far],
            [-w/(2*K.fx)*far, h/(2*K.fy)*far, far],
        ])
        
        # Transform to world frame
        R = pose[:3, :3]
        t = pose[:3, 3]
        corners_world = (R @ corners_cam.T).T + t
        
        # Define frustum edges (as line segments)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Near plane
            (4, 5), (5, 6), (6, 7), (7, 4),  # Far plane
            (0, 4), (1, 5), (2, 6), (3, 7),  # Connecting edges
        ]
        
        # Log line segments
        for i, (start, end) in enumerate(edges):
            points = np.array([corners_world[start], corners_world[end]])
            rr.log(
                f"world/camera_frustum/edge_{i}",
                rr.LineStrips3D([points], colors=self.config.frustum_color),
            )
    
    def _log_point_cloud(self, frame: UnifiedFrame) -> None:
        """Log point cloud in world frame"""
        points = frame.point_cloud
        
        if len(points) == 0:
            return
        
        # Prepare colors (use RGB if available, else use heights for coloring)
        if frame.point_colors is not None and len(frame.point_colors) == len(points):
            colors = frame.point_colors.astype(np.uint8)
        else:
            # Color by height (Z coordinate)
            z_min, z_max = points[:, 2].min(), points[:, 2].max()
            if z_max > z_min:
                z_normalized = (points[:, 2] - z_min) / (z_max - z_min)
                colors = np.stack([
                    (z_normalized * 255).astype(np.uint8),  # Red = height
                    np.zeros(len(points), dtype=np.uint8),
                    ((1 - z_normalized) * 255).astype(np.uint8),  # Blue = inverse height
                ], axis=1)
            else:
                colors = np.tile([128, 128, 128], (len(points), 1)).astype(np.uint8)
        
        rr.log(
            "world/point_cloud",
            rr.Points3D(
                positions=points,
                colors=colors,
                radii=self.config.point_cloud_radius,
            ),
        )
    
    def _log_depth_map(self, frame: UnifiedFrame) -> None:
        """Log depth map with colormap visualization"""
        depth = frame.depth_map
        if depth is None or len(depth) == 0:
            return
        
        # Normalize depth to 0-255 for visualization
        depth_valid = depth[depth > 0]
        if len(depth_valid) == 0:
            return
        
        depth_min, depth_max = depth_valid.min(), depth_valid.max()
        depth_normalized = np.clip((depth - depth_min) / (depth_max - depth_min + 1e-8), 0, 1)
        depth_viz = (depth_normalized * 255).astype(np.uint8)
        
        rr.log(
            "rgb/depth_map",
            rr.Image(depth_viz),
        )
    
    def _log_objects_3d(self, frame: UnifiedFrame) -> None:
        """Log 3D bounding boxes for detected objects"""
        for obj in frame.objects_3d:
            self._log_single_object(obj, frame)
    
    def _log_single_object(self, obj: Object3D, frame: UnifiedFrame) -> None:
        """Log a single 3D bounding box"""
        # Extract 2D box dimensions from bbox_2d
        x1, y1, x2, y2 = obj.bbox_2d
        bbox_2d_w = x2 - x1
        bbox_2d_h = y2 - y1
        
        # Estimate 3D box dimensions (heuristic: assume depth similar to width)
        estimated_depth = max(bbox_2d_w, bbox_2d_h) * 0.3  # Rough estimate
        
        # Create 3D bounding box corners (centered at object position)
        half_w = bbox_2d_w * 0.3 / 2
        half_h = bbox_2d_h * 0.3 / 2
        half_d = estimated_depth / 2
        
        corners = np.array([
            [-half_w, -half_h, -half_d],
            [half_w, -half_h, -half_d],
            [half_w, half_h, -half_d],
            [-half_w, half_h, -half_d],
            [-half_w, -half_h, half_d],
            [half_w, -half_h, half_d],
            [half_w, half_h, half_d],
            [-half_w, half_h, half_d],
        ])
        
        # Translate to object position
        corners_world = corners + obj.position_3d
        
        # Define box edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Front face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Back face
            (0, 4), (1, 5), (2, 6), (3, 7),  # Connecting edges
        ]
        
        # Determine color based on confidence
        color = (
            int(obj.confidence * 255),  # Red intensity = confidence
            int((1 - obj.confidence) * 255),  # Green intensity = 1 - confidence
            100,  # Blue constant
        )
        
        # Log box edges
        for i, (start, end) in enumerate(edges):
            points = np.array([corners_world[start], corners_world[end]])
            rr.log(
                f"world/objects/obj_{obj.id}/edge_{i}",
                rr.LineStrips3D([points], colors=[color]),
            )
        
        # Log object center and label
        rr.log(
            f"world/objects/obj_{obj.id}",
            rr.Points3D(
                positions=[obj.position_3d],
                colors=[color],
                radii=0.05,
            ),
        )
        
        # Log label
        rr.log(
            f"world/objects/obj_{obj.id}/label",
            rr.TextDocument(
                f"{obj.class_name} (#{obj.id})\nConf: {obj.confidence:.2f}\n"
                f"Pos: ({obj.position_3d[0]:.2f}, {obj.position_3d[1]:.2f}, {obj.position_3d[2]:.2f})"
            ),
        )
    
    def _log_heatmaps(self, frame: UnifiedFrame) -> None:
        """Log motion, change, and attention heatmaps"""
        if frame.motion_heatmap is not None:
            motion_viz = (np.clip(frame.motion_heatmap, 0, 1) * 255).astype(np.uint8)
            rr.log("rgb/motion_heatmap", rr.Image(motion_viz))
        
        if frame.change_heatmap is not None:
            change_viz = (np.clip(frame.change_heatmap, 0, 1) * 255).astype(np.uint8)
            rr.log("rgb/change_heatmap", rr.Image(change_viz))
        
        if frame.attention_heatmap is not None:
            attention_viz = (np.clip(frame.attention_heatmap, 0, 1) * 255).astype(np.uint8)
            rr.log("rgb/attention_heatmap", rr.Image(attention_viz))
    
    def _log_semantic_embeddings(self, frame: UnifiedFrame) -> None:
        """
        Log semantic embeddings visualization.
        
        Creates a 2D projection of CLIP embeddings for objects.
        """
        if len(frame.clip_embeddings) == 0:
            return
        
        # Simple 2D projection: take first 2 PCA components
        # (In production, would use proper dimensionality reduction)
        embeddings = np.array(list(frame.clip_embeddings.values()))
        
        if len(embeddings) < 2:
            return
        
        # For now, just log as text summary
        text = f"Embeddings: {len(embeddings)} objects with CLIP features"
        rr.log("info/embeddings", rr.TextDocument(text))
    
    def save_recording(self, output_path: str) -> None:
        """Save recording to RRD file"""
        rr.save(output_path)
        print(f"✅ Recording saved to {output_path}")
    
    def shutdown(self) -> None:
        """Shutdown Rerun"""
        rr.disconnect()


def visualize_unified_frames(
    frames: List[UnifiedFrame],
    config: Optional[VisualizationConfig] = None,
    recording_path: Optional[str] = None,
) -> None:
    """
    Convenience function to visualize a list of UnifiedFrames.
    
    Args:
        frames: List of UnifiedFrame objects
        config: Visualization config
        recording_path: Path to save recording (.rrd file)
    """
    config = config or VisualizationConfig()
    if recording_path:
        config.recording_path = recording_path
    
    visualizer = UnifiedRerunVisualizer(config)
    
    print(f"🎥 Logging {len(frames)} frames to Rerun...")
    for frame in frames:
        visualizer.log_frame(frame)
    
    print(f"✅ Logged {len(frames)} frames")
    print(f"📊 Frame range: {frames[0].timestamp:.2f}s - {frames[-1].timestamp:.2f}s")
    
    if recording_path:
        visualizer.save_recording(recording_path)
