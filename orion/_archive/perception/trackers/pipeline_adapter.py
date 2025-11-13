"""
Pipeline Adapter: Convert existing outputs to UnifiedFrame

Allows gradual migration without breaking existing code.
"""

from typing import Optional, Dict, List
import numpy as np
from orion.perception.unified_frame import UnifiedFrame, Object3D
from orion.perception.types import CameraIntrinsics


class UnifiedFrameBuilder:
    """Build UnifiedFrame from pipeline component outputs"""
    
    def __init__(self):
        """Initialize with default camera intrinsics"""
        # Will be overridden per frame if intrinsics provided
        self.camera_intrinsics = None
    
    def build(self,
              frame_id: int,
              timestamp: float,
              fps: float,
              # Inputs
              rgb_frame: np.ndarray,
              depth_map: np.ndarray,
              camera_pose: np.ndarray,
              yolo_detections: List[Dict],
              point_cloud: np.ndarray,
              point_confidences: np.ndarray,
              camera_intrinsics: Optional[CameraIntrinsics] = None,
              voxel_grid: Optional[object] = None,
              clip_embeddings: Optional[Dict] = None,
              # Timing metrics
              depth_time_ms: float = 0.0,
              yolo_time_ms: float = 0.0,
              slam_time_ms: float = 0.0,
              spatial_map_time_ms: float = 0.0,
              tracking_time_ms: float = 0.0,
              clip_time_ms: float = 0.0,
              ) -> UnifiedFrame:
        """
        Build complete UnifiedFrame from component outputs.
        
        Args:
            frame_id: Frame number
            timestamp: Time in seconds
            fps: Adaptive FPS
            rgb_frame: (H, W, 3) input frame
            depth_map: (H, W) depth in meters
            camera_pose: (4, 4) transformation matrix
            yolo_detections: List of {bbox, class, confidence, track_id?}
            point_cloud: (N, 3) in world frame
            point_confidences: (N,) confidence values
            camera_intrinsics: CameraIntrinsics or None
            voxel_grid: Optional VoxelGrid instance
            clip_embeddings: Optional dict of {object_id: embedding}
            ... timing metrics ...
        
        Returns:
            UnifiedFrame instance
        """
        
        # Store intrinsics for future use
        if camera_intrinsics is not None:
            self.camera_intrinsics = camera_intrinsics
        
        # Build 3D objects from detections
        objects_3d = self._build_3d_objects(
            yolo_detections=yolo_detections,
            depth_map=depth_map,
            camera_pose=camera_pose,
            camera_intrinsics=self.camera_intrinsics,
            clip_embeddings=clip_embeddings or {}
        )
        
        # Sample colors from RGB frame for point cloud
        point_colors = self._sample_point_colors(
            point_cloud=point_cloud,
            camera_pose=camera_pose,
            rgb_frame=rgb_frame,
            camera_intrinsics=self.camera_intrinsics
        )
        
        # Create frame
        frame = UnifiedFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            fps=fps,
            camera_pose=camera_pose,
            camera_intrinsics=self.camera_intrinsics,
            point_cloud=point_cloud,
            point_confidences=point_confidences,
            point_colors=point_colors,
            point_ages=np.zeros(len(point_cloud), dtype=np.int32),
            objects_3d=objects_3d,
            voxel_grid=voxel_grid,
            clip_embeddings=clip_embeddings or {},
            rgb_frame=rgb_frame,
            depth_map=depth_map,
        )
        
        # Set metrics
        frame.metrics.fps = fps
        frame.metrics.depth_time_ms = depth_time_ms
        frame.metrics.yolo_time_ms = yolo_time_ms
        frame.metrics.slam_time_ms = slam_time_ms
        frame.metrics.spatial_map_time_ms = spatial_map_time_ms
        frame.metrics.tracking_time_ms = tracking_time_ms
        frame.metrics.clip_time_ms = clip_time_ms
        frame.metrics.n_points = len(point_cloud)
        frame.metrics.n_objects = len(objects_3d)
        
        if voxel_grid is not None:
            frame.metrics.n_voxels = int(np.sum(voxel_grid.grid > 0.5))
        
        return frame
    
    def _build_3d_objects(self,
                         yolo_detections: List[Dict],
                         depth_map: np.ndarray,
                         camera_pose: np.ndarray,
                         camera_intrinsics: Optional[CameraIntrinsics],
                         clip_embeddings: Dict = None
                         ) -> List[Object3D]:
        """Convert YOLO detections to Object3D instances"""
        
        objects_3d = []
        
        if camera_intrinsics is None:
            return objects_3d
        
        for i, det in enumerate(yolo_detections):
            # Get 3D position from depth at object center
            x1, y1, x2, y2 = det['bbox']
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            # Sample depth robustly within bbox (median of valid samples)
            h, w = depth_map.shape
            if 0 <= cy < h and 0 <= cx < w:
                # Extract depth region within bbox
                y_min = max(0, int(y1))
                y_max = min(h, int(y2))
                x_min = max(0, int(x1))
                x_max = min(w, int(x2))
                
                if y_max > y_min and x_max > x_min:
                    depth_roi = depth_map[y_min:y_max, x_min:x_max]
                    valid_depths = depth_roi[depth_roi > 0.05]
                    
                    if len(valid_depths) > 0:
                        depth_z = np.median(valid_depths)  # Robust median
                    else:
                        depth_z = 0.0
                else:
                    depth_z = 0.0
                
                if depth_z > 0.05:  # Valid depth
                    # Backproject to camera frame
                    p_camera = self._backproject_point(
                        cx, cy, depth_z, camera_intrinsics
                    )
                    
                    # Transform to world frame
                    R = camera_pose[:3, :3]
                    t = camera_pose[:3, 3]
                    p_world = R @ p_camera + t
                    
                    # Get tracking ID if available, otherwise use index
                    obj_id = det.get('track_id', i)
                    
                    # Create Object3D
                    obj = Object3D(
                        id=obj_id,
                        class_name=det.get('class', 'unknown'),
                        confidence=det.get('confidence', 0.5),
                        bbox_2d=(int(x1), int(y1), int(x2), int(y2)),
                        position_3d=p_world,
                        depth_confidence=0.8,
                        clip_embedding=clip_embeddings.get(obj_id) if clip_embeddings else None,
                    )
                    
                    objects_3d.append(obj)
        
        return objects_3d
    
    def _backproject_point(self,
                          u: float, v: float, z: float,
                          K: CameraIntrinsics) -> np.ndarray:
        """Backproject pixel to 3D camera frame"""
        x = (u - K.cx) / K.fx * z
        y = (v - K.cy) / K.fy * z
        return np.array([x, y, z])
    
    def _sample_point_colors(self,
                            point_cloud: np.ndarray,
                            camera_pose: np.ndarray,
                            rgb_frame: np.ndarray,
                            camera_intrinsics: Optional[CameraIntrinsics]
                            ) -> np.ndarray:
        """Sample RGB colors for point cloud by reprojection"""
        
        colors = np.ones((len(point_cloud), 3), dtype=np.uint8) * 128
        
        if camera_intrinsics is None:
            return colors
        
        try:
            # Transform world points to camera frame
            R_inv = camera_pose[:3, :3].T
            t_inv = -R_inv @ camera_pose[:3, 3]
            points_camera = (R_inv @ point_cloud.T).T + t_inv
            
            h, w = rgb_frame.shape[:2]
            
            for i, p_cam in enumerate(points_camera):
                if p_cam[2] <= 0:
                    continue
                
                # Project to image
                u = camera_intrinsics.fx * p_cam[0] / p_cam[2] + camera_intrinsics.cx
                v = camera_intrinsics.fy * p_cam[1] / p_cam[2] + camera_intrinsics.cy
                
                u, v = int(u), int(v)
                if 0 <= u < w and 0 <= v < h:
                    colors[i] = rgb_frame[v, u]
        except Exception:
            pass  # Fallback to default colors
        
        return colors
