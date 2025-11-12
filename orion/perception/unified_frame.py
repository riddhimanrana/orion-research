"""
Unified Frame: Single data structure for all frame-level perception.

Instead of scattered outputs:
  depth_map, detections, pose, points, tracks = ...

We now have:
  unified_frame = UnifiedFrame(...)
  
All components read/write to this single source of truth.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Object3D:
    """Single object in 3D world space"""
    
    # Identity
    id: int
    class_name: str
    confidence: float
    
    # 2D Detection (from YOLO)
    bbox_2d: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    
    # 3D Position (computed from depth + YOLO box)
    position_3d: np.ndarray  # (x, y, z) in world frame
    bbox_3d_corners: Optional[np.ndarray] = None  # (8, 3) world coords
    
    # Temporal tracking
    age: int = 0  # frames visible
    last_seen_frame: int = 0
    is_visible: bool = True  # visible in current frame?
    
    # Semantic information
    clip_embedding: Optional[np.ndarray] = None  # (512,) or (256,)
    embedding_age: int = 0  # how old is this embedding?
    
    # Uncertainty
    depth_confidence: float = 0.5  # 0-1, higher = more confident
    
    def get_3d_center(self) -> np.ndarray:
        """Return 3D position"""
        return self.position_3d.copy()
    
    def project_to_2d(self, K: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Project 3D position back to 2D image using intrinsics.
        
        Useful for reprojection consistency checks.
        
        Args:
            K: (3, 3) camera intrinsics
        
        Returns:
            (u, v) in image coordinates, or None if behind camera
        """
        x, y, z = self.position_3d
        
        if z <= 0:
            return None  # Behind camera
        
        u = K[0, 0] * x / z + K[0, 2]
        v = K[1, 1] * y / z + K[1, 2]
        
        return (u, v)


@dataclass
class UnifiedFrame:
    """
    Complete perception state for a single frame.
    
    This is the single source of truth for all frame-level data.
    """
    
    # ========== FRAME METADATA ==========
    frame_id: int
    timestamp: float  # seconds since video start
    fps: float  # adaptive FPS (1-10)
    
    # ========== CAMERA GEOMETRY ==========
    camera_pose: np.ndarray  # (4, 4) transformation matrix [R|t; 0|1]
    camera_intrinsics: 'CameraIntrinsics'  # fx, fy, cx, cy
    
    # ========== 3D SPATIAL MAP ==========
    point_cloud: np.ndarray  # (N, 3) in world coordinates
    point_confidences: np.ndarray  # (N,) in [0, 1]
    point_colors: np.ndarray  # (N, 3) RGB in [0, 255]
    point_ages: np.ndarray  # (N,) frames since added
    
    voxel_grid: Optional[object] = None  # VoxelGrid instance
    
    # ========== OBJECT DETECTIONS ==========
    objects_3d: List[Object3D] = field(default_factory=list)
    """All detected objects with 3D positions"""
    
    # ========== SEMANTIC INFORMATION ==========
    clip_embeddings: Dict[int, np.ndarray] = field(default_factory=dict)
    """object_id → embedding"""
    
    # ========== TEMPORAL HEATMAPS ==========
    motion_heatmap: Optional[np.ndarray] = None  # (H, W) float
    change_heatmap: Optional[np.ndarray] = None  # (H, W) float
    attention_heatmap: Optional[np.ndarray] = None  # (H, W) float
    
    # ========== PERFORMANCE METRICS ==========
    @dataclass
    class Metrics:
        fps: float = 0.0
        frame_time_ms: float = 0.0
        depth_time_ms: float = 0.0
        yolo_time_ms: float = 0.0
        slam_time_ms: float = 0.0
        spatial_map_time_ms: float = 0.0
        tracking_time_ms: float = 0.0
        clip_time_ms: float = 0.0
        
        n_points: int = 0
        n_objects: int = 0
        n_voxels: int = 0
    
    metrics: Metrics = field(default_factory=Metrics)
    
    # ========== RAW INPUTS ==========
    rgb_frame: Optional[np.ndarray] = None  # (H, W, 3) for reference
    depth_map: Optional[np.ndarray] = None  # (H, W) depth in meters
    
    def get_point_cloud_in_range(self, 
                                  max_distance: float = 5.0
                                  ) -> np.ndarray:
        """Return point cloud filtered by distance from camera"""
        distances = np.linalg.norm(self.point_cloud, axis=1)
        mask = distances < max_distance
        return self.point_cloud[mask]
    
    def get_visible_objects(self) -> List[Object3D]:
        """Return only objects visible in current frame"""
        return [obj for obj in self.objects_3d if obj.is_visible]
    
    def get_object_by_id(self, obj_id: int) -> Optional[Object3D]:
        """Lookup object by ID"""
        for obj in self.objects_3d:
            if obj.id == obj_id:
                return obj
        return None
    
    def summary(self) -> str:
        """Human-readable summary"""
        return (
            f"Frame {self.frame_id} (t={self.timestamp:.2f}s)\n"
            f"  FPS: {self.fps:.1f}\n"
            f"  Points: {len(self.point_cloud)}\n"
            f"  Objects: {len(self.objects_3d)}\n"
            f"  Visible: {len(self.get_visible_objects())}\n"
            f"  Voxels occupied: {self.metrics.n_voxels}\n"
            f"  Time: {self.metrics.frame_time_ms:.1f}ms"
        )
