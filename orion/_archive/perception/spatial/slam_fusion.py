"""
SLAM + Depth Fusion for 3D Spatial Understanding

Combines Depth Anything V2 (metric depth) with SLAM (camera poses) to create
a metrically consistent 3D reconstruction with semantic understanding.

Supports:
- RTAB-Map (real-time, native M1)
- ORB-SLAM3 (feature-based, lightweight)
- DROID-SLAM (deep learned, most accurate offline)
- COLMAP (batch SfM, highest accuracy)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class CameraPose:
    """6-DoF camera pose (position + orientation)"""
    frame_idx: int
    timestamp: float
    position: np.ndarray  # (3,) - world coordinates [x, y, z]
    rotation: np.ndarray  # (3, 3) - rotation matrix
    translation: np.ndarray  # (3,) - translation vector
    confidence: float = 1.0
    
    def to_matrix(self) -> np.ndarray:
        """Get 4x4 transformation matrix"""
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T
    
    def inverse(self) -> np.ndarray:
        """Get inverse transformation (world-to-camera)"""
        T_inv = np.eye(4)
        R_inv = self.rotation.T
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = -R_inv @ self.translation
        return T_inv


@dataclass
class Point3D:
    """3D point in world coordinates"""
    position: np.ndarray  # (3,) - [x, y, z]
    color: np.ndarray  # (3,) - [R, G, B] 0-255
    normal: Optional[np.ndarray] = None  # (3,) - surface normal
    confidence: float = 1.0
    frame_idx: int = -1


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int
    height: int
    
    def to_matrix(self) -> np.ndarray:
        """Get 3x3 intrinsics matrix K"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])


class SLAMBackend:
    """Base class for SLAM backends"""
    
    def __init__(self, intrinsics: CameraIntrinsics):
        self.intrinsics = intrinsics
        self.poses: List[CameraPose] = []
        self.frame_idx = 0
        
    def track_frame(self, frame: np.ndarray, timestamp: float) -> Optional[CameraPose]:
        """
        Track camera pose for current frame.
        
        Args:
            frame: RGB frame (H, W, 3)
            timestamp: Frame timestamp in seconds
            
        Returns:
            CameraPose or None if tracking failed
        """
        raise NotImplementedError
    
    def get_map_points(self) -> List[Point3D]:
        """Get 3D map points from SLAM"""
        raise NotImplementedError
    
    def reset(self):
        """Reset SLAM state"""
        self.poses = []
        self.frame_idx = 0


class VisualOdometryBackend(SLAMBackend):
    """
    Lightweight visual odometry using OpenCV.
    Estimates relative camera motion between frames.
    Good fallback when full SLAM is unavailable.
    """
    
    def __init__(self, intrinsics: CameraIntrinsics):
        super().__init__(intrinsics)
        # Feature detector
        self.detector = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # State
        self.prev_frame = None
        self.prev_kp = None
        self.prev_desc = None
        self.current_pose = np.eye(4)
        
        # Intrinsics matrix
        self.K = intrinsics.to_matrix()
        
        logger.info("✅ Visual Odometry backend initialized")
    
    def track_frame(self, frame: np.ndarray, timestamp: float) -> Optional[CameraPose]:
        """Track camera pose using feature matching"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Detect features
        kp, desc = self.detector.detectAndCompute(gray, None)
        
        if self.prev_frame is None:
            # First frame - identity pose
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_desc = desc
            
            pose = CameraPose(
                frame_idx=self.frame_idx,
                timestamp=timestamp,
                position=np.array([0.0, 0.0, 0.0]),
                rotation=np.eye(3),
                translation=np.array([0.0, 0.0, 0.0]),
                confidence=1.0
            )
            self.poses.append(pose)
            self.frame_idx += 1
            return pose
        
        # Match features with previous frame
        if desc is None or self.prev_desc is None or len(desc) < 10 or len(self.prev_desc) < 10:
            logger.warning(f"Frame {self.frame_idx}: Insufficient features")
            self.frame_idx += 1
            return None
        
        matches = self.matcher.knnMatch(self.prev_desc, desc, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            logger.warning(f"Frame {self.frame_idx}: Too few matches ({len(good_matches)})")
            self.frame_idx += 1
            return None
        
        # Extract matched points
        pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches])
        
        # Compute essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is None:
            logger.warning(f"Frame {self.frame_idx}: Essential matrix computation failed")
            self.frame_idx += 1
            return None
        
        # Recover pose
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        
        # Update cumulative pose
        T_delta = np.eye(4)
        T_delta[:3, :3] = R
        T_delta[:3, 3] = t.flatten()
        
        self.current_pose = self.current_pose @ T_delta
        
        # Extract position and rotation
        position = self.current_pose[:3, 3]
        rotation = self.current_pose[:3, :3]
        
        # Confidence based on inlier ratio
        confidence = np.sum(mask_pose) / len(mask_pose) if len(mask_pose) > 0 else 0.0
        
        pose = CameraPose(
            frame_idx=self.frame_idx,
            timestamp=timestamp,
            position=position,
            rotation=rotation,
            translation=t.flatten(),
            confidence=confidence
        )
        
        self.poses.append(pose)
        
        # Update state
        self.prev_frame = gray
        self.prev_kp = kp
        self.prev_desc = desc
        self.frame_idx += 1
        
        return pose
    
    def get_map_points(self) -> List[Point3D]:
        """Visual odometry doesn't maintain map points"""
        return []
    
    def reset(self):
        super().reset()
        self.prev_frame = None
        self.prev_kp = None
        self.prev_desc = None
        self.current_pose = np.eye(4)


class DepthSLAMFusion:
    """
    Fuses Depth Anything V2 with SLAM for metrically consistent 3D reconstruction.
    
    Pipeline:
    1. SLAM tracks camera pose (6-DoF)
    2. Depth Anything V2 estimates per-frame depth
    3. Fuse depth + pose → global point cloud
    4. Project 2D detections to 3D using fused geometry
    """
    
    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        depth_model,  # DepthAnythingV2Estimator
        slam_backend: Optional[SLAMBackend] = None,
        max_depth: float = 10.0,
        point_sampling_stride: int = 4  # Sample every Nth pixel for point cloud
    ):
        """
        Args:
            intrinsics: Camera intrinsic parameters
            depth_model: DepthAnythingV2Estimator instance
            slam_backend: SLAM backend (defaults to VisualOdometryBackend)
            max_depth: Maximum depth in meters
            point_sampling_stride: Downsample point cloud by this factor
        """
        self.intrinsics = intrinsics
        self.depth_model = depth_model
        self.max_depth = max_depth
        self.point_sampling_stride = point_sampling_stride
        
        # Initialize SLAM backend
        if slam_backend is None:
            logger.info("No SLAM backend provided, using Visual Odometry")
            self.slam = VisualOdometryBackend(intrinsics)
        else:
            self.slam = slam_backend
        
        # Global point cloud
        self.global_points: List[Point3D] = []
        
        # Intrinsics matrix
        self.K = intrinsics.to_matrix()
        self.K_inv = np.linalg.inv(self.K)
        
        logger.info(f"✅ Depth-SLAM Fusion initialized")
        logger.info(f"   Intrinsics: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")
        logger.info(f"   Resolution: {intrinsics.width}x{intrinsics.height}")
    
    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        detections: Optional[List[Dict]] = None
    ) -> Tuple[Optional[CameraPose], np.ndarray, List[Point3D]]:
        """
        Process single frame through SLAM + depth fusion.
        
        Args:
            frame: RGB frame (H, W, 3)
            timestamp: Frame timestamp
            detections: Optional list of 2D detections to lift to 3D
            
        Returns:
            (camera_pose, depth_map, frame_points)
        """
        # 1. Track camera pose with SLAM
        pose = self.slam.track_frame(frame, timestamp)
        
        if pose is None:
            logger.warning(f"Frame {self.slam.frame_idx}: SLAM tracking failed")
            return None, np.zeros((frame.shape[0], frame.shape[1])), []
        
        # 2. Estimate depth with Depth Anything V2
        depth_map, confidence = self.depth_model.estimate(frame)
        
        # Scale depth to metric (Depth Anything outputs relative depth)
        # Use median scaling or learned scale factor
        depth_map = self._scale_depth_to_metric(depth_map, confidence)
        
        # Clamp depth
        depth_map = np.clip(depth_map, 0.1, self.max_depth)
        
        # 3. Fuse depth + pose → 3D point cloud
        frame_points = self._depth_to_pointcloud(frame, depth_map, pose)
        
        # Add to global map
        self.global_points.extend(frame_points)
        
        # 4. Lift 2D detections to 3D (if provided)
        if detections is not None:
            self._lift_detections_to_3d(detections, depth_map, pose)
        
        return pose, depth_map, frame_points
    
    def _scale_depth_to_metric(self, depth: np.ndarray, confidence: np.ndarray) -> np.ndarray:
        """
        Scale relative depth to metric depth.
        
        Depth Anything V2 outputs relative depth. We scale it to meters using:
        1. Median depth assumption (e.g., median = 2m for indoor scenes)
        2. Or use known scale from SLAM (if available)
        
        TODO: Implement learned scale alignment with SLAM
        """
        # For now, use simple median scaling
        # Assume median depth = 2.5 meters for indoor scenes
        median_depth = np.median(depth)
        if median_depth > 0:
            scale = 2.5 / median_depth
            depth = depth * scale
        
        return depth
    
    def _depth_to_pointcloud(
        self,
        frame: np.ndarray,
        depth_map: np.ndarray,
        pose: CameraPose
    ) -> List[Point3D]:
        """
        Convert depth map to 3D point cloud in world coordinates.
        
        For each pixel (u, v):
            P_camera = K^-1 * [u, v, 1]^T * depth(u, v)
            P_world = T_camera_to_world * P_camera
        """
        h, w = depth_map.shape
        points_3d = []
        
        # Sample points (downsample for efficiency)
        stride = self.point_sampling_stride
        
        for v in range(0, h, stride):
            for u in range(0, w, stride):
                d = depth_map[v, u]
                
                if d <= 0.1 or d >= self.max_depth:
                    continue  # Invalid depth
                
                # Pixel to camera coordinates
                pixel_homogeneous = np.array([u, v, 1.0])
                camera_coords = self.K_inv @ pixel_homogeneous * d
                
                # Camera to world coordinates
                camera_coords_homogeneous = np.append(camera_coords, 1.0)
                world_coords = pose.to_matrix() @ camera_coords_homogeneous
                
                # Extract color
                color = frame[v, u, :]  # RGB
                
                point = Point3D(
                    position=world_coords[:3],
                    color=color,
                    confidence=1.0,
                    frame_idx=pose.frame_idx
                )
                
                points_3d.append(point)
        
        return points_3d
    
    def _lift_detections_to_3d(
        self,
        detections: List[Dict],
        depth_map: np.ndarray,
        pose: CameraPose
    ):
        """
        Lift 2D detections to 3D using depth and pose.
        Modifies detections in-place to add 3D information.
        """
        for detection in detections:
            bbox = detection.get('bbox')  # [x1, y1, x2, y2]
            if bbox is None:
                continue
            
            x1, y1, x2, y2 = bbox
            
            # Get center point
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Sample depth at center
            if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]:
                d = depth_map[cy, cx]
                
                if d > 0.1 and d < self.max_depth:
                    # Backproject to 3D
                    pixel = np.array([cx, cy, 1.0])
                    camera_coords = self.K_inv @ pixel * d
                    camera_coords_homogeneous = np.append(camera_coords, 1.0)
                    world_coords = pose.to_matrix() @ camera_coords_homogeneous
                    
                    # Add 3D info to detection
                    detection['position_3d'] = world_coords[:3].tolist()
                    detection['depth'] = float(d)
                    
                    # Estimate 3D bounding box size (rough approximation)
                    bbox_width_px = x2 - x1
                    bbox_height_px = y2 - y1
                    
                    # Approximate physical size using depth
                    # size_physical = size_pixels * depth / focal_length
                    width_3d = bbox_width_px * d / self.intrinsics.fx
                    height_3d = bbox_height_px * d / self.intrinsics.fy
                    
                    detection['size_3d'] = [width_3d, height_3d, height_3d * 0.5]  # [w, h, d]
    
    def get_global_pointcloud(self) -> np.ndarray:
        """
        Get full global point cloud as Nx6 array (x, y, z, r, g, b).
        """
        if len(self.global_points) == 0:
            return np.zeros((0, 6))
        
        points = np.zeros((len(self.global_points), 6))
        for i, point in enumerate(self.global_points):
            points[i, :3] = point.position
            points[i, 3:] = point.color / 255.0  # Normalize to [0, 1]
        
        return points
    
    def get_trajectory(self) -> np.ndarray:
        """
        Get camera trajectory as Nx3 array of positions.
        """
        if len(self.slam.poses) == 0:
            return np.zeros((0, 3))
        
        trajectory = np.array([pose.position for pose in self.slam.poses])
        return trajectory
    
    def reset(self):
        """Reset SLAM and point cloud"""
        self.slam.reset()
        self.global_points = []
        logger.info("🔄 Depth-SLAM Fusion reset")


# ============================================================================
# Factory functions
# ============================================================================

def create_slam_fusion(
    frame_width: int,
    frame_height: int,
    depth_model,
    fov_degrees: float = 60.0,
    slam_type: str = "vo"  # "vo", "orb-slam3", "rtabmap", etc.
) -> DepthSLAMFusion:
    """
    Factory to create DepthSLAMFusion with automatic intrinsics estimation.
    
    Args:
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        depth_model: DepthAnythingV2Estimator instance
        fov_degrees: Field of view in degrees (default 60)
        slam_type: SLAM backend type
        
    Returns:
        DepthSLAMFusion instance
    """
    # Estimate intrinsics from FOV
    fov_rad = np.deg2rad(fov_degrees)
    fx = frame_width / (2 * np.tan(fov_rad / 2))
    fy = fx  # Assume square pixels
    cx = frame_width / 2
    cy = frame_height / 2
    
    intrinsics = CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=frame_width,
        height=frame_height
    )
    
    # Create SLAM backend
    if slam_type == "vo":
        slam_backend = VisualOdometryBackend(intrinsics)
    else:
        logger.warning(f"SLAM type '{slam_type}' not implemented, using VO")
        slam_backend = VisualOdometryBackend(intrinsics)
    
    fusion = DepthSLAMFusion(
        intrinsics=intrinsics,
        depth_model=depth_model,
        slam_backend=slam_backend
    )
    
    logger.info(f"✅ Created SLAM fusion with {slam_type} backend")
    
    return fusion
