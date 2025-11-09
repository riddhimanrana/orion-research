"""
SLAM Engine - Visual SLAM / Visual Odometry for Camera Pose Estimation

Implements feature-based visual odometry using OpenCV to estimate
camera motion and maintain consistent world coordinates.
Falls back to depth-based ICP odometry for low-texture scenes.

Author: Orion Research Team
Date: November 2025
"""

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path

from orion.slam.depth_odometry import DepthOdometry, DepthOdometryConfig


@dataclass
class SLAMConfig:
    """Configuration for SLAM engine"""
    method: str = "opencv"  # "opencv" for now
    num_features: int = 1500  # ORB features to track
    match_ratio_test: float = 0.75  # Lowe's ratio test threshold
    ransac_threshold: float = 1.0  # RANSAC inlier threshold (pixels)
    min_matches: int = 15  # Minimum matches for pose estimation
    scale_factor: float = 1.0  # Scale for monocular (will be estimated)
    use_imu: bool = False  # Future: IMU integration


class SLAMEngine:
    """
    Visual SLAM Engine for camera pose estimation
    
    Provides camera trajectory and transforms observations to world frame.
    """
    
    def __init__(self, config: Optional[SLAMConfig] = None):
        """
        Initialize SLAM engine
        
        Args:
            config: SLAM configuration
        """
        self.config = config or SLAMConfig()
        
        if self.config.method == "opencv":
            self.slam = OpenCVSLAM(self.config)
        else:
            raise ValueError(f"Unknown SLAM method: {self.config.method}")
        
        # State
        self.poses: List[np.ndarray] = []  # 4x4 transformation matrices
        self.trajectory: List[np.ndarray] = []  # Camera positions over time
        self.tracking_status: List[bool] = []  # Tracking success per frame
        self.map_points: List[np.ndarray] = []  # 3D map points (future)
    
    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_idx: int
    ) -> Optional[np.ndarray]:
        """
        Process frame and return camera pose
        
        Args:
            frame: Input frame (BGR)
            timestamp: Frame timestamp
            frame_idx: Frame index
        
        Returns:
            4x4 transformation matrix [R|t] or None if tracking lost
        """
        pose = self.slam.track(frame, timestamp, frame_idx)
        
        if pose is not None:
            self.poses.append(pose)
            self.trajectory.append(pose[:3, 3])  # Translation component
            self.tracking_status.append(True)
        else:
            # Tracking lost
            if self.poses:
                # Use last known pose
                pose = self.poses[-1].copy()
            else:
                # Initialize with identity
                pose = np.eye(4)
            
            self.poses.append(pose)
            if self.poses:
                self.trajectory.append(pose[:3, 3])
            self.tracking_status.append(False)
        
        return pose
    
    def transform_to_world(
        self,
        point_camera: np.ndarray,  # (x, y, z) in camera frame
        pose_idx: int
    ) -> np.ndarray:
        """
        Transform point from camera frame to world frame
        
        Args:
            point_camera: (x, y, z) in mm, camera frame
            pose_idx: Index of camera pose to use
        
        Returns:
            (x, y, z) in mm, world frame
        """
        if pose_idx >= len(self.poses):
            # Pose not available, return camera coords
            return point_camera
        
        pose = self.poses[pose_idx]
        
        # Convert to homogeneous coordinates
        point_h = np.append(point_camera, 1.0)
        
        # Apply transformation
        point_world_h = pose @ point_h
        
        # Return 3D coordinates
        return point_world_h[:3]
    
    def get_trajectory(self) -> np.ndarray:
        """
        Get camera trajectory as (N, 3) array
        
        Returns:
            Array of shape (N, 3) with camera positions
        """
        if not self.trajectory:
            return np.array([]).reshape(0, 3)
        return np.array(self.trajectory)
    
    def get_tracking_quality(self) -> float:
        """
        Get tracking quality (fraction of frames with successful tracking)
        
        Returns:
            Quality score [0, 1]
        """
        if not self.tracking_status:
            return 0.0
        return sum(self.tracking_status) / len(self.tracking_status)
    
    def save_trajectory(self, output_path: str):
        """
        Save trajectory in TUM format
        
        Format: timestamp x y z qx qy qz qw
        """
        from scipy.spatial.transform import Rotation
        
        with open(output_path, 'w') as f:
            for i, pose in enumerate(self.poses):
                # Extract translation
                t = pose[:3, 3]
                
                # Extract rotation and convert to quaternion
                R = pose[:3, :3]
                quat = Rotation.from_matrix(R).as_quat()  # (x, y, z, w)
                
                # TUM format: timestamp x y z qx qy qz qw
                f.write(f"{i} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                       f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n")
        
        print(f"✓ Trajectory saved to: {output_path}")
    
    def get_statistics(self) -> Dict:
        """Get SLAM statistics"""
        return {
            'total_frames': len(self.poses),
            'tracking_success_rate': self.get_tracking_quality(),
            'trajectory_length_m': self._compute_trajectory_length(),
            'avg_translation_per_frame_m': self._compute_avg_motion(),
        }
    
    def _compute_trajectory_length(self) -> float:
        """Compute total trajectory length in meters"""
        if len(self.trajectory) < 2:
            return 0.0
        
        trajectory_np = self.get_trajectory()
        diffs = np.diff(trajectory_np, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return float(np.sum(distances) / 1000.0)  # mm → meters
    
    def _compute_avg_motion(self) -> float:
        """Compute average motion per frame in meters"""
        traj_length = self._compute_trajectory_length()
        if len(self.poses) <= 1:
            return 0.0
        return traj_length / (len(self.poses) - 1)


class OpenCVSLAM:
    """
    Simplified visual odometry using OpenCV
    
    Uses ORB features + essential matrix + pose recovery.
    Note: Monocular scale is ambiguous - we use heuristic scale.
    """
    
    def __init__(self, config: SLAMConfig):
        """
        Initialize OpenCV SLAM
        
        Args:
            config: SLAM configuration
        """
        self.config = config
        
        # Feature detector (ORB) - More aggressive settings for low-texture scenes
        self.detector = cv2.ORB_create(
            nfeatures=config.num_features * 2,  # Double features for better coverage
            scaleFactor=1.1,  # Smaller scale pyramid for finer features
            nlevels=12,  # More pyramid levels
            edgeThreshold=5,  # Lower threshold to detect more features
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=21  # Smaller patch for finer details
        )
        
        # Add FAST detector as backup for low-texture scenes
        self.fast_detector = cv2.FastFeatureDetector_create(
            threshold=10,  # Low threshold for more features
            nonmaxSuppression=True,
            type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
        )
        
        # Feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Depth odometry fallback for low-texture scenes
        self.depth_odometry = None  # Will be initialized when we get camera intrinsics
        
        # Camera intrinsics (will be estimated from frame size)
        self.K: Optional[np.ndarray] = None
        
        # State
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_keypoints: Optional[List] = None
        self.prev_descriptors: Optional[np.ndarray] = None
        self.prev_depth: Optional[np.ndarray] = None  # Store depth for scale
        self.current_pose: np.ndarray = np.eye(4)  # Start at origin
        self.scale: float = config.scale_factor  # Scale for monocular
        self.use_depth_scale = True  # Use depth for scale when available
        
        # Statistics
        self.num_matches_history: List[int] = []
        self.num_inliers_history: List[int] = []
    
    def track(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_idx: int,
        depth_map: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Track camera motion using feature matching
        
        Args:
            frame: Input frame (BGR)
            timestamp: Frame timestamp
            frame_idx: Frame index
            depth_map: Optional depth map for scale recovery (HxW, values in mm)
        
        Returns:
            4x4 pose matrix or None if tracking failed
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize camera matrix if needed
        if self.K is None:
            h, w = gray.shape
            # Approximate focal length (common heuristic)
            fx = fy = w * 1.2  # Slightly wider FOV
            cx, cy = w / 2, h / 2
            self.K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float64)
            
            # Initialize depth odometry fallback now that we have K
            if self.depth_odometry is None and depth_map is not None:
                self.depth_odometry = DepthOdometry(self.K, DepthOdometryConfig())
                print("[SLAM] Initialized depth odometry fallback")
        
        # Detect features
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if self.prev_frame is None:
            # First frame - initialize
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return self.current_pose.copy()
        
        if descriptors is None or len(keypoints) < 5:  # Very low threshold
            # Not enough features - try optical flow fallback
            if self.prev_frame is not None:
                flow_pose = self._track_with_optical_flow(gray, self.prev_frame)
                if flow_pose is not None:
                    self.prev_frame = gray
                    return flow_pose
            
            print(f"[SLAM] Frame {frame_idx}: Too few features ({len(keypoints) if keypoints else 0})")
            return None
        
        # Match features
        matches = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < self.config.match_ratio_test * n.distance:
                    good_matches.append(m)
        
        self.num_matches_history.append(len(good_matches))
        
        if len(good_matches) < 8:  # Lowered from 15 to 8
            print(f"[SLAM] Frame {frame_idx}: Too few matches ({len(good_matches)})")
            
            # Try depth odometry fallback if available
            if self.depth_odometry is not None and depth_map is not None:
                depth_pose = self.depth_odometry.track(depth_map)
                if depth_pose is not None:
                    self.prev_frame = gray
                    self.prev_keypoints = keypoints
                    self.prev_descriptors = descriptors
                    self.prev_depth = depth_map
                    self.current_pose = depth_pose
                    return depth_pose
            
            # Try optical flow fallback
            if self.prev_frame is not None:
                flow_pose = self._track_with_optical_flow(gray, self.prev_frame)
                if flow_pose is not None:
                    self.prev_frame = gray
                    self.prev_keypoints = keypoints
                    self.prev_descriptors = descriptors
                    self.prev_depth = depth_map
                    return flow_pose
            
            return None
        
        # Extract matched points
        pts1 = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([keypoints[m.trainIdx].pt for m in good_matches])
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=self.config.ransac_threshold
        )
        
        if E is None:
            print(f"[SLAM] Frame {frame_idx}: Essential matrix estimation failed")
            return None
        
        # Count inliers
        num_inliers = int(mask.sum())
        self.num_inliers_history.append(num_inliers)
        
        if num_inliers < self.config.min_matches:
            print(f"[SLAM] Frame {frame_idx}: Too few inliers ({num_inliers})")
            return None
        
        # Recover pose
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        
        # Estimate scale from depth if available
        if depth_map is not None and self.prev_depth is not None and self.use_depth_scale:
            # Use depth to estimate real-world scale
            scale_from_depth = self._estimate_scale_from_depth(
                pts1, pts2, good_matches, depth_map, self.prev_depth
            )
            if scale_from_depth is not None and scale_from_depth > 0:
                self.scale = scale_from_depth
        elif self.scale == 1.0:
            # Heuristic fallback: typical human walking speed ~1.4 m/s
            # At 30 FPS: ~47mm per frame
            motion_magnitude = np.linalg.norm(t)
            if motion_magnitude > 1e-6:
                self.scale = 47.0 / motion_magnitude  # mm per frame
        
        # Apply scale
        t_scaled = t * self.scale
        
        # Update current pose
        # T_new = T_current @ T_relative
        T_relative = np.eye(4)
        T_relative[:3, :3] = R
        T_relative[:3, 3] = t_scaled.flatten()
        
        self.current_pose = self.current_pose @ T_relative
        
        # Update previous frame and depth
        self.prev_frame = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_depth = depth_map
        
        return self.current_pose.copy()
    
    def _estimate_scale_from_depth(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        matches: List,
        curr_depth: np.ndarray,
        prev_depth: np.ndarray
    ) -> Optional[float]:
        """
        Estimate scale using depth measurements at matched keypoints
        
        Returns:
            Estimated scale factor (mm per unit) or None if failed
        """
        try:
            scales = []
            for i, m in enumerate(matches):
                # Get pixel coordinates
                u1, v1 = int(pts1[i][0]), int(pts1[i][1])
                u2, v2 = int(pts2[i][0]), int(pts2[i][1])
                
                # Get depths (in mm)
                if (0 <= v1 < prev_depth.shape[0] and 0 <= u1 < prev_depth.shape[1] and
                    0 <= v2 < curr_depth.shape[0] and 0 <= u2 < curr_depth.shape[1]):
                    
                    d1 = prev_depth[v1, u1]
                    d2 = curr_depth[v2, u2]
                    
                    # Valid depth readings
                    if d1 > 100 and d2 > 100 and d1 < 10000 and d2 < 10000:
                        # 3D point triangulation from depth
                        # Average depth change gives us scale
                        depth_change = abs(d2 - d1)
                        if depth_change < 500:  # < 50cm change (reasonable)
                            scales.append(depth_change)
            
            if len(scales) > 5:
                # Robust median scale
                median_scale = float(np.median(scales))
                if 10 < median_scale < 200:  # Reasonable range: 1cm-20cm per frame
                    return median_scale
            
            return None
        except Exception:
            return None
    
    def _track_with_optical_flow(self, curr_frame: np.ndarray, prev_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Fallback tracking using optical flow when feature matching fails.
        Returns 4x4 pose matrix if successful.
        """
        try:
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, curr_frame,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            
            # Get median flow (robust to outliers)
            flow_x = np.median(flow[:, :, 0])
            flow_y = np.median(flow[:, :, 1])
            flow_magnitude = np.sqrt(flow_x**2 + flow_y**2)
            
            # Minimum motion threshold
            if flow_magnitude < 2.0:  # < 2 pixels
                return None
            
            # Convert pixel motion to camera motion estimate
            # This is approximate - assumes planar motion at median depth
            fx = self.K[0, 0]
            fy = self.K[1, 1]
            
            # Estimate translation (very rough, better than nothing)
            t_x = -flow_x / fx * 1000  # Convert to mm
            t_y = -flow_y / fy * 1000
            t_z = 47.0  # Assume average forward motion (47mm/frame)
            
            # Construct pose (identity rotation, estimated translation)
            pose = np.eye(4)
            pose[:3, 3] = [t_x, t_y, t_z]
            
            # Update cumulative pose
            self.cumulative_pose = self.cumulative_pose @ pose
            
            print(f"[SLAM] Optical flow tracking: flow={flow_magnitude:.1f}px, t=[{t_x:.0f}, {t_y:.0f}, {t_z:.0f}]mm")
            return self.cumulative_pose.copy()
            
        except Exception as e:
            print(f"[SLAM] Optical flow failed: {e}")
            return None
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics"""
        if not self.num_matches_history:
            return {
                'avg_matches': 0,
                'avg_inliers': 0,
                'scale': self.scale,
            }
        
        return {
            'avg_matches': np.mean(self.num_matches_history),
            'avg_inliers': np.mean(self.num_inliers_history),
            'scale': self.scale,
        }
