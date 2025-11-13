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
from scipy.ndimage import median_filter
from collections import deque

# Import loop closure components
from orion.slam.loop_closure import LoopClosureDetector, LoopClosure
from orion.slam.pose_graph import PoseGraphOptimizer

# Import depth processing utilities
from orion.slam.depth_utils import (
    DepthUncertaintyEstimator,
    TemporalDepthFilter,
    ScaleKalmanFilter,
    check_depth_consistency,
)


class SimpleKalmanFilter:
    """Simple 1D Kalman filter for smoothing scale estimates"""
    def __init__(self, process_variance: float = 0.01, measurement_variance: float = 0.1):
        self.process_variance = process_variance  # How much we expect value to change
        self.measurement_variance = measurement_variance  # Measurement noise
        self.estimate = None
        self.estimate_variance = 1.0
    
    def update(self, measurement: float, measurement_confidence: float = 1.0) -> float:
        """Update filter with new measurement, returns smoothed estimate"""
        if self.estimate is None:
            self.estimate = measurement
            return self.estimate
        
        # Adjust measurement variance based on confidence
        adjusted_measurement_var = self.measurement_variance / measurement_confidence
        
        # Prediction step (assume constant)
        predicted_estimate = self.estimate
        predicted_variance = self.estimate_variance + self.process_variance
        
        # Update step
        kalman_gain = predicted_variance / (predicted_variance + adjusted_measurement_var)
        self.estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
        self.estimate_variance = (1 - kalman_gain) * predicted_variance
        
        return self.estimate


def preprocess_depth_map(depth: np.ndarray, bilateral_d: int = 5, sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
    """
    Pre-process depth map to reduce noise and remove outliers.
    
    Args:
        depth: Raw depth map (HxW) in mm
        bilateral_d: Filter diameter (larger = more smoothing)
        sigma_color: Color space sigma for bilateral filter
        sigma_space: Coordinate space sigma for bilateral filter
    
    Returns:
        Cleaned depth map
    """
    if depth is None or depth.size == 0:
        return depth
    
    # Convert to float32 for filtering
    depth_float = depth.astype(np.float32)
    
    # 1. Bilateral filter - smooth while preserving edges
    # This removes sensor noise while keeping object boundaries sharp
    depth_filtered = cv2.bilateralFilter(depth_float, bilateral_d, sigma_color, sigma_space)
    
    # 2. Detect and remove outliers using median filtering
    # If a pixel's depth differs too much from its neighbors, it's likely noise
    median = median_filter(depth_filtered, size=5)
    diff = np.abs(depth_filtered - median)
    
    # Mark outliers (depth differs by >20% from local median)
    outlier_threshold = 0.2 * median
    outliers = diff > outlier_threshold
    
    # Replace outliers with median value
    depth_filtered[outliers] = median[outliers]
    
    # 3. Remove invalid depths (too close/far, likely sensor errors)
    min_valid = 100.0  # 10cm minimum
    max_valid = 10000.0  # 10m maximum
    invalid = (depth_filtered < min_valid) | (depth_filtered > max_valid)
    depth_filtered[invalid] = 0.0
    
    return depth_filtered


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
    
    # Loop closure detection
    enable_loop_closure: bool = True  # Enable loop closure detection
    min_loop_interval: int = 30  # Minimum frames between loops
    min_loop_inliers: int = 30  # Minimum RANSAC inliers for valid loop
    bow_similarity_threshold: float = 0.70  # BoW similarity threshold
    
    # Pose graph optimization
    enable_pose_graph_optimization: bool = True  # Optimize trajectory after loops
    loop_closure_weight: float = 100.0  # Weight for loop closure constraints
    optimize_every_n_loops: int = 5  # Run optimization every N loop closures
    
    # Depth integration (Phase 3)
    use_depth_uncertainty: bool = True  # Estimate depth uncertainty
    use_temporal_depth_filter: bool = True  # Temporal depth smoothing
    use_scale_kalman: bool = True  # Kalman filter for scale
    depth_temporal_alpha: float = 0.7  # Temporal filter weight (0.7 = 70% new, 30% old)
    scale_process_noise: float = 0.01  # Scale Kalman process noise
    
    # Hybrid odometry (Phase 3 Week 6)
    enable_pose_fusion: bool = True  # Fuse visual and depth poses
    rotation_weight_visual: float = 0.8  # Visual rotation weight (0-1)
    translation_fusion_mode: str = "weighted"  # "weighted", "visual", "depth"
    min_confidence_threshold: float = 0.3  # Min confidence to use pose
    
    # Depth consistency (Phase 3 Week 6 Day 2)
    enable_depth_consistency: bool = True  # Check depth consistency
    epipolar_threshold: float = 1.0  # Epipolar error threshold (pixels)
    depth_ratio_threshold: float = 0.3  # Max depth change ratio (30%)
    
    # Multi-frame depth fusion (Phase 3 Week 6 Day 3)
    enable_multi_frame_fusion: bool = True  # Fuse depth across frames
    fusion_window_size: int = 5  # Sliding window size
    fusion_outlier_threshold: float = 100.0  # Temporal outlier threshold (mm)
    fusion_min_confidence: float = 0.3  # Min confidence to use frame


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
        
        # Loop closure callbacks (Phase 4 Week 2)
        self.loop_closure_callbacks: List[callable] = []
        
        if self.config.method == "opencv":
            self.slam = OpenCVSLAM(self.config)
        else:
            raise ValueError(f"Unknown SLAM method: {self.config.method}")
        
        # State
        self.poses: List[np.ndarray] = []  # 4x4 transformation matrices
        self.trajectory: List[np.ndarray] = []  # Camera positions over time
        self.tracking_status: List[bool] = []  # Tracking success per frame
        self.map_points: List[np.ndarray] = []  # 3D map points (future)
        
        # Loop closure
        self.loop_detector: Optional[LoopClosureDetector] = None
        self.pose_optimizer: Optional[PoseGraphOptimizer] = None
        
        if self.config.enable_loop_closure:
            self.loop_detector = LoopClosureDetector(
                min_loop_interval=self.config.min_loop_interval,
                min_bow_similarity=self.config.bow_similarity_threshold,
                min_inliers=self.config.min_loop_inliers,
                enable_pose_graph_optimization=self.config.enable_pose_graph_optimization,
            )
            
            if self.config.enable_pose_graph_optimization:
                self.pose_optimizer = PoseGraphOptimizer(
                    loop_closure_weight=self.config.loop_closure_weight,
                )
            
            print("[SLAM] Loop closure detection enabled")
        
        # Depth processing (Phase 3)
        self.depth_uncertainty: Optional[DepthUncertaintyEstimator] = None
        self.temporal_depth_filter: Optional[TemporalDepthFilter] = None
        self.scale_kalman: Optional[ScaleKalmanFilter] = None
        
        if self.config.use_depth_uncertainty:
            self.depth_uncertainty = DepthUncertaintyEstimator()
            print("[SLAM] Depth uncertainty estimation enabled")
        
        if self.config.use_temporal_depth_filter:
            self.temporal_depth_filter = TemporalDepthFilter(
                alpha=self.config.depth_temporal_alpha
            )
            print("[SLAM] Temporal depth filtering enabled")
        
        if self.config.use_scale_kalman:
            self.scale_kalman = ScaleKalmanFilter(
                process_noise=self.config.scale_process_noise
            )
            print("[SLAM] Scale Kalman filter enabled")
        
        # Hybrid odometry (Phase 3 Week 6)
        self.hybrid_odometry: Optional['HybridOdometry'] = None
        if self.config.enable_pose_fusion:
            from orion.slam.hybrid_odometry import HybridOdometry
            self.hybrid_odometry = HybridOdometry(
                rotation_weight_visual=self.config.rotation_weight_visual,
                translation_fusion_mode=self.config.translation_fusion_mode,
                min_confidence_threshold=self.config.min_confidence_threshold,
            )
            # Pass to SLAM instance
            self.slam.hybrid_odometry = self.hybrid_odometry
            print("[SLAM] Hybrid visual-depth pose fusion enabled")
        
        # Depth consistency checking (Phase 3 Week 6 Day 2)
        self.depth_consistency: Optional['DepthConsistencyChecker'] = None
        if self.config.enable_depth_consistency:
            from orion.slam.depth_consistency import DepthConsistencyChecker
            self.depth_consistency = DepthConsistencyChecker(
                epipolar_threshold=self.config.epipolar_threshold,
                temporal_threshold=100.0,  # 100mm
                depth_ratio_threshold=self.config.depth_ratio_threshold,
            )
            # Pass to SLAM instance
            self.slam.depth_consistency = self.depth_consistency
            print("[SLAM] Depth consistency checking enabled")
        
        # Multi-frame depth fusion (Phase 3 Week 6 Day 3)
        self.multi_frame_fusion: Optional['MultiFrameDepthFusion'] = None
        if self.config.enable_multi_frame_fusion:
            from orion.slam.multi_frame_depth_fusion import MultiFrameDepthFusion
            self.multi_frame_fusion = MultiFrameDepthFusion(
                window_size=self.config.fusion_window_size,
                outlier_threshold_mm=self.config.fusion_outlier_threshold,
                min_confidence=self.config.fusion_min_confidence,
            )
            print("[SLAM] Multi-frame depth fusion enabled")
    
    def register_loop_closure_callback(self, callback):
        """
        Register callback to be called when loop closure is detected.
        
        Callback signature: callback(loop_closure: LoopClosure)
        
        Args:
            callback: Function to call on loop closure detection
        """
        self.loop_closure_callbacks.append(callback)
    
    def get_latest_pose(self) -> Optional[np.ndarray]:
        """Get most recent camera pose (4x4 transform) or None if no frames processed yet."""
        return self.poses[-1] if self.poses else None
    
    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_idx: int,
        depth_map: Optional[np.ndarray] = None  # Add depth parameter
    ) -> Optional[np.ndarray]:
        """
        Process frame and return camera pose
        
        Args:
            frame: Input frame (BGR)
            timestamp: Frame timestamp
            frame_idx: Frame index
            depth_map: Optional depth map for scale recovery
        
        Returns:
            4x4 transformation matrix [R|t] or None if tracking lost
        """
        # Process depth map and compute uncertainty if available
        depth_uncertainty_map = None
        filtered_depth_map = depth_map
        
        if depth_map is not None and self.depth_uncertainty is not None:
            # Estimate depth uncertainty
            depth_quality = self.depth_uncertainty.estimate(depth_map, frame)
            depth_uncertainty_map = depth_quality.uncertainty_map
            
            # Apply temporal filtering if enabled
            if self.temporal_depth_filter is not None:
                filtered_depth_map = self.temporal_depth_filter.update(
                    depth_map, depth_uncertainty_map
                )
        
        # Track frame with depth and uncertainty
        pose = self.slam.track(
            frame, timestamp, frame_idx, 
            depth_map=filtered_depth_map,
            uncertainty_map=depth_uncertainty_map
        )
        
        if pose is not None:
            # Apply multi-frame depth fusion (Phase 3 Week 6 Day 3)
            if self.multi_frame_fusion is not None and filtered_depth_map is not None:
                # Add current frame to sliding window
                confidence_map = 1.0 - depth_uncertainty_map if depth_uncertainty_map is not None else None
                self.multi_frame_fusion.add_frame(
                    filtered_depth_map, pose, confidence_map, frame_idx
                )
                
                # Fuse depth from window (only if we have enough frames)
                if len(self.multi_frame_fusion.depth_window) >= 2:
                    try:
                        fused_depth, fused_confidence, fusion_stats = \
                            self.multi_frame_fusion.fuse_to_current_frame(self.slam.K)
                        
                        # Log fusion stats occasionally
                        if frame_idx % 30 == 0:
                            print(f"[SLAM] Multi-frame fusion: {fusion_stats['num_frames_used']} frames, "
                                  f"{fusion_stats['valid_ratio']:.1%} valid, "
                                  f"avg {fusion_stats['avg_frames_per_fusion']:.1f} frames/fusion")
                    except Exception as e:
                        print(f"[SLAM] Multi-frame fusion warning: {e}")
            
            self.poses.append(pose)
            self.trajectory.append(pose[:3, 3])  # Translation component
            self.tracking_status.append(True)
            
            # Check for loop closure on keyframes
            if (self.loop_detector is not None and 
                hasattr(self.slam, 'num_keyframes') and
                self.slam.num_keyframes > 0 and
                frame_idx == self.slam.last_keyframe_idx):
                
                # This is a keyframe - check for loop closure
                self._handle_loop_closure(frame, frame_idx, pose)
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
    
    def get_depth_consistency_stats(self) -> dict:
        """
        Get depth consistency statistics
        
        Returns:
            Dictionary with consistency stats
        """
        if self.depth_consistency is None:
            return {}
        return self.depth_consistency.get_statistics()
    
    def get_multi_frame_fusion_stats(self) -> dict:
        """
        Get multi-frame depth fusion statistics
        
        Returns:
            Dictionary with fusion stats
        """
        if self.multi_frame_fusion is None:
            return {}
        return self.multi_frame_fusion.get_statistics()
    
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
    
    def _handle_loop_closure(
        self,
        frame: np.ndarray,
        frame_idx: int,
        current_pose: np.ndarray,
    ) -> None:
        """Handle loop closure detection and pose graph optimization"""
        if self.loop_detector is None:
            return
        
        # Extract features for this keyframe
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.slam.detector.detectAndCompute(gray, None)
        
        if descriptors is None or len(descriptors) == 0:
            return
        
        # Add to loop detector database
        self.loop_detector.add_keyframe(
            frame_id=frame_idx,
            pose=current_pose.copy(),
            keypoints=keypoints,
            descriptors=descriptors,
            image_gray=gray,
        )
        
        # Check for loop closure
        loop = self.loop_detector.detect_loop(
            query_id=frame_idx,
            camera_matrix=self.slam.K,
        )
        
        if loop is not None:
            print(f"\n[SLAM] ✓ Loop closure: frame {loop.query_id} → {loop.match_id}")
            print(f"       Inliers: {loop.inliers}, Confidence: {loop.confidence:.2f}")
            
            # Trigger loop closure callbacks (Phase 4 Week 2)
            for callback in self.loop_closure_callbacks:
                try:
                    callback(loop)
                except Exception as e:
                    print(f"⚠️  Loop closure callback error: {e}")
            
            # Run pose graph optimization if enabled
            if (self.config.enable_pose_graph_optimization and
                self.pose_optimizer is not None and 
                len(self.loop_detector.loop_closures) % self.config.optimize_every_n_loops == 0):
                
                print(f"\n[SLAM] Running pose graph optimization...")
                print(f"       Poses: {len(self.poses)}, Loops: {len(self.loop_detector.loop_closures)}")
                
                self._optimize_trajectory()
    
    def _optimize_trajectory(self) -> None:
        """Run pose graph optimization and update poses"""
        if self.pose_optimizer is None or len(self.poses) < 10:
            return
        
        # Build odometry edges (sequential)
        odometry_edges = []
        for i in range(len(self.poses) - 1):
            relative_pose = np.linalg.inv(self.poses[i]) @ self.poses[i+1]
            odometry_edges.append((i, i+1, relative_pose))
        
        # Get loop closure edges
        loop_edges = self.loop_detector.get_loop_constraints()
        
        if len(loop_edges) == 0:
            return
        
        # Optimize
        optimized_poses = self.pose_optimizer.optimize(
            poses=self.poses.copy(),
            odometry_edges=odometry_edges,
            loop_edges=loop_edges,
        )
        
        # Update poses
        self.poses = optimized_poses
        self.trajectory = [pose[:3, 3] for pose in optimized_poses]
        
        # Update SLAM engine's current pose
        if hasattr(self.slam, 'current_pose'):
            self.slam.current_pose = optimized_poses[-1].copy()
        
        print(f"[SLAM] ✓ Pose graph optimization complete")


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
        
        # Hybrid odometry (passed from parent SLAMEngine)
        self.hybrid_odometry = None  # Will be set by SLAMEngine if enabled
        
        # Depth consistency checker (Phase 3 Week 6 Day 2)
        self.depth_consistency = None  # Will be set by SLAMEngine if enabled
        
        # Camera intrinsics (will be estimated from frame size)
        self.K: Optional[np.ndarray] = None
        
        # State
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_keypoints: Optional[List] = None
        self.prev_descriptors: Optional[np.ndarray] = None
        self.prev_depth: Optional[np.ndarray] = None  # Store depth for scale
        self.prev_uncertainty: Optional[np.ndarray] = None  # Store uncertainty map
        self.current_pose: np.ndarray = np.eye(4)  # Start at origin
        self.scale: float = config.scale_factor  # Scale for monocular
        self.use_depth_scale = True  # Use depth for scale when available
        
        # Keyframe management (reduce drift by only processing key frames)
        self.last_keyframe_idx: int = -1
        self.keyframe_interval: int = 2  # AGGRESSIVE: Process every 2nd frame (was 3)
        self.frames_since_keyframe: int = 0
        self.last_keyframe_pose: np.ndarray = np.eye(4)
        self.motion_since_keyframe: np.ndarray = np.eye(4)  # Accumulated motion
        
        # Temporal smoothing for drift reduction
        self.scale_filter = SimpleKalmanFilter(process_variance=0.005, measurement_variance=0.05)
        self.translation_history = deque(maxlen=5)  # Moving average over last 5 frames
        self.initial_height: Optional[float] = None  # Ground plane constraint
        self.prev_scale: Optional[float] = None  # Track scale for clamping
        
        # Statistics
        self.num_matches_history: List[int] = []
        self.num_inliers_history: List[int] = []
        self.total_poses_tracked: int = 0  # Counter for successfully tracked poses
        self.num_keyframes: int = 0
    
    def _select_features_with_depth(
        self,
        keypoints: List,
        descriptors: np.ndarray,
        depth_map: np.ndarray,
        uncertainty_map: Optional[np.ndarray] = None,
        max_features: int = 1500
    ) -> Tuple[List, np.ndarray]:
        """
        Select best features for tracking based on depth validity and uncertainty.
        
        Prioritizes features with:
        1. High feature response (ORB corner strength)
        2. Valid depth (not too close/far)
        3. Low uncertainty (confident depth estimate)
        
        Args:
            keypoints: List of cv2.KeyPoint objects
            descriptors: Feature descriptors (Nx128 for ORB)
            depth_map: Depth map (HxW) in mm
            uncertainty_map: Optional uncertainty map [0,1]
            max_features: Maximum features to return
        
        Returns:
            (selected_keypoints, selected_descriptors)
        """
        if len(keypoints) <= max_features:
            return keypoints, descriptors
        
        try:
            # Score each feature
            scores = []
            for i, kp in enumerate(keypoints):
                u, v = int(kp.pt[0]), int(kp.pt[1])
                
                # Bounds check
                if not (0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]):
                    scores.append(0.0)
                    continue
                
                # 1. Feature response (ORB corner strength)
                response_score = min(kp.response / 100.0, 1.0)  # Normalize to [0,1]
                
                # 2. Depth validity
                depth = depth_map[v, u]
                if 100 < depth < 8000:  # Valid range (10cm to 8m)
                    depth_score = 1.0
                elif depth > 0:  # Outside ideal range but still valid
                    depth_score = 0.5
                else:
                    depth_score = 0.0
                
                # 3. Uncertainty (lower is better)
                if uncertainty_map is not None:
                    uncertainty = uncertainty_map[v, u]
                    certainty_score = 1.0 - uncertainty
                else:
                    certainty_score = 0.7  # Default moderate certainty
                
                # Combined score (weighted average)
                total_score = (
                    response_score * 0.4 +
                    depth_score * 0.3 +
                    certainty_score * 0.3
                )
                scores.append(total_score)
            
            # Select top features
            scores = np.array(scores)
            top_indices = np.argsort(scores)[-max_features:]
            
            selected_kp = [keypoints[i] for i in top_indices]
            selected_desc = descriptors[top_indices]
            
            return selected_kp, selected_desc
            
        except Exception as e:
            print(f"[SLAM] Depth-guided feature selection failed: {e}, using all features")
            return keypoints[:max_features], descriptors[:max_features]
    
    def track(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_idx: int,
        depth_map: Optional[np.ndarray] = None,
        uncertainty_map: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Track camera motion using feature matching
        
        Args:
            frame: Input frame (BGR)
            timestamp: Frame timestamp
            frame_idx: Frame index
            depth_map: Optional depth map for scale recovery (HxW, values in mm)
            uncertainty_map: Optional depth uncertainty map (HxW, values 0-1)
        
        Returns:
            4x4 pose matrix or None if tracking failed
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Preprocess depth map if provided
        if depth_map is not None:
            depth_map = preprocess_depth_map(depth_map, bilateral_d=5, sigma_color=50.0, sigma_space=50.0)
        
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
        
        # Apply depth-guided feature selection if depth is available
        if depth_map is not None and len(keypoints) > 1500:
            keypoints, descriptors = self._select_features_with_depth(
                keypoints, descriptors, depth_map,
                uncertainty_map=uncertainty_map,
                max_features=1500
            )
            print(f"[SLAM] Depth-guided feature selection: selected {len(keypoints)} features")
        
        if self.prev_frame is None:
            # First frame - initialize as keyframe
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_depth = depth_map
            self.last_keyframe_idx = frame_idx
            self.last_keyframe_pose = self.current_pose.copy()
            self.num_keyframes += 1
            return self.current_pose.copy()
        
        # Keyframe selection: only process full SLAM on keyframes
        # For non-keyframes, use motion model to predict pose
        self.frames_since_keyframe += 1
        is_keyframe = (self.frames_since_keyframe >= self.keyframe_interval)
        
        # Also trigger keyframe if motion is large (adaptive)
        if not is_keyframe and len(keypoints) > 50:
            # Quick check: if many features moved significantly, force keyframe
            if self.prev_keypoints is not None and len(self.prev_keypoints) > 50:
                # Sample some features to check motion
                sample_size = min(50, len(keypoints), len(self.prev_keypoints))
                avg_motion = 0
                for i in range(sample_size):
                    pt1 = self.prev_keypoints[i].pt
                    pt2 = keypoints[i].pt
                    avg_motion += np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
                avg_motion /= sample_size
                
                # If average motion > 20 pixels, force keyframe
                if avg_motion > 20:
                    is_keyframe = True
        
        if not is_keyframe:
            # Non-keyframe: use constant velocity model
            # Assume same motion as last frame
            self.current_pose = self.current_pose @ self.motion_since_keyframe
            self.total_poses_tracked += 1
            return self.current_pose.copy()
        
        # This is a keyframe - do full SLAM processing
        self.frames_since_keyframe = 0
        self.last_keyframe_idx = frame_idx
        self.num_keyframes += 1
        
        # Loop closure is handled by SLAMEngine after this method returns
        
        if descriptors is None or len(keypoints) < 5:  # Very low threshold
            # Not enough features - try depth odometry first if available
            if depth_map is not None and self.depth_odometry is not None:
                print(f"[SLAM] Frame {frame_idx}: Using DEPTH ODOMETRY (features: {len(keypoints) if keypoints else 0})")
                depth_pose = self._track_depth_odometry(depth_map, gray)
                if depth_pose is not None:
                    self.prev_frame = gray
                    return depth_pose
            
            # Try optical flow fallback
            if self.prev_frame is not None:
                print(f"[SLAM] Frame {frame_idx}: Trying optical flow (features: {len(keypoints) if keypoints else 0})")
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
                print(f"[SLAM] Frame {frame_idx}: Using depth odometry (matches: {len(good_matches)})")
                depth_pose = self._track_depth_odometry(depth_map, gray)
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
        
        # Dampen small rotations (reduce rotation drift)
        # Extract rotation angle
        trace_R = np.trace(R)
        angle = np.arccos(np.clip((trace_R - 1) / 2, -1.0, 1.0))
        
        # If rotation is tiny (< 2 degrees), likely noise - dampen it
        if angle < np.radians(2):
            # Blend toward identity (no rotation)
            damping = 0.5  # Keep 50% of small rotations
            R = R * damping + np.eye(3) * (1 - damping)
            # Re-orthogonalize
            U, S, Vt = np.linalg.svd(R)
            R = U @ Vt
        
        # Apply depth consistency filtering (Phase 3 Week 6 Day 2)
        pts1_filtered = pts1
        pts2_filtered = pts2
        inlier_consistency_ratio = 1.0
        
        if (self.depth_consistency is not None and 
            depth_map is not None and 
            self.prev_depth is not None and 
            E is not None):
            
            # Sample depth at matched features
            depths1 = np.array([
                self.prev_depth[int(pt[1]), int(pt[0])] 
                if 0 <= int(pt[1]) < self.prev_depth.shape[0] and 
                   0 <= int(pt[0]) < self.prev_depth.shape[1]
                else 0.0
                for pt in pts1
            ])
            
            depths2 = np.array([
                depth_map[int(pt[1]), int(pt[0])] 
                if 0 <= int(pt[1]) < depth_map.shape[0] and 
                   0 <= int(pt[0]) < depth_map.shape[1]
                else 0.0
                for pt in pts2
            ])
            
            # Filter outliers using depth consistency
            pts1_filtered, pts2_filtered, depths1_f, depths2_f, inlier_consistency_ratio = \
                self.depth_consistency.check_and_filter(
                    pts1, pts2, depths1, depths2, E, self.K
                )
            
            # Log filtering results
            print(f"[SLAM] Frame {frame_idx}: Depth consistency filtering: "
                  f"{len(pts1_filtered)}/{len(pts1)} ({inlier_consistency_ratio:.1%} inliers)")
            
            # Update points for subsequent processing
            pts1 = pts1_filtered
            pts2 = pts2_filtered
        
        # Estimate scale from depth if available
        scale_confidence = 0.3  # Default low confidence
        if depth_map is not None and self.prev_depth is not None and self.use_depth_scale:
            # Try robust scale estimation first (with RANSAC + uncertainty)
            scale_result = self._estimate_scale_robust(
                pts1, pts2, good_matches, self.prev_depth, depth_map,
                uncertainty_prev=self.prev_uncertainty,
                uncertainty_curr=uncertainty_map
            )
            
            # Fallback to simpler method if robust fails
            if scale_result is None:
                scale_result = self._estimate_scale_from_depth(
                    pts1, pts2, good_matches, depth_map, self.prev_depth
                )
            
            if scale_result is not None:
                scale_estimate, scale_confidence = scale_result
                # Use Kalman filter for temporal smoothing
                self.scale = self.scale_filter.update(scale_estimate, scale_confidence)
                print(f"[SLAM] Scale: {self.scale:.1f}mm/unit (confidence: {scale_confidence:.2f})")
        elif self.scale == 1.0:
            # Heuristic fallback: typical human walking speed ~1.4 m/s
            # At 30 FPS: ~47mm per frame
            motion_magnitude = np.linalg.norm(t)
            if motion_magnitude > 1e-6:
                fallback_scale = 47.0 / motion_magnitude
                self.scale = self.scale_filter.update(fallback_scale, 0.5)  # Medium confidence
        
        # Apply scale
        t_scaled = t * self.scale
        
        # Apply translation smoothing (moving average)
        self.translation_history.append(t_scaled.flatten())
        if len(self.translation_history) > 1:
            # Weighted average: recent frames get more weight
            weights = np.exp(np.linspace(-1, 0, len(self.translation_history)))
            weights /= weights.sum()
            t_smoothed = np.zeros(3)
            for i, trans in enumerate(self.translation_history):
                t_smoothed += weights[i] * trans
            t_scaled = t_smoothed.reshape(-1, 1)
        
        # Hybrid pose fusion (Phase 3 Week 6)
        if self.hybrid_odometry is not None and depth_map is not None and self.depth_odometry is not None:
            # Build visual pose
            visual_pose = np.eye(4)
            visual_pose[:3, :3] = R
            visual_pose[:3, 3] = t_scaled.flatten()
            
            # Compute depth-based pose
            depth_pose = self._track_depth_odometry(depth_map, gray)
            
            # Compute texture score
            from orion.slam.hybrid_odometry import compute_texture_score, check_depth_range
            texture_score = compute_texture_score(gray)
            
            # Check depth range validity
            depth_range_ok = check_depth_range(depth_map, min_depth=100.0, max_depth=10000.0)
            
            # Compute confidences
            visual_conf = self.hybrid_odometry.estimate_visual_confidence(
                inlier_ratio=num_inliers / len(good_matches) if len(good_matches) > 0 else 0.0,
                num_matches=len(good_matches),
                texture_score=texture_score
            )
            
            depth_conf = self.hybrid_odometry.estimate_depth_confidence(
                uncertainty_map=uncertainty_map,
                valid_ratio=np.sum(depth_map > 0) / depth_map.size if depth_map is not None else 0.0,
                depth_range_ok=depth_range_ok
            )
            
            # Fuse poses
            fused_pose, fusion_mode = self.hybrid_odometry.fuse_poses(
                visual_pose, depth_pose, visual_conf, depth_conf
            )
            
            if fused_pose is not None and fusion_mode == "fusion":
                # Use fused pose
                R = fused_pose[:3, :3]
                t_scaled = fused_pose[:3, 3].reshape(-1, 1)
                print(f"[SLAM] Pose fusion: visual={visual_conf:.2f}, depth={depth_conf:.2f}, mode={fusion_mode}")
        
        # Ground plane constraint: lock vertical motion to reduce drift
        if self.initial_height is None:
            # Initialize from first good pose
            self.initial_height = self.current_pose[1, 3]  # Y coordinate (height)
        else:
            # Constrain vertical drift: allow only small changes (±10cm)
            predicted_height = self.current_pose[1, 3] + t_scaled[1]
            height_drift = predicted_height - self.initial_height
            if abs(height_drift) > 100:  # > 10cm drift
                # Clamp translation to stay within bounds
                t_scaled[1] = self.initial_height + np.sign(height_drift) * 100 - self.current_pose[1, 3]
        
        # Update current pose
        # T_new = T_current @ T_relative
        T_relative = np.eye(4)
        T_relative[:3, :3] = R
        T_relative[:3, 3] = t_scaled.flatten()
        
        # Store this as motion since last keyframe (for non-keyframe prediction)
        self.motion_since_keyframe = T_relative.copy()
        
        self.current_pose = self.current_pose @ T_relative
        self.last_keyframe_pose = self.current_pose.copy()
        
        # Increment pose counter
        self.total_poses_tracked += 1
        
        # Update previous frame and depth
        self.prev_frame = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_depth = depth_map
        self.prev_uncertainty = uncertainty_map  # Store uncertainty for next frame
        
        return self.current_pose.copy()
    
    def _estimate_scale_robust(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        good_matches: List,
        depth_prev: np.ndarray,
        depth_curr: np.ndarray,
        uncertainty_prev: Optional[np.ndarray] = None,
        uncertainty_curr: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Robust scale estimation using RANSAC + uncertainty weighting.
        
        Args:
            pts1: Points in previous frame (Nx2)
            pts2: Points in current frame (Nx2)
            good_matches: List of good matches
            depth_prev: Previous depth map (HxW)
            depth_curr: Current depth map (HxW)
            uncertainty_prev: Optional uncertainty map for previous frame
            uncertainty_curr: Optional uncertainty map for current frame
        
        Returns:
            (scale, confidence) or None if estimation failed
        """
        from orion.slam.depth_utils import backproject_point
        
        try:
            # 1. Sample depth at matched features
            scales = []
            weights = []
            
            for i, match in enumerate(good_matches):
                try:
                    u1, v1 = int(pts1[i][0]), int(pts1[i][1])
                    u2, v2 = int(pts2[i][0]), int(pts2[i][1])
                    
                    # Bounds check
                    if not (0 <= v1 < depth_prev.shape[0] and 0 <= u1 < depth_prev.shape[1] and
                            0 <= v2 < depth_curr.shape[0] and 0 <= u2 < depth_curr.shape[1]):
                        continue
                    
                    # Get depth values
                    d1 = depth_prev[v1, u1]
                    d2 = depth_curr[v2, u2]
                    
                    # Get uncertainty (default to 0.5 if not provided)
                    if uncertainty_prev is not None and uncertainty_curr is not None:
                        unc1 = uncertainty_prev[v1, u1]
                        unc2 = uncertainty_curr[v2, u2]
                    else:
                        unc1 = 0.5
                        unc2 = 0.5
                    
                    # Skip if invalid or too uncertain
                    if d1 < 100 or d1 > 10000 or d2 < 100 or d2 > 10000:
                        continue
                    if unc1 > 0.7 or unc2 > 0.7:
                        continue
                    
                    # Compute 3D motion
                    p1_3d = backproject_point(u1, v1, d1, self.K)
                    p2_3d = backproject_point(u2, v2, d2, self.K)
                    depth_motion = np.linalg.norm(p2_3d - p1_3d)
                    
                    # Compute 2D motion (pixels)
                    pixel_motion = np.linalg.norm(pts2[i] - pts1[i])
                    
                    if pixel_motion > 1.0 and 1.0 < depth_motion < 500:  # Reasonable range
                        scale = depth_motion
                        scales.append(scale)
                        
                        # Weight by certainty
                        certainty = (1 - unc1) * (1 - unc2)
                        weights.append(certainty)
                        
                except (IndexError, ValueError):
                    continue
            
            if len(scales) < 10:
                return None
            
            # 2. RANSAC to remove outliers
            scales = np.array(scales)
            weights = np.array(weights)
            
            best_scale = None
            best_inliers = 0
            best_inlier_mask = None
            
            n_iterations = min(50, len(scales))
            for _ in range(n_iterations):
                # Sample random scale (weighted by confidence)
                idx = np.random.choice(len(scales), p=weights/weights.sum())
                candidate = scales[idx]
                
                # Count inliers (within 20% of candidate)
                inlier_mask = np.abs(scales - candidate) < (0.2 * candidate)
                num_inliers = np.sum(inlier_mask)
                
                if num_inliers > best_inliers:
                    best_inliers = num_inliers
                    best_scale = candidate
                    best_inlier_mask = inlier_mask
            
            if best_inliers < len(scales) * 0.3:  # Need 30% inliers
                return None
            
            # 3. Weighted median of inliers
            inlier_scales = scales[best_inlier_mask]
            inlier_weights = weights[best_inlier_mask]
            
            # Weighted median
            sorted_idx = np.argsort(inlier_scales)
            sorted_scales = inlier_scales[sorted_idx]
            sorted_weights = inlier_weights[sorted_idx]
            cumsum = np.cumsum(sorted_weights)
            median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
            
            final_scale = sorted_scales[median_idx]
            confidence = float(best_inliers) / len(scales)
            
            # Sanity check: 5mm to 500mm per frame
            if 5 < final_scale < 500:
                return (final_scale, confidence)
            
            return None
            
        except Exception as e:
            print(f"[SLAM] Robust scale estimation failed: {e}")
            return None
    
    def _estimate_scale_from_depth(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        matches: List,
        curr_depth: np.ndarray,
        prev_depth: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """
        Improved scale estimation using multiple depth samples with confidence weighting.
        
        Uses three methods:
        1. Matched keypoint depth ratios (most reliable)
        2. Ground plane fitting (for floor/ground detection)
        3. Optical flow magnitude correlation
        
        Returns:
            Tuple of (scale_estimate, confidence) or None if failed
            - scale_estimate: Scale factor in mm per unit
            - confidence: 0.0-1.0 indicating reliability
        """
        try:
            scale_estimates = []  # (scale, confidence) pairs
            
            # Method 1: Matched keypoint depth analysis
            depth_scales = []
            depth_confidences = []
            
            for i, m in enumerate(matches):
                # Get pixel coordinates
                u1, v1 = int(pts1[i][0]), int(pts1[i][1])
                u2, v2 = int(pts2[i][0]), int(pts2[i][1])
                
                # Bounds check
                if not (0 <= v1 < prev_depth.shape[0] and 0 <= u1 < prev_depth.shape[1] and
                        0 <= v2 < curr_depth.shape[0] and 0 <= u2 < curr_depth.shape[1]):
                    continue
                
                d1 = prev_depth[v1, u1]
                d2 = curr_depth[v2, u2]
                
                # Valid depth check (100mm to 10m)
                if not (100 < d1 < 10000 and 100 < d2 < 10000):
                    continue
                
                # Sample neighboring depths for local consistency check
                neighborhood_size = 3
                v1_min, v1_max = max(0, v1-neighborhood_size), min(prev_depth.shape[0], v1+neighborhood_size+1)
                u1_min, u1_max = max(0, u1-neighborhood_size), min(prev_depth.shape[1], u1+neighborhood_size+1)
                
                local_depths_prev = prev_depth[v1_min:v1_max, u1_min:u1_max]
                depth_variance_prev = np.std(local_depths_prev[local_depths_prev > 100])
                
                # Low variance = high confidence (smooth surface)
                confidence = 1.0 / (1.0 + depth_variance_prev / 100.0)  # Higher variance = lower confidence
                confidence = np.clip(confidence, 0.1, 1.0)
                
                # Compute 3D motion from depth change
                # Project to 3D, compute distance
                fx, fy = self.K[0, 0], self.K[1, 1]
                cx, cy = self.K[0, 2], self.K[1, 2]
                
                # 3D points in camera frame
                x1 = (u1 - cx) * d1 / fx
                y1 = (v1 - cy) * d1 / fy
                z1 = d1
                
                x2 = (u2 - cx) * d2 / fx
                y2 = (v2 - cy) * d2 / fy
                z2 = d2
                
                # 3D motion magnitude
                motion_3d = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                
                # 2D pixel motion
                motion_2d = np.sqrt((u2-u1)**2 + (v2-v1)**2)
                
                # Scale = 3D motion / 2D motion (but we work with normalized coords)
                # For now, use depth change as proxy
                if motion_3d > 1.0 and motion_3d < 500:  # 1mm to 50cm reasonable
                    depth_scales.append(motion_3d)
                    depth_confidences.append(confidence)
            
            # Weighted average of depth-based scales
            if len(depth_scales) >= 5:
                depth_scales = np.array(depth_scales)
                depth_confidences = np.array(depth_confidences)
                
                # Remove outliers (beyond 2 std devs)
                median = np.median(depth_scales)
                std = np.std(depth_scales)
                inliers = np.abs(depth_scales - median) < 2 * std
                
                if np.sum(inliers) >= 3:
                    depth_scales_clean = depth_scales[inliers]
                    depth_confidences_clean = depth_confidences[inliers]
                    
                    # Weighted average
                    weighted_scale = np.average(depth_scales_clean, weights=depth_confidences_clean)
                    overall_confidence = np.mean(depth_confidences_clean)
                    
                    scale_estimates.append((weighted_scale, overall_confidence))
            
            # Method 2: Ground plane estimation
            # Sample bottom 20% of image (likely floor/ground)
            h, w = curr_depth.shape
            ground_region = curr_depth[int(h*0.8):, :]
            ground_depths = ground_region[ground_region > 100]
            
            if len(ground_depths) > 100:
                # Median ground depth
                ground_depth = np.median(ground_depths)
                ground_std = np.std(ground_depths)
                
                # If ground is relatively flat (low variance), use it as reference
                if ground_std < 200:  # Less than 20cm variance
                    # Assume typical human walking: ~1.4 m/s = 47mm per frame at 30fps
                    # Ground depth change correlates with forward motion
                    ground_confidence = 0.7  # Medium confidence
                    scale_estimates.append((47.0, ground_confidence))
            
            # Method 3: Combine estimates
            if len(scale_estimates) == 0:
                # Fallback: human walking heuristic
                return (47.0, 0.3)  # Low confidence
            
            # Weighted fusion of all estimates
            scales = np.array([s[0] for s in scale_estimates])
            confidences = np.array([s[1] for s in scale_estimates])
            
            final_scale = np.average(scales, weights=confidences)
            final_confidence = np.max(confidences)  # Use best confidence
            
            # Sanity check: scale should be 5mm-500mm per frame
            if 5 < final_scale < 500:
                return (final_scale, final_confidence)
            
            return None
            
        except Exception as e:
            # Silent fail, use fallback
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
            self.current_pose = self.current_pose @ pose
            
            print(f"[SLAM] Optical flow tracking: flow={flow_magnitude:.1f}px, t=[{t_x:.0f}, {t_y:.0f}, {t_z:.0f}]mm")
            return self.current_pose.copy()
            
        except Exception as e:
            print(f"[SLAM] Optical flow failed: {e}")
            return None
    
    def _track_depth_odometry(self, depth_map: np.ndarray, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Fallback tracking using depth odometry when features completely fail.
        Uses ICP on 3D point clouds from depth.
        """
        try:
            if self.depth_odometry is None:
                return None
            
            # Compute odometry from depth maps
            result = self.depth_odometry.compute_odometry(self.prev_depth, depth_map)
            
            if result is not None:
                pose = result['pose']
                self.current_pose = self.current_pose @ pose
                
                t = pose[:3, 3]
                print(f"[SLAM] Depth odometry: t=[{t[0]:.0f}, {t[1]:.0f}, {t[2]:.0f}]mm, points={result.get('num_points', 0)}")
                return self.current_pose.copy()
            
            return None
            
        except Exception as e:
            print(f"[SLAM] Depth odometry failed: {e}")
            return None
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics"""
        if not self.num_matches_history:
            return {
                'avg_matches': 0,
                'avg_inliers': 0,
                'scale': self.scale,
                'total_poses': self.total_poses_tracked,
                'num_keyframes': self.num_keyframes,
                'keyframe_ratio': 0.0,
            }
        
        keyframe_ratio = self.num_keyframes / self.total_poses_tracked if self.total_poses_tracked > 0 else 0.0
        
        return {
            'avg_matches': np.mean(self.num_matches_history),
            'avg_inliers': np.mean(self.num_inliers_history),
            'scale': self.scale,
            'total_poses': self.total_poses_tracked,
            'num_keyframes': self.num_keyframes,
            'keyframe_ratio': keyframe_ratio,
        }


@dataclass
class DepthOdometryConfig:
    """Configuration for depth-based odometry fallback"""
    max_depth: float = 10000.0  # mm
    min_depth: float = 100.0    # mm
    sample_rate: int = 4        # Sample every Nth pixel
    max_icp_iterations: int = 30
    convergence_threshold: float = 0.001


class DepthOdometry:
    """
    Depth-based odometry using ICP on 3D point clouds.
    Fallback when visual features completely fail.
    """
    def __init__(self, K: np.ndarray, config: DepthOdometryConfig):
        self.K = K
        self.config = config
        self.prev_points = None
    
    def compute_odometry(self, prev_depth: Optional[np.ndarray], curr_depth: np.ndarray) -> Optional[Dict]:
        """
        Compute camera motion from depth maps using ICP.
        Returns dict with 'pose' (4x4 matrix) and 'num_points'.
        """
        try:
            if prev_depth is None:
                return None
            
            # Convert depth maps to 3D point clouds
            curr_points = self._depth_to_points(curr_depth)
            if curr_points is None or len(curr_points) < 100:
                return None
            
            if self.prev_points is None or len(self.prev_points) < 100:
                self.prev_points = curr_points
                return None
            
            # Simple ICP: find transformation from prev to curr
            R, t = self._icp(self.prev_points, curr_points)
            
            if R is None or t is None:
                self.prev_points = curr_points
                return None
            
            # Construct 4x4 pose matrix
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
            
            self.prev_points = curr_points
            
            return {
                'pose': pose,
                'num_points': len(curr_points)
            }
            
        except Exception as e:
            print(f"[DepthOdometry] Error: {e}")
            return None
    
    def _depth_to_points(self, depth: np.ndarray) -> Optional[np.ndarray]:
        """Convert depth map to 3D point cloud"""
        try:
            h, w = depth.shape
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]
            
            points = []
            sample = self.config.sample_rate
            
            for v in range(0, h, sample):
                for u in range(0, w, sample):
                    d = depth[v, u]
                    
                    # Valid depth check
                    if self.config.min_depth < d < self.config.max_depth:
                        # Project to 3D (in mm)
                        x = (u - cx) * d / fx
                        y = (v - cy) * d / fy
                        z = d
                        points.append([x, y, z])
            
            if len(points) < 100:
                return None
            
            return np.array(points, dtype=np.float32)
            
        except Exception:
            return None
    
    def _icp(self, src_points: np.ndarray, dst_points: np.ndarray, 
             max_iterations: int = 30) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Simple ICP algorithm to find R, t such that dst = R @ src + t
        """
        try:
            # Sample points if too many (for speed)
            if len(src_points) > 1000:
                indices = np.random.choice(len(src_points), 1000, replace=False)
                src_points = src_points[indices]
            if len(dst_points) > 1000:
                indices = np.random.choice(len(dst_points), 1000, replace=False)
                dst_points = dst_points[indices]
            
            # Initialize with identity
            R = np.eye(3)
            t = np.zeros(3)
            
            prev_error = float('inf')
            
            for iteration in range(max_iterations):
                # Transform source points
                src_transformed = (R @ src_points.T).T + t
                
                # Find nearest neighbors (simple brute force for small sets)
                distances = np.linalg.norm(dst_points[:, np.newaxis, :] - src_transformed[np.newaxis, :, :], axis=2)
                nearest_indices = np.argmin(distances, axis=0)
                
                # Get matched pairs
                matched_dst = dst_points[nearest_indices]
                
                # Compute centroids
                centroid_src = np.mean(src_transformed, axis=0)
                centroid_dst = np.mean(matched_dst, axis=0)
                
                # Center the point sets
                src_centered = src_transformed - centroid_src
                dst_centered = matched_dst - centroid_dst
                
                # Compute rotation using SVD
                H = src_centered.T @ dst_centered
                U, _, Vt = np.linalg.svd(H)
                R_delta = Vt.T @ U.T
                
                # Ensure proper rotation (det(R) = 1)
                if np.linalg.det(R_delta) < 0:
                    Vt[-1, :] *= -1
                    R_delta = Vt.T @ U.T
                
                # Update transformation
                t_delta = centroid_dst - R_delta @ centroid_src
                
                R = R_delta @ R
                t = R_delta @ t + t_delta
                
                # Compute error
                error = np.mean(np.linalg.norm(matched_dst - src_transformed, axis=1))
                
                # Check convergence
                if abs(prev_error - error) < self.config.convergence_threshold:
                    break
                
                prev_error = error
            
            # Sanity check: reasonable translation (< 1m per frame)
            if np.linalg.norm(t) > 1000:  # > 1 meter
                return None, None
            
            return R, t
            
        except Exception as e:
            print(f"[ICP] Error: {e}")
            return None, None
    
    def add_keyframe_to_loop_detector(
        self,
        frame_id: int,
        keypoints: List,
        descriptors: np.ndarray,
        gray: Optional[np.ndarray] = None,
    ) -> None:
        """
        Expose method for parent SLAM engine to add keyframes to loop detector.
        
        This is called from SLAMEngine which manages the loop closure detector.
        """
        # Placeholder - implementation happens in SLAMEngine
        pass
