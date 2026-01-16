"""
Enhanced Spatial Reasoning for Orion
=====================================

Improves 3D spatial understanding with:
1. Better camera motion estimation (rotation, translation)
2. Improved depth consistency and scaling
3. More accurate scene graph relations
4. Better zone classification and spatial memory

Key improvements:
- Proper camera intrinsic estimation
- Relative motion tracking (left/right/forward)
- Depth map refinement and consistency
- Multi-frame spatial reasoning
- Better support/containment detection

Author: Orion Research Team
Date: November 2025
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R


@dataclass
class CameraMotion:
    """Camera motion between frames"""
    rotation: np.ndarray  # 3x3 rotation matrix
    translation: np.ndarray  # 3D translation vector
    rotation_euler: np.ndarray  # (roll, pitch, yaw) in degrees
    
    def describe(self) -> str:
        """Human-readable motion description"""
        roll, pitch, yaw = self.rotation_euler
        tx, ty, tz = self.translation
        
        descriptions = []
        
        # Rotation
        if abs(yaw) > 5:
            if yaw > 0:
                descriptions.append(f"turning right {abs(yaw):.1f}°")
            else:
                descriptions.append(f"turning left {abs(yaw):.1f}°")
        
        if abs(pitch) > 5:
            if pitch > 0:
                descriptions.append(f"tilting up {abs(pitch):.1f}°")
            else:
                descriptions.append(f"tilting down {abs(pitch):.1f}°")
        
        if abs(roll) > 5:
            if roll > 0:
                descriptions.append(f"rolling right {abs(roll):.1f}°")
            else:
                descriptions.append(f"rolling left {abs(roll):.1f}°")
        
        # Translation
        if abs(tz) > 0.1:
            if tz > 0:
                descriptions.append(f"moving forward {abs(tz):.2f}m")
            else:
                descriptions.append(f"moving backward {abs(tz):.2f}m")
        
        if abs(tx) > 0.1:
            if tx > 0:
                descriptions.append(f"moving right {abs(tx):.2f}m")
            else:
                descriptions.append(f"moving left {abs(tx):.2f}m")
        
        if abs(ty) > 0.1:
            if ty > 0:
                descriptions.append(f"moving down {abs(ty):.2f}m")
            else:
                descriptions.append(f"moving up {abs(ty):.2f}m")
        
        return ", ".join(descriptions) if descriptions else "stationary"


class EnhancedSpatialReasoning:
    """
    Enhanced spatial reasoning with accurate 3D understanding
    """
    
    def __init__(self):
        """Initialize spatial reasoner"""
        # Camera parameters (estimated from frame size)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        
        # Previous frame data for motion estimation
        self.prev_frame_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        # ORB feature detector for motion
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Depth consistency tracking
        self.depth_scale_factor = 1.0
        self.depth_history = []
    
    def estimate_camera_intrinsics(self, frame_width: int, frame_height: int):
        """
        Estimate camera intrinsic parameters from frame size
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            dict with fx, fy, cx, cy
        """
        # Typical FOV for webcam/phone is ~60-70 degrees
        fov_deg = 65.0
        fov_rad = np.deg2rad(fov_deg)
        
        # Focal length from FOV
        self.fx = frame_width / (2 * np.tan(fov_rad / 2))
        self.fy = self.fx  # Assume square pixels
        
        # Principal point at center
        self.cx = frame_width / 2
        self.cy = frame_height / 2
        
        print(f"[SpatialReasoning] Camera intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")
        
        return {
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy
        }
    
    def estimate_camera_motion(self, 
                              frame: np.ndarray,
                              depth_map: np.ndarray) -> Optional[CameraMotion]:
        """
        Estimate camera motion (rotation + translation) between frames
        
        Args:
            frame: Current frame (BGR)
            depth_map: Depth map (H, W) in millimeters
            
        Returns:
            CameraMotion or None if not enough features
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if self.prev_keypoints is None:
            # First frame, just store
            self.prev_frame_gray = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return None
        
        if descriptors is None or len(keypoints) < 10:
            return None
        
        # Match features
        matches = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 8:
            return None
        
        # Get matched point coordinates
        prev_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches])
        curr_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches])
        
        # Get depth for matched points
        prev_depths = []
        curr_depths = []
        valid_matches = []
        
        for i, (pp, cp) in enumerate(zip(prev_pts, curr_pts)):
            px, py = int(pp[0]), int(pp[1])
            cx_pt, cy_pt = int(cp[0]), int(cp[1])
            
            # Get depth at keypoint
            h, w = depth_map.shape
            if 0 <= py < h and 0 <= px < w and 0 <= cy_pt < h and 0 <= cx_pt < w:
                # Use median depth around keypoint for robustness
                prev_depth = depth_map[max(0, py-2):min(h, py+3), max(0, px-2):min(w, px+3)].mean()
                curr_depth = depth_map[max(0, cy_pt-2):min(h, cy_pt+3), max(0, cx_pt-2):min(w, cx_pt+3)].mean()
                
                if prev_depth > 0 and curr_depth > 0:
                    prev_depths.append(prev_depth)
                    curr_depths.append(curr_depth)
                    valid_matches.append(i)
        
        if len(valid_matches) < 8:
            return None
        
        # Filter matched points
        prev_pts = prev_pts[valid_matches]
        curr_pts = curr_pts[valid_matches]
        prev_depths = np.array(prev_depths)
        curr_depths = np.array(curr_depths)
        
        # Back-project to 3D (convert mm to meters)
        prev_pts_3d = self._backproject_to_3d(prev_pts, prev_depths / 1000.0)
        curr_pts_3d = self._backproject_to_3d(curr_pts, curr_depths / 1000.0)
        
        # Estimate rigid transform using RANSAC
        try:
            # Use Kabsch algorithm with RANSAC
            R_mat, t_vec = self._estimate_rigid_transform(prev_pts_3d, curr_pts_3d)
            
            # Convert rotation to Euler angles
            rot = R.from_matrix(R_mat)
            euler = rot.as_euler('xyz', degrees=True)
            
            motion = CameraMotion(
                rotation=R_mat,
                translation=t_vec,
                rotation_euler=euler
            )
            
            # Update for next frame
            self.prev_frame_gray = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            
            return motion
            
        except Exception as e:
            print(f"[SpatialReasoning] Motion estimation failed: {e}")
            return None
    
    def _backproject_to_3d(self, points_2d: np.ndarray, depths: np.ndarray) -> np.ndarray:
        """
        Back-project 2D points to 3D using depth
        
        Args:
            points_2d: (N, 2) array of (x, y) pixel coordinates
            depths: (N,) array of depths in meters
            
        Returns:
            (N, 3) array of 3D points
        """
        if self.fx is None:
            raise ValueError("Camera intrinsics not set")
        
        N = len(points_2d)
        points_3d = np.zeros((N, 3))
        
        for i in range(N):
            x, y = points_2d[i]
            z = depths[i]
            
            # Back-project
            points_3d[i, 0] = (x - self.cx) * z / self.fx
            points_3d[i, 1] = (y - self.cy) * z / self.fy
            points_3d[i, 2] = z
        
        return points_3d
    
    def _estimate_rigid_transform(self, src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate rigid transform (R, t) from src to dst using Kabsch with RANSAC
        
        Args:
            src: (N, 3) source points
            dst: (N, 3) destination points
            
        Returns:
            R: 3x3 rotation matrix
            t: 3D translation vector
        """
        # RANSAC parameters
        n_iterations = 100
        inlier_threshold = 0.05  # 5cm
        best_inliers = 0
        best_R = np.eye(3)
        best_t = np.zeros(3)
        
        N = len(src)
        for _ in range(n_iterations):
            # Sample 3 random points
            indices = np.random.choice(N, min(3, N), replace=False)
            src_sample = src[indices]
            dst_sample = dst[indices]
            
            # Compute transform
            R_mat, t_vec = self._kabsch(src_sample, dst_sample)
            
            # Count inliers
            transformed = (R_mat @ src.T).T + t_vec
            errors = np.linalg.norm(transformed - dst, axis=1)
            inliers = np.sum(errors < inlier_threshold)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_R = R_mat
                best_t = t_vec
        
        # Refine with all inliers
        if best_inliers > 3:
            transformed = (best_R @ src.T).T + best_t
            errors = np.linalg.norm(transformed - dst, axis=1)
            inlier_mask = errors < inlier_threshold
            
            if np.sum(inlier_mask) > 3:
                best_R, best_t = self._kabsch(src[inlier_mask], dst[inlier_mask])
        
        return best_R, best_t
    
    def _kabsch(self, src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Kabsch algorithm to find optimal rotation + translation
        
        Args:
            src: (N, 3) source points
            dst: (N, 3) destination points
            
        Returns:
            R: 3x3 rotation matrix
            t: 3D translation vector
        """
        # Center the points
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        
        src_centered = src - src_mean
        dst_centered = dst - dst_mean
        
        # Compute covariance matrix
        H = src_centered.T @ dst_centered
        
        # SVD
        U, S, Vt = np.linalg.svd(H)
        
        # Rotation
        R_mat = Vt.T @ U.T
        
        # Ensure proper rotation (det = 1)
        if np.linalg.det(R_mat) < 0:
            Vt[-1, :] *= -1
            R_mat = Vt.T @ U.T
        
        # Translation
        t_vec = dst_mean - R_mat @ src_mean
        
        return R_mat, t_vec
    
    def refine_depth_consistency(self, depth_map: np.ndarray, motion: Optional[CameraMotion] = None) -> np.ndarray:
        """
        Refine depth map for temporal consistency
        
        Args:
            depth_map: Raw depth map in millimeters
            motion: Optional camera motion for warping
            
        Returns:
            Refined depth map
        """
        # Temporal filtering
        self.depth_history.append(depth_map.copy())
        if len(self.depth_history) > 5:
            self.depth_history.pop(0)
        
        if len(self.depth_history) >= 3:
            # Median filter across frames for stability
            depth_stack = np.stack(self.depth_history, axis=0)
            refined_depth = np.median(depth_stack, axis=0)
        else:
            refined_depth = depth_map
        
        # Bilateral filter for edge-preserving smoothing
        refined_depth = cv2.bilateralFilter(
            refined_depth.astype(np.float32),
            d=5,
            sigmaColor=50,
            sigmaSpace=50
        )
        
        return refined_depth
    
    def compute_spatial_statistics(self, 
                                   depth_map: np.ndarray,
                                   detections: dict) -> dict:
        """
        Compute spatial statistics for better scene understanding
        
        Args:
            depth_map: Depth map in millimeters
            detections: Detection dict with boxes
            
        Returns:
            dict with spatial statistics
        """
        stats = {}
        
        # Overall scene depth
        valid_depth = depth_map[depth_map > 0]
        if len(valid_depth) > 0:
            stats['depth_min'] = valid_depth.min()
            stats['depth_max'] = valid_depth.max()
            stats['depth_median'] = np.median(valid_depth)
            stats['depth_std'] = valid_depth.std()
        
        # Per-object depth
        if detections.get("num_detections", 0) > 0:
            boxes = detections["boxes"]
            object_depths = []
            
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                roi_depth = depth_map[y1:y2, x1:x2]
                valid_roi = roi_depth[roi_depth > 0]
                
                if len(valid_roi) > 0:
                    object_depths.append(np.median(valid_roi))
            
            if object_depths:
                stats['object_depth_range'] = (min(object_depths), max(object_depths))
                stats['avg_object_distance'] = np.mean(object_depths) / 1000.0  # to meters
        
        return stats
