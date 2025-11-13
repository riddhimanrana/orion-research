"""
Semantic SLAM - Hybrid Visual + Object Landmark SLAM

Combines traditional ORB features with semantic object landmarks (detected by YOLO)
to improve tracking in texture-less areas.

Author: Orion Research Team
Date: November 2025
"""

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SemanticLandmark:
    """Semantic landmark from YOLO detection"""
    object_id: str
    class_name: str
    centroid_2d: Tuple[float, float]  # (x, y) in pixels
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    frame_idx: int
    
    # Optional 3D info
    depth_mm: Optional[float] = None
    centroid_3d: Optional[Tuple[float, float, float]] = None


class SemanticSLAM:
    """
    Hybrid SLAM using both visual features and semantic landmarks
    
    Workflow:
    1. Track ORB features (existing)
    2. Track semantic landmarks (beds, doors, furniture)
    3. Fuse both for robust pose estimation
    4. Fall back to landmarks when features fail
    """
    
    def __init__(
        self,
        base_slam,  # OpenCVSLAM instance
        use_landmarks: bool = True,
        landmark_weight: float = 0.3,  # Weight for landmark matching
        stable_object_classes: Optional[List[str]] = None
    ):
        """
        Initialize Semantic SLAM
        
        Args:
            base_slam: Base visual SLAM engine (OpenCVSLAM)
            use_landmarks: Enable semantic landmark tracking
            landmark_weight: Weight for landmark-based tracking (0-1)
            stable_object_classes: Object classes to use as landmarks
        """
        self.base_slam = base_slam
        self.use_landmarks = use_landmarks
        self.landmark_weight = landmark_weight
        
        # Stable objects make good landmarks (non-moving)
        self.stable_classes = stable_object_classes or [
            'bed', 'couch', 'chair', 'dining table', 'potted plant',
            'tv', 'laptop', 'refrigerator', 'oven', 'microwave',
            'sink', 'toilet', 'door', 'window', 'desk', 'cabinet',
            'bookshelf', 'wardrobe'
        ]
        
        # Landmark tracking
        self.landmarks: Dict[str, SemanticLandmark] = {}  # object_id -> landmark
        self.landmark_history: List[Dict[str, SemanticLandmark]] = []
        self.prev_landmarks: Optional[Dict[str, SemanticLandmark]] = None
        
        logger.info(f"[SemanticSLAM] Initialized with {len(self.stable_classes)} stable object classes")
    
    def track(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_idx: int,
        yolo_detections: Optional[List[Dict]] = None,
        depth_map: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Track camera pose using visual features + semantic landmarks
        
        Args:
            frame: Input frame (BGR)
            timestamp: Frame timestamp
            frame_idx: Frame index
            yolo_detections: YOLO detections with keys:
                - bbox: (x1, y1, x2, y2)
                - class: class name
                - confidence: detection confidence
                - entity_id: unique ID (optional)
                - depth_mm: depth in mm (optional)
                - centroid_3d_mm: (x, y, z) in mm (optional)
            depth_map: Optional depth map for scale recovery (HxW, mm)
        
        Returns:
            4x4 transformation matrix [R|t] or None if tracking lost
        """
        # 1. Track using base visual SLAM (ORB features) with depth
        visual_pose = self.base_slam.track(frame, timestamp, frame_idx, depth_map)
        
        # Check visual tracking success
        visual_success = (
            visual_pose is not None and 
            len(self.base_slam.num_inliers_history) > 0 and
            self.base_slam.num_inliers_history[-1] >= self.base_slam.config.min_matches
        )
        
        # 2. Extract semantic landmarks from detections
        current_landmarks = {}
        if yolo_detections and self.use_landmarks:
            current_landmarks = self._extract_landmarks(yolo_detections, frame_idx)
        
        # 3. Track landmarks and estimate pose
        landmark_pose = None
        landmark_success = False
        
        if current_landmarks and self.prev_landmarks:
            landmark_pose, landmark_success = self._track_landmarks(
                self.prev_landmarks,
                current_landmarks,
                frame.shape
            )
        
        # 4. Fuse poses based on confidence
        if visual_success and landmark_success:
            # Both succeeded: weighted fusion
            fused_pose = self._fuse_poses(
                visual_pose,
                landmark_pose,
                weight_visual=1.0 - self.landmark_weight
            )
            logger.debug(f"[SemanticSLAM] Frame {frame_idx}: Fused visual + landmark pose")
            final_pose = fused_pose
            
        elif visual_success:
            # Only visual succeeded
            logger.debug(f"[SemanticSLAM] Frame {frame_idx}: Using visual pose")
            final_pose = visual_pose
            
        elif landmark_success:
            # Only landmarks succeeded (rescue in texture-less area!)
            logger.info(f"[SemanticSLAM] Frame {frame_idx}: RESCUED by semantic landmarks")
            final_pose = landmark_pose
            
        else:
            # Both failed (normal for first frame)
            if frame_idx > 0:  # Only warn after first frame
                logger.warning(f"[SemanticSLAM] Frame {frame_idx}: Both visual and landmark tracking failed")
            final_pose = None
        
        # Update history
        self.prev_landmarks = current_landmarks if current_landmarks else self.prev_landmarks
        self.landmark_history.append(current_landmarks)
        
        return final_pose
    
    def _extract_landmarks(
        self,
        detections: List[Dict],
        frame_idx: int
    ) -> Dict[str, SemanticLandmark]:
        """Extract semantic landmarks from YOLO detections"""
        landmarks = {}
        
        for det in detections:
            class_name = det.get('class', '')
            
            # Only use stable objects as landmarks
            if class_name not in self.stable_classes:
                continue
            
            # Extract detection info
            bbox = det.get('bbox')
            confidence = det.get('confidence', 0.0)
            
            if bbox is None or confidence < 0.5:  # Filter low confidence
                continue
            
            x1, y1, x2, y2 = bbox
            centroid_x = (x1 + x2) / 2.0
            centroid_y = (y1 + y2) / 2.0
            
            # Create landmark
            object_id = det.get('entity_id', f"{class_name}_{frame_idx}")
            
            landmark = SemanticLandmark(
                object_id=object_id,
                class_name=class_name,
                centroid_2d=(centroid_x, centroid_y),
                bbox=bbox,
                confidence=confidence,
                frame_idx=frame_idx,
                depth_mm=det.get('depth_mm'),
                centroid_3d=det.get('centroid_3d_mm')
            )
            
            landmarks[object_id] = landmark
        
        if landmarks:
            logger.debug(f"[SemanticSLAM] Extracted {len(landmarks)} landmarks: {list(landmarks.keys())}")
        
        return landmarks
    
    def _track_landmarks(
        self,
        prev_landmarks: Dict[str, SemanticLandmark],
        curr_landmarks: Dict[str, SemanticLandmark],
        frame_shape: Tuple[int, int, int]
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Track landmarks between frames and estimate pose
        
        Returns:
            Tuple of (pose_matrix, success_flag)
        """
        # Match landmarks between frames
        matches = self._match_landmarks(prev_landmarks, curr_landmarks)
        
        if len(matches) < 3:  # Need at least 3 correspondences
            logger.debug(f"[SemanticSLAM] Too few landmark matches: {len(matches)}")
            return None, False
        
        # Extract matched points
        prev_points = np.array([prev_landmarks[m[0]].centroid_2d for m in matches], dtype=np.float32)
        curr_points = np.array([curr_landmarks[m[1]].centroid_2d for m in matches], dtype=np.float32)
        
        # Estimate transformation (Essential matrix approach simplified for 2D)
        # Use homography for now (works for planar scenes)
        try:
            H, mask = cv2.findHomography(
                prev_points,
                curr_points,
                cv2.RANSAC,
                ransacReprojThreshold=5.0
            )
            
            if H is None:
                return None, False
            
            inliers = np.sum(mask)
            if inliers < 3:
                return None, False
            
            # Convert homography to pose (simplified)
            # For full 3D, would need Essential matrix with camera intrinsics
            pose = self._homography_to_pose(H)
            
            logger.debug(f"[SemanticSLAM] Landmark tracking: {inliers}/{len(matches)} inliers")
            return pose, True
            
        except Exception as e:
            logger.warning(f"[SemanticSLAM] Landmark tracking failed: {e}")
            return None, False
    
    def _match_landmarks(
        self,
        prev_landmarks: Dict[str, SemanticLandmark],
        curr_landmarks: Dict[str, SemanticLandmark]
    ) -> List[Tuple[str, str]]:
        """
        Match landmarks between frames
        
        Returns:
            List of (prev_id, curr_id) matches
        """
        matches = []
        
        # Simple matching: same class + nearest neighbor
        for prev_id, prev_lm in prev_landmarks.items():
            best_match = None
            best_dist = float('inf')
            
            for curr_id, curr_lm in curr_landmarks.items():
                # Must be same class
                if curr_lm.class_name != prev_lm.class_name:
                    continue
                
                # Euclidean distance between centroids
                dx = curr_lm.centroid_2d[0] - prev_lm.centroid_2d[0]
                dy = curr_lm.centroid_2d[1] - prev_lm.centroid_2d[1]
                dist = np.sqrt(dx**2 + dy**2)
                
                # Threshold: object shouldn't move more than 200 pixels between frames
                if dist < 200 and dist < best_dist:
                    best_dist = dist
                    best_match = curr_id
            
            if best_match:
                matches.append((prev_id, best_match))
        
        return matches
    
    def _homography_to_pose(self, H: np.ndarray) -> np.ndarray:
        """
        Convert homography to pose matrix (simplified)
        
        This is an approximation. For accurate 3D pose, would need:
        - Camera intrinsics
        - Essential matrix decomposition
        - Proper R|t recovery
        """
        # Extract rotation and translation components
        # H ≈ K [R | t] where K is camera matrix
        
        # Simplified: assume H encodes small motion
        pose = np.eye(4)
        
        # Extract translation from last column
        pose[0, 3] = H[0, 2]  # tx
        pose[1, 3] = H[1, 2]  # ty
        
        # Extract rotation (approximate from 2x2 submatrix)
        rotation_2x2 = H[:2, :2]
        
        # Orthogonalize to get proper rotation
        U, _, Vt = np.linalg.svd(rotation_2x2)
        R_2x2 = U @ Vt
        
        pose[:2, :2] = R_2x2
        
        return pose
    
    def _fuse_poses(
        self,
        pose1: np.ndarray,
        pose2: np.ndarray,
        weight_visual: float = 0.7
    ) -> np.ndarray:
        """
        Fuse two pose estimates with weights
        
        Simple linear interpolation on translation, SLERP on rotation
        """
        weight_landmark = 1.0 - weight_visual
        
        # Extract components
        R1 = pose1[:3, :3]
        t1 = pose1[:3, 3]
        R2 = pose2[:3, :3]
        t2 = pose2[:3, 3]
        
        # Fuse translation (weighted average)
        t_fused = weight_visual * t1 + weight_landmark * t2
        
        # Fuse rotation (simplified: average)
        # Proper method: SLERP (Spherical Linear Interpolation)
        R_fused = weight_visual * R1 + weight_landmark * R2
        
        # Orthogonalize to ensure valid rotation
        U, _, Vt = np.linalg.svd(R_fused)
        R_fused = U @ Vt
        
        # Build fused pose
        pose_fused = np.eye(4)
        pose_fused[:3, :3] = R_fused
        pose_fused[:3, 3] = t_fused
        
        return pose_fused
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics"""
        base_stats = self.base_slam.get_statistics() if hasattr(self.base_slam, 'get_statistics') else {}
        
        total_frames = len(self.landmark_history)
        frames_with_landmarks = sum(1 for lms in self.landmark_history if lms)
        
        return {
            **base_stats,
            'semantic_slam_enabled': self.use_landmarks,
            'total_frames_with_landmarks': frames_with_landmarks,
            'landmark_usage_rate': frames_with_landmarks / total_frames if total_frames > 0 else 0.0,
            'stable_object_classes': len(self.stable_classes)
        }
    
    def register_loop_closure_callback(self, callback):
        """
        Register callback for loop closure detection.
        
        Delegates to base SLAM engine.
        (Phase 4 Week 2 - Day 4)
        
        Args:
            callback: Function to call when loop closure detected
        """
        if hasattr(self.base_slam, 'register_loop_closure_callback'):
            self.base_slam.register_loop_closure_callback(callback)
        else:
            logger.warning("[SemanticSLAM] Base SLAM does not support loop closure callbacks")

