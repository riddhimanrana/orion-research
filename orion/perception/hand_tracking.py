"""
Hand tracking using MediaPipe with 3D projection.
"""

import time
from typing import List, Tuple, Optional
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[HandTracker] Warning: MediaPipe not installed. Hand tracking will be disabled.")

from .types import Hand, HandPose, CameraIntrinsics
from .camera_intrinsics import backproject_point


class HandTracker:
    """
    Detect hands and project to 3D using depth map.
    
    Uses MediaPipe Hands for 2D detection, then backprojects using depth.
    """
    
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.3,  # Lowered from 0.5 for egocentric views
        min_tracking_confidence: float = 0.3,   # Lowered from 0.5 for egocentric views
    ):
        """
        Initialize hand tracker.
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection (0.3 for egocentric)
            min_tracking_confidence: Minimum confidence for hand tracking (0.3 for egocentric)
        """
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is required for hand tracking. Install: pip install mediapipe")
        
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        print(f"[HandTracker] Initialized with max_hands={max_num_hands}, "
              f"min_detection_conf={min_detection_confidence}, "
              f"min_tracking_conf={min_tracking_confidence}")
    
    def detect(
        self,
        frame: np.ndarray,
        depth_map: np.ndarray,
        intrinsics: CameraIntrinsics
    ) -> List[Hand]:
        """
        Detect hands in frame and project to 3D.
        
        Args:
            frame: RGB frame (H, W, 3)
            depth_map: Depth map (H, W) in millimeters
            intrinsics: Camera intrinsics for backprojection
            
        Returns:
            List of detected hands with 3D landmarks
        """
        start_time = time.time()
        
        # Convert BGR to RGB if needed (MediaPipe expects RGB)
        if frame.shape[2] == 3:
            frame_rgb = frame
        else:
            frame_rgb = frame[:, :, ::-1]
        
        # Run MediaPipe detection
        results = self.hands.process(frame_rgb)
        
        hands = []
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get handedness
                handedness = results.multi_handedness[idx].classification[0].label
                confidence = results.multi_handedness[idx].classification[0].score
                
                # Extract 2D landmarks (normalized 0-1)
                landmarks_2d = []
                for landmark in hand_landmarks.landmark:
                    landmarks_2d.append((landmark.x, landmark.y))
                
                # Convert to pixel coordinates and backproject to 3D
                landmarks_3d = []
                for x_norm, y_norm in landmarks_2d:
                    # Convert normalized to pixel coordinates
                    x_px = x_norm * intrinsics.width
                    y_px = y_norm * intrinsics.height
                    
                    # Sample depth at landmark location
                    depth_z = self._sample_depth(depth_map, x_px, y_px)
                    
                    # Backproject to 3D
                    point_3d = backproject_point(x_px, y_px, depth_z, intrinsics)
                    landmarks_3d.append(point_3d)
                
                # Compute palm center (average of base landmarks)
                # MediaPipe hand landmarks: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
                palm_indices = [0, 1, 5, 9, 13, 17]  # wrist + base of fingers
                palm_points = [landmarks_3d[i] for i in palm_indices]
                palm_center_3d = tuple(np.mean(palm_points, axis=0).tolist())
                
                # Classify hand pose
                pose = self._classify_pose(landmarks_3d, landmarks_2d)
                
                # Create Hand object
                hand = Hand(
                    id=f"hand_{idx}_{handedness.lower()}",
                    landmarks_2d=landmarks_2d,
                    landmarks_3d=landmarks_3d,
                    palm_center_3d=palm_center_3d,
                    pose=pose,
                    confidence=confidence,
                    handedness=handedness,
                )
                
                hands.append(hand)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return hands
    
    def _sample_depth(
        self,
        depth_map: np.ndarray,
        x_px: float,
        y_px: float,
        window_size: int = 5
    ) -> float:
        """
        Sample depth at a pixel location with local averaging.
        
        Args:
            depth_map: Depth map (H, W)
            x_px, y_px: Pixel coordinates
            window_size: Size of averaging window
            
        Returns:
            Depth value in millimeters
        """
        h, w = depth_map.shape
        
        # Clamp to valid range
        x_px = np.clip(x_px, 0, w - 1)
        y_px = np.clip(y_px, 0, h - 1)
        
        # Extract window around point
        half_size = window_size // 2
        x1 = max(0, int(x_px) - half_size)
        x2 = min(w, int(x_px) + half_size + 1)
        y1 = max(0, int(y_px) - half_size)
        y2 = min(h, int(y_px) + half_size + 1)
        
        window = depth_map[y1:y2, x1:x2]
        
        # Filter out invalid depths
        valid_depths = window[(window > 0) & np.isfinite(window)]
        
        if len(valid_depths) > 0:
            return float(np.median(valid_depths))
        else:
            # Fallback to center pixel
            depth = depth_map[int(y_px), int(x_px)]
            if depth > 0 and np.isfinite(depth):
                return float(depth)
            else:
                return 1000.0  # Default 1 meter
    
    def _classify_pose(
        self,
        landmarks_3d: List[Tuple[float, float, float]],
        landmarks_2d: List[Tuple[float, float]]
    ) -> HandPose:
        """
        Classify hand pose based on finger configurations.
        
        Simple heuristic-based classification:
        - OPEN: All fingers extended
        - CLOSED: All fingers curled
        - PINCH: Thumb and index close together
        - POINT: Index extended, others curled
        
        Args:
            landmarks_3d: 21 3D landmarks
            landmarks_2d: 21 2D landmarks (normalized)
            
        Returns:
            HandPose classification
        """
        # MediaPipe landmark indices:
        # 0: wrist
        # 4: thumb tip, 8: index tip, 12: middle tip, 16: ring tip, 20: pinky tip
        # 3: thumb ip, 7: index pip, 11: middle pip, 15: ring pip, 19: pinky pip
        
        try:
            # Get fingertip and knuckle positions (in 2D for simplicity)
            thumb_tip = landmarks_2d[4]
            index_tip = landmarks_2d[8]
            middle_tip = landmarks_2d[12]
            ring_tip = landmarks_2d[16]
            pinky_tip = landmarks_2d[20]
            
            thumb_ip = landmarks_2d[3]
            index_pip = landmarks_2d[6]
            middle_pip = landmarks_2d[10]
            ring_pip = landmarks_2d[14]
            pinky_pip = landmarks_2d[18]
            
            wrist = landmarks_2d[0]
            
            # Check if fingers are extended (tip farther from wrist than knuckle)
            def is_extended(tip, knuckle, wrist_pos):
                tip_dist = np.linalg.norm(np.array(tip) - np.array(wrist_pos))
                knuckle_dist = np.linalg.norm(np.array(knuckle) - np.array(wrist_pos))
                return tip_dist > knuckle_dist * 1.1
            
            thumb_extended = is_extended(thumb_tip, thumb_ip, wrist)
            index_extended = is_extended(index_tip, index_pip, wrist)
            middle_extended = is_extended(middle_tip, middle_pip, wrist)
            ring_extended = is_extended(ring_tip, ring_pip, wrist)
            pinky_extended = is_extended(pinky_tip, pinky_pip, wrist)
            
            # Count extended fingers
            extended_count = sum([
                thumb_extended,
                index_extended,
                middle_extended,
                ring_extended,
                pinky_extended
            ])
            
            # Check for pinch (thumb and index close)
            thumb_index_dist = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
            is_pinching = thumb_index_dist < 0.05  # normalized distance threshold
            
            # Classify
            if is_pinching:
                return HandPose.PINCH
            elif extended_count >= 4:
                return HandPose.OPEN
            elif extended_count <= 1:
                return HandPose.CLOSED
            elif index_extended and not middle_extended:
                return HandPose.POINT
            else:
                return HandPose.UNKNOWN
                
        except Exception as e:
            # If classification fails, return unknown
            return HandPose.UNKNOWN
    
    def cleanup(self) -> None:
        """Release resources."""
        if hasattr(self, 'hands'):
            self.hands.close()
