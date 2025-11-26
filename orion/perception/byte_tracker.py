"""
Object Tracker
==============

Temporal tracking with ByteTrack-inspired algorithm.
Pure Python implementation for M-series compatibility.

Features:
- Kalman filter for motion prediction
- IoU-based association
- Track lifecycle management (tentative → confirmed → deleted)
- Re-ID recovery hooks (for Phase 2)

References:
- ByteTrack: https://arxiv.org/abs/2110.06864
- OC-SORT: https://arxiv.org/abs/2203.14360
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class KalmanFilter:
    """
    Simple 2D Kalman filter for bbox tracking.
    
    State: [x, y, w, h, vx, vy, vw, vh]
    """
    
    def __init__(self):
        # State transition matrix (constant velocity model)
        self.F = np.eye(8)
        self.F[0, 4] = 1  # x += vx
        self.F[1, 5] = 1  # y += vy
        self.F[2, 6] = 1  # w += vw
        self.F[3, 7] = 1  # h += vh
        
        # Measurement matrix (observe position only)
        self.H = np.eye(4, 8)
        
        # Process noise
        self.Q = np.eye(8) * 0.01
        
        # Measurement noise
        self.R = np.eye(4) * 0.1
        
        # State covariance
        self.P = np.eye(8) * 10
        
        # State
        self.x = np.zeros(8)
    
    def init(self, bbox: List[float]):
        """Initialize with first detection [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0])
        self.P = np.eye(8) * 10
    
    def predict(self) -> np.ndarray:
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.get_bbox()
    
    def update(self, bbox: List[float]):
        """Update with measurement [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        z = np.array([cx, cy, w, h])
        
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
    
    def get_bbox(self) -> np.ndarray:
        """Get current bbox [x1, y1, x2, y2]."""
        cx, cy, w, h = self.x[:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2])


def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Compute IoU between two bboxes [x1, y1, x2, y2]."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)


class Track:
    """Single object track."""
    
    # Track states
    TENTATIVE = 0
    CONFIRMED = 1
    DELETED = 2
    
    def __init__(
        self,
        track_id: int,
        detection: Dict[str, Any],
        tentative_threshold: int = 3,
    ):
        self.track_id = track_id
        self.category = detection["category"]
        self.confidence = detection["confidence"]
        
        # Kalman filter
        self.kf = KalmanFilter()
        self.kf.init(detection["bbox"])
        
        # Track state
        self.state = self.TENTATIVE
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        
        # History
        self.history: List[Dict[str, Any]] = [detection]
        
        # Thresholds
        self.tentative_threshold = tentative_threshold
        
        # Re-ID hook (Phase 2)
        self.embedding_id: Optional[str] = None
    
    def predict(self) -> np.ndarray:
        """Predict next position."""
        self.age += 1
        self.time_since_update += 1
        return self.kf.predict()
    
    def update(self, detection: Dict[str, Any]):
        """Update with new detection."""
        self.kf.update(detection["bbox"])
        self.hits += 1
        self.time_since_update = 0
        self.confidence = detection["confidence"]
        self.history.append(detection)
        
        # Promote tentative to confirmed
        if self.state == self.TENTATIVE and self.hits >= self.tentative_threshold:
            self.state = self.CONFIRMED
    
    def mark_missed(self):
        """Mark as missed in current frame."""
        if self.state == self.TENTATIVE:
            self.state = self.DELETED
    
    def is_tentative(self) -> bool:
        return self.state == self.TENTATIVE
    
    def is_confirmed(self) -> bool:
        return self.state == self.CONFIRMED
    
    def is_deleted(self) -> bool:
        return self.state == self.DELETED
    
    def get_current_bbox(self) -> np.ndarray:
        """Get current predicted bbox."""
        return self.kf.get_bbox()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export track info."""
        return {
            "track_id": self.track_id,
            "category": self.category,
            "state": self.state,
            "hits": self.hits,
            "age": self.age,
            "time_since_update": self.time_since_update,
            "bbox": self.get_current_bbox().tolist(),
        }


class ObjectTracker:
    """
    ByteTrack-inspired multi-object tracker.
    
    Features:
    - Two-stage association (high/low confidence)
    - Kalman filter motion prediction
    - Track lifecycle management
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        tentative_threshold: int = 3,
        high_conf_threshold: float = 0.5,
        low_conf_threshold: float = 0.1,
    ):
        """
        Initialize tracker.
        
        Args:
            iou_threshold: Min IoU for association
            max_age: Frames to keep unmatched tracks
            tentative_threshold: Hits needed to confirm track
            high_conf_threshold: Threshold for high-confidence detections
            low_conf_threshold: Threshold for low-confidence detections
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tentative_threshold = tentative_threshold
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold
        
        # Track management
        self.tracks: List[Track] = []
        self.next_id = 1
        
        # Statistics
        self.total_tracks = 0
        self.frame_count = 0
    
    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dicts from current frame
            
        Returns:
            List of track assignments with track_id added
        """
        self.frame_count += 1
        
        # Predict all tracks
        for track in self.tracks:
            track.predict()
        
        # Split detections by confidence (ByteTrack approach)
        high_conf_dets = [d for d in detections if d["confidence"] >= self.high_conf_threshold]
        low_conf_dets = [d for d in detections if self.low_conf_threshold <= d["confidence"] < self.high_conf_threshold]
        
        # Stage 1: Associate high-confidence detections
        matches, unmatched_tracks, unmatched_dets = self._associate(
            self.tracks, high_conf_dets
        )
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(high_conf_dets[det_idx])
        
        # Stage 2: Associate low-confidence with unmatched tracks
        if len(low_conf_dets) > 0 and len(unmatched_tracks) > 0:
            unmatched_track_objs = [self.tracks[i] for i in unmatched_tracks]
            matches_low, unmatched_tracks_low, _ = self._associate(
                unmatched_track_objs, low_conf_dets
            )
            
            # Update with low-confidence matches
            for track_idx, det_idx in matches_low:
                unmatched_track_objs[track_idx].update(low_conf_dets[det_idx])
            
            # Update unmatched tracks list
            unmatched_tracks = [unmatched_tracks[i] for i in unmatched_tracks_low]
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Create new tracks from unmatched high-confidence detections
        for det_idx in unmatched_dets:
            self._create_track(high_conf_dets[det_idx])
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted() and t.time_since_update <= self.max_age]
        
        # Build output (confirmed tracks only)
        outputs = []
        for track in self.tracks:
            if track.is_confirmed() and track.time_since_update == 0:
                # Last detection in history
                last_det = track.history[-1].copy()
                last_det["track_id"] = track.track_id
                last_det["embedding_id"] = track.embedding_id
                outputs.append(last_det)
        
        return outputs
    
    def _associate(
        self,
        tracks: List[Track],
        detections: List[Dict[str, Any]],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate tracks with detections using IoU.
        
        Returns:
            matches: List of (track_idx, det_idx) pairs
            unmatched_tracks: List of track indices
            unmatched_dets: List of detection indices
        """
        if len(tracks) == 0:
            return [], [], list(range(len(detections)))
        
        if len(detections) == 0:
            return [], list(range(len(tracks))), []
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            track_bbox = track.get_current_bbox()
            for j, det in enumerate(detections):
                det_bbox = np.array(det["bbox"])
                
                # Only match same class
                if track.category != det["category"]:
                    iou_matrix[i, j] = 0.0
                else:
                    iou_matrix[i, j] = compute_iou(track_bbox, det_bbox)
        
        # Greedy matching (can be improved with Hungarian algorithm)
        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))
        
        # Sort by IoU (highest first)
        iou_pairs = []
        for i in range(len(tracks)):
            for j in range(len(detections)):
                if iou_matrix[i, j] >= self.iou_threshold:
                    iou_pairs.append((iou_matrix[i, j], i, j))
        
        iou_pairs.sort(reverse=True, key=lambda x: x[0])
        
        matched_tracks = set()
        matched_dets = set()
        
        for _, track_idx, det_idx in iou_pairs:
            if track_idx not in matched_tracks and det_idx not in matched_dets:
                matches.append((track_idx, det_idx))
                matched_tracks.add(track_idx)
                matched_dets.add(det_idx)
        
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_dets = [j for j in range(len(detections)) if j not in matched_dets]
        
        return matches, unmatched_tracks, unmatched_dets
    
    def _create_track(self, detection: Dict[str, Any]) -> Track:
        """Create new track from detection."""
        track = Track(
            track_id=self.next_id,
            detection=detection,
            tentative_threshold=self.tentative_threshold,
        )
        
        self.tracks.append(track)
        self.next_id += 1
        self.total_tracks += 1
        
        return track
    
    def get_active_tracks(self) -> List[Track]:
        """Get all confirmed active tracks."""
        return [t for t in self.tracks if t.is_confirmed()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        return {
            "total_tracks": self.total_tracks,
            "active_tracks": len(self.get_active_tracks()),
            "all_tracks": len(self.tracks),
            "frame_count": self.frame_count,
        }
