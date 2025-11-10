"""
Phase 2: Entity Tracking Engine with Bayesian Beliefs and Temporal Identity

This module implements persistent entity tracking across video frames using:
- Bayesian belief states for class probabilities
- Hungarian algorithm for optimal data association
- Exponential decay for disappearance handling
- Re-identification based on appearance similarity

Author: Orion Research Team
Date: November 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
import time

# NEW: Geometric Re-ID for spatial consistency
from orion.perception.geometric_reid import GeometricReID


@dataclass
class BayesianEntityBelief:
    """
    Represents the belief state of a tracked entity with Bayesian class probabilities.
    
    This encapsulates:
    - Temporal identity (entity_id)
    - Position (centroid_2d, centroid_3d_mm)
    - Class posterior distribution (belief over all YOLO classes)
    - Appearance embedding for re-identification
    - Motion state (velocity, acceleration)
    - Lifecycle tracking (birth, disappearance, re-ID)
    """
    
    entity_id: int
    """Unique temporal identifier for this entity"""
    
    # Spatial state
    centroid_2d: np.ndarray  # (x, y) in pixels
    centroid_3d_mm: Optional[np.ndarray] = None  # (x, y, z) in millimeters
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4))  # (x1, y1, x2, y2)
    
    # Belief state
    class_posterior: Dict[str, float] = field(default_factory=dict)
    """Posterior probability distribution over YOLO classes"""
    
    most_likely_class: str = "unknown"
    """Current most probable class (argmax of posterior)"""
    
    # Appearance for re-identification
    appearance_embedding: Optional[np.ndarray] = None
    """Visual feature vector for re-ID (e.g., from CLIP or ResNet)"""
    
    # Motion state
    velocity_2d: np.ndarray = field(default_factory=lambda: np.zeros(2))  # pixels/frame
    velocity_3d_mm: Optional[np.ndarray] = None  # mm/frame
    
    # Temporal tracking
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    total_detections: int = 0
    consecutive_misses: int = 0
    
    # Lifecycle flags
    is_disappeared: bool = False
    disappearance_frame: Optional[int] = None
    reidentified_times: int = 0
    
    def update_belief(self, detection_class: str, detection_conf: float, 
                     all_classes: List[str], learning_rate: float = 0.3):
        """
        Bayesian update of class posterior given new detection.
        
        Uses a simple weighted update rule:
            P(class | observation) ∝ P(observation | class) * P(class)
        
        Args:
            detection_class: YOLO class name from new detection
            detection_conf: Detection confidence [0, 1]
            all_classes: List of all possible YOLO classes
            learning_rate: How much to weight new observation (0=ignore, 1=replace)
        """
        # Initialize uniform prior if first detection
        if not self.class_posterior:
            self.class_posterior = {c: 1.0 / len(all_classes) for c in all_classes}
        
        # Bayesian update: posterior = (1 - lr) * prior + lr * likelihood
        # Likelihood is peaked at detected class with confidence
        for cls in all_classes:
            prior = self.class_posterior.get(cls, 1.0 / len(all_classes))
            likelihood = detection_conf if cls == detection_class else (1 - detection_conf) / (len(all_classes) - 1)
            self.class_posterior[cls] = (1 - learning_rate) * prior + learning_rate * likelihood
        
        # Normalize to sum to 1
        total = sum(self.class_posterior.values())
        self.class_posterior = {k: v / total for k, v in self.class_posterior.items()}
        
        # Update most likely class
        self.most_likely_class = max(self.class_posterior, key=self.class_posterior.get)
    
    def update_motion(self, new_centroid_2d: np.ndarray, new_centroid_3d_mm: Optional[np.ndarray] = None):
        """Update velocity estimates based on new position."""
        self.velocity_2d = new_centroid_2d - self.centroid_2d
        
        if new_centroid_3d_mm is not None and self.centroid_3d_mm is not None:
            self.velocity_3d_mm = new_centroid_3d_mm - self.centroid_3d_mm
        
        self.centroid_2d = new_centroid_2d
        if new_centroid_3d_mm is not None:
            self.centroid_3d_mm = new_centroid_3d_mm


@dataclass
class TrackingConfig:
    """Configuration for entity tracking."""
    
    # Association thresholds
    max_distance_pixels: float = 150.0
    """Maximum 2D distance for association (pixels)"""
    
    max_distance_3d_mm: float = 1500.0  # 1.5 meters
    """Maximum 3D distance for association (millimeters)"""
    
    appearance_similarity_threshold: float = 0.7
    """Minimum cosine similarity for re-ID (0=orthogonal, 1=identical)"""
    
    # Disappearance handling
    ttl_frames: int = 30  # ~1 second at 30fps
    """Time-to-live: frames before declaring entity disappeared"""
    
    reid_window_frames: int = 90  # ~3 seconds
    """Window for attempting re-identification after disappearance"""
    
    # Bayesian update
    class_belief_lr: float = 0.3
    """Learning rate for Bayesian class belief updates"""
    
    # Motion prediction
    use_motion_prediction: bool = True
    """Whether to use velocity for predicting next position"""


class EntityTracker3D:
    """
    Tracks entities across video frames with temporal identity preservation.
    
    Uses Hungarian algorithm for optimal detection-to-track association,
    Bayesian updates for class beliefs, and appearance-based re-identification.
    """
    
    def __init__(self, config: TrackingConfig, yolo_classes: List[str]):
        """
        Initialize the tracker.
        
        Args:
            config: Tracking configuration
            yolo_classes: List of all YOLO class names
        """
        self.config = config
        self.yolo_classes = yolo_classes
        
        # Active tracks
        self.tracks: Dict[int, BayesianEntityBelief] = {}
        
        # Recently disappeared tracks (for re-ID)
        self.disappeared_tracks: Dict[int, BayesianEntityBelief] = {}
        
        # ID counter
        self.next_entity_id = 0
        
        # Statistics
        self.total_entities_seen = 0
        self.reidentifications = 0
        self.id_switches = 0  # For evaluation
        
        # NEW: Geometric Re-ID for spatial consistency
        self.geometric_reid = GeometricReID(
            max_distance_mm=2000.0,  # 2m max movement per frame
            max_velocity_change_mm_s=3000.0,  # 3m/s max acceleration
            history_size=10,
            temporal_weight_decay=0.95
        )
    
    def track_frame(
        self, 
        detections: List[Dict],
        frame_number: int,
        timestamp: float
    ) -> List[BayesianEntityBelief]:
        """
        Process one frame's detections and update all tracks.
        
        Args:
            detections: List of YOLO detections with keys:
                - bbox: [x1, y1, x2, y2]
                - class_name: str
                - confidence: float
                - centroid_2d: [x, y] (optional, computed if missing)
                - centroid_3d_mm: [x, y, z] (optional, from 3D perception)
                - appearance_embedding: np.ndarray (optional, for re-ID)
            frame_number: Current frame index
            timestamp: Current timestamp (seconds)
        
        Returns:
            List of updated BayesianEntityBelief objects (all active tracks)
        """
        # Compute centroids if not provided
        for det in detections:
            if 'centroid_2d' not in det:
                bbox = det['bbox']
                det['centroid_2d'] = np.array([
                    (bbox[0] + bbox[2]) / 2,
                    (bbox[1] + bbox[3]) / 2
                ])
        
        # Step 1: Predict track positions (using motion model)
        self._predict_track_positions()
        
        # Step 2: Compute cost matrix (detection-to-track distances)
        cost_matrix = self._compute_cost_matrix(detections)
        
        # Step 3: Hungarian algorithm for optimal assignment
        matched_det_ids, matched_track_ids, unmatched_det_ids, unmatched_track_ids = \
            self._hungarian_assignment(cost_matrix, detections)
        
        # Step 4: Update matched tracks
        for det_idx, track_id in zip(matched_det_ids, matched_track_ids):
            self._update_track(track_id, detections[det_idx], frame_number)
        
        # Step 5: Handle unmatched tracks (disappearances)
        for track_id in unmatched_track_ids:
            self._handle_unmatched_track(track_id, frame_number)
        
        # Step 6: Handle unmatched detections (new entities or re-IDs)
        for det_idx in unmatched_det_ids:
            self._handle_unmatched_detection(detections[det_idx], frame_number)
        
        # Step 7: Cleanup disappeared tracks beyond re-ID window
        self._cleanup_disappeared_tracks(frame_number)
        
        # Step 8: NEW - Cleanup old geometric Re-ID states (every 100 frames)
        if frame_number % 100 == 0:
            self.geometric_reid.cleanup_old_entities(frame_number, max_frames_gap=100)
        
        return list(self.tracks.values())
    
    def _predict_track_positions(self):
        """Use velocity to predict next position for each track."""
        if not self.config.use_motion_prediction:
            return
        
        for track in self.tracks.values():
            # Simple linear prediction: next_pos = current_pos + velocity
            predicted_2d = track.centroid_2d + track.velocity_2d
            track.centroid_2d = predicted_2d
            
            if track.centroid_3d_mm is not None and track.velocity_3d_mm is not None:
                predicted_3d = track.centroid_3d_mm + track.velocity_3d_mm
                track.centroid_3d_mm = predicted_3d
    
    def _compute_cost_matrix(self, detections: List[Dict]) -> np.ndarray:
        """
        Compute cost matrix for detection-to-track assignment.
        
        Cost is a weighted combination of:
        - 2D Euclidean distance
        - 3D Euclidean distance (if available)
        - Appearance dissimilarity (1 - cosine similarity)
        - Geometric consistency (NEW: spatial continuity check)
        
        Returns:
            cost_matrix of shape (num_detections, num_tracks)
        """
        num_detections = len(detections)
        num_tracks = len(self.tracks)
        
        if num_detections == 0 or num_tracks == 0:
            return np.zeros((num_detections, num_tracks))
        
        cost_matrix = np.zeros((num_detections, num_tracks))
        
        track_ids = list(self.tracks.keys())
        
        for i, det in enumerate(detections):
            det_centroid_2d = det['centroid_2d']
            det_centroid_3d = det.get('centroid_3d_mm')
            det_embedding = det.get('appearance_embedding')
            
            for j, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                
                # 2D distance (always available)
                dist_2d = np.linalg.norm(det_centroid_2d - track.centroid_2d)
                
                # 3D distance (if available)
                dist_3d = 0.0
                if det_centroid_3d is not None and track.centroid_3d_mm is not None:
                    dist_3d = np.linalg.norm(det_centroid_3d - track.centroid_3d_mm)
                    # Normalize to similar scale as 2D (divide by max threshold)
                    dist_3d = dist_3d / self.config.max_distance_3d_mm * self.config.max_distance_pixels
                
                # Appearance dissimilarity (if available)
                appearance_cost = 0.0
                if det_embedding is not None and track.appearance_embedding is not None:
                    similarity = 1 - cosine(det_embedding, track.appearance_embedding)
                    appearance_cost = (1 - similarity) * self.config.max_distance_pixels
                
                # NEW: Geometric consistency score (spatial continuity)
                geometric_score = 0.5  # Default neutral score
                if det_centroid_3d is not None and track.centroid_3d_mm is not None:
                    # Check if match is geometrically plausible
                    # Use frame number as timestamp approximation (30fps)
                    timestamp = track.last_seen_frame / 30.0
                    
                    if not self.geometric_reid.is_plausible_match(
                        det_centroid_3d, track.entity_id, timestamp
                    ):
                        # Physically impossible match - reject with huge penalty
                        cost_matrix[i, j] = 1e8
                        continue
                    
                    # Compute geometric consistency score (0-1, higher = better)
                    geometric_score = self.geometric_reid.compute_geometric_score(
                        det_centroid_3d, track.entity_id, 
                        track.last_seen_frame, timestamp
                    )
                    
                    # Convert score to cost (invert: cost = 1 - score)
                    geometric_cost = (1 - geometric_score) * self.config.max_distance_pixels
                else:
                    geometric_cost = 0.0  # No penalty if 3D unavailable
                
                # Combined cost (weighted sum)
                # NEW WEIGHTS: 40% 2D, 20% 3D, 20% appearance, 20% geometric
                cost = (0.4 * dist_2d + 
                       0.2 * dist_3d + 
                       0.2 * appearance_cost +
                       0.2 * geometric_cost)
                
                # Penalize if distance exceeds threshold
                if dist_2d > self.config.max_distance_pixels:
                    cost += 1e6  # Large penalty
                
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def _hungarian_assignment(
        self, 
        cost_matrix: np.ndarray,
        detections: List[Dict]
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Perform Hungarian algorithm for optimal assignment.
        
        Returns:
            - matched_det_ids: detection indices that were matched
            - matched_track_ids: corresponding track IDs
            - unmatched_det_ids: detection indices with no match
            - unmatched_track_ids: track IDs with no match
        """
        num_detections = cost_matrix.shape[0]
        num_tracks = cost_matrix.shape[1]
        
        if num_detections == 0 or num_tracks == 0:
            return [], [], list(range(num_detections)), list(self.tracks.keys())
        
        # Run Hungarian algorithm
        det_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        # Filter out assignments with cost > threshold
        matched_det_ids = []
        matched_track_ids = []
        track_ids_list = list(self.tracks.keys())
        
        for det_idx, track_idx in zip(det_indices, track_indices):
            cost = cost_matrix[det_idx, track_idx]
            if cost < self.config.max_distance_pixels * 1.5:  # Allow some slack
                matched_det_ids.append(det_idx)
                matched_track_ids.append(track_ids_list[track_idx])
        
        # Find unmatched
        unmatched_det_ids = [i for i in range(num_detections) if i not in matched_det_ids]
        unmatched_track_ids = [tid for tid in self.tracks.keys() if tid not in matched_track_ids]
        
        return matched_det_ids, matched_track_ids, unmatched_det_ids, unmatched_track_ids
    
    def _update_track(self, track_id: int, detection: Dict, frame_number: int):
        """Update an existing track with new detection."""
        track = self.tracks[track_id]
        
        # Update Bayesian belief
        track.update_belief(
            detection['class_name'],
            detection['confidence'],
            self.yolo_classes,
            self.config.class_belief_lr
        )
        
        # Update position and motion
        new_centroid_2d = detection['centroid_2d']
        new_centroid_3d = detection.get('centroid_3d_mm')
        track.update_motion(new_centroid_2d, new_centroid_3d)
        
        # Update bbox
        track.bbox = np.array(detection['bbox'])
        
        # Update appearance
        if 'appearance_embedding' in detection:
            # Exponential moving average
            if track.appearance_embedding is not None:
                alpha = 0.7  # Weight for new observation
                track.appearance_embedding = alpha * detection['appearance_embedding'] + \
                                           (1 - alpha) * track.appearance_embedding
            else:
                track.appearance_embedding = detection['appearance_embedding']
        
        # Update temporal tracking
        track.last_seen_frame = frame_number
        track.total_detections += 1
        track.consecutive_misses = 0
        
        # NEW: Update geometric Re-ID spatial state
        if new_centroid_3d is not None:
            timestamp = frame_number / 30.0  # Assume 30fps
            self.geometric_reid.update_spatial_state(
                entity_id=track.entity_id,
                position_3d=new_centroid_3d,
                timestamp=timestamp,
                frame_idx=frame_number
            )
        
        # If was disappeared, mark as re-identified
        if track.is_disappeared:
            track.is_disappeared = False
            track.reidentified_times += 1
            self.reidentifications += 1
    
    def _handle_unmatched_track(self, track_id: int, frame_number: int):
        """Handle a track that wasn't matched (potential disappearance)."""
        track = self.tracks[track_id]
        track.consecutive_misses += 1
        
        # Check if should be marked as disappeared
        if track.consecutive_misses >= self.config.ttl_frames:
            track.is_disappeared = True
            track.disappearance_frame = frame_number
            
            # Move to disappeared tracks for potential re-ID
            self.disappeared_tracks[track_id] = track
            del self.tracks[track_id]
    
    def _handle_unmatched_detection(self, detection: Dict, frame_number: int):
        """
        Handle an unmatched detection: either create new track or re-identify.
        """
        # First, try to re-identify from disappeared tracks
        reid_track_id = self._attempt_reidentification(detection)
        
        if reid_track_id is not None:
            # Re-identified! Move back to active tracks FIRST
            track = self.disappeared_tracks[reid_track_id]
            self.tracks[reid_track_id] = track  # Move to active BEFORE update
            del self.disappeared_tracks[reid_track_id]
            
            # Now update the track (requires it to be in self.tracks)
            track.is_disappeared = False
            track.consecutive_misses = 0
            track.reidentified_times += 1
            self.reidentifications += 1
            
            # Update with new detection
            self._update_track(reid_track_id, detection, frame_number)
        else:
            # Create new track
            self._create_new_track(detection, frame_number)
    
    def _attempt_reidentification(self, detection: Dict) -> Optional[int]:
        """
        Try to re-identify detection with a disappeared track.
        
        Returns:
            track_id if re-identified, None otherwise
        """
        if not self.disappeared_tracks:
            return None
        
        det_embedding = detection.get('appearance_embedding')
        if det_embedding is None:
            return None  # Cannot re-ID without appearance
        
        best_track_id = None
        best_similarity = 0.0
        
        for track_id, track in self.disappeared_tracks.items():
            if track.appearance_embedding is None:
                continue
            
            # Compute cosine similarity
            similarity = 1 - cosine(det_embedding, track.appearance_embedding)
            
            # Check if passes threshold
            if similarity > self.config.appearance_similarity_threshold and \
               similarity > best_similarity:
                best_similarity = similarity
                best_track_id = track_id
        
        return best_track_id
    
    def _create_new_track(self, detection: Dict, frame_number: int):
        """Create a new track from detection."""
        entity_id = self.next_entity_id
        self.next_entity_id += 1
        self.total_entities_seen += 1
        
        # Initialize belief with uniform prior
        track = BayesianEntityBelief(
            entity_id=entity_id,
            centroid_2d=detection['centroid_2d'].copy(),
            centroid_3d_mm=detection.get('centroid_3d_mm'),
            bbox=np.array(detection['bbox']),
            appearance_embedding=detection.get('appearance_embedding'),
            first_seen_frame=frame_number,
            last_seen_frame=frame_number,
            total_detections=1,
            consecutive_misses=0
        )
        
        # Initial belief update
        track.update_belief(
            detection['class_name'],
            detection['confidence'],
            self.yolo_classes,
            self.config.class_belief_lr
        )
        
        self.tracks[entity_id] = track
    
    def _cleanup_disappeared_tracks(self, frame_number: int):
        """Remove disappeared tracks beyond re-ID window."""
        to_remove = []
        
        for track_id, track in self.disappeared_tracks.items():
            if track.disappearance_frame is None:
                continue
            
            frames_since_disappearance = frame_number - track.disappearance_frame
            if frames_since_disappearance > self.config.reid_window_frames:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.disappeared_tracks[track_id]
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics for evaluation."""
        # Get geometric Re-ID stats
        geo_stats = self.geometric_reid.get_statistics()
        
        return {
            'total_active_tracks': len(self.tracks),
            'total_disappeared_tracks': len(self.disappeared_tracks),
            'total_entities_seen': self.total_entities_seen,
            'reidentifications': self.reidentifications,
            'id_switches': self.id_switches,
            # NEW: Geometric Re-ID stats
            'geometric_tracked_entities': geo_stats['total_tracked_entities'],
            'geometric_avg_history': geo_stats['avg_history_size'],
        }
