"""
Multi-Hypothesis Tracking
==========================

Links detections across frames to form persistent tracks.
Handles occlusions, re-appearances, and assigns stable track IDs.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class Track:
    """Represents a persistent track of an object across frames"""
    
    _id_counter = 1000
    
    def __init__(
        self,
        detection: Dict,
        frame_idx: int,
        track_id: Optional[int] = None
    ):
        """
        Initialize track from detection.
        
        Args:
            detection: Detection dict with class, bbox, embedding, bbox_3d
            frame_idx: Frame index where track starts
            track_id: Optional explicit ID (default: auto-increment)
        """
        if track_id is None:
            self.id = Track._id_counter
            Track._id_counter += 1
        else:
            self.id = track_id
        
        self.class_name = detection.get("class", "unknown")
        self.detections = [detection]
        self.frame_indices = [frame_idx]
        self.age = 1  # Frames since birth
        self.consecutive_misses = 0  # Frames without detection
        self.confirmed = False  # Confirmed after N frames
        self.confidence_history = [detection.get("confidence", 0.5)]
    
    def update(self, detection: Dict, frame_idx: int):
        """Add detection to track"""
        self.detections.append(detection)
        self.frame_indices.append(frame_idx)
        self.age += 1
        self.consecutive_misses = 0
        self.confidence_history.append(detection.get("confidence", 0.5))
        
        # Confirm after 3 consistent detections
        if self.age >= 3 and not self.confirmed:
            self.confirmed = True
    
    def miss(self):
        """Track missed detection this frame (occlusion)"""
        self.age += 1
        self.consecutive_misses += 1
    
    def is_active(self) -> bool:
        """Check if track is still active"""
        return self.consecutive_misses < 30  # Allow up to 30 frame gaps
    
    def get_latest(self) -> Dict:
        """Get most recent detection"""
        return self.detections[-1]
    
    def get_centroid_3d(self) -> Optional[np.ndarray]:
        """Get latest 3D centroid"""
        latest = self.get_latest()
        if "bbox_3d" in latest:
            return np.array(latest["bbox_3d"]["centroid_3d"])
        return None
    
    def get_embedding(self) -> Optional[np.ndarray]:
        """Get latest embedding"""
        latest = self.get_latest()
        if "embedding" in latest:
            return latest["embedding"]
        return None
    
    def avg_confidence(self) -> float:
        """Get average confidence across detections"""
        return float(np.mean(self.confidence_history)) if self.confidence_history else 0.0


class MultiHypothesisTracker:
    """
    Track objects across frames using appearance + spatial cues.
    
    Maintains multiple hypotheses for each object and resolves conflicts.
    """
    
    def __init__(
        self,
        embedding_similarity_threshold: float = 0.6,
        spatial_distance_threshold: float = 1.0,
        max_age: int = 100,
        confirm_threshold: int = 3
    ):
        """
        Initialize tracker.
        
        Args:
            embedding_similarity_threshold: Min cosine sim for match
            spatial_distance_threshold: Max 3D distance for match (meters)
            max_age: Max frames to keep inactive track
            confirm_threshold: Frames needed to confirm track
        """
        self.embedding_sim_thresh = embedding_similarity_threshold
        self.spatial_dist_thresh = spatial_distance_threshold
        self.max_age = max_age
        self.confirm_threshold = confirm_threshold
        
        self.tracks: Dict[int, Track] = {}  # track_id -> Track
        self.frame_count = 0
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new frame detections.
        
        Args:
            detections: List of detection dicts from current frame
            
        Returns:
            tracked_detections: Detections with track IDs assigned
        """
        self.frame_count += 1
        
        # Get predictions from existing tracks
        predictions = self._predict()
        
        # Match detections to tracks
        matches, unmatched_dets, unmatched_tracks = self._match(
            predictions,
            detections
        )
        
        # Update matched tracks
        for pred_idx, det_idx in matches:
            track_id = predictions[pred_idx]["track_id"]
            self.tracks[track_id].update(detections[det_idx], self.frame_count)
            detections[det_idx]["track_id"] = track_id
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = Track(detections[det_idx], self.frame_count)
            self.tracks[new_track.id] = new_track
            detections[det_idx]["track_id"] = new_track.id
        
        # Mark unmatched tracks as missed
        for pred_idx in unmatched_tracks:
            track_id = predictions[pred_idx]["track_id"]
            self.tracks[track_id].miss()
        
        # Remove dead tracks
        self.tracks = {
            tid: track for tid, track in self.tracks.items()
            if track.is_active()
        }
        
        return detections
    
    def _predict(self) -> List[Dict]:
        """Get predictions from active tracks"""
        predictions = []
        for track_id, track in self.tracks.items():
            if track.is_active():
                latest_det = track.get_latest()
                pred = latest_det.copy()
                pred["track_id"] = track_id
                pred["track_age"] = track.age
                pred["confirmed"] = track.confirmed
                predictions.append(pred)
        return predictions
    
    def _match(
        self,
        predictions: List[Dict],
        detections: List[Dict]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match predictions to detections.
        
        Returns:
            matches: [(pred_idx, det_idx), ...]
            unmatched_dets: [det_idx, ...]
            unmatched_tracks: [pred_idx, ...]
        """
        matches = []
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(range(len(predictions)))
        
        if not predictions or not detections:
            return matches, list(unmatched_dets), list(unmatched_tracks)
        
        # Compute cost matrix
        cost_matrix = np.full((len(predictions), len(detections)), np.inf)
        
        for i, pred in enumerate(predictions):
            for j, det in enumerate(detections):
                cost = self._compute_cost(pred, det)
                if cost < np.inf:
                    cost_matrix[i, j] = cost
        
        # Greedy matching (Hungarian would be better for large sets)
        for i in range(len(predictions)):
            best_j = None
            best_cost = np.inf
            
            for j in range(len(detections)):
                if j in unmatched_dets and cost_matrix[i, j] < best_cost:
                    best_j = j
                    best_cost = cost_matrix[i, j]
            
            if best_j is not None and best_cost < 1.0:  # Cost threshold
                matches.append((i, best_j))
                unmatched_dets.remove(best_j)
                unmatched_tracks.remove(i)
        
        return matches, list(unmatched_dets), list(unmatched_tracks)
    
    def _compute_cost(self, pred: Dict, det: Dict) -> float:
        """
        Compute cost of matching prediction to detection.
        Lower cost = better match. np.inf = impossible match.
        
        Returns:
            cost: Float in [0, inf]
        """
        # Class must match
        if pred.get("class") != det.get("class"):
            return np.inf
        
        cost = 0.0
        
        # Embedding distance (negative similarity)
        if "embedding" in pred and "embedding" in det:
            pred_emb = pred["embedding"]
            det_emb = det["embedding"]
            
            # Handle None embeddings (fallback case)
            if pred_emb is not None and det_emb is not None:
                emb_sim = float(np.dot(pred_emb, det_emb))
                emb_cost = 1.0 - emb_sim  # Convert sim to cost
                cost += 0.6 * emb_cost
        
        # Spatial distance
        if "bbox_3d" in pred and "bbox_3d" in det:
            pred_pos = np.array(pred["bbox_3d"]["centroid_3d"])
            det_pos = np.array(det["bbox_3d"]["centroid_3d"])
            distance = np.linalg.norm(pred_pos - det_pos)
            
            # Distance cost: infinity if too far
            if distance > self.spatial_dist_thresh * 3:
                return np.inf
            
            spatial_cost = distance / self.spatial_dist_thresh
            cost += 0.4 * spatial_cost
        
        return float(cost)
    
    def get_confirmed_tracks(self) -> Dict[int, Track]:
        """Get only confirmed tracks"""
        return {tid: t for tid, t in self.tracks.items() if t.confirmed}
    
    def get_track_stats(self) -> Dict:
        """Get tracking statistics"""
        confirmed = len(self.get_confirmed_tracks())
        total = len(self.tracks)
        
        return {
            "frame_count": self.frame_count,
            "total_tracks": total,
            "confirmed_tracks": confirmed,
            "unconfirmed_tracks": total - confirmed
        }
