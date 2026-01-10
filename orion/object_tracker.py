"""
Simple Object Tracker for UnifiedFrame objects.

Tracks objects across frames using:
1. 2D centroid proximity
2. 3D position (when available)
3. Appearance embeddings (CLIP)
4. Re-ID features (future)

Goal: Convert 72 detections (4/frame × 20 frames) into ~6 unique tracked objects
by recognizing the same object across frames.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from orion.perception.unified_frame import Object3D


@dataclass
class TrackedObject:
    """Object tracked across frames"""
    track_id: int                                      # Unique ID
    class_name: str                                   # Detection class
    first_seen: int                                   # Frame ID when first detected
    last_seen: int                                    # Frame ID when last detected
    age: int                                          # Number of frames observed
    confidence_history: list[float] = field(default_factory=list)  # Confidence per frame
    positions_2d: list[tuple[int, int]] = field(default_factory=list)  # Image centroids
    positions_3d: list[np.ndarray] = field(default_factory=list)  # World positions
    embeddings: list[Optional[np.ndarray]] = field(default_factory=list)  # CLIP embeddings
    
    @property
    def current_confidence(self) -> float:
        """Average confidence across frames"""
        return np.mean(self.confidence_history) if self.confidence_history else 0.0
    
    @property
    def current_position_2d(self) -> tuple[int, int]:
        """Last observed 2D position"""
        return self.positions_2d[-1] if self.positions_2d else (0, 0)
    
    @property
    def current_position_3d(self) -> Optional[np.ndarray]:
        """Last observed 3D position"""
        return self.positions_3d[-1] if self.positions_3d else None
    
    def update(self, obj: Object3D, pos_2d: tuple[int, int], frame_idx: int):
        """Update track with new observation"""
        self.last_seen = frame_idx
        self.age += 1
        self.confidence_history.append(obj.confidence)
        self.positions_2d.append(pos_2d)
        self.positions_3d.append(obj.position_3d)
        if obj.clip_embedding is not None:
            self.embeddings.append(obj.clip_embedding)


class ObjectTracker:
    """
    Tracks objects across frames.
    
    Strategy:
    1. Match detected objects to existing tracks using 2D proximity + appearance
    2. Create new track if no match found
    3. Age out tracks that aren't seen for N frames
    """
    
    def __init__(
        self,
        max_distance_px: float = 100.0,       # Max pixel distance for 2D matching
        max_distance_3d: float = 1.0,         # Max meters for 3D matching
        embedding_threshold: float = 0.7,     # Cosine similarity for CLIP matching
        max_age_frames: int = 5,              # Frames before track expires
    ):
        self.max_distance_px = max_distance_px
        self.max_distance_3d = max_distance_3d
        self.embedding_threshold = embedding_threshold
        self.max_age_frames = max_age_frames
        
        self.tracks: dict[int, TrackedObject] = {}
        self.next_track_id = 0
        self.frame_idx = 0
    
    def update(self, detections_3d: list[Object3D], centroids_2d: list[tuple[int, int]], frame_idx: int) -> dict[int, TrackedObject]:
        """
        Update tracker with detections from current frame.
        
        Args:
            detections_3d: list of Object3D detections
            centroids_2d: list of (x, y) image centroids corresponding to detections
            frame_idx: Current frame index
        
        Returns:
            dictionary of {track_id: TrackedObject} for current frame
        """
        self.frame_idx = frame_idx
        frame_tracks = {}
        
        # Track which detections have been matched
        matched_detections = set()
        
        # Try to match detections to existing tracks
        for track_id, track in list(self.tracks.items()):
            best_match_idx = -1
            best_distance = float('inf')
            
            for det_idx, (obj, pos_2d) in enumerate(zip(detections_3d, centroids_2d)):
                if det_idx in matched_detections:
                    continue  # Already matched
                
                # Skip if class doesn't match
                if obj.class_name != track.class_name:
                    continue
                
                # Compute matching score
                score = self._compute_match_score(track, obj, pos_2d)
                
                if score < best_distance:
                    best_distance = score
                    best_match_idx = det_idx
            
            # Accept match if score is good
            if best_match_idx >= 0 and best_distance < 1.0:  # Normalized score < 1.0
                det_idx = best_match_idx
                obj = detections_3d[det_idx]
                pos_2d = centroids_2d[det_idx]
                
                track.update(obj, pos_2d, frame_idx)
                frame_tracks[track_id] = track
                matched_detections.add(det_idx)
            else:
                # Track not matched this frame - mark for potential expiration
                pass
        
        # Remove old tracks
        expired_ids = [
            tid for tid, track in self.tracks.items()
            if frame_idx - track.last_seen > self.max_age_frames
        ]
        for tid in expired_ids:
            del self.tracks[tid]
        
        # Create new tracks for unmatched detections
        for det_idx, (obj, pos_2d) in enumerate(zip(detections_3d, centroids_2d)):
            if det_idx not in matched_detections:
                track = TrackedObject(
                    track_id=self.next_track_id,
                    class_name=obj.class_name,
                    first_seen=frame_idx,
                    last_seen=frame_idx,
                    age=1,
                    confidence_history=[obj.confidence],
                    positions_2d=[pos_2d],
                    positions_3d=[obj.position_3d],
                    embeddings=[obj.clip_embedding] if obj.clip_embedding is not None else [],
                )
                self.tracks[self.next_track_id] = track
                frame_tracks[self.next_track_id] = track
                self.next_track_id += 1
        
        return frame_tracks
    
    def _compute_match_score(self, track: TrackedObject, detection: Object3D, pos_2d: tuple[int, int]) -> float:
        """
        Compute matching score between track and detection.
        Lower score = better match. Score should be 0-1 for good matches.
        
        Combines multiple cues:
        - 2D proximity (pixel distance)
        - 3D proximity (world distance)
        - Embedding similarity (CLIP)
        """
        score_parts = []
        
        # 1. 2D distance (normalized by max_distance)
        track_pos_2d = track.current_position_2d
        dist_2d = np.sqrt((pos_2d[0] - track_pos_2d[0])**2 + (pos_2d[1] - track_pos_2d[1])**2)
        score_2d = min(1.0, dist_2d / self.max_distance_px)  # 0-1, lower better
        score_parts.append(('2d', 0.6, score_2d))
        
        # 2. 3D distance (if available)
        if track.current_position_3d is not None and detection.position_3d is not None:
            dist_3d = np.linalg.norm(detection.position_3d - track.current_position_3d)
            score_3d = min(1.0, dist_3d / self.max_distance_3d)  # 0-1, lower better
            score_parts.append(('3d', 0.3, score_3d))
        
        # 3. Embedding similarity (CLIP)
        if track.embeddings and detection.clip_embedding is not None:
            recent_embedding = track.embeddings[-1]
            if recent_embedding is not None:
                # Cosine similarity
                sim = np.dot(recent_embedding, detection.clip_embedding) / (
                    np.linalg.norm(recent_embedding) * np.linalg.norm(detection.clip_embedding) + 1e-8
                )
                # Convert similarity (1=match, -1=opposite) to distance (0=match, 1=opposite)
                score_embedding = (1.0 - sim) / 2.0
                score_parts.append(('embedding', 0.1, score_embedding))
        
        # Weighted average
        total_weight = sum(w for _, w, _ in score_parts)
        if total_weight == 0:
            return 1.0
        
        final_score = sum(w * s for _, w, s in score_parts) / total_weight
        return final_score
    
    def get_active_tracks(self) -> dict[int, TrackedObject]:
        """Get all active tracks"""
        return self.tracks.copy()
    
    def get_statistics(self) -> dict:
        """Get tracking statistics"""
        return {
            'total_tracks': len(self.tracks),
            'total_track_ids_ever': self.next_track_id,
            'avg_age': np.mean([t.age for t in self.tracks.values()]) if self.tracks else 0,
            'tracks_by_class': self._group_by_class(),
        }
    
    def _group_by_class(self) -> dict[str, int]:
        """Group active tracks by class"""
        groups = {}
        for track in self.tracks.values():
            groups[track.class_name] = groups.get(track.class_name, 0) + 1
        return groups
