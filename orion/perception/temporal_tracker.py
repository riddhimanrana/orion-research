#!/usr/bin/env python3
"""
Temporal Object Tracker with Re-ID
===================================

Implements robust object tracking across frames with:
- ByteTrack-style association
- CLIP embeddings for re-identification  
- Temporal smoothing of positions
- Motion prediction for occlusions
- Confidence-based filtering

Addresses the duplicate object problem by maintaining consistent IDs.

Author: Orion Research
Date: November 11, 2025
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import cv2


@dataclass
class TrackedObject:
    """Represents a tracked object across frames"""
    track_id: int
    class_name: str
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    embedding: Optional[np.ndarray] = None  # CLIP embedding for re-ID
    position_3d: Optional[np.ndarray] = None  # World space position
    velocity_3d: Optional[np.ndarray] = None  # 3D velocity
    
    # Temporal history
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=10))
    embedding_history: deque = field(default_factory=lambda: deque(maxlen=5))
    position_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # State
    frames_since_update: int = 0
    total_frames_tracked: int = 0
    is_confirmed: bool = False  # Confirmed after 3 consecutive detections
    
    def update(self, bbox: np.ndarray, confidence: float, 
               embedding: Optional[np.ndarray] = None,
               position_3d: Optional[np.ndarray] = None):
        """Update track with new detection"""
        self.bbox = bbox
        self.confidence = confidence
        self.frames_since_update = 0
        self.total_frames_tracked += 1
        
        # Update history
        self.bbox_history.append(bbox.copy())
        
        if embedding is not None:
            self.embedding = embedding
            self.embedding_history.append(embedding.copy())
        
        if position_3d is not None:
            # Compute velocity if we have previous position
            if self.position_3d is not None:
                self.velocity_3d = position_3d - self.position_3d
            self.position_3d = position_3d
            self.position_history.append(position_3d.copy())
        
        # Confirm after 3 consecutive frames
        if not self.is_confirmed and self.total_frames_tracked >= 3:
            self.is_confirmed = True
    
    def predict(self):
        """Predict next position using motion model"""
        self.frames_since_update += 1
        
        # Simple constant velocity prediction
        if self.velocity_3d is not None and self.position_3d is not None:
            self.position_3d = self.position_3d + self.velocity_3d
    
    def get_smoothed_bbox(self) -> np.ndarray:
        """Get temporally smoothed bounding box"""
        if len(self.bbox_history) < 2:
            return self.bbox
        
        # Exponential moving average
        boxes = np.array(list(self.bbox_history))
        weights = np.exp(np.linspace(-1, 0, len(boxes)))
        weights /= weights.sum()
        
        return (boxes.T @ weights).astype(np.float32)
    
    def get_average_embedding(self) -> Optional[np.ndarray]:
        """Get averaged embedding for robust matching"""
        if not self.embedding_history:
            return self.embedding
        
        embeddings = np.array(list(self.embedding_history))
        avg_embedding = embeddings.mean(axis=0)
        # Normalize
        return avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)


class TemporalTracker:
    """
    Temporal object tracker with re-identification
    
    Features:
    - ByteTrack-style data association (high + low confidence)
    - CLIP embedding matching for re-ID
    - Motion prediction for occluded objects
    - Temporal smoothing
    - Automatic track lifecycle management
    """
    
    def __init__(self,
                 iou_threshold: float = 0.3,
                 embedding_threshold: float = 0.7,
                 max_age: int = 30,
                 min_hits: int = 3):
        """
        Args:
            iou_threshold: IoU threshold for matching
            embedding_threshold: Cosine similarity threshold for re-ID
            max_age: Max frames to keep track without update
            min_hits: Min detections to confirm track
        """
        self.iou_threshold = iou_threshold
        self.embedding_threshold = embedding_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        
        self.tracks: List[TrackedObject] = []
        self.next_id = 0
        self.frame_count = 0
        
        print(f"  ✓ Temporal tracker initialized")
        print(f"    IoU threshold: {iou_threshold}")
        print(f"    Embedding similarity: {embedding_threshold}")
        print(f"    Max age: {max_age} frames")
    
    def update(self, 
               detections: List[Dict],
               embeddings: Optional[List[np.ndarray]] = None) -> List[TrackedObject]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of detection dicts with 'bbox', 'score', 'class', etc.
            embeddings: Optional CLIP embeddings for each detection
        
        Returns:
            List of active TrackedObject instances
        """
        self.frame_count += 1
        
        # Predict existing tracks
        for track in self.tracks:
            track.predict()
        
        if not detections:
            # No detections - age out tracks
            self.tracks = [t for t in self.tracks 
                          if t.frames_since_update < self.max_age]
            return self.tracks
        
        # Split detections by confidence (ByteTrack style)
        high_conf = []
        low_conf = []
        high_conf_idx = []
        low_conf_idx = []
        
        for i, det in enumerate(detections):
            if det['score'] >= 0.5:
                high_conf.append(det)
                high_conf_idx.append(i)
            else:
                low_conf.append(det)
                low_conf_idx.append(i)
        
        # First association: high confidence detections
        matched, unmatched_tracks, unmatched_dets = self._associate(
            self.tracks, high_conf, high_conf_idx, embeddings
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            det = high_conf[det_idx]
            emb = embeddings[high_conf_idx[det_idx]] if embeddings else None
            
            self.tracks[track_idx].update(
                bbox=np.array(det['bbox']),
                confidence=det['score'],
                embedding=emb,
                position_3d=det.get('position_3d')
            )
        
        # Second association: unmatched tracks vs low confidence detections
        unmatched_tracks_obj = [self.tracks[i] for i in unmatched_tracks]
        matched_low, unmatched_tracks_2, unmatched_dets_low = self._associate(
            unmatched_tracks_obj, low_conf, low_conf_idx, embeddings,
            iou_threshold=0.5  # Higher threshold for low conf
        )
        
        # Update from low confidence matches
        for track_idx, det_idx in matched_low:
            det = low_conf[det_idx]
            emb = embeddings[low_conf_idx[det_idx]] if embeddings else None
            
            track_idx_global = unmatched_tracks[track_idx]
            self.tracks[track_idx_global].update(
                bbox=np.array(det['bbox']),
                confidence=det['score'],
                embedding=emb,
                position_3d=det.get('position_3d')
            )
        
        # Create new tracks for unmatched high confidence detections
        for det_idx in unmatched_dets:
            det = high_conf[det_idx]
            emb = embeddings[high_conf_idx[det_idx]] if embeddings else None
            
            new_track = TrackedObject(
                track_id=self.next_id,
                class_name=det['class'],
                bbox=np.array(det['bbox']),
                confidence=det['score'],
                embedding=emb,
                position_3d=det.get('position_3d')
            )
            new_track.update(new_track.bbox, new_track.confidence, emb, new_track.position_3d)
            
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks 
                      if t.frames_since_update < self.max_age]
        
        return self.tracks
    
    def _associate(self,
                   tracks: List[TrackedObject],
                   detections: List[Dict],
                   det_indices: List[int],
                   embeddings: Optional[List[np.ndarray]],
                   iou_threshold: Optional[float] = None) -> Tuple[List, List, List]:
        """
        Associate tracks with detections using IoU + embedding similarity
        
        Returns:
            (matched, unmatched_tracks, unmatched_detections)
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        iou_thresh = iou_threshold or self.iou_threshold
        
        # Compute cost matrix (IoU + embedding similarity)
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                # IoU cost
                iou = self._compute_iou(track.bbox, np.array(det['bbox']))
                cost = 1.0 - iou  # Lower is better
                
                # Embedding similarity (if available)
                if embeddings and track.embedding is not None:
                    det_emb = embeddings[det_indices[j]]
                    track_emb = track.get_average_embedding()
                    
                    if det_emb is not None and track_emb is not None:
                        # Cosine similarity
                        similarity = np.dot(det_emb, track_emb) / (
                            np.linalg.norm(det_emb) * np.linalg.norm(track_emb) + 1e-8
                        )
                        
                        # Combine with IoU (weighted average)
                        cost = 0.6 * cost + 0.4 * (1.0 - similarity)
                
                cost_matrix[i, j] = cost
        
        # Hungarian algorithm (greedy for speed)
        matched = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))
        
        # Greedy matching
        while True:
            if len(unmatched_tracks) == 0 or len(unmatched_dets) == 0:
                break
            
            # Find minimum cost
            min_cost = float('inf')
            min_i, min_j = -1, -1
            
            for i in unmatched_tracks:
                for j in unmatched_dets:
                    if cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        min_i, min_j = i, j
            
            # Check if match is good enough
            if min_cost > (1.0 - iou_thresh):
                break
            
            matched.append((min_i, min_j))
            unmatched_tracks.remove(min_i)
            unmatched_dets.remove(min_j)
        
        return matched, unmatched_tracks, unmatched_dets
    
    @staticmethod
    def _compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute IoU between two bounding boxes [x1, y1, x2, y2]"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-8)
    
    def get_confirmed_tracks(self) -> List[TrackedObject]:
        """Get only confirmed tracks (tracked for min_hits frames)"""
        return [t for t in self.tracks if t.is_confirmed]
    
    def reset(self):
        """Reset tracker state"""
        self.tracks = []
        self.next_id = 0
        self.frame_count = 0
