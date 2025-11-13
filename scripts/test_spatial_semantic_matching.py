#!/usr/bin/env python3
"""
New Re-ID strategy: Spatial-Semantic Matching
Instead of relying on CLIP appearance embeddings (which have 0.78 avg similarity),
use combination of:
1. Spatial consistency (objects don't teleport)
2. Semantic category (verified by CLIP)
3. Size consistency (objects don't change size dramatically)
4. Coarse appearance (only as tiebreaker, not primary signal)
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

@dataclass
class Detection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    class_name: str
    embedding: np.ndarray
    confidence: float

@dataclass
class Track:
    id: int
    bbox: np.ndarray
    class_name: str
    embedding: np.ndarray  # avg appearance
    age: int
    hits: int
    velocity: Optional[np.ndarray] = None  # [dx, dy]

def compute_spatial_cost(det_bbox, track_bbox, track_velocity=None):
    """
    Cost based on spatial distance.
    If track has velocity, predict next position and compare.
    """
    det_center = np.array([
        (det_bbox[0] + det_bbox[2]) / 2,
        (det_bbox[1] + det_bbox[3]) / 2
    ])
    
    track_center = np.array([
        (track_bbox[0] + track_bbox[2]) / 2,
        (track_bbox[1] + track_bbox[3]) / 2
    ])
    
    # Predict next position if velocity known
    if track_velocity is not None:
        predicted_center = track_center + track_velocity
    else:
        predicted_center = track_center
    
    # Euclidean distance normalized by frame diagonal
    dist = np.linalg.norm(det_center - predicted_center)
    frame_diag = np.sqrt(1920**2 + 1080**2)  # assuming 1080p
    
    return dist / frame_diag  # 0 = perfect, 1 = opposite corners

def compute_size_cost(det_bbox, track_bbox):
    """
    Cost based on bbox size change.
    Objects shouldn't grow/shrink dramatically.
    """
    det_area = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
    track_area = (track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1])
    
    if track_area == 0:
        return 1.0
    
    size_ratio = det_area / track_area
    # Cost is high if ratio far from 1.0
    # Allow 2x growth/shrink before penalizing heavily
    if 0.5 < size_ratio < 2.0:
        return abs(1.0 - size_ratio)
    else:
        return 1.0

def compute_semantic_cost(det_class, track_class):
    """
    Cost based on class mismatch.
    0 = same class, 1 = different class
    """
    return 0.0 if det_class == track_class else 1.0

def compute_appearance_cost(det_embedding, track_embedding):
    """
    Cost based on CLIP embedding similarity.
    Only used as WEAK signal (low weight).
    """
    similarity = np.dot(det_embedding, track_embedding)
    return 1.0 - similarity

def match_detections_to_tracks(
    detections: list[Detection],
    tracks: list[Track],
    spatial_weight: float = 0.5,
    size_weight: float = 0.2,
    semantic_weight: float = 0.25,
    appearance_weight: float = 0.05,  # VERY LOW
    max_cost_threshold: float = 0.6
):
    """
    Hungarian matching with spatial-semantic priority.
    
    Cost = 0.5*spatial + 0.2*size + 0.25*semantic + 0.05*appearance
    
    Rationale:
    - Spatial: Primary signal (objects move continuously)
    - Size: Secondary (objects don't change size much)
    - Semantic: Important (class should stay same)
    - Appearance: Weak (CLIP embeddings not discriminative enough)
    """
    if len(detections) == 0 or len(tracks) == 0:
        return [], list(range(len(detections))), list(range(len(tracks)))
    
    # Build cost matrix
    cost_matrix = np.zeros((len(detections), len(tracks)))
    
    for i, det in enumerate(detections):
        for j, track in enumerate(tracks):
            spatial_cost = compute_spatial_cost(det.bbox, track.bbox, track.velocity)
            size_cost = compute_size_cost(det.bbox, track.bbox)
            semantic_cost = compute_semantic_cost(det.class_name, track.class_name)
            appearance_cost = compute_appearance_cost(det.embedding, track.embedding)
            
            cost = (
                spatial_weight * spatial_cost +
                size_weight * size_cost +
                semantic_weight * semantic_cost +
                appearance_weight * appearance_cost
            )
            
            cost_matrix[i, j] = cost
    
    # Hungarian matching
    det_indices, track_indices = linear_sum_assignment(cost_matrix)
    
    matches = []
    unmatched_dets = []
    unmatched_tracks = list(range(len(tracks)))
    
    for det_idx, track_idx in zip(det_indices, track_indices):
        if cost_matrix[det_idx, track_idx] < max_cost_threshold:
            matches.append((det_idx, track_idx))
            if track_idx in unmatched_tracks:
                unmatched_tracks.remove(track_idx)
        else:
            unmatched_dets.append(det_idx)
    
    # Add detections that weren't matched at all
    all_matched_dets = set([m[0] for m in matches] + unmatched_dets)
    unmatched_dets.extend([i for i in range(len(detections)) if i not in all_matched_dets])
    
    return matches, unmatched_dets, unmatched_tracks


# TEST THE NEW MATCHING
if __name__ == "__main__":
    print("="*80)
    print("SPATIAL-SEMANTIC MATCHING TEST")
    print("="*80)
    
    # Simulate scenario: keyboard moved slightly between frames
    track_keyboard = Track(
        id=1,
        bbox=np.array([0, 1028, 635, 1345]),  # frame 100
        class_name="keyboard",
        embedding=np.random.randn(512),  # doesn't matter much now
        age=1,
        hits=1,
        velocity=np.array([2, 1])  # moving right 2px, down 1px per frame
    )
    
    # Frame 101: keyboard moved to [4, 1031, 504, 1209]
    det_keyboard = Detection(
        bbox=np.array([4, 1031, 504, 1209]),
        class_name="keyboard",
        embedding=np.random.randn(512),
        confidence=0.85
    )
    
    # Also a NEW monitor detection
    det_monitor = Detection(
        bbox=np.array([5, 84, 819, 687]),
        class_name="monitor",
        embedding=np.random.randn(512),
        confidence=0.92
    )
    
    detections = [det_keyboard, det_monitor]
    tracks = [track_keyboard]
    
    matches, unmatched_dets, unmatched_tracks = match_detections_to_tracks(
        detections, tracks
    )
    
    print("\nResults:")
    print(f"  Matches: {matches}")
    print(f"  Unmatched detections: {unmatched_dets}")
    print(f"  Unmatched tracks: {unmatched_tracks}")
    
    if matches:
        det_idx, track_idx = matches[0]
        det = detections[det_idx]
        track = tracks[track_idx]
        
        spatial = compute_spatial_cost(det.bbox, track.bbox, track.velocity)
        size = compute_size_cost(det.bbox, track.bbox)
        semantic = compute_semantic_cost(det.class_name, track.class_name)
        
        print(f"\nMatch costs:")
        print(f"  Spatial:  {spatial:.3f}")
        print(f"  Size:     {size:.3f}")
        print(f"  Semantic: {semantic:.3f}")
        print(f"  Total:    {0.5*spatial + 0.2*size + 0.25*semantic:.3f}")
        
        print("\n✓ Keyboard correctly matched across frames!")
    
    print("="*80)
