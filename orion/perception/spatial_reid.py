"""
Spatial-Temporal Re-ID using SLAM coordinates for viewpoint-invariant matching.

Key insight: In egocentric video, objects have persistent 3D positions in world space.
A monitor at position (X,Y,Z) is the SAME monitor regardless of camera viewpoint.

This solves the fundamental problem: CLIP embeddings are viewpoint-sensitive (0.78 avg
similarity for different objects), but world coordinates are viewpoint-invariant.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx


@dataclass
class SpatialHistory:
    """Track object's spatial presence over time."""
    object_id: int
    class_name: str
    
    # Spatial memory (world coordinates)
    world_positions: List[np.ndarray] = field(default_factory=list)  # [(x,y,z), ...]
    last_world_pos: Optional[np.ndarray] = None
    position_variance: float = 0.0  # How much object moves
    
    # Temporal context
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    total_observations: int = 0
    
    # Co-occurrence (which objects appear together)
    typical_neighbors: Set[str] = field(default_factory=set)  # {class_names}
    neighbor_distances: Dict[str, float] = field(default_factory=dict)  # {class: avg_dist}
    
    # Appearance (weak signal)
    appearance_embeddings: List[np.ndarray] = field(default_factory=list)
    avg_appearance: Optional[np.ndarray] = None
    
    def update(self, world_pos: np.ndarray, frame_idx: int, 
               embedding: Optional[np.ndarray] = None):
        """Update spatial history with new observation."""
        self.world_positions.append(world_pos)
        self.last_world_pos = world_pos
        self.last_seen_frame = frame_idx
        self.total_observations += 1
        
        # Update position variance (how much object moves)
        if len(self.world_positions) > 1:
            displacements = np.diff(self.world_positions, axis=0)
            self.position_variance = np.std(np.linalg.norm(displacements, axis=1))
        
        # Update appearance (EMA)
        if embedding is not None:
            self.appearance_embeddings.append(embedding)
            if self.avg_appearance is None:
                self.avg_appearance = embedding
            else:
                # EMA with alpha=0.9
                self.avg_appearance = 0.9 * self.avg_appearance + 0.1 * embedding


class SpatialTemporalReID:
    """
    SLAM-aware Re-ID using spatial persistence + temporal context.
    
    Core principle: Objects persist at world coordinates, regardless of camera viewpoint.
    
    Matching strategy:
    - 50% spatial proximity (world coordinates from SLAM)
    - 25% temporal plausibility (physics constraints)
    - 15% semantic context (co-occurrence patterns)
    - 10% appearance (CLIP, downweighted due to viewpoint issues)
    """
    
    def __init__(
        self,
        spatial_threshold: float = 200.0,  # mm - objects <20cm apart likely same
        velocity_threshold: float = 500.0,  # mm/s - max object velocity
        temporal_window: int = 90,  # frames - max gap for Re-ID (3 sec @ 30fps)
    ):
        self.spatial_threshold = spatial_threshold
        self.velocity_threshold = velocity_threshold
        self.temporal_window = temporal_window
        
        # Spatial memory database
        self.spatial_memory: Dict[int, SpatialHistory] = {}
        self.next_object_id = 0
        
        # Co-occurrence graph
        self.cooccurrence_graph = nx.Graph()
        
        # Zone classification (desk, shelf, floor)
        self.zones: Dict[str, List[np.ndarray]] = {}  # {zone_id: [positions]}
    
    def match_detection(
        self,
        world_pos: np.ndarray,
        class_name: str,
        embedding: Optional[np.ndarray],
        frame_idx: int,
        nearby_objects: Optional[List[str]] = None,
    ) -> Tuple[Optional[int], float]:
        """
        Match detection to existing objects using spatial-temporal cues.
        
        Returns:
            (object_id, confidence) or (None, 0.0) if no match
        """
        if not self.spatial_memory:
            return None, 0.0
        
        best_match_id = None
        best_score = 0.0
        
        for obj_id, history in self.spatial_memory.items():
            # Skip if different class (after normalization)
            if not self._classes_compatible(class_name, history.class_name):
                continue
            
            # 1. Spatial score (most reliable - viewpoint invariant!)
            spatial_dist = np.linalg.norm(world_pos - history.last_world_pos)
            spatial_score = np.exp(-spatial_dist / self.spatial_threshold)
            
            # 2. Temporal plausibility (physics constraints)
            time_gap_frames = frame_idx - history.last_seen_frame
            time_gap_sec = time_gap_frames / 30.0  # Assume 30fps
            
            # Max displacement based on velocity + object variance
            max_displacement = (
                self.velocity_threshold * time_gap_sec + 
                3.0 * history.position_variance  # 3-sigma
            )
            
            if spatial_dist > max_displacement:
                temporal_score = 0.0  # Physically impossible
            else:
                # Penalize long gaps (occlusion is fine, but prefer recent)
                temporal_score = np.exp(-time_gap_frames / 30.0)
            
            # 3. Semantic context score (co-occurrence)
            context_score = 0.5  # Neutral default
            if nearby_objects and history.typical_neighbors:
                overlap = len(set(nearby_objects) & history.typical_neighbors)
                context_score = min(1.0, overlap / max(len(history.typical_neighbors), 1))
            
            # 4. Appearance score (weak signal)
            appearance_score = 0.5  # Neutral default
            if embedding is not None and history.avg_appearance is not None:
                similarity = np.dot(embedding, history.avg_appearance)
                appearance_score = (similarity + 1.0) / 2.0  # Normalize to [0,1]
            
            # Weighted fusion (spatial dominates!)
            final_score = (
                0.50 * spatial_score +      # SLAM position (viewpoint-invariant!)
                0.25 * temporal_score +     # Physics constraints
                0.15 * context_score +      # Co-occurrence patterns
                0.10 * appearance_score     # CLIP (downweighted)
            )
            
            if final_score > best_score:
                best_score = final_score
                best_match_id = obj_id
        
        # Threshold for accepting match
        if best_score > 0.6:  # Require strong evidence
            return best_match_id, best_score
        else:
            return None, 0.0
    
    def create_new_object(
        self,
        world_pos: np.ndarray,
        class_name: str,
        frame_idx: int,
        embedding: Optional[np.ndarray] = None,
    ) -> int:
        """Create new object in spatial memory."""
        obj_id = self.next_object_id
        self.next_object_id += 1
        
        history = SpatialHistory(
            object_id=obj_id,
            class_name=class_name,
            first_seen_frame=frame_idx,
            last_seen_frame=frame_idx,
        )
        history.update(world_pos, frame_idx, embedding)
        
        self.spatial_memory[obj_id] = history
        return obj_id
    
    def update_object(
        self,
        obj_id: int,
        world_pos: np.ndarray,
        frame_idx: int,
        embedding: Optional[np.ndarray] = None,
        nearby_objects: Optional[List[str]] = None,
    ):
        """Update existing object's spatial history."""
        history = self.spatial_memory[obj_id]
        history.update(world_pos, frame_idx, embedding)
        
        # Update co-occurrence graph
        if nearby_objects:
            for neighbor_class in nearby_objects:
                if neighbor_class != history.class_name:
                    history.typical_neighbors.add(neighbor_class)
                    
                    # Update graph edge weight
                    if self.cooccurrence_graph.has_edge(history.class_name, neighbor_class):
                        self.cooccurrence_graph[history.class_name][neighbor_class]['weight'] += 1
                    else:
                        self.cooccurrence_graph.add_edge(
                            history.class_name, 
                            neighbor_class, 
                            weight=1
                        )
    
    def get_nearby_objects(
        self, 
        world_pos: np.ndarray, 
        radius: float = 500.0,
        exclude_id: Optional[int] = None
    ) -> List[str]:
        """Get class names of objects within radius (for context matching)."""
        nearby = []
        for obj_id, history in self.spatial_memory.items():
            if exclude_id and obj_id == exclude_id:
                continue
            
            dist = np.linalg.norm(world_pos - history.last_world_pos)
            if dist < radius:
                nearby.append(history.class_name)
        
        return nearby
    
    def cleanup_old_objects(self, current_frame: int):
        """Remove objects not seen recently (left the scene)."""
        to_remove = []
        for obj_id, history in self.spatial_memory.items():
            if current_frame - history.last_seen_frame > self.temporal_window:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.spatial_memory[obj_id]
    
    def _classes_compatible(self, class1: str, class2: str) -> bool:
        """Check if two classes could be the same object (after normalization)."""
        # Normalize common variants
        normalize_map = {
            'laptop keyboard': 'keyboard',
            'musical keyboard': 'keyboard',
            'desktop computer': 'laptop',
            'computer monitor': 'monitor',
            'diaper bag': 'backpack',
            'water bottle': 'bottle',
            'thermos': 'bottle',
            'thermos bottle': 'bottle',
        }
        
        c1 = normalize_map.get(class1.lower(), class1.lower())
        c2 = normalize_map.get(class2.lower(), class2.lower())
        
        return c1 == c2
    
    def get_statistics(self) -> Dict:
        """Get Re-ID statistics for monitoring."""
        return {
            'total_objects': len(self.spatial_memory),
            'active_objects': sum(
                1 for h in self.spatial_memory.values() 
                if h.total_observations >= 3
            ),
            'avg_observations_per_object': np.mean([
                h.total_observations for h in self.spatial_memory.values()
            ]) if self.spatial_memory else 0,
            'cooccurrence_edges': self.cooccurrence_graph.number_of_edges(),
        }
