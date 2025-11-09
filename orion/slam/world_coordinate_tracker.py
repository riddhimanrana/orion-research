"""
World Coordinate Tracker

Maintains entity positions in world coordinates using SLAM transforms.

Author: Orion Research Team  
Date: November 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class WorldCoordinateTracker:
    """
    Tracker that maintains entity positions in world coordinates
    
    Uses SLAM poses to transform camera-frame observations to world frame.
    """
    
    def __init__(self, slam_engine):
        """
        Initialize world coordinate tracker
        
        Args:
            slam_engine: SLAMEngine instance
        """
        self.slam = slam_engine
        
        # Storage: entity_id → [(timestamp, pos_world, frame_idx), ...]
        self.entities_world: Dict[str, List[Tuple[float, np.ndarray, int]]] = defaultdict(list)
        
        # Statistics
        self.total_observations = 0
        self.failed_transforms = 0
    
    def add_observation(
        self,
        entity_id: str,
        timestamp: float,
        pos_camera: np.ndarray,  # (x, y, z) in camera frame (mm)
        frame_idx: int
    ):
        """
        Add entity observation and transform to world coordinates
        
        Args:
            entity_id: Unique entity ID
            timestamp: Observation timestamp
            pos_camera: (x, y, z) in mm, camera frame
            frame_idx: Frame index (for pose lookup)
        """
        self.total_observations += 1
        
        # Transform to world coordinates
        try:
            pos_world = self.slam.transform_to_world(pos_camera, frame_idx)
            self.entities_world[entity_id].append((timestamp, pos_world, frame_idx))
        except Exception as e:
            self.failed_transforms += 1
            # Fall back to camera coordinates (not ideal but better than nothing)
            self.entities_world[entity_id].append((timestamp, pos_camera, frame_idx))
    
    def get_entity_world_centroid(self, entity_id: str) -> Optional[np.ndarray]:
        """
        Get mean world position for entity
        
        Args:
            entity_id: Entity ID
        
        Returns:
            (x, y, z) world coordinates in mm, or None if not found
        """
        if entity_id not in self.entities_world:
            return None
        
        positions = [obs[1] for obs in self.entities_world[entity_id]]
        return np.mean(positions, axis=0)
    
    def get_entity_observations(self, entity_id: str) -> List[Tuple[float, np.ndarray, int]]:
        """
        Get all world-coordinate observations for an entity
        
        Args:
            entity_id: Entity ID
        
        Returns:
            List of (timestamp, pos_world, frame_idx)
        """
        return self.entities_world.get(entity_id, [])
    
    def get_all_entity_centroids(self) -> Dict[str, np.ndarray]:
        """
        Get world centroids for all entities
        
        Returns:
            Dict mapping entity_id → (x, y, z) world coordinates
        """
        centroids = {}
        for entity_id in self.entities_world:
            centroid = self.get_entity_world_centroid(entity_id)
            if centroid is not None:
                centroids[entity_id] = centroid
        return centroids
    
    def get_entities_in_region(
        self,
        center: np.ndarray,
        radius_mm: float
    ) -> List[str]:
        """
        Get entities within a spatial region
        
        Args:
            center: (x, y, z) center in world coordinates (mm)
            radius_mm: Radius in mm
        
        Returns:
            List of entity IDs within region
        """
        entities_in_region = []
        
        for entity_id, centroid in self.get_all_entity_centroids().items():
            dist = np.linalg.norm(centroid - center)
            if dist <= radius_mm:
                entities_in_region.append(entity_id)
        
        return entities_in_region
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics"""
        return {
            'total_entities': len(self.entities_world),
            'total_observations': self.total_observations,
            'failed_transforms': self.failed_transforms,
            'success_rate': 1.0 - (self.failed_transforms / max(1, self.total_observations)),
            'avg_observations_per_entity': (
                self.total_observations / len(self.entities_world)
                if self.entities_world else 0
            ),
        }
    
    def export_to_dict(self) -> Dict:
        """
        Export all entity world positions to dict
        
        Returns:
            Dict suitable for JSON serialization
        """
        export = {}
        
        for entity_id, observations in self.entities_world.items():
            export[entity_id] = {
                'centroid_world': self.get_entity_world_centroid(entity_id).tolist(),
                'num_observations': len(observations),
                'first_seen': observations[0][0],
                'last_seen': observations[-1][0],
                'observations': [
                    {
                        'timestamp': obs[0],
                        'position_world': obs[1].tolist(),
                        'frame_idx': obs[2]
                    }
                    for obs in observations
                ]
            }
        
        return export
