"""
Entity Tracker
==============

Clusters observations into unique tracked entities using HDBSCAN.

Responsibilities:
- Cluster observations by embedding similarity
- Group observations into unique entities
- Compute entity statistics (appearance count, temporal bounds)
- Select best observation per entity for description
- Provide lightweight motion tracking utilities for causal inference

Author: Orion Research Team
Date: October 2025
"""

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from orion.perception.types import Observation, PerceptionEntity, ObjectClass
from orion.perception.config import PerceptionConfig

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    hdbscan = None
    HDBSCAN_AVAILABLE = False

logger = logging.getLogger(__name__)

Vector2 = Tuple[float, float]
BBox = Sequence[Union[float, int]]


def bbox_to_centroid(bbox: BBox) -> Vector2:
    x1, y1, x2, y2 = map(float, bbox)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = map(float, bbox)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_overlap_area(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    return (inter_x2 - inter_x1) * (inter_y2 - inter_y1)


def calculate_distance(point_a: Vector2, point_b: Vector2) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


@dataclass
class MotionData:
    centroid: Vector2
    velocity: Vector2
    speed: float
    direction: float
    timestamp: float

    def is_moving_towards(
        self,
        target: Vector2,
        angle_threshold: float = math.pi / 4,
        min_speed: float = 0.5,
    ) -> bool:
        if self.speed < min_speed:
            return False
        dx = target[0] - self.centroid[0]
        dy = target[1] - self.centroid[1]
        distance = math.hypot(dx, dy)
        if distance == 0.0:
            return True
        dot = self.velocity[0] * dx + self.velocity[1] * dy
        cos_angle = dot / (self.speed * distance)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle = math.acos(cos_angle)
        return angle <= angle_threshold


class MotionTracker:
    """Simple centroid-based motion tracker with sliding-window smoothing."""

    def __init__(self, smoothing_window: int = 5):
        self.smoothing_window = max(1, int(smoothing_window))
        self.history: Dict[str, Deque[Tuple[float, Vector2]]] = {}
        self._last_motion: Dict[str, MotionData] = {}

    def update(self, object_id: str, timestamp: float, bbox: BBox) -> MotionData:
        centroid = bbox_to_centroid(bbox)
        history = self.history.setdefault(object_id, deque(maxlen=self.smoothing_window))
        history.append((timestamp, centroid))

        if len(history) < 2:
            motion = MotionData(centroid=centroid, velocity=(0.0, 0.0), speed=0.0, direction=0.0, timestamp=timestamp)
            self._last_motion[object_id] = motion
            return motion

        oldest_ts, oldest_centroid = history[0]
        newest_ts, newest_centroid = history[-1]
        dt = max(newest_ts - oldest_ts, 1e-6)
        velocity = (
            (newest_centroid[0] - oldest_centroid[0]) / dt,
            (newest_centroid[1] - oldest_centroid[1]) / dt,
        )
        speed = math.hypot(*velocity)
        direction = math.atan2(velocity[1], velocity[0]) if speed > 0 else 0.0
        motion = MotionData(centroid=centroid, velocity=velocity, speed=speed, direction=direction, timestamp=timestamp)
        self._last_motion[object_id] = motion
        return motion

    def get_motion_at_time(
        self,
        object_id: str,
        timestamp: float,
        tolerance: float = 0.25,
    ) -> Optional[MotionData]:
        history = self.history.get(object_id)
        if not history:
            return None
        closest_ts, closest_centroid = min(history, key=lambda item: abs(item[0] - timestamp))
        if abs(closest_ts - timestamp) > tolerance:
            return None
        motion = self._last_motion.get(object_id)
        if motion and abs(motion.timestamp - closest_ts) <= tolerance:
            return motion
        return MotionData(
            centroid=closest_centroid,
            velocity=(0.0, 0.0),
            speed=0.0,
            direction=0.0,
            timestamp=closest_ts,
        )

    def clear_old_history(self, cutoff_timestamp: float) -> None:
        for object_id in list(self.history.keys()):
            history = self.history[object_id]
            while history and history[0][0] < cutoff_timestamp:
                history.popleft()
            if not history:
                del self.history[object_id]
                self._last_motion.pop(object_id, None)


class EntityTracker:
    """
    Tracks entities by clustering observations.
    
    Uses HDBSCAN to cluster observations by embedding similarity,
    identifying unique objects across the video.
    """
    
    def __init__(self, config: PerceptionConfig):
        """
        Initialize tracker.
        
        Args:
            config: Perception configuration
        """
        self.config = config
        
        if not HDBSCAN_AVAILABLE:
            logger.warning("HDBSCAN not available - falling back to simple clustering")
        
        logger.debug(
            f"EntityTracker initialized: min_cluster_size=3, "
            f"metric=euclidean"
        )
    
    def cluster_observations(
        self,
        observations: List[Observation],
    ) -> List[PerceptionEntity]:
        """
        Cluster observations into entities.
        
        Args:
            observations: List of observations from detection + embedding
            
        Returns:
            List of tracked entities
        """
        logger.info("="*80)
        logger.info("PHASE 1C: ENTITY CLUSTERING & TRACKING")
        logger.info("="*80)
        
        if not observations:
            logger.warning("No observations to cluster")
            return []
        
        logger.info(f"Clustering {len(observations)} observations...")
        
        if not HDBSCAN_AVAILABLE:
            return self._fallback_clustering(observations)
        
        # Extract embeddings
        embeddings = np.array([obs.visual_embedding for obs in observations])
        logger.info(f"  Embedding shape: {embeddings.shape}")
        
        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=3,
            min_samples=1,
            metric="euclidean",
            cluster_selection_epsilon=0.35,
            cluster_selection_method="eom",
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        # Analyze clustering results
        unique_labels = set(labels)
        noise_count = np.sum(labels == -1)
        entity_count = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        logger.info(f"  Clusters found: {entity_count}")
        logger.info(f"  Noise points: {noise_count} ({100 * noise_count / len(labels):.1f}%)")
        
        # Group observations by cluster
        clusters = defaultdict(list)
        for obs, label in zip(observations, labels):
            if label != -1:  # Skip noise
                clusters[label].append(obs)
        
        # Create entities
        entities = []
        for label, cluster_obs in clusters.items():
            # Determine entity class (majority vote)
            class_counts = defaultdict(int)
            for obs in cluster_obs:
                class_counts[obs.object_class] += 1
            
            entity_class = max(class_counts, key=class_counts.get)
            
            # Create entity
            entity = PerceptionEntity(
                entity_id=f"entity_{label}",
                object_class=entity_class,
                observations=cluster_obs,
            )
            
            # Compute average embedding
            entity.compute_average_embedding()
            
            entities.append(entity)
        
        # Sort by appearance count (descending)
        entities.sort(key=lambda e: e.appearance_count, reverse=True)
        
        # Log summary
        logger.info(f"\n✓ Tracked {len(entities)} unique entities")
        logger.info("  Top entities by appearance:")
        for i, entity in enumerate(entities[:10]):
            logger.info(
                f"    {i+1}. {entity.entity_id}: {entity.object_class.value} "
                f"({entity.appearance_count} appearances, "
                f"frames {entity.first_seen_frame}-{entity.last_seen_frame})"
            )
        
        if len(entities) > 10:
            logger.info(f"    ... and {len(entities) - 10} more")
        
        logger.info("="*80 + "\n")
        
        return entities
    
    def _fallback_clustering(
        self,
        observations: List[Observation],
    ) -> List[PerceptionEntity]:
        """
        Simple fallback clustering when HDBSCAN unavailable.
        
        Groups observations by class name + temporal proximity.
        """
        logger.warning("Using fallback clustering (class + temporal)")
        
        # Group by class
        class_groups = defaultdict(list)
        for obs in observations:
            class_groups[obs.object_class].append(obs)
        
        entities = []
        entity_id = 0
        
        for object_class, class_obs in class_groups.items():
            # Sort by frame number
            class_obs.sort(key=lambda o: o.frame_number)
            
            # Create entities (one per continuous appearance window)
            current_entity_obs = []
            last_frame = -100
            
            for obs in class_obs:
                # Start new entity if gap > 30 frames
                if obs.frame_number - last_frame > 30:
                    if current_entity_obs:
                        entity = PerceptionEntity(
                            entity_id=f"entity_{entity_id}",
                            object_class=object_class,
                            observations=current_entity_obs,
                        )
                        entity.compute_average_embedding()
                        entities.append(entity)
                        entity_id += 1
                    
                    current_entity_obs = []
                
                current_entity_obs.append(obs)
                last_frame = obs.frame_number
            
            # Add final entity
            if current_entity_obs:
                entity = PerceptionEntity(
                    entity_id=f"entity_{entity_id}",
                    object_class=object_class,
                    observations=current_entity_obs,
                )
                entity.compute_average_embedding()
                entities.append(entity)
                entity_id += 1
        
        logger.info(f"✓ Created {len(entities)} entities (fallback method)")
        
        return entities
