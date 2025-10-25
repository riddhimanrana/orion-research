"""
Spatial Utilities for Zone Detection
=====================================

HDBSCAN-based spatial clustering to detect zones like:
- desk_area (keyboard, mouse, monitor)
- bedroom_area (bed, pillow)
- kitchen_area (appliances)

Author: Orion Research Team
Date: October 2025 - Phase 2
"""

import logging
from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Zone classification patterns
ZONE_PATTERNS = {
    'desk_area': ['keyboard', 'mouse', 'monitor', 'tv', 'laptop', 'remote', 'cell phone'],
    'bedroom_area': ['bed', 'pillow', 'clock'],
    'kitchen_area': ['refrigerator', 'oven', 'microwave', 'sink', 'toaster'],
    'living_area': ['couch', 'tv', 'remote', 'chair'],
    'workspace': ['chair', 'desk', 'laptop', 'book'],
    'entertainment_area': ['tv', 'remote', 'couch', 'chair'],
}


@dataclass
class SpatialZone:
    """Represents a spatial zone detected via clustering."""
    zone_id: str
    label: str  # e.g., "desk_area", "bedroom_area"
    entity_ids: List[str]
    centroid: Tuple[float, float]  # (x, y) normalized [0, 1]
    bounding_box: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized
    confidence: float
    dominant_classes: List[str] = field(default_factory=list)
    summary: str = ""
    
    # Optional relationships
    adjacent_zones: List[str] = field(default_factory=list)
    contained_zones: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'zone_id': self.zone_id,
            'label': self.label,
            'entity_ids': self.entity_ids,
            'centroid': list(self.centroid),
            'bounding_box': list(self.bounding_box),
            'confidence': self.confidence,
            'dominant_classes': self.dominant_classes,
            'summary': self.summary,
            'adjacent_zones': self.adjacent_zones,
            'contained_zones': self.contained_zones,
        }


def extract_spatial_features(
    entities: List,
    feature_weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Extract spatial features for HDBSCAN clustering.
    
    Features per entity:
    1. Centroid X (normalized 0-1)
    2. Centroid Y (normalized 0-1)
    3. Temporal co-occurrence score
    4. Bbox area (normalized, log-scaled)
    
    Args:
        entities: List of consolidated entities
        feature_weights: Optional weights for features
        
    Returns:
        (N, 4) numpy array of features
    """
    if not entities:
        return np.array([])

    features = []

    # Default weights
    if feature_weights is None:
        feature_weights = {'position': 0.6, 'temporal': 0.2, 'size': 0.2}

    # Get temporal range
    all_timestamps = []
    for entity in entities:
        if getattr(entity, 'first_timestamp', None) is not None and getattr(entity, 'last_timestamp', None) is not None:
            all_timestamps.extend([entity.first_timestamp, entity.last_timestamp])

    min_time = min(all_timestamps) if all_timestamps else 0.0
    max_time = max(all_timestamps) if all_timestamps else 1.0
    time_range = max(max_time - min_time, 1.0)

    for entity in entities:
        frame_width = getattr(entity, 'frame_width', 1920.0) or 1920.0
        frame_height = getattr(entity, 'frame_height', 1080.0) or 1080.0

        # Feature 1 & 2: Spatial position (centroid)
        if getattr(entity, 'average_centroid', None) is not None:
            cx, cy = entity.average_centroid
        elif getattr(entity, 'observations', None):
            centroids = [obs.centroid for obs in entity.observations if getattr(obs, 'centroid', None) is not None]
            if centroids:
                cx = float(np.mean([c[0] for c in centroids]))
                cy = float(np.mean([c[1] for c in centroids]))
            else:
                cx, cy = frame_width / 2.0, frame_height / 2.0
        else:
            cx, cy = frame_width / 2.0, frame_height / 2.0

        cx = float(np.clip(cx / frame_width, 0.0, 1.0))
        cy = float(np.clip(cy / frame_height, 0.0, 1.0))

        # Feature 3: Temporal co-occurrence (how much of video entity appears in)
        if getattr(entity, 'first_timestamp', None) is not None and getattr(entity, 'last_timestamp', None) is not None:
            temporal_span = max(entity.last_timestamp - entity.first_timestamp, 0.0)
            temporal_score = temporal_span / time_range if time_range > 0 else 0.5
        else:
            temporal_score = 0.5

        # Feature 4: Bbox area (normalized, log-scaled for scale invariance)
        bbox = getattr(entity, 'average_bbox', None)
        if bbox is not None:
            area = bbox.area if hasattr(bbox, 'area') else (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            normalized_area = max(area / (frame_width * frame_height), 1e-6)
            area_feature = np.log10(normalized_area)
        else:
            area_feature = -6.0  # Very small area by default

        feature_vector = [
            cx * feature_weights['position'],
            cy * feature_weights['position'],
            temporal_score * feature_weights['temporal'],
            (area_feature + 6.0) / 6.0 * feature_weights['size'],  # Normalize log scale to [0, 1]
        ]

        features.append(feature_vector)

    return np.array(features)


def cluster_entities_hdbscan(
    features: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster entities using HDBSCAN.
    
    Args:
        features: (N, D) array of spatial features
        min_cluster_size: Minimum cluster size
        min_samples: Minimum samples for core point
        
    Returns:
        Tuple of (labels, probabilities) each with shape (N,)
        where labels contain cluster ids (-1 = noise) and
        probabilities capture membership strength per entity.
    """
    if len(features) < min_cluster_size:
        logger.warning(
            "Not enough entities (%s) for clustering (min=%s)",
            len(features),
            min_cluster_size,
        )
        return np.full(len(features), -1), np.zeros(len(features))

    try:
        from hdbscan import HDBSCAN
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_epsilon=0.0,
        )

        labels = clusterer.fit_predict(features_scaled)
        probabilities = getattr(clusterer, "probabilities_", np.ones(len(labels)))

        n_clusters = len({label for label in labels if label != -1})
        n_noise = int(np.count_nonzero(labels == -1))

        logger.info("HDBSCAN clustering: %s zones, %s noise entities", n_clusters, n_noise)

        return labels, probabilities

    except ImportError:
        logger.error("HDBSCAN not installed. Run: pip install hdbscan")
        return np.full(len(features), -1), np.zeros(len(features))
    except Exception as exc:
        logger.error("HDBSCAN clustering failed: %s", exc)
        return np.full(len(features), -1), np.zeros(len(features))


def label_zone(
    entity_classes: List[str],
    centroid: Tuple[float, float],
) -> str:
    """
    Infer zone label from entity classes using pattern matching.
    
    Args:
        entity_classes: List of object classes in the zone
        centroid: Zone centroid (x, y) in [0, 1]
        
    Returns:
        Zone label (e.g., "desk_area", "bedroom_area")
    """
    if not entity_classes:
        return "unknown_area"
    
    # Normalize classes to lowercase
    entity_classes_lower = [cls.lower().replace('_', ' ') for cls in entity_classes]
    
    # Count matches for each pattern
    scores: Dict[str, int] = {}
    for zone_label, patterns in ZONE_PATTERNS.items():
        score = sum(1 for cls in entity_classes_lower if any(pattern in cls for pattern in patterns))
        scores[zone_label] = score
    
    # Return best match
    if scores and max(scores.values()) > 0:
        best_label = max(scores.items(), key=lambda item: item[1])[0]
        logger.debug("Zone labeled as '%s' based on entities: %s", best_label, entity_classes)
        return best_label
    
    # Fallback: position-based heuristic
    x, y = centroid
    if y < 0.3:
        return 'upper_area'
    elif y > 0.7:
        return 'lower_area'
    elif x < 0.3:
        return 'left_area'
    elif x > 0.7:
        return 'right_area'
    else:
        return 'central_area'


def compute_zone_centroid(entities: List) -> Tuple[float, float]:
    """Compute centroid of a zone from its entities."""
    centroids = []
    for entity in entities:
        if hasattr(entity, 'average_centroid'):
            centroids.append(entity.average_centroid)
        elif hasattr(entity, 'observations') and entity.observations:
            obs_centroids = [obs.centroid for obs in entity.observations if hasattr(obs, 'centroid')]
            if obs_centroids:
                avg_cx = np.mean([c[0] for c in obs_centroids])
                avg_cy = np.mean([c[1] for c in obs_centroids])
                centroids.append((avg_cx, avg_cy))
    
    if not centroids:
        return (0.5, 0.5)
    
    avg_x = np.mean([c[0] for c in centroids])
    avg_y = np.mean([c[1] for c in centroids])
    return (float(avg_x), float(avg_y))


def compute_zone_bbox(entities: List) -> Tuple[float, float, float, float]:
    """Compute bounding box of a zone from its entities."""
    all_bboxes = []
    for entity in entities:
        if hasattr(entity, 'average_bbox'):
            bbox = entity.average_bbox
            if hasattr(bbox, 'x1'):
                all_bboxes.append((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
            else:
                all_bboxes.append(bbox)
    
    if not all_bboxes:
        return (0.0, 0.0, 1.0, 1.0)
    
    # Compute union of all bboxes
    x1 = min(bbox[0] for bbox in all_bboxes)
    y1 = min(bbox[1] for bbox in all_bboxes)
    x2 = max(bbox[2] for bbox in all_bboxes)
    y2 = max(bbox[3] for bbox in all_bboxes)
    
    return (float(x1), float(y1), float(x2), float(y2))


def compute_zone_relationships(zones: List[SpatialZone]) -> None:
    """
    Compute adjacency relationships between zones (in-place).
    
    Two zones are adjacent if their bounding boxes are close.
    """
    adjacency_threshold = 0.15  # Normalized distance
    
    for i, zone1 in enumerate(zones):
        for j, zone2 in enumerate(zones):
            if i >= j:
                continue
            
            # Compute distance between centroids
            dist = np.sqrt(
                (zone1.centroid[0] - zone2.centroid[0]) ** 2 +
                (zone1.centroid[1] - zone2.centroid[1]) ** 2
            )
            
            if dist < adjacency_threshold:
                zone1.adjacent_zones.append(zone2.zone_id)
                zone2.adjacent_zones.append(zone1.zone_id)
