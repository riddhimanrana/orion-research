"""
Spatial utilities for semantic analysis.

This module provides functions for detecting spatial zones, clustering entities,
and computing spatial relationships. It is designed to work with the data
structures from `orion.semantic.types`.

Key functions:
- `extract_spatial_features`: Creates feature vectors for clustering.
- `cluster_entities_hdbscan`: Groups entities into spatial zones.
- `compute_zone_centroid`: Calculates the center of a zone.
- `compute_zone_bbox`: Calculates the bounding box of a zone.
- `label_zone`: Generates a descriptive label for a zone.
"""
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np

# Assuming SemanticEntity and BoundingBox are defined in orion.semantic.types
# and orion.perception.types respectively. We create dummy classes for standalone execution.
try:
    from orion.semantic.types import SemanticEntity
    from orion.perception.types import BoundingBox
except ImportError:
    from dataclasses import dataclass, field
    @dataclass
    class BoundingBox:
        x1: float; y1: float; x2: float; y2: float
        def to_list(self): return [self.x1, self.y1, self.x2, self.y2]
        @property
        def center(self): return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @dataclass
    class SemanticEntity:
        entity_id: str
        class_label: str
        average_bbox: Optional[BoundingBox] = None
        average_centroid: Optional[Tuple[float, float]] = None
        frame_width: float = 1920.0
        frame_height: float = 1080.0


logger = logging.getLogger(__name__)


@dataclass
class SpatialZone:
    """Represents a detected spatial zone containing a cluster of entities."""
    zone_id: str
    label: str
    entity_ids: List[str]
    centroid: Tuple[float, float]
    bounding_box: BoundingBox
    confidence: float
    dominant_classes: List[str]
    summary: str
    relationships: Dict[str, Any] = field(default_factory=dict)


def extract_spatial_features(
    entities: List[SemanticEntity],
    feature_weights: Dict[str, float],
) -> np.ndarray:
    """
    Extracts weighted spatial features from entities for clustering.

    Args:
        entities: List of semantic entities.
        feature_weights: Dictionary with weights for 'centroid' and 'size'.

    Returns:
        A numpy array of feature vectors.
    """
    features = []
    if not entities:
        return np.array([])

    # Get frame dimensions from the first entity (assuming they are consistent)
    frame_w = entities[0].frame_width or 1.0
    frame_h = entities[0].frame_height or 1.0

    for entity in entities:
        if not entity.average_bbox or not entity.average_centroid:
            continue

        # Normalized centroid (x, y)
        norm_cx = entity.average_centroid[0] / frame_w
        norm_cy = entity.average_centroid[1] / frame_h

        # Normalized size (width, height)
        norm_w = entity.average_bbox.width / frame_w
        norm_h = entity.average_bbox.height / frame_h

        # Apply weights
        w_centroid = feature_weights.get('centroid', 1.0)
        w_size = feature_weights.get('size', 1.0)

        feature_vector = [
            norm_cx * w_centroid,
            norm_cy * w_centroid,
            norm_w * w_size,
            norm_h * w_size,
        ]
        features.append(feature_vector)

    return np.array(features)


def cluster_entities_hdbscan(
    features: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clusters entities using HDBSCAN.

    Args:
        features: The feature vectors to cluster.
        min_cluster_size: The minimum size of a cluster.
        min_samples: The number of samples in a neighborhood for a point to be a core point.

    Returns:
        A tuple containing cluster labels and probabilities.
    """
    if features.shape[0] < min_cluster_size:
        logger.warning(f"Not enough features ({features.shape[0]}) for HDBSCAN, skipping.")
        return np.array([-1] * features.shape[0]), np.array([0.0] * features.shape[0])

    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            prediction_data=True,
        )
        clusterer.fit(features)
        return clusterer.labels_, clusterer.probabilities_
    except ImportError:
        logger.error("hdbscan not installed. Please run 'pip install hdbscan'. Clustering disabled.")
        return np.array([-1] * features.shape[0]), np.array([0.0] * features.shape[0])


def label_zone(entity_classes: List[str], centroid: Tuple[float, float]) -> str:
    """
    Generates a descriptive label for a spatial zone.

    Args:
        entity_classes: A list of class labels for entities in the zone.
        centroid: The (x, y) centroid of the zone.

    Returns:
        A descriptive string label.
    """
    from collections import Counter
    if not entity_classes:
        return "unlabeled_zone"

    class_counts = Counter(entity_classes)
    # Get the most common class, handling ties by taking the first one
    dominant_class = class_counts.most_common(1)[0][0]

    # Simple positional labeling
    pos_x = "center"
    if centroid[0] < 0.33: pos_x = "left"
    elif centroid[0] > 0.66: pos_x = "right"

    return f"{pos_x}_{dominant_class}_group"


def compute_zone_centroid(entities: List[SemanticEntity]) -> Tuple[float, float]:
    """Computes the average centroid of a list of entities."""
    if not entities:
        return (0.0, 0.0)
    
    centroids = [e.average_centroid for e in entities if e.average_centroid]
    if not centroids:
        return (0.0, 0.0)

    avg_x = np.mean([c[0] for c in centroids])
    avg_y = np.mean([c[1] for c in centroids])
    return (float(avg_x), float(avg_y))


def compute_zone_bbox(entities: List[SemanticEntity]) -> BoundingBox:
    """Computes the encompassing bounding box for a list of entities."""
    if not entities:
        return BoundingBox(0, 0, 0, 0)

    bboxes = [e.average_bbox for e in entities if e.average_bbox]
    if not bboxes:
        return BoundingBox(0, 0, 0, 0)

    x1 = min(b.x1 for b in bboxes)
    y1 = min(b.y1 for b in bboxes)
    x2 = max(b.x2 for b in bboxes)
    y2 = max(b.y2 for b in bboxes)
    return BoundingBox(x1, y1, x2, y2)


def compute_zone_relationships(zones: List[SpatialZone]):
    """
    Computes spatial relationships (e.g., 'left_of', 'above') between zones.
    This is a simplified example.
    """
    for i, zone_a in enumerate(zones):
        for j, zone_b in enumerate(zones):
            if i == j:
                continue

            # Horizontal relationship
            if zone_a.centroid[0] < zone_b.centroid[0]:
                zone_a.relationships[zone_b.zone_id] = "left_of"
            else:
                zone_a.relationships[zone_b.zone_id] = "right_of"

            # Vertical relationship
            if zone_a.centroid[1] < zone_b.centroid[1]:
                if 'left_of' in zone_a.relationships[zone_b.zone_id] or 'right_of' in zone_a.relationships[zone_b.zone_id]:
                    zone_a.relationships[zone_b.zone_id] += "_and_above"
                else:
                    zone_a.relationships[zone_b.zone_id] = "above"
            else:
                if 'left_of' in zone_a.relationships[zone_b.zone_id] or 'right_of' in zone_a.relationships[zone_b.zone_id]:
                    zone_a.relationships[zone_b.zone_id] += "_and_below"
                else:
                    zone_a.relationships[zone_b.zone_id] = "below"
