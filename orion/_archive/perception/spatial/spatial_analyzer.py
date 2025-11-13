"""
Spatial Analysis for Perception
================================

Provides spatial context for detected objects:
- Zone classification (ceiling, wall, floor)
- Horizontal positioning (left, center, right)
- Scene type inference from object distributions

Integrated into perception pipeline for richer entity context.

Author: Orion Research Team
Date: October 2025
"""

import logging
from typing import Dict, Iterable, List, Sequence, Tuple, Union

from orion.perception.types import BoundingBox, SpatialContext, ObjectClass

logger = logging.getLogger(__name__)


def calculate_spatial_zone(
    bbox: BoundingBox,
    frame_width: float = 1920.0,
    frame_height: float = 1080.0,
) -> SpatialContext:
    """
    Classify spatial zone from bounding box.
    
    Args:
        bbox: Bounding box of detected object
        frame_width: Video frame width
        frame_height: Video frame height
        
    Returns:
        SpatialContext with zone type and positioning
    """
    if frame_width <= 0 or frame_height <= 0:
        return SpatialContext(
            zone_type="unknown",
            confidence=0.0,
            x_position="unknown",
            y_position="unknown",
            reasoning=["Invalid frame dimensions"]
        )
    
    # Calculate centroid
    cx = (bbox.x1 + bbox.x2) / 2.0
    cy = (bbox.y1 + bbox.y2) / 2.0
    
    # Normalize to [0, 1]
    nx = cx / frame_width
    ny = cy / frame_height
    
    nx = max(0.0, min(1.0, nx))
    ny = max(0.0, min(1.0, ny))
    
    # Horizontal position
    horizontal = "left" if nx < 0.33 else ("right" if nx > 0.66 else "center")
    
    # Vertical zone classification
    if ny < 0.15:
        return SpatialContext(
            zone_type="ceiling",
            confidence=0.9,
            x_position=horizontal,
            y_position="top",
            reasoning=[f"Top {ny * 100:.1f}% of frame"]
        )
    elif ny < 0.35:
        return SpatialContext(
            zone_type="wall_upper",
            confidence=0.85,
            x_position=horizontal,
            y_position="top",
            reasoning=[f"Upper wall ({ny * 100:.1f}% from top)"]
        )
    elif ny < 0.65:
        return SpatialContext(
            zone_type="wall_middle",
            confidence=0.9,
            x_position=horizontal,
            y_position="middle",
            reasoning=[f"Mid-height ({ny * 100:.1f}%) - typical for hardware"]
        )
    elif ny < 0.75:
        return SpatialContext(
            zone_type="wall_lower",
            confidence=0.85,
            x_position=horizontal,
            y_position="middle",
            reasoning=[f"Lower wall ({ny * 100:.1f}%)"]
        )
    else:
        return SpatialContext(
            zone_type="floor",
            confidence=0.9,
            x_position=horizontal,
            y_position="bottom",
            reasoning=[f"Near floor ({ny * 100:.1f}%)"]
        )


def infer_scene_type(object_classes: Iterable[ObjectClass]) -> str:
    """
    Infer scene type from distribution of detected objects.
    
    Args:
        object_classes: Collection of detected object classes
        
    Returns:
        Scene type: 'kitchen', 'bedroom', 'office', 'living_room', or 'general'
    """
    class_counter: Dict[str, int] = {}
    
    for obj_class in object_classes:
        if not obj_class or not obj_class.name:
            continue
        key = obj_class.name.lower()
        class_counter[key] = class_counter.get(key, 0) + 1
    
    if not class_counter:
        return "general"
    
    def _score(required: Sequence[str]) -> int:
        return sum(1 for name in required if class_counter.get(name, 0) > 0)
    
    # Kitchen indicators
    if _score(["oven", "microwave", "refrigerator", "stove", "sink"]) >= 2:
        return "kitchen"
    
    # Bedroom indicators
    if _score(["bed", "nightstand", "dresser"]) >= 1:
        return "bedroom"
    
    # Office indicators
    if _score(["laptop", "keyboard", "mouse", "monitor", "desk"]) >= 2:
        return "office"
    
    # Living room indicators
    if _score(["couch", "tv", "sofa"]) >= 1:
        return "living_room"
    
    return "general"


def enrich_observation_with_spatial_context(
    observation,
    frame_width: float = 1920.0,
    frame_height: float = 1080.0,
):
    """
    Add spatial zone information to an observation.
    
    Args:
        observation: Observation to enrich
        frame_width: Video frame width
        frame_height: Video frame height
        
    Returns:
        Observation with spatial_zone populated
    """
    if observation.bounding_box:
        observation.spatial_zone = calculate_spatial_zone(
            observation.bounding_box,
            frame_width,
            frame_height
        )
    return observation
