"""
Spatial Zone Manager
====================

Assigns entities to spatial zones based on their 3D position or semantic class.
"""

from typing import Dict
from orion.perception.types import PerceptionEntity

class ZoneManager:
    """
    Manages spatial zones and assigns entities to them.
    """
    
    def __init__(self):
        self.zones: Dict[int, str] = {
            0: "Unknown",
            1: "Workspace",
            2: "Living Area",
            3: "Kitchen",
            4: "Bedroom"
        }
        
        # Semantic mapping (heuristic)
        self.semantic_map = {
            "laptop": 1, "mouse": 1, "keyboard": 1, "monitor": 1,
            "couch": 2, "sofa": 2, "tv": 2, "remote": 2,
            "refrigerator": 3, "oven": 3, "microwave": 3, "sink": 3,
            "bed": 4, "pillow": 4, "lamp": 4
        }

    def assign_zone(self, entity: PerceptionEntity) -> int:
        """
        Assign a zone ID to an entity.
        
        Strategy:
        1. If 3D position is available, use geometric zones (TODO).
        2. Fallback to semantic class mapping.
        3. Default to 0.
        """
        # TODO: Implement geometric zones if we have a map
        
        # Semantic fallback
        cls = entity.object_class.value if hasattr(entity.object_class, 'value') else str(entity.object_class)
        return self.assign_zone_from_class(cls)

    def assign_zone_from_class(self, class_name: str) -> int:
        """Lightweight helper when only a class label is available."""
        return self.semantic_map.get(class_name.lower(), 0)

    def get_zone_name(self, zone_id: int) -> str:
        return self.zones.get(zone_id, "Unknown")
