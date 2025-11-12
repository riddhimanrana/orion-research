"""
Spatial Natural Language Generation for ORION

Converts technical spatial data into natural, conversational descriptions.
Transforms: "zone 0, 3.4m, bbox [x,y,w,h]" → "on the coffee table near the couch"

Key Features:
- Support surface detection (ON relationships)
- Proximity-based descriptions (NEAR relationships)
- Room-aware location naming
- Temporal context ("last seen", "currently")
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from orion.graph.spatial_memory import SpatialEntity


@dataclass
class SpatialRelationship:
    """Describes spatial relationship between entities"""
    subject_id: int
    relation_type: str  # "ON", "NEAR", "ABOVE", "BELOW", "IN"
    object_id: int
    confidence: float
    distance: float  # meters


class SpatialNLG:
    """
    Natural Language Generation for spatial descriptions
    
    Converts raw spatial data into friendly, contextual descriptions
    that feel natural for a helpful AI assistant.
    """
    
    # Distance thresholds for proximity
    VERY_CLOSE = 0.5  # meters
    CLOSE = 1.5
    NEAR = 3.0
    FAR = 6.0
    
    def __init__(self):
        """Initialize spatial NLG"""
        # Landmark objects that are good spatial references
        self.landmark_objects = {
            'couch', 'sofa', 'table', 'desk', 'bed', 'counter',
            'chair', 'cabinet', 'shelf', 'tv', 'door', 'window'
        }
        
        # Support surfaces (things objects can be ON)
        self.support_surfaces = {
            'table', 'desk', 'counter', 'shelf', 'floor',
            'couch', 'chair', 'bed', 'cabinet'
        }
    
    def describe_location(
        self,
        entity: SpatialEntity,
        nearby_entities: List[SpatialEntity],
        room_context: Optional[str] = None
    ) -> str:
        """
        Generate natural location description
        
        Args:
            entity: Entity to describe
            nearby_entities: Other entities in the scene
            room_context: Room type (e.g., "living room")
            
        Returns:
            Natural language description like "on the coffee table near the couch"
        """
        # Strategy 1: Check for support surface (ON relationship)
        support = self._find_support_surface(entity, nearby_entities)
        if support:
            # "on the coffee table"
            base = f"on the {support.class_name}"
            
            # Add proximity context if relevant
            landmark = self._find_nearby_landmark(entity, nearby_entities, exclude=[support.entity_id])
            if landmark:
                proximity = self._get_proximity_description(entity, landmark)
                return f"{base} {proximity} the {landmark.class_name}"
            
            return base
        
        # Strategy 2: Near a prominent landmark
        landmark = self._find_nearby_landmark(entity, nearby_entities)
        if landmark:
            proximity = self._get_proximity_description(entity, landmark)
            return f"{proximity} the {landmark.class_name}"
        
        # Strategy 3: Room area with direction
        if room_context:
            direction = self._get_frame_direction(entity)
            if direction:
                return f"in the {direction} area of the {room_context}"
            return f"in the {room_context}"
        
        # Fallback: Frame position
        return self._get_frame_position_description(entity)
    
    def _find_support_surface(
        self,
        entity: SpatialEntity,
        nearby_entities: List[SpatialEntity]
    ) -> Optional[SpatialEntity]:
        """
        Detect if entity is ON a support surface
        
        Heuristic: Entity's bottom is close to surface's top, and there's overlap
        """
        entity_pos = self._get_3d_position(entity)
        if not entity_pos:
            return None
        
        entity_x, entity_y, entity_z = entity_pos
        
        for surface in nearby_entities:
            if surface.class_name not in self.support_surfaces:
                continue
            
            if surface.entity_id == entity.entity_id:
                continue
            
            surface_pos = self._get_3d_position(surface)
            if not surface_pos:
                continue
            
            surface_x, surface_y, surface_z = surface_pos
            
            # Check vertical alignment (entity above surface)
            height_diff = entity_y - surface_y
            if 0 < height_diff < 0.5:  # Entity slightly above surface
                # Check horizontal proximity
                horiz_dist = np.sqrt((entity_x - surface_x)**2 + (entity_z - surface_z)**2)
                if horiz_dist < 1.0:  # Close enough horizontally
                    return surface
        
        return None
    
    def _find_nearby_landmark(
        self,
        entity: SpatialEntity,
        nearby_entities: List[SpatialEntity],
        exclude: List[int] = None
    ) -> Optional[SpatialEntity]:
        """
        Find the most prominent nearby landmark for spatial reference
        
        Prioritizes:
        1. Landmark objects (couch, table, etc.)
        2. Tracked for longer duration (more stable)
        3. Higher detection confidence
        """
        exclude = exclude or []
        
        candidates = [
            e for e in nearby_entities
            if e.entity_id != entity.entity_id
            and e.entity_id not in exclude
            and e.class_name in self.landmark_objects
        ]
        
        if not candidates:
            # No landmarks, use any nearby object
            candidates = [e for e in nearby_entities if e.entity_id != entity.entity_id and e.entity_id not in exclude]
        
        if not candidates:
            return None
        
        # Score by proximity and stability
        scored = []
        entity_pos = self._get_3d_position(entity)
        
        for candidate in candidates:
            cand_pos = self._get_3d_position(candidate)
            if not entity_pos or not cand_pos:
                continue
            
            # Calculate distance
            dist = np.linalg.norm(np.array(entity_pos) - np.array(cand_pos))
            
            if dist < self.FAR:  # Within reasonable range
                # Score: closer is better, longer tracking is better
                tracking_score = len(candidate.observations) / 100.0
                distance_score = 1.0 / (1.0 + dist)
                total_score = distance_score + tracking_score
                
                scored.append((candidate, total_score, dist))
        
        if not scored:
            return None
        
        # Return closest high-scoring landmark
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]
    
    def _get_proximity_description(
        self,
        entity: SpatialEntity,
        reference: SpatialEntity
    ) -> str:
        """
        Get proximity descriptor based on distance
        
        Returns: "right next to", "near", "across from", etc.
        """
        entity_pos = self._get_3d_position(entity)
        ref_pos = self._get_3d_position(reference)
        
        if not entity_pos or not ref_pos:
            return "near"
        
        dist = np.linalg.norm(np.array(entity_pos) - np.array(ref_pos))
        
        if dist < self.VERY_CLOSE:
            return "right next to"
        elif dist < self.CLOSE:
            return "next to"
        elif dist < self.NEAR:
            return "near"
        else:
            return "across from"
    
    def _get_frame_direction(self, entity: SpatialEntity) -> Optional[str]:
        """
        Get frame-relative direction (left/right/center)
        
        Based on entity's average or last known 2D position
        This is a simplified heuristic - actual implementation would need bbox data
        """
        # Since we don't have bbox in SpatialEntity, we can't determine frame position
        # This would need to be added to tracking data
        # For now, return None to skip this
        return None
    
    def _get_frame_position_description(self, entity: SpatialEntity) -> str:
        """
        Fallback: describe position in frame
        
        Used when no better spatial reference is available
        """
        direction = self._get_frame_direction(entity)
        if direction:
            return f"in the {direction} area"
        return "in the scene"
    
    def _get_3d_position(self, entity: SpatialEntity) -> Optional[Tuple[float, float, float]]:
        """
        Get 3D position from entity
        
        Handles multiple possible attribute names for compatibility
        """
        # Try different attribute names
        for attr in ['avg_position_3d', 'last_known_position_3d', 'position_3d', 'centroid_3d']:
            if hasattr(entity, attr):
                pos = getattr(entity, attr)
                if pos is not None and len(pos) == 3:
                    return tuple(pos)
        
        # Try movement_history as fallback
        if hasattr(entity, 'movement_history') and entity.movement_history:
            # Get most recent position
            last_entry = entity.movement_history[-1]
            if isinstance(last_entry, tuple) and len(last_entry) >= 2:
                timestamp, pos = last_entry[0], last_entry[1]
                if pos and len(pos) == 3:
                    return tuple(pos)
        
        return None
    
    def describe_temporal_context(
        self,
        entity: SpatialEntity,
        current_time: float
    ) -> str:
        """
        Add temporal context to descriptions
        
        Returns: "currently", "last seen 5 minutes ago", "was here earlier"
        """
        # Use last_seen from SpatialEntity
        if not hasattr(entity, 'last_seen') or not entity.last_seen:
            return ""
        
        last_time = entity.last_seen
        time_diff = current_time - last_time
        
        if time_diff < 5:  # Within 5 seconds
            return "currently"
        elif time_diff < 60:  # Within 1 minute
            return f"{int(time_diff)} seconds ago"
        elif time_diff < 3600:  # Within 1 hour
            mins = int(time_diff / 60)
            return f"{mins} minute{'s' if mins > 1 else ''} ago"
        else:
            hours = int(time_diff / 3600)
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
    
    def generate_entity_description(
        self,
        entity: SpatialEntity,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate complete entity description with all context
        
        Args:
            entity: Entity to describe
            context: Dict with 'nearby_entities', 'room_type', 'current_time'
            
        Returns:
            Full natural description: 
            "a book on the coffee table near the couch in the living room"
        """
        parts = []
        
        # Article + class name
        article = "an" if entity.class_name[0] in "aeiou" else "a"
        parts.append(f"{article} {entity.class_name}")
        
        # Spatial location
        location = self.describe_location(
            entity,
            context.get('nearby_entities', []),
            context.get('room_type')
        )
        parts.append(location)
        
        # Room context (if not already included)
        room = context.get('room_type')
        if room and room not in location:
            parts.append(f"in the {room}")
        
        # Temporal context (if relevant)
        current_time = context.get('current_time')
        if current_time:
            temporal = self.describe_temporal_context(entity, current_time)
            if temporal and temporal != "currently":
                parts.append(f"(last seen {temporal})")
        
        return ' '.join(parts)
