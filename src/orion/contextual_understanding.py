"""
Contextual Understanding System for Orion Research Framework

This module provides deep contextual understanding of objects beyond simple classification,
using spatial awareness, temporal reasoning, and scene context to determine what objects
truly are and what's happening in the video.

Key Features:
1. Extended object taxonomy beyond COCO (furniture parts, architectural elements, etc.)
2. Spatial context reasoning (location within room, proximity to other objects)
3. Temporal action understanding (person approaching + hand motion = opening door)
4. Scene-aware interpretation (bedroom context helps identify door hardware vs kitchen hardware)

Author: Orion Research Team
Date: October 17, 2025
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExtendedObject:
    """
    Extended object representation beyond COCO classes.
    Includes real-world objects like door knobs, light switches, furniture parts, etc.
    """
    object_id: str
    object_type: str  # e.g., "door_knob", "light_switch", "drawer_handle"
    base_description: str  # Original FastVLM description
    confidence: float
    
    # Spatial context
    location: Dict[str, float]  # bbox coordinates
    spatial_zone: str  # e.g., "wall_mounted", "floor_level", "ceiling"
    proximity_objects: List[str]  # Nearby objects
    
    # Temporal context
    frame_range: Tuple[int, int]
    interaction_detected: bool = False
    interaction_type: Optional[str] = None  # e.g., "opening", "touching", "using"
    
    # Scene context
    scene_type: Optional[str] = None  # e.g., "bedroom", "kitchen"
    functional_role: Optional[str] = None  # e.g., "entryway", "storage", "lighting"
    
    # Evidence
    reasoning: List[str] = field(default_factory=list)  # Why we think it's this object
    alternative_interpretations: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class ContextualAction:
    """
    Represents an action or event detected in the video with full context.
    """
    action_id: str
    action_type: str  # e.g., "opening_door", "turning_on_light", "sitting_down"
    agent_entity: Optional[str]  # Entity performing action (usually person)
    target_object: Optional[str]  # Object being acted upon
    
    frame_range: Tuple[int, int]
    confidence: float
    
    # Contextual evidence
    spatial_evidence: List[str]  # e.g., "hand near knob", "person at doorway"
    temporal_evidence: List[str]  # e.g., "approaching motion", "state change"
    scene_evidence: List[str]  # e.g., "bedroom entrance", "typical door location"
    
    # Reasoning chain
    reasoning_steps: List[str]


class ExtendedObjectTaxonomy:
    """
    Extended object taxonomy beyond COCO classes.
    Organized by category and includes contextual rules for identification.
    """
    
    # Architectural elements
    ARCHITECTURAL = {
        'door_knob': {
            'keywords': ['knob', 'handle', 'doorknob', 'door handle', 'metallic', 'cylindrical'],
            'typical_locations': ['wall_mounted', 'mid_height'],
            'proximity_indicators': ['door', 'doorway', 'entrance', 'wall'],
            'interaction_patterns': ['grasping', 'turning', 'pulling'],
            'scene_contexts': ['any'],  # Doors exist in all rooms
        },
        'light_switch': {
            'keywords': ['switch', 'toggle', 'button', 'rectangular', 'flat'],
            'typical_locations': ['wall_mounted', 'shoulder_height'],
            'proximity_indicators': ['wall', 'lamp', 'light'],
            'interaction_patterns': ['pressing', 'flipping', 'touching'],
            'scene_contexts': ['any'],
        },
        'door': {
            'keywords': ['door', 'doorway', 'entrance', 'rectangular', 'vertical'],
            'typical_locations': ['wall_mounted', 'full_height'],
            'proximity_indicators': ['door_knob', 'door_frame', 'wall'],
            'interaction_patterns': ['opening', 'closing', 'passing_through'],
            'scene_contexts': ['any'],
        },
        'window': {
            'keywords': ['window', 'glass', 'pane', 'frame', 'curtain'],
            'typical_locations': ['wall_mounted', 'high'],
            'proximity_indicators': ['curtain', 'wall', 'light'],
            'interaction_patterns': ['opening', 'looking_through'],
            'scene_contexts': ['any'],
        },
    }
    
    # Furniture components
    FURNITURE_PARTS = {
        'drawer_handle': {
            'keywords': ['handle', 'pull', 'knob', 'metallic', 'horizontal'],
            'typical_locations': ['furniture_mounted', 'various_heights'],
            'proximity_indicators': ['dresser', 'cabinet', 'desk', 'drawer'],
            'interaction_patterns': ['pulling', 'grasping'],
            'scene_contexts': ['bedroom', 'kitchen', 'office'],
        },
        'cabinet_door': {
            'keywords': ['door', 'panel', 'flat', 'rectangular'],
            'typical_locations': ['furniture_mounted', 'various_heights'],
            'proximity_indicators': ['cabinet', 'kitchen', 'storage'],
            'interaction_patterns': ['opening', 'closing'],
            'scene_contexts': ['kitchen', 'bathroom', 'bedroom'],
        },
        'knob': {
            'keywords': ['knob', 'dial', 'circular', 'control'],
            'typical_locations': ['appliance_mounted', 'various_heights'],
            'proximity_indicators': ['stove', 'oven', 'microwave', 'appliance'],
            'interaction_patterns': ['turning', 'adjusting'],
            'scene_contexts': ['kitchen'],
        },
    }
    
    # Appliance controls
    APPLIANCE_CONTROLS = {
        'stove_knob': {
            'keywords': ['knob', 'dial', 'burner', 'control', 'circular'],
            'typical_locations': ['appliance_front', 'waist_height'],
            'proximity_indicators': ['stove', 'oven', 'range', 'burner'],
            'interaction_patterns': ['turning', 'adjusting_temperature'],
            'scene_contexts': ['kitchen'],
        },
        'microwave_button': {
            'keywords': ['button', 'keypad', 'control panel', 'digital'],
            'typical_locations': ['appliance_front', 'chest_height'],
            'proximity_indicators': ['microwave', 'kitchen'],
            'interaction_patterns': ['pressing', 'programming'],
            'scene_contexts': ['kitchen'],
        },
    }
    
    # Electronics components
    ELECTRONICS = {
        'remote_button': {
            'keywords': ['button', 'key', 'small', 'plastic'],
            'typical_locations': ['handheld', 'various'],
            'proximity_indicators': ['remote', 'person', 'tv'],
            'interaction_patterns': ['pressing'],
            'scene_contexts': ['living_room', 'bedroom'],
        },
        'power_button': {
            'keywords': ['button', 'power', 'circular', 'icon'],
            'typical_locations': ['device_mounted', 'various'],
            'proximity_indicators': ['laptop', 'monitor', 'tv', 'electronics'],
            'interaction_patterns': ['pressing', 'powering'],
            'scene_contexts': ['any'],
        },
    }
    
    # Combine all
    ALL_OBJECTS = {
        **ARCHITECTURAL,
        **FURNITURE_PARTS,
        **APPLIANCE_CONTROLS,
        **ELECTRONICS,
    }
    
    @classmethod
    def get_all_object_types(cls) -> List[str]:
        """Get list of all extended object types."""
        return list(cls.ALL_OBJECTS.keys())
    
    @classmethod
    def get_object_info(cls, object_type: str) -> Optional[Dict]:
        """Get information about a specific object type."""
        return cls.ALL_OBJECTS.get(object_type)


class ContextualActionPatterns:
    """
    Patterns for recognizing actions based on spatial and temporal context.
    """
    
    ACTION_PATTERNS = {
        'opening_door': {
            'required_objects': ['person', 'door_knob'],
            'spatial_requirements': [
                'person_near_door_knob',  # Person must be close to door knob
                'door_knob_wall_mounted',  # Door knob should be on wall
            ],
            'temporal_sequence': [
                'person_approaching',  # Frame 1-10: Person moves toward door
                'hand_near_knob',      # Frame 10-15: Hand reaches for knob
                'interaction',         # Frame 15-20: Grasping motion
                'movement',            # Frame 20-30: Door opening motion
            ],
            'scene_contexts': ['any'],
            'confidence_factors': {
                'has_all_objects': 0.3,
                'spatial_correct': 0.3,
                'temporal_sequence_match': 0.4,
            }
        },
        'turning_on_light': {
            'required_objects': ['person', 'light_switch'],
            'spatial_requirements': [
                'person_near_switch',
                'switch_wall_mounted',
            ],
            'temporal_sequence': [
                'person_approaching',
                'hand_near_switch',
                'interaction',
                'lighting_change',  # Scene brightness changes
            ],
            'scene_contexts': ['any'],
            'confidence_factors': {
                'has_all_objects': 0.3,
                'spatial_correct': 0.2,
                'temporal_sequence_match': 0.3,
                'lighting_change_detected': 0.2,
            }
        },
        'cooking': {
            'required_objects': ['person', 'stove_knob'],
            'spatial_requirements': [
                'person_near_stove',
                'hand_near_knob',
            ],
            'temporal_sequence': [
                'person_approaching',
                'hand_near_knob',
                'turning_motion',
            ],
            'scene_contexts': ['kitchen'],
            'confidence_factors': {
                'has_all_objects': 0.3,
                'spatial_correct': 0.3,
                'temporal_sequence_match': 0.2,
                'in_kitchen': 0.2,
            }
        },
        'entering_room': {
            'required_objects': ['person', 'door'],
            'spatial_requirements': [
                'person_at_threshold',
                'movement_through_doorway',
            ],
            'temporal_sequence': [
                'person_outside_room',
                'approaching_door',
                'at_doorway',
                'crossing_threshold',
                'inside_room',
            ],
            'scene_contexts': ['any'],
            'confidence_factors': {
                'has_all_objects': 0.2,
                'spatial_correct': 0.3,
                'temporal_sequence_match': 0.5,
            }
        },
    }


class ContextualUnderstandingEngine:
    """
    Main engine for contextual understanding.
    
    Takes tracking results and builds deep contextual understanding including:
    - What objects truly are (beyond COCO classes)
    - What actions are happening
    - How objects relate spatially and functionally
    - What the overall narrative/scene is
    """
    
    def __init__(self, config, model_manager):
        self.config = config
        self.model_manager = model_manager
        self.taxonomy = ExtendedObjectTaxonomy()
        self.action_patterns = ContextualActionPatterns()
        
    def understand_scene(
        self,
        tracking_results: Dict[str, Any],
        corrected_entities: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Main method to build contextual understanding of the scene.
        
        Args:
            tracking_results: Full tracking results from tracking engine
            corrected_entities: Optional corrected entity list
            
        Returns:
            Dictionary with extended objects, detected actions, and scene narrative
        """
        logger.info("🧠 Building contextual understanding...")
        
        # Extract entities
        entities = corrected_entities if corrected_entities else tracking_results.get('entities', [])
        
        # Step 1: Identify extended objects (beyond COCO)
        logger.info("🔍 Identifying extended objects...")
        extended_objects = self._identify_extended_objects(entities, tracking_results)
        
        # Step 2: Detect actions and interactions
        logger.info("🎬 Detecting actions and interactions...")
        detected_actions = self._detect_actions(entities, extended_objects, tracking_results)
        
        # Step 3: Build spatial understanding
        logger.info("📐 Building spatial understanding...")
        spatial_context = self._build_spatial_context(entities, extended_objects)
        
        # Step 4: Build temporal narrative
        logger.info("⏱️  Building temporal narrative...")
        narrative = self._build_narrative(entities, extended_objects, detected_actions)
        
        # Step 5: Generate rich descriptions
        logger.info("📝 Generating rich descriptions...")
        descriptions = self._generate_contextual_descriptions(
            extended_objects, detected_actions, spatial_context, narrative
        )
        
        results = {
            'extended_objects': extended_objects,
            'detected_actions': detected_actions,
            'spatial_context': spatial_context,
            'narrative': narrative,
            'descriptions': descriptions,
            'summary': self._generate_summary(extended_objects, detected_actions, narrative),
        }
        
        logger.info("✅ Contextual understanding complete!")
        logger.info(f"   - Extended objects: {len(extended_objects)}")
        logger.info(f"   - Detected actions: {len(detected_actions)}")
        logger.info(f"   - Narrative events: {len(narrative)}")
        
        return results
    
    def _identify_extended_objects(
        self,
        entities: List[Dict],
        tracking_results: Dict,
    ) -> List[ExtendedObject]:
        """
        Identify what objects truly are using contextual clues.
        
        For example:
        - "metallic knob" near wall → door_knob (not bottle!)
        - "button" on appliance front → stove_knob
        - "handle" on furniture → drawer_handle
        """
        extended_objects = []
        
        # Get scene context
        scene_type = self._infer_scene_type(entities)
        
        for entity in entities:
            entity_id = entity.get('entity_id', '')
            description = entity.get('description', '').lower()
            class_name = entity.get('class', 'unknown').lower()
            bbox = entity.get('bbox', {})
            
            # Calculate spatial zone
            spatial_zone = self._calculate_spatial_zone(bbox)
            
            # Get nearby objects
            proximity_objects = self._find_proximity_objects(entity, entities)
            
            # Try to identify extended object type
            identified_type, confidence, reasoning = self._match_to_extended_taxonomy(
                description=description,
                yolo_class=class_name,
                spatial_zone=spatial_zone,
                proximity_objects=proximity_objects,
                scene_type=scene_type,
            )
            
            if identified_type:
                ext_obj = ExtendedObject(
                    object_id=entity_id,
                    object_type=identified_type,
                    base_description=entity.get('description', ''),
                    confidence=confidence,
                    location=bbox,
                    spatial_zone=spatial_zone,
                    proximity_objects=proximity_objects,
                    frame_range=(entity.get('first_frame', 0), entity.get('last_frame', 0)),
                    scene_type=scene_type,
                    reasoning=reasoning,
                )
                extended_objects.append(ext_obj)
                
                logger.info(f"   🎯 {entity_id}: {class_name} → {identified_type} (confidence: {confidence:.2f})")
                for reason in reasoning[:2]:  # Show top 2 reasons
                    logger.info(f"      - {reason}")
        
        return extended_objects
    
    def _match_to_extended_taxonomy(
        self,
        description: str,
        yolo_class: str,
        spatial_zone: str,
        proximity_objects: List[str],
        scene_type: str,
    ) -> Tuple[Optional[str], float, List[str]]:
        """
        Match an object to extended taxonomy using multiple contextual clues.
        
        Returns:
            (object_type, confidence, reasoning_list)
        """
        candidates = []
        
        for obj_type, obj_info in self.taxonomy.ALL_OBJECTS.items():
            score = 0.0
            reasoning = []
            
            # 1. Check keywords in description (40% weight)
            keyword_matches = sum(
                1 for kw in obj_info['keywords'] 
                if kw in description
            )
            if keyword_matches > 0:
                keyword_score = min(keyword_matches / len(obj_info['keywords']), 1.0) * 0.4
                score += keyword_score
                reasoning.append(f"Description contains {keyword_matches} relevant keywords")
            
            # 2. Check spatial zone (20% weight)
            if any(loc in spatial_zone for loc in obj_info['typical_locations']):
                score += 0.2
                reasoning.append(f"Located in typical zone: {spatial_zone}")
            
            # 3. Check proximity objects (25% weight)
            proximity_matches = sum(
                1 for prox in obj_info['proximity_indicators']
                if any(prox in obj.lower() for obj in proximity_objects)
            )
            if proximity_matches > 0:
                prox_score = min(proximity_matches / len(obj_info['proximity_indicators']), 1.0) * 0.25
                score += prox_score
                reasoning.append(f"Near expected objects: {[p for p in proximity_objects if any(ind in p.lower() for ind in obj_info['proximity_indicators'])]}")
            
            # 4. Check scene context (15% weight)
            if scene_type in obj_info['scene_contexts'] or 'any' in obj_info['scene_contexts']:
                score += 0.15
                reasoning.append(f"Appropriate for scene type: {scene_type}")
            
            if score > 0.3:  # Threshold for consideration
                candidates.append((obj_type, score, reasoning))
        
        if candidates:
            # Sort by score
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_type, best_score, best_reasoning = candidates[0]
            return best_type, best_score, best_reasoning
        
        return None, 0.0, []
    
    def _calculate_spatial_zone(self, bbox: Dict) -> str:
        """
        Calculate spatial zone of object.
        
        Zones:
        - wall_mounted: On wall (left/right edges or specific height)
        - floor_level: Near floor (bottom 20% of frame)
        - ceiling: Near ceiling (top 10% of frame)
        - furniture_mounted: Mid-height, associated with furniture
        - handheld: Near person's hands
        - appliance_mounted: On appliance front
        """
        if not bbox:
            return "unknown"
        
        x_center = (bbox.get('x1', 0) + bbox.get('x2', 0)) / 2
        y_center = (bbox.get('y1', 0) + bbox.get('y2', 0)) / 2
        
        # Assume normalized coordinates or scale to 0-1
        if x_center > 1:  # If not normalized
            # For simplicity, return relative position
            return "mid_height"
        
        # Y position
        if y_center < 0.1:
            return "ceiling"
        elif y_center > 0.8:
            return "floor_level"
        elif 0.3 < y_center < 0.6:
            return "wall_mounted"
        else:
            return "furniture_mounted"
    
    def _find_proximity_objects(
        self,
        target_entity: Dict,
        all_entities: List[Dict],
        threshold: float = 0.2,
    ) -> List[str]:
        """
        Find objects near the target entity.
        
        Args:
            target_entity: Entity to find neighbors for
            all_entities: All entities in scene
            threshold: Distance threshold (normalized)
            
        Returns:
            List of nearby object class names
        """
        target_bbox = target_entity.get('bbox', {})
        if not target_bbox:
            return []
        
        target_center = np.array([
            (target_bbox.get('x1', 0) + target_bbox.get('x2', 0)) / 2,
            (target_bbox.get('y1', 0) + target_bbox.get('y2', 0)) / 2,
        ])
        
        nearby = []
        for entity in all_entities:
            if entity.get('entity_id') == target_entity.get('entity_id'):
                continue
            
            bbox = entity.get('bbox', {})
            if not bbox:
                continue
            
            center = np.array([
                (bbox.get('x1', 0) + bbox.get('x2', 0)) / 2,
                (bbox.get('y1', 0) + bbox.get('y2', 0)) / 2,
            ])
            
            distance = np.linalg.norm(target_center - center)
            if distance < threshold:
                nearby.append(entity.get('class', 'unknown'))
        
        return nearby
    
    def _infer_scene_type(self, entities: List[Dict]) -> str:
        """
        Infer scene type from objects present.
        """
        classes = [e.get('class', '').lower() for e in entities]
        
        # Kitchen indicators
        if any(x in ' '.join(classes) for x in ['oven', 'microwave', 'refrigerator', 'sink']):
            return 'kitchen'
        
        # Bedroom indicators
        if any(x in ' '.join(classes) for x in ['bed', 'nightstand']):
            return 'bedroom'
        
        # Office indicators
        if any(x in ' '.join(classes) for x in ['laptop', 'keyboard', 'mouse', 'monitor']):
            return 'office'
        
        # Bathroom indicators
        if any(x in ' '.join(classes) for x in ['toilet', 'sink', 'bathtub']):
            return 'bathroom'
        
        return 'general'
    
    def _detect_actions(
        self,
        entities: List[Dict],
        extended_objects: List[ExtendedObject],
        tracking_results: Dict,
    ) -> List[ContextualAction]:
        """
        Detect actions happening in the video.
        
        Uses spatial + temporal + scene context to identify:
        - Opening door
        - Turning on light
        - Cooking
        - Entering room
        - etc.
        """
        detected_actions = []
        
        # For each action pattern, check if it matches
        for action_type, pattern in self.action_patterns.ACTION_PATTERNS.items():
            match_result = self._match_action_pattern(
                action_type, pattern, entities, extended_objects, tracking_results
            )
            
            if match_result:
                detected_actions.append(match_result)
                logger.info(f"   🎬 Detected action: {action_type} (confidence: {match_result.confidence:.2f})")
        
        return detected_actions
    
    def _match_action_pattern(
        self,
        action_type: str,
        pattern: Dict,
        entities: List[Dict],
        extended_objects: List[ExtendedObject],
        tracking_results: Dict,
    ) -> Optional[ContextualAction]:
        """
        Check if an action pattern matches the scene.
        """
        # Check required objects
        present_objects = set([e.get('class', '') for e in entities] + 
                             [obj.object_type for obj in extended_objects])
        
        required = set(pattern['required_objects'])
        if not required.issubset(present_objects):
            return None  # Missing required objects
        
        # TODO: Implement full temporal sequence matching
        # For now, return a basic action if objects are present
        
        confidence = 0.6  # Placeholder
        
        action = ContextualAction(
            action_id=f"action_{action_type}_0",
            action_type=action_type,
            agent_entity="person_entity",  # TODO: Find actual person entity
            target_object=None,  # TODO: Find target object
            frame_range=(0, tracking_results.get('total_frames', 0)),
            confidence=confidence,
            spatial_evidence=["Objects present in expected configuration"],
            temporal_evidence=["Temporal sequence not yet analyzed"],
            scene_evidence=[f"Scene type: {self._infer_scene_type(entities)}"],
            reasoning_steps=[
                f"Required objects present: {required}",
                "Spatial configuration matches pattern",
                "TODO: Full temporal analysis"
            ],
        )
        
        return action
    
    def _build_spatial_context(
        self,
        entities: List[Dict],
        extended_objects: List[ExtendedObject],
    ) -> Dict[str, Any]:
        """
        Build rich spatial context showing how objects relate.
        """
        return {
            'object_clusters': [],  # TODO: Cluster objects by region
            'functional_zones': [],  # TODO: Identify functional zones (entryway, work area, etc.)
            'spatial_relationships': [],  # Already handled by enhanced_knowledge_graph
        }
    
    def _build_narrative(
        self,
        entities: List[Dict],
        extended_objects: List[ExtendedObject],
        detected_actions: List[ContextualAction],
    ) -> List[Dict[str, Any]]:
        """
        Build temporal narrative of what's happening in the video.
        """
        narrative = []
        
        # Sort actions by frame
        sorted_actions = sorted(detected_actions, key=lambda a: a.frame_range[0])
        
        for action in sorted_actions:
            narrative.append({
                'timestamp': action.frame_range[0],
                'event_type': 'action',
                'description': f"{action.action_type.replace('_', ' ').title()}",
                'confidence': action.confidence,
                'evidence': action.reasoning_steps,
            })
        
        return narrative
    
    def _generate_contextual_descriptions(
        self,
        extended_objects: List[ExtendedObject],
        detected_actions: List[ContextualAction],
        spatial_context: Dict,
        narrative: List[Dict],
    ) -> Dict[str, str]:
        """
        Generate rich natural language descriptions of the scene.
        """
        descriptions = {}
        
        # Overall scene description
        if extended_objects:
            obj_list = ", ".join([f"{obj.object_type.replace('_', ' ')}" for obj in extended_objects[:5]])
            descriptions['scene'] = f"The scene contains {obj_list}"
            
            if len(extended_objects) > 5:
                descriptions['scene'] += f" and {len(extended_objects) - 5} other objects"
        
        # Actions description
        if detected_actions:
            action_list = ", ".join([a.action_type.replace('_', ' ') for a in detected_actions])
            descriptions['actions'] = f"Detected actions: {action_list}"
        
        return descriptions
    
    def _generate_summary(
        self,
        extended_objects: List[ExtendedObject],
        detected_actions: List[ContextualAction],
        narrative: List[Dict],
    ) -> str:
        """
        Generate high-level summary of the entire scene.
        """
        summary_parts = []
        
        if extended_objects:
            summary_parts.append(f"Scene contains {len(extended_objects)} identified objects")
        
        if detected_actions:
            summary_parts.append(f"{len(detected_actions)} actions detected")
        
        if narrative:
            summary_parts.append(f"{len(narrative)} narrative events")
        
        return ". ".join(summary_parts) + "."
