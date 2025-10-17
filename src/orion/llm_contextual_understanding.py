"""
LLM-Enhanced Contextual Understanding

This module uses Gemma3 to provide deep reasoning about:
1. What objects truly are (not just COCO mapping)
2. Why we think they're that object
3. What actions are happening
4. The overall narrative

Author: Orion Research Team
Date: October 17, 2025
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict

from .contextual_understanding import (
    ExtendedObject,
    ContextualAction,
    ContextualUnderstandingEngine,
    ExtendedObjectTaxonomy,
)

logger = logging.getLogger(__name__)


class LLMContextualReasoning:
    """
    Uses LLM (Gemma3) to provide deep contextual reasoning.
    
    The LLM can:
    1. Understand what an object truly is based on full context
    2. Explain its reasoning (why it thinks knob → door_knob vs stove_knob)
    3. Detect complex actions with multi-step reasoning
    4. Generate narrative descriptions
    """
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.taxonomy = ExtendedObjectTaxonomy()
        
    def identify_object_with_reasoning(
        self,
        description: str,
        yolo_class: str,
        spatial_context: Dict[str, Any],
        temporal_context: Dict[str, Any],
        scene_context: Dict[str, Any],
    ) -> Tuple[str, float, List[str]]:
        """
        Use LLM to identify what an object truly is with full reasoning.
        
        Args:
            description: FastVLM description of object
            yolo_class: What YOLO thinks it is
            spatial_context: Where it is, what's nearby
            temporal_context: When it appears, how it moves
            scene_context: What room/scene it's in
            
        Returns:
            (object_type, confidence, reasoning_chain)
        """
        # Build rich prompt
        prompt = self._build_object_identification_prompt(
            description, yolo_class, spatial_context, temporal_context, scene_context
        )
        
        # Get LLM response
        try:
            response = self.model_manager.generate_with_ollama(prompt)
            
            # Parse response
            result = self._parse_object_identification_response(response)
            return result
            
        except Exception as e:
            logger.error(f"LLM object identification failed: {e}")
            return yolo_class, 0.5, ["LLM reasoning failed, using YOLO class"]
    
    def detect_action_with_reasoning(
        self,
        person_entity: Dict,
        target_object: Optional[Dict],
        spatial_sequence: List[Dict],
        scene_type: str,
    ) -> Optional[Tuple[str, float, List[str]]]:
        """
        Use LLM to detect what action is happening with reasoning.
        
        Args:
            person_entity: Person performing action
            target_object: Object being acted upon
            spatial_sequence: Sequence of spatial states over time
            scene_type: Type of room/scene
            
        Returns:
            (action_type, confidence, reasoning_chain) or None
        """
        prompt = self._build_action_detection_prompt(
            person_entity, target_object, spatial_sequence, scene_type
        )
        
        try:
            response = self.model_manager.generate_with_ollama(prompt)
            result = self._parse_action_detection_response(response)
            return result
            
        except Exception as e:
            logger.error(f"LLM action detection failed: {e}")
            return None
    
    def generate_scene_narrative(
        self,
        extended_objects: List[ExtendedObject],
        detected_actions: List[ContextualAction],
        temporal_sequence: List[Dict],
    ) -> str:
        """
        Generate rich narrative description of entire scene.
        
        Returns natural language description like:
        "A person enters a bedroom by opening the door (identified by door knob 
        near wall). They approach a laptop on the desk and sit down. The room 
        contains a bed, suggesting this is a bedroom/workspace hybrid."
        """
        prompt = self._build_narrative_prompt(
            extended_objects, detected_actions, temporal_sequence
        )
        
        try:
            response = self.model_manager.generate_with_ollama(prompt)
            return response.strip()
            
        except Exception as e:
            logger.error(f"LLM narrative generation failed: {e}")
            return "Unable to generate scene narrative."
    
    def _build_object_identification_prompt(
        self,
        description: str,
        yolo_class: str,
        spatial_context: Dict,
        temporal_context: Dict,
        scene_context: Dict,
    ) -> str:
        """
        Build prompt for object identification.
        """
        # Get valid object types
        valid_types = self.taxonomy.get_all_object_types()
        
        prompt = f"""You are an expert at identifying objects in videos using full contextual understanding.

OBJECT TO IDENTIFY:
- YOLO detected class: "{yolo_class}"
- Visual description: "{description}"

SPATIAL CONTEXT:
- Location: {spatial_context.get('zone', 'unknown')}
- Nearby objects: {', '.join(spatial_context.get('nearby_objects', [])) or 'none'}
- Position: {spatial_context.get('position_description', 'unknown')}

TEMPORAL CONTEXT:
- Appears in frames: {temporal_context.get('frame_range', 'unknown')}
- Movement: {temporal_context.get('movement', 'static')}
- Interactions: {', '.join(temporal_context.get('interactions', [])) or 'none'}

SCENE CONTEXT:
- Room type: {scene_context.get('room_type', 'unknown')}
- Typical objects in scene: {', '.join(scene_context.get('typical_objects', [])) or 'none'}

VALID OBJECT TYPES (choose from these or use YOLO class if none fit):
{', '.join(valid_types)}

EXAMPLES:
1. "metallic cylindrical knob" near wall at mid-height in bedroom → door_knob (not stove_knob, not bottle)
2. "circular dial" on appliance front in kitchen → stove_knob (not door_knob)
3. "rectangular button" on wall in any room → light_switch

TASK:
1. Determine what this object TRULY is (not just COCO class)
2. Consider ALL context: visual + spatial + temporal + scene
3. Provide confidence score (0-1)
4. Explain your reasoning step by step

Respond in JSON format:
{{
    "object_type": "door_knob",
    "confidence": 0.85,
    "reasoning": [
        "Description mentions 'knob' which matches door hardware",
        "Located at mid-height on wall, typical for door knobs",
        "In bedroom context, door knobs are common architectural elements",
        "Nearby wall suggests architectural mounting, not appliance",
        "YOLO class 'hair drier' is clearly wrong given context"
    ],
    "alternative_interpretations": [
        {{"type": "drawer_handle", "confidence": 0.15, "reason": "Could be furniture hardware"}}
    ]
}}

Respond with ONLY the JSON, no other text.
"""
        return prompt
    
    def _build_action_detection_prompt(
        self,
        person_entity: Dict,
        target_object: Optional[Dict],
        spatial_sequence: List[Dict],
        scene_type: str,
    ) -> str:
        """
        Build prompt for action detection.
        """
        prompt = f"""You are an expert at detecting human actions in videos using spatial and temporal reasoning.

SCENE TYPE: {scene_type}

PERSON:
- Entity ID: {person_entity.get('entity_id', 'unknown')}
- Appears in frames: {person_entity.get('first_frame', 0)} to {person_entity.get('last_frame', 0)}

TARGET OBJECT (if any):
{json.dumps(target_object, indent=2) if target_object else "No specific target object"}

SPATIAL SEQUENCE (how person's position changes over time):
{json.dumps(spatial_sequence[:10], indent=2)}  # First 10 frames

COMMON ACTIONS:
- opening_door: Person approaches door knob, hand reaches for it, turning motion, door opens
- turning_on_light: Person approaches light switch, hand reaches up, pressing motion
- entering_room: Person moves through doorway, crosses threshold
- sitting_down: Person approaches chair/bed, lowering motion
- picking_up_object: Person reaches for object, grasping motion, lifting
- using_laptop: Person sits at desk, hands on keyboard/trackpad
- cooking: Person at stove, hand on knob, turning motion

TASK:
1. Analyze the spatial sequence to detect what action is happening
2. Consider scene context (kitchen vs bedroom affects interpretation)
3. Look for characteristic motion patterns
4. Provide confidence score
5. Explain reasoning step by step

Respond in JSON format:
{{
    "action_detected": true,
    "action_type": "opening_door",
    "confidence": 0.82,
    "reasoning": [
        "Person moves from frame 10 to 50, approaching left side of frame",
        "Spatial position suggests moving toward wall-mounted object",
        "Target object is door_knob based on context",
        "Sequence shows approach → reach → interaction pattern",
        "In bedroom context, door opening is common action"
    ],
    "frame_range": [10, 50],
    "key_moments": [
        {{"frame": 10, "event": "person_starts_approaching"}},
        {{"frame": 30, "event": "person_reaches_door_knob"}},
        {{"frame": 45, "event": "interaction_with_knob"}},
        {{"frame": 50, "event": "door_opening_motion"}}
    ]
}}

If NO action is clearly detected, respond:
{{
    "action_detected": false,
    "reasoning": ["Explain why no clear action was found"]
}}

Respond with ONLY the JSON, no other text.
"""
        return prompt
    
    def _build_narrative_prompt(
        self,
        extended_objects: List[ExtendedObject],
        detected_actions: List[ContextualAction],
        temporal_sequence: List[Dict],
    ) -> str:
        """
        Build prompt for narrative generation.
        """
        # Summarize objects
        obj_summary = []
        for obj in extended_objects[:10]:  # First 10 objects
            obj_summary.append({
                'type': obj.object_type,
                'description': obj.base_description[:100],  # Truncate
                'confidence': obj.confidence,
                'location': obj.spatial_zone,
            })
        
        # Summarize actions
        action_summary = []
        for action in detected_actions:
            action_summary.append({
                'type': action.action_type,
                'frames': action.frame_range,
                'confidence': action.confidence,
            })
        
        prompt = f"""You are an expert at generating natural, engaging narratives from video analysis.

OBJECTS DETECTED:
{json.dumps(obj_summary, indent=2)}

ACTIONS DETECTED:
{json.dumps(action_summary, indent=2)}

TEMPORAL SEQUENCE:
{json.dumps(temporal_sequence[:20], indent=2)}  # First 20 events

TASK:
Generate a natural language narrative that:
1. Describes what's happening in the video
2. Explains the scene and context
3. Mentions key objects and their roles
4. Describes actions in chronological order
5. Uses the TRUE object names (e.g., "door knob" not "hair drier")
6. Provides spatial context ("near the wall", "on the desk")
7. Is 3-5 sentences long

EXAMPLE OUTPUT:
"A person enters a bedroom by opening the door, identified by a metallic door knob mounted on the wall. They walk toward the center of the room where a laptop sits on a desk. The room contains a bed along the far wall, confirming this is a bedroom. The person sits down at the desk and begins using the laptop, suggesting they're starting work or study."

Generate the narrative now (3-5 sentences):
"""
        return prompt
    
    def _parse_object_identification_response(
        self, response: str
    ) -> Tuple[str, float, List[str]]:
        """
        Parse LLM response for object identification.
        """
        try:
            # Try to extract JSON from response
            if '```json' in response:
                json_start = response.index('```json') + 7
                json_end = response.index('```', json_start)
                response = response[json_start:json_end].strip()
            elif '```' in response:
                json_start = response.index('```') + 3
                json_end = response.index('```', json_start)
                response = response[json_start:json_end].strip()
            
            result = json.loads(response)
            
            object_type = result.get('object_type', 'unknown')
            confidence = result.get('confidence', 0.5)
            reasoning = result.get('reasoning', [])
            
            return object_type, confidence, reasoning
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Response was: {response[:200]}")
            return 'unknown', 0.3, ["Failed to parse LLM response"]
    
    def _parse_action_detection_response(
        self, response: str
    ) -> Optional[Tuple[str, float, List[str]]]:
        """
        Parse LLM response for action detection.
        """
        try:
            # Extract JSON
            if '```json' in response:
                json_start = response.index('```json') + 7
                json_end = response.index('```', json_start)
                response = response[json_start:json_end].strip()
            elif '```' in response:
                json_start = response.index('```') + 3
                json_end = response.index('```', json_start)
                response = response[json_start:json_end].strip()
            
            result = json.loads(response)
            
            if not result.get('action_detected', False):
                return None
            
            action_type = result.get('action_type', 'unknown')
            confidence = result.get('confidence', 0.5)
            reasoning = result.get('reasoning', [])
            
            return action_type, confidence, reasoning
            
        except Exception as e:
            logger.error(f"Failed to parse action detection response: {e}")
            return None


class EnhancedContextualUnderstandingEngine(ContextualUnderstandingEngine):
    """
    Enhanced version that uses LLM reasoning for deeper understanding.
    """
    
    def __init__(self, config, model_manager):
        super().__init__(config, model_manager)
        self.llm_reasoning = LLMContextualReasoning(model_manager)
    
    def _identify_extended_objects(
        self,
        entities: List[Dict],
        tracking_results: Dict,
    ) -> List[ExtendedObject]:
        """
        Override to use LLM reasoning for object identification.
        """
        extended_objects = []
        
        scene_type = self._infer_scene_type(entities)
        
        logger.info("🧠 Using LLM for deep contextual object identification...")
        
        for entity in entities:
            entity_id = entity.get('entity_id', '')
            description = entity.get('description', '')
            class_name = entity.get('class', 'unknown')
            bbox = entity.get('bbox', {})
            
            # Build context
            spatial_zone = self._calculate_spatial_zone(bbox)
            proximity_objects = self._find_proximity_objects(entity, entities)
            
            spatial_context = {
                'zone': spatial_zone,
                'nearby_objects': proximity_objects,
                'position_description': f"Located at {spatial_zone}",
            }
            
            temporal_context = {
                'frame_range': (entity.get('first_frame', 0), entity.get('last_frame', 0)),
                'movement': 'static',  # TODO: Detect movement
                'interactions': [],  # TODO: Detect interactions
            }
            
            scene_context = {
                'room_type': scene_type,
                'typical_objects': [e.get('class') for e in entities],
            }
            
            # Use LLM to identify object
            identified_type, confidence, reasoning = self.llm_reasoning.identify_object_with_reasoning(
                description=description,
                yolo_class=class_name,
                spatial_context=spatial_context,
                temporal_context=temporal_context,
                scene_context=scene_context,
            )
            
            ext_obj = ExtendedObject(
                object_id=entity_id,
                object_type=identified_type,
                base_description=description,
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
            if reasoning:
                logger.info(f"      Reasoning: {reasoning[0]}")
        
        return extended_objects
    
    def understand_scene(
        self,
        tracking_results: Dict[str, Any],
        corrected_entities: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Override to add LLM narrative generation.
        """
        # Call parent method
        results = super().understand_scene(tracking_results, corrected_entities)
        
        # Add LLM-generated narrative
        logger.info("📖 Generating narrative with LLM...")
        narrative_text = self.llm_reasoning.generate_scene_narrative(
            results['extended_objects'],
            results['detected_actions'],
            results['narrative'],
        )
        
        results['narrative_text'] = narrative_text
        logger.info(f"   Narrative: {narrative_text}")
        
        return results
