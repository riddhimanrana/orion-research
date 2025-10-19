"""
Contextual Understanding Engine
================================

Provides contextual understanding for detected objects including:
- Spatial zone detection (wall, floor, ceiling positions)
- Object classification correction (fix YOLO errors)
- Scene type inference (bedroom, kitchen, office, etc.)
- Batch LLM processing for efficiency

This engine processes perception data and adds rich context before
building the knowledge graph.

Author: Orion Research Team
Date: October 2025
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import json

from .config import OrionConfig
from .class_correction import ClassCorrector
from .spatial_utils import SpatialZone, calculate_spatial_zone_from_bbox, infer_scene_type

logger = logging.getLogger(__name__)
class ContextualEngine:
    """
    Contextual understanding engine with optimized processing.
    
    Key features:
    - Batch LLM calls by frame (not per-object)
    - Smart filtering (skip obvious cases)
    - Fixed spatial zone calculation
    - Evidence-based scene inference
    """
    
    def __init__(self, config=None, model_manager=None):
        self.config = config or OrionConfig()
        self.model_manager = model_manager
        self.corrector = ClassCorrector(config=self.config, model_manager=model_manager)
        self.scene_cache = {}  # Cache scene analysis
        self.stats = {
            'objects_processed': 0,
            'llm_calls': 0,
            'cache_hits': 0,
            'corrections': 0,
            'spatial_zones_detected': 0,
        }
    
    def process(self, perception_log: List[Dict], progress_callback=None, **kwargs) -> List[Dict]:
        """
        Main processing method.
        
        Args:
            perception_log: List of detected objects from perception engine
            progress_callback: Optional callback for progress updates
            
        Returns:
            Enhanced perception log with spatial zones, corrections, etc.
        """
        logger.info(f"Processing {len(perception_log)} objects...")
        
        if progress_callback:
            progress_callback("contextual.start", {"total": len(perception_log)})
        
        self.stats = {
            'objects_processed': 0,
            'llm_calls': 0,
            'cache_hits': 0,
            'corrections': 0,
            'spatial_zones_detected': 0,
        }

        if not perception_log:
            return []
        
        # Convert to entity format
        entities = self._convert_to_entities(perception_log)
        
        # Fast heuristic processing (no LLM)
        if progress_callback:
            progress_callback("contextual.heuristics", {
                "message": "Computing spatial zones & scene type"
            })
        entities = self._apply_heuristics(entities)
        
        # Batch LLM analysis for ambiguous cases
        entities_needing_llm = [e for e in entities if e.get('needs_llm')]
        if entities_needing_llm:
            if progress_callback:
                progress_callback("contextual.llm", {
                    "total": len(entities_needing_llm),
                    "message": f"Correcting {len(entities_needing_llm)} classifications"
                })
            self._batch_llm_analysis(entities_needing_llm)
        
        # Convert back to perception log format
        enhanced_log = self._convert_to_perception_log(entities, perception_log)
        
        if progress_callback:
            progress_callback("contextual.complete", {
                "total": len(enhanced_log),
                "spatial_zones": self.stats['spatial_zones_detected'],
                "corrections": self.stats['corrections'],
                "llm_calls": self.stats['llm_calls'],
                "message": f"✓ {self.stats['spatial_zones_detected']}/{len(entities)} spatial zones, {self.stats['corrections']} corrections"
            })
        
        logger.info(f"✓ {self.stats['spatial_zones_detected']}/{len(entities)} spatial zones detected")
        logger.info(f"✓ {self.stats['corrections']} classifications corrected")
        logger.info(f"✓ {self.stats['llm_calls']} LLM calls (vs {len(entities)} objects)")
        
        return enhanced_log
    
    def _convert_to_entities(self, perception_log: List[Dict]) -> List[Dict]:
        """Convert perception objects to entity format"""
        entities = []
        for i, obj in enumerate(perception_log):
            entity = {
                'entity_id': obj.get('temp_id') or f"entity_{i:06d}",
                'class': obj.get('object_class', 'unknown'),
                'description': obj.get('rich_description', ''),
                'confidence': obj.get('detection_confidence', 0.0),
                'bbox': self._normalize_bbox(obj.get('bounding_box', [])),
                'frame': obj.get('frame_number', 0),
                'timestamp': obj.get('timestamp', 0.0),
                '_original': obj,
            }
            entities.append(entity)
        return entities
    
    def _normalize_bbox(self, bbox: Any) -> Dict:
        """Normalize bbox to dict format"""
        if isinstance(bbox, dict):
            return bbox
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return {
                'x1': float(bbox[0]), 'y1': float(bbox[1]),
                'x2': float(bbox[2]), 'y2': float(bbox[3]),
                'frame_width': 1920, 'frame_height': 1080
            }
        return {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
    
    def _apply_heuristics(self, entities: List[Dict]) -> List[Dict]:
        """Apply fast heuristics without LLM"""
        self.stats['objects_processed'] = len(entities)
        
        # Infer scene type using shared heuristics
        scene_type = infer_scene_type(
            entity.get('class', '') for entity in entities if entity.get('class')
        )
        
        for entity in entities:
            # Calculate spatial zone
            zone = calculate_spatial_zone_from_bbox(entity['bbox'])
            entity['spatial_zone'] = zone.zone_type
            entity['spatial_zone_confidence'] = zone.confidence
            entity['spatial_reasoning'] = zone.reasoning
            entity['x_position'] = zone.x_position
            entity['y_position'] = zone.y_position
            entity['scene_type'] = scene_type
            
            if zone.zone_type != 'unknown':
                self.stats['spatial_zones_detected'] += 1
            
        self._apply_class_corrections(entities)

        for entity in entities:
            force_llm = bool(entity.pop('_force_llm', False))
            if entity.get('corrected_class'):
                entity['needs_llm'] = False
            else:
                entity['needs_llm'] = force_llm or self._needs_llm_analysis(entity)
        
        return entities

    def _apply_class_corrections(self, entities: List[Dict]) -> None:
        """Run deterministic class correction heuristics."""
        if not self.corrector:
            return

        enable_canonical = bool(getattr(self.config.correction, "enable_canonical_labels", True))

        for entity in entities:
            original_class = entity.get('class', '')
            description = entity.get('description', '')
            confidence = float(entity.get('confidence', 0.0))

            if not original_class or not description:
                continue

            try:
                needs_correction = self.corrector.should_correct(
                    original_class,
                    description,
                    confidence,
                    clip_verified=False,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Class correction heuristic failed: %s", exc)
                continue

            canonical_label: Optional[str] = None
            if enable_canonical:
                canonical_label = self.corrector._canonical_from_description(description, original_class)  # noqa: SLF001
                if canonical_label and not entity.get('canonical_label'):
                    entity['canonical_label'] = canonical_label

            if needs_correction:
                corrected_class, corr_conf = self.corrector.extract_corrected_class(
                    original_class,
                    description,
                    use_llm=False,
                )

                if corrected_class and corrected_class != original_class:
                    entity['corrected_class'] = corrected_class
                    entity['correction_confidence'] = corr_conf
                    entity['correction_reason'] = 'rule_based'
                    entity['original_class'] = original_class
                    if canonical_label and not entity.get('canonical_label'):
                        entity['canonical_label'] = canonical_label
                    self.stats['corrections'] += 1
                else:
                    # Flag for LLM follow-up when heuristics cannot finalize
                    entity['_force_llm'] = True
            elif canonical_label and canonical_label != original_class:
                # Canonical label differs from YOLO class: ask LLM to reconcile mapping
                entity['_force_llm'] = True
    
    
    
    def _needs_llm_analysis(self, entity: Dict) -> bool:
        """Determine if entity needs expensive LLM analysis"""
        yolo_class = (entity.get('class') or '').lower()
        description = (entity.get('description') or '').lower()
        confidence = float(entity.get('confidence', 0.0))

        if not yolo_class:
            return False

        if entity.get('corrected_class'):
            return False
        
        # High confidence + matching description = skip
        if confidence > 0.7 and yolo_class in description:
            return False

        # High confidence with synonym mention
        if confidence > 0.7 and self.corrector:
            synonym_map = {
                'tv': ['monitor', 'screen', 'display'],
                'laptop': ['computer', 'notebook'],
                'cell phone': ['phone', 'smartphone'],
            }
            for syn in synonym_map.get(yolo_class, []):
                if syn in description:
                    return False
        
        # Unambiguous objects
        clear_objects = {
            'person', 'chair', 'laptop', 'book', 'cup', 'bottle',
            'bed', 'couch', 'keyboard', 'clock'
        }
        if yolo_class in clear_objects and confidence > 0.6:
            return False
        
        # Known problematic YOLO classes
        problematic = {'hair drier', 'cell phone', 'remote', 'potted plant'}
        if self.corrector:
            problematic.update(cls.lower() for cls in self.corrector.COMMON_CORRECTIONS.keys())
        if yolo_class in problematic:
            return True
        
        # Low confidence or mismatch
        if confidence < 0.5 or (yolo_class not in description and confidence < 0.7):
            return True
        
        return False
    
    def _batch_llm_analysis(self, entities: List[Dict]):
        """Batch LLM analysis by frame"""
        if not self.model_manager:
            return
        
        # Group by frame
        frame_groups = defaultdict(list)
        for entity in entities:
            frame_groups[entity['frame']].append(entity)
        
        logger.info(f"Batching {len(entities)} entities into {len(frame_groups)} frames")
        
        # Process each frame
        for frame_num, frame_entities in frame_groups.items():
            cache_key = f"frame_{frame_num}"
            
            if cache_key in self.scene_cache:
                self.stats['cache_hits'] += 1
                continue
            
            try:
                result = self._call_llm_batch(frame_entities)
                self.stats['llm_calls'] += 1
                
                # Apply results
                for i, entity in enumerate(frame_entities):
                    if i < len(result.get('objects', [])):
                        obj_result = result['objects'][i]
                        if obj_result.get('corrected_class'):
                            entity['corrected_class'] = obj_result['corrected_class']
                            entity['correction_confidence'] = obj_result.get('confidence', 0.5)
                            entity['correction_reason'] = obj_result.get('reason', '')
                            self.stats['corrections'] += 1
                
                self.scene_cache[cache_key] = result
            except Exception as e:
                logger.error(f"Batch LLM failed for frame {frame_num}: {e}")
    
    def _call_llm_batch(self, entities: List[Dict]) -> Dict:
        """Call LLM for batch of objects"""
        objects_summary = [
            {
                'index': i,
                'yolo_class': e['class'],
                'description': e['description'][:200],
                'confidence': e['confidence'],
                'spatial_zone': e.get('spatial_zone', 'unknown'),
            }
            for i, e in enumerate(entities)
        ]
        
        model_manager = self.model_manager
        if model_manager is None:
            return {'objects': [{'index': i, 'needs_correction': False} for i in range(len(objects_summary))]}

        prompt = f"""Analyze these {len(objects_summary)} objects and correct misclassifications.

OBJECTS:
{json.dumps(objects_summary, indent=2)}

Common issues:
- "hair drier" often wrong for knobs/handles
- "cell phone" often wrong for remotes
- Consider spatial context (wall_middle = door hardware)

Respond with JSON:
{{
    "objects": [
        {{"index": 0, "needs_correction": true/false, "corrected_class": "door_knob", "confidence": 0.85, "reason": "..."}},
        ...
    ]
}}

JSON only:"""
        
        try:
            response = model_manager.generate_with_ollama(prompt)
            
            # Parse JSON
            if '```json' in response:
                start = response.index('```json') + 7
                end = response.index('```', start)
                response = response[start:end].strip()
            elif '```' in response:
                start = response.index('```') + 3
                end = response.index('```', start)
                response = response[start:end].strip()
            
            return json.loads(response)
        except Exception as e:
            logger.error(f"LLM parse error: {e}")
            return {'objects': [{'index': i, 'needs_correction': False} for i in range(len(entities))]}
    
    def _convert_to_perception_log(self, entities: List[Dict], original_log: List[Dict]) -> List[Dict]:
        """Convert entities back to perception log format"""
        entity_map = {e['entity_id']: e for e in entities}
        enhanced_log = []
        
        for orig_obj in original_log:
            entity_id = orig_obj.get('temp_id') or orig_obj.get('entity_id')
            entity = entity_map.get(entity_id)
            
            if not entity:
                enhanced_log.append(orig_obj.copy())
                continue
            
            enhanced_obj = orig_obj.copy()
            
            # Add spatial info
            enhanced_obj['spatial_zone'] = entity.get('spatial_zone', 'unknown')
            enhanced_obj['spatial_zone_confidence'] = entity.get('spatial_zone_confidence', 0.0)
            enhanced_obj['spatial_reasoning'] = entity.get('spatial_reasoning', [])
            enhanced_obj['x_position'] = entity.get('x_position', 'unknown')
            enhanced_obj['y_position'] = entity.get('y_position', 'unknown')
            enhanced_obj['scene_type'] = entity.get('scene_type', 'general')
            
            # Add corrections if any
            if entity.get('corrected_class'):
                enhanced_obj['original_class'] = enhanced_obj.get('object_class')
                enhanced_obj['object_class'] = entity['corrected_class']
                enhanced_obj['correction_confidence'] = entity.get('correction_confidence', 0.0)
                enhanced_obj['correction_reason'] = entity.get('correction_reason', '')
                enhanced_obj['was_corrected'] = True
            else:
                enhanced_obj['was_corrected'] = False
            
            if entity.get('canonical_label'):
                enhanced_obj['canonical_label'] = entity['canonical_label']

            # Ensure entity_id is set
            if not enhanced_obj.get('entity_id'):
                enhanced_obj['entity_id'] = entity['entity_id']
            
            enhanced_log.append(enhanced_obj)
        
        return enhanced_log


# Backwards compatibility
def apply_contextual_understanding(perception_log: List[Dict], model_manager, config=None, progress_callback=None) -> List[Dict]:
    """Apply contextual understanding to perception log"""
    engine = ContextualEngine(config, model_manager)
    return engine.process(perception_log, progress_callback=progress_callback)
