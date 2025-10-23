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
from collections import OrderedDict
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
        self.scene_cache = {}  # Cache scene analysis per frame
        self._correction_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.stats = {
            'raw_observations': 0,
            'unique_entities': 0,
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
        raw_count = len(perception_log)
        logger.info("Processing %d observations...", raw_count)

        self.stats = {
            'raw_observations': raw_count,
            'unique_entities': 0,
            'llm_calls': 0,
            'cache_hits': 0,
            'corrections': 0,
            'spatial_zones_detected': 0,
        }

        if not perception_log:
            return []
        
        # Convert to entity format
        entities = self._convert_to_entities(perception_log)
        unique_count = len(entities)
        self.stats['unique_entities'] = unique_count

        logger.info("Consolidated to %d unique entities from %d observations", unique_count, raw_count)

        if progress_callback:
            progress_callback(
                "contextual.start",
                {
                    "total": raw_count,
                    "unique": unique_count,
                    "raw": raw_count,
                    "message": f"Processing {unique_count} unique entities ({raw_count} observations)",
                },
            )
        
        # Fast heuristic processing (no LLM)
        if progress_callback:
            progress_callback(
                "contextual.heuristics",
                {
                    "message": f"Computing spatial zones & scene type for {unique_count} unique entities",
                    "unique": unique_count,
                },
            )
        entities = self._apply_heuristics(entities)
        
        # Batch LLM analysis for ambiguous cases
        entities_needing_llm = [e for e in entities if e.get('needs_llm')]
        if entities_needing_llm:
            if progress_callback:
                progress_callback(
                    "contextual.llm",
                    {
                        "total": len(entities_needing_llm),
                        "unique": unique_count,
                        "raw": raw_count,
                        "message": (
                            f"LLM review for {len(entities_needing_llm)}/{unique_count} unique entities"
                        ),
                    },
                )
            self._batch_llm_analysis(entities_needing_llm, progress_callback=progress_callback)
        elif progress_callback:
            progress_callback(
                "contextual.llm",
                {
                    "total": 0,
                    "unique": unique_count,
                    "raw": raw_count,
                    "message": "LLM review not required",
                },
            )
        
        # Convert back to perception log format
        enhanced_log = self._convert_to_perception_log(entities, perception_log)
        
        if progress_callback:
            progress_callback(
                "contextual.complete",
                {
                    "total": len(enhanced_log),
                    "unique": unique_count,
                    "raw": raw_count,
                    "spatial_zones": self.stats['spatial_zones_detected'],
                    "corrections": self.stats['corrections'],
                    "llm_calls": self.stats['llm_calls'],
                    "message": (
                        f"✓ {self.stats['spatial_zones_detected']}/{unique_count} spatial zones, "
                        f"{self.stats['corrections']} corrections"
                    ),
                },
            )
        
        logger.info(
            "✓ %d/%d spatial zones detected",
            self.stats['spatial_zones_detected'],
            unique_count,
        )
        logger.info("✓ %d classifications corrected", self.stats['corrections'])
        logger.info(
            "✓ %d LLM calls (reviewed %d unique entities)",
            self.stats['llm_calls'],
            len(entities_needing_llm),
        )
        logger.info("✓ %d contextual cache hits", self.stats['cache_hits'])
        
        return enhanced_log
    
    def _convert_to_entities(self, perception_log: List[Dict]) -> List[Dict]:
        """Aggregate perception observations into unique entity profiles."""
        entity_map: Dict[str, Dict[str, Any]] = {}

        for index, obj in enumerate(perception_log):
            raw_id = (
                obj.get('entity_id')
                or obj.get('track_id')
                or obj.get('temp_id')
                or obj.get('object_id')
                or f"entity_{index:06d}"
            )
            entity_id = str(raw_id)

            normalized_bbox = self._normalize_bbox(obj.get('bounding_box', []))
            description = obj.get('rich_description') or ''
            confidence = float(obj.get('detection_confidence', 0.0))
            frame = int(obj.get('frame_number', 0))
            timestamp = float(obj.get('timestamp', 0.0))

            entry = entity_map.get(entity_id)
            if entry is None:
                entry = {
                    'entity_id': entity_id,
                    'class': obj.get('object_class', 'unknown'),
                    'description': description,
                    'description_samples': [description] if description else [],
                    'confidence': confidence,
                    'bbox': normalized_bbox,
                    'frame': frame,
                    'timestamp': timestamp,
                    'first_timestamp': timestamp,
                    'first_frame': frame,
                    'observations': [obj],
                    'observation_count': 1,
                    'frames': {frame},
                    'timestamps': [timestamp],
                    '_original': obj,
                    'needs_llm': False,
                    'scene_type': obj.get('scene_type', 'unknown') or 'unknown',
                    'spatial_zone': obj.get('spatial_zone', 'unknown') or 'unknown',
                    'spatial_zone_confidence': float(obj.get('spatial_zone_confidence', 0.0)),
                    'spatial_reasoning': obj.get('spatial_reasoning', []) or [],
                    'x_position': obj.get('x_position', 'unknown') or 'unknown',
                    'y_position': obj.get('y_position', 'unknown') or 'unknown',
                }
                entity_map[entity_id] = entry
            else:
                entry['observations'].append(obj)
                entry['observation_count'] += 1
                entry['frames'].add(frame)
                entry['timestamps'].append(timestamp)
                if description:
                    entry['description_samples'].append(description)
                    if not entry.get('description'):
                        entry['description'] = description
                # Prefer the highest confidence observation as canonical
                if confidence > entry['confidence']:
                    entry['confidence'] = confidence
                    entry['description'] = description or entry.get('description', '')
                    entry['bbox'] = normalized_bbox
                    if obj.get('object_class'):
                        entry['class'] = obj.get('object_class')
                    entry['frame'] = frame
                    entry['timestamp'] = timestamp
                    entry['_original'] = obj
                    entry['scene_type'] = obj.get('scene_type', entry.get('scene_type', 'unknown')) or entry.get('scene_type', 'unknown')
                    entry['spatial_zone'] = obj.get('spatial_zone', entry.get('spatial_zone', 'unknown')) or entry.get('spatial_zone', 'unknown')
                    entry['spatial_zone_confidence'] = float(obj.get('spatial_zone_confidence', entry.get('spatial_zone_confidence', 0.0)))
                    entry['spatial_reasoning'] = obj.get('spatial_reasoning', entry.get('spatial_reasoning', [])) or entry.get('spatial_reasoning', [])
                    entry['x_position'] = obj.get('x_position', entry.get('x_position', 'unknown')) or entry.get('x_position', 'unknown')
                    entry['y_position'] = obj.get('y_position', entry.get('y_position', 'unknown')) or entry.get('y_position', 'unknown')
                if timestamp < entry.get('first_timestamp', timestamp):
                    entry['first_timestamp'] = timestamp
                    entry['first_frame'] = frame

            entry['_cache_key'] = self._build_cache_key(entry)

        entities = sorted(
            entity_map.values(),
            key=lambda e: (e.get('first_timestamp', e.get('timestamp', 0.0)), e['entity_id']),
        )
        return entities

    def _build_cache_key(self, entity: Dict) -> str:
        """Create a stable cache key for an entity description."""
        parts = [
            (entity.get('class') or '').strip().lower(),
            (entity.get('description') or '').strip().lower(),
            (entity.get('scene_type') or '').strip().lower(),
            (entity.get('spatial_zone') or '').strip().lower(),
        ]
        return "|".join(parts)

    def _maybe_apply_cache(self, entity: Dict) -> bool:
        """Apply cached LLM correction if available."""
        if not getattr(self.config.correction, 'enable_llm_cache', True):
            return False

        cache_key = entity.get('_cache_key') or self._build_cache_key(entity)
        if not cache_key:
            return False

        cached = self._correction_cache.get(cache_key)
        if cached is None:
            return False

        # move to the end (LRU)
        self._correction_cache.move_to_end(cache_key)
        entity['needs_llm'] = False

        corrected_class = cached.get('corrected_class')
        if corrected_class:
            entity['corrected_class'] = corrected_class
            entity['correction_confidence'] = cached.get('correction_confidence', 0.5)
            entity['correction_reason'] = cached.get('correction_reason', 'cached')
            self.stats['corrections'] += 1

        self.stats['cache_hits'] += 1
        return True

    def _store_cache_entry(self, entity: Dict, result: Dict[str, Any]) -> None:
        if not getattr(self.config.correction, 'enable_llm_cache', True):
            return

        cache_key = entity.get('_cache_key') or self._build_cache_key(entity)
        if not cache_key:
            return

        payload = {
            'corrected_class': result.get('corrected_class'),
            'correction_confidence': result.get('confidence', 0.5),
            'correction_reason': result.get('reason', 'llm'),
        }

        self._correction_cache[cache_key] = payload
        self._correction_cache.move_to_end(cache_key)

        max_cache = getattr(self.config.correction, 'llm_cache_size', 256)
        while len(self._correction_cache) > max_cache > 0:
            self._correction_cache.popitem(last=False)

    def _apply_object_result(self, entity: Dict, obj_result: Dict[str, Any], source: str) -> None:
        corrected_class = obj_result.get('corrected_class')
        if corrected_class:
            entity['corrected_class'] = corrected_class
            entity['correction_confidence'] = obj_result.get('confidence', 0.5)
            entity['correction_reason'] = obj_result.get('reason', source)
            entity['needs_llm'] = False
            self.stats['corrections'] += 1
        else:
            entity['needs_llm'] = False

        self._store_cache_entry(entity, obj_result)

    def _apply_batch_result(self, frame_entities: List[Dict], result: Dict[str, Any], source: str) -> None:
        objects = result.get('objects', []) if isinstance(result, dict) else []
        for index, entity in enumerate(frame_entities):
            obj_result = objects[index] if index < len(objects) else {}
            self._apply_object_result(entity, obj_result, source=source)
    
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
        self.stats['unique_entities'] = len(entities)

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

            # Update cache key with enriched context
            entity['_cache_key'] = self._build_cache_key(entity)

        self._apply_class_corrections(entities)

        for entity in entities:
            entity['_cache_key'] = self._build_cache_key(entity)

            if self._maybe_apply_cache(entity):
                continue

            force_llm = bool(entity.pop('_force_llm', False))
            if entity.get('corrected_class'):
                entity['needs_llm'] = False
                continue

            if force_llm:
                entity['needs_llm'] = True
                continue

            entity['needs_llm'] = self._needs_llm_analysis(entity)

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

        conf_floor = getattr(self.config.correction, 'llm_confidence_floor', 0.45)
        conf_ceiling = getattr(self.config.correction, 'llm_confidence_ceiling', 0.65)
        high_confidence_threshold = getattr(self.config.detection, 'high_confidence_threshold', conf_ceiling)

        if not yolo_class:
            return False

        if entity.get('corrected_class'):
            return False
        
        # High confidence + matching description = skip
        if confidence >= conf_ceiling and yolo_class and yolo_class in description:
            return False

        # High confidence with synonym mention
        if confidence >= conf_ceiling and self.corrector:
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
        if yolo_class in clear_objects and confidence >= high_confidence_threshold:
            return False
        
        # Known problematic YOLO classes
        problematic = {'hair drier', 'cell phone', 'remote', 'potted plant'}
        if self.corrector:
            problematic.update(cls.lower() for cls in self.corrector.COMMON_CORRECTIONS.keys())
        if yolo_class in problematic:
            return True
        
        # Low confidence or mismatch
        if confidence <= conf_floor:
            return True

        if yolo_class not in description and confidence < conf_ceiling:
            return True
        
        return False
    
    def _batch_llm_analysis(self, entities: List[Dict], progress_callback=None):
        """Batch LLM analysis by frame"""
        if not self.model_manager or not entities:
            return
        
        # Group by frame
        frame_groups: Dict[int, List[Dict]] = {}
        for entity in entities:
            frame_groups.setdefault(entity['frame'], []).append(entity)
        
        logger.info(f"Batching {len(entities)} entities into {len(frame_groups)} frames")
        
        total_entities = len(entities)
        processed_entities = 0
        total_batches = len(frame_groups)

        # Process each frame
        for batch_index, (frame_num, frame_entities) in enumerate(sorted(frame_groups.items()), start=1):
            cache_key = f"frame_{frame_num}"
            
            if cache_key in self.scene_cache:
                cached_result = self.scene_cache[cache_key]
                self.stats['cache_hits'] += 1
                self._apply_batch_result(frame_entities, cached_result, source='cache_frame')
                processed_entities += len(frame_entities)
                if progress_callback:
                    progress_callback(
                        "contextual.llm.cache_hit",
                        {
                            "frame": frame_num,
                            "processed": processed_entities,
                            "total": total_entities,
                            "batch": batch_index,
                            "batches": total_batches,
                            "message": (
                                f"Frame {frame_num} served from cache "
                                f"({processed_entities}/{total_entities} unique entities)"
                            ),
                        },
                    )
                continue
            
            try:
                if progress_callback:
                    progress_callback(
                        "contextual.llm.batch",
                        {
                            "frame": frame_num,
                            "processed": processed_entities,
                            "total": total_entities,
                            "batch": batch_index,
                            "batches": total_batches,
                            "message": (
                                f"Batch {batch_index}/{total_batches}: "
                                f"sending {len(frame_entities)} unique entities to LLM"
                            ),
                        },
                    )

                result = self._call_llm_batch(frame_entities)
                self.stats['llm_calls'] += 1
                
                # Apply results
                self._apply_batch_result(frame_entities, result, source='llm')
                
                self.scene_cache[cache_key] = result
                processed_entities += len(frame_entities)

                if progress_callback:
                    progress_callback(
                        "contextual.llm.batch",
                        {
                            "frame": frame_num,
                            "processed": processed_entities,
                            "total": total_entities,
                            "batch": batch_index,
                            "batches": total_batches,
                            "message": (
                                f"Batch {batch_index}/{total_batches}: "
                                f"resolved {processed_entities}/{total_entities} unique entities"
                            ),
                        },
                    )
            except Exception as e:
                logger.error(f"Batch LLM failed for frame {frame_num}: {e}")
    
    def _call_llm_batch(self, entities: List[Dict]) -> Dict:
        """Call LLM for batch of objects"""
        objects_summary = []
        for i, entity in enumerate(entities):
            description = entity.get('description') or next(
                (sample for sample in entity.get('description_samples', []) if sample),
                '',
            )
            objects_summary.append(
                {
                    'index': i,
                    'yolo_class': entity.get('class', 'unknown'),
                    'description': description[:200],
                    'confidence': entity.get('confidence', 0.0),
                    'spatial_zone': entity.get('spatial_zone', 'unknown'),
                    'observations': entity.get('observation_count', 1),
                }
            )
        
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
                # Store original YOLO class if not already stored
                if not enhanced_obj.get('original_yolo_class'):
                    enhanced_obj['original_yolo_class'] = enhanced_obj.get('yolo_class', enhanced_obj.get('object_class'))
                enhanced_obj['original_class'] = enhanced_obj.get('object_class')
                # Update both object_class and yolo_class to corrected value
                enhanced_obj['object_class'] = entity['corrected_class']
                # Keep yolo_class showing the original detection for transparency
                # but object_class shows the corrected value
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


    def understand_scene(
        self,
        tracking_results: Dict[str, Any],
        corrected_entities: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Enhanced scene understanding with entity analysis.
        
        Used during pipeline for semantic enhancement stage.
        Provides extended objects, detected actions, narrative, and summary.
        
        Args:
            tracking_results: Results from tracking engine
            corrected_entities: Optional corrected entities list
            
        Returns:
            Dictionary with extended_objects, detected_actions, narrative_text, summary
        """
        from dataclasses import dataclass, field
        
        @dataclass
        class ExtendedObject:
            object_id: str
            object_type: str
            confidence: float
            spatial_zone: str = "unknown"
            proximity_objects: List[str] = field(default_factory=list)
            reasoning: List[str] = field(default_factory=list)

        @dataclass
        class DetectedAction:
            action_type: str
            confidence: float
            frame_range: Tuple[int, int] = (0, 0)
            reasoning_steps: List[str] = field(default_factory=list)
        
        entities = corrected_entities or tracking_results.get("entities", [])
        if not entities:
            return {
                "extended_objects": [],
                "detected_actions": [],
                "narrative_text": "No entities found.",
                "summary": "Empty scene",
            }

        extended: List[ExtendedObject] = []
        classes = []
        canonical_terms = []

        # Build extended objects list
        for e in entities:
            oid = e.get("id") or e.get("entity_id") or "unknown"
            cls = e.get("class", "unknown")
            canon = e.get("canonical_label")
            obj_type = canon or cls
            conf = float(e.get("correction_confidence") or e.get("confidence") or 0.6)
            zone = e.get("spatial_zone", "unknown")

            reasoning = []
            if canon:
                reasoning.append(f"Canonical label inferred from description: {canon}")
            desc = e.get("description")
            if desc:
                reasoning.append(f"Description: {desc[:80]}...")

            extended.append(ExtendedObject(
                object_id=oid,
                object_type=obj_type,
                confidence=conf,
                spatial_zone=zone,
                reasoning=reasoning,
            ))
            classes.append(cls)
            if canon:
                canonical_terms.append(canon)

        # Simple action inference heuristics
        actions: List[DetectedAction] = []
        has_person = any((e.get("class", "").lower() == "person") for e in entities)
        has_knob_like = any(((e.get("canonical_label") or "").lower() in ("knob", "handle")) for e in entities)
        has_door_like = any(e.get("class", "").lower() in ("refrigerator", "oven", "door") for e in entities)
        if self.config.correction.infer_events and has_person and has_knob_like and has_door_like:
            actions.append(DetectedAction(
                action_type="OPENS_DOOR",
                confidence=0.6,
                frame_range=(0, 0),
                reasoning_steps=[
                    "Person present",
                    "Knob/handle detected (canonical)",
                    "Door-like object present (refrigerator/oven/door)",
                ],
            ))

        # Narrative
        unique_classes = sorted(set(classes))
        unique_canon = sorted(set(canonical_terms))
        narrative_parts = []
        if unique_classes:
            narrative_parts.append(f"The scene contains: {', '.join(unique_classes[:8])}.")
        if unique_canon:
            narrative_parts.append(f"Fine-grained items detected: {', '.join(unique_canon[:6])}.")
        if actions:
            verb_list = ', '.join(a.action_type for a in actions)
            narrative_parts.append(f"Possible actions: {verb_list}.")
        narrative = " ".join(narrative_parts) or "A scene with detected objects."

        # Summary
        summary = f"Objects: {len(entities)}; Actions: {len(actions)}; Canonical terms: {', '.join(unique_canon[:5]) if unique_canon else 'none'}."

        return {
            "extended_objects": extended,
            "detected_actions": actions,
            "narrative_text": narrative,
            "summary": summary,
        }


# Backwards compatibility
def apply_contextual_understanding(perception_log: List[Dict], model_manager, config=None, progress_callback=None) -> List[Dict]:
    """Apply contextual understanding to perception log"""
    engine = ContextualEngine(config, model_manager)
    return engine.process(perception_log, progress_callback=progress_callback)