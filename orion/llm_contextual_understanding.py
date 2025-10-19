"""
Lightweight Contextual Understanding Engine
==========================================

Deterministic, fast contextual understanding for debugging and demos.
- Builds extended object summaries with canonical labels
- Provides simple action hypotheses (optional)
- Generates a concise narrative and summary

Does not require Neo4j. LLM is optional and off by default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging

from .config import OrionConfig

logger = logging.getLogger(__name__)


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


class EnhancedContextualUnderstandingEngine:
    def __init__(self, config: Optional[OrionConfig] = None, model_manager: Optional[Any] = None):
        self.config = config or OrionConfig()
        self.model_manager = model_manager

    def understand_scene(
        self,
        tracking_results: Dict[str, Any],
        corrected_entities: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
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

        # Simple action inference heuristics (non-temporal)
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
