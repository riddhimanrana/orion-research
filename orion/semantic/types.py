"""Typed semantic structures shared across Orion phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple


@dataclass
class StateChange:
    """Summarizes how an entity evolved across a temporal window."""

    entity_id: str
    class_label: str
    frame_before: int
    frame_after: int
    timestamp_before: float
    timestamp_after: float
    centroid_before: Optional[Tuple[float, float]] = None
    centroid_after: Optional[Tuple[float, float]] = None
    centroid_3d_before: Optional[Tuple[float, float, float]] = None
    centroid_3d_after: Optional[Tuple[float, float, float]] = None
    velocity_3d: Optional[Tuple[float, float, float]] = None
    velocity_2d: Optional[Tuple[float, float]] = None
    bounding_box_before: Optional[Sequence[float]] = None
    bounding_box_after: Optional[Sequence[float]] = None
    change_magnitude: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def duration_seconds(self) -> float:
        return max(0.0, float(self.timestamp_after - self.timestamp_before))

    def to_dict(self) -> Dict[str, Any]:
        def _maybe_list(value):
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                return [float(v) for v in value]
            return value

        return {
            "entity_id": self.entity_id,
            "class_label": self.class_label,
            "frame_before": self.frame_before,
            "frame_after": self.frame_after,
            "timestamp_before": self.timestamp_before,
            "timestamp_after": self.timestamp_after,
            "centroid_before": _maybe_list(self.centroid_before),
            "centroid_after": _maybe_list(self.centroid_after),
            "centroid_3d_before": _maybe_list(self.centroid_3d_before),
            "centroid_3d_after": _maybe_list(self.centroid_3d_after),
            "velocity_3d": _maybe_list(self.velocity_3d),
            "velocity_2d": _maybe_list(self.velocity_2d),
            "bounding_box_before": _maybe_list(self.bounding_box_before),
            "bounding_box_after": _maybe_list(self.bounding_box_after),
            "change_magnitude": self.change_magnitude,
            "metadata": self.metadata,
        }


@dataclass
class CausalLink:
    """Represents a directional causal hypothesis between two entities."""

    agent_id: str
    patient_id: str
    agent_change: StateChange
    patient_change: StateChange
    influence_score: float
    justification: str
    features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "patient_id": self.patient_id,
            "influence_score": float(self.influence_score),
            "justification": self.justification,
            "features": {k: float(v) for k, v in self.features.items()},
            "agent_change": self.agent_change.to_dict(),
            "patient_change": self.patient_change.to_dict(),
        }


__all__ = ["StateChange", "CausalLink"]