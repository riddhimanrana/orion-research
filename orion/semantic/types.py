"""
Semantic Engine Data Types & Contracts
======================================

Clean, typed data structures for Phase 2 (Semantic Understanding).

These types flow through:
    PerceptionResult → SemanticEntity → StateChange → Event → Graph

Author: Orion Research Team
Date: October 2025
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from orion.perception.types import Observation, PerceptionEntity, BoundingBox
else:
    from orion.perception.types import BoundingBox


class ChangeType(str, Enum):
    """Types of semantic state changes"""
    POSITION = "position"
    VELOCITY = "velocity"
    APPEARANCE = "appearance"
    DISAPPEARANCE = "disappearance"
    INTERACTION = "interaction"
    UNKNOWN = "unknown"


@dataclass
class StateChange:
    """
    Detected semantic state change in an entity.
    
    Represents a transition from one description/state to another.
    """
    
    # Identity
    entity_id: str
    
    # Temporal bounds
    timestamp_before: float
    timestamp_after: float
    frame_before: int
    frame_after: int
    
    # State descriptions
    description_before: str
    description_after: str
    
    # Similarity metrics
    similarity_score: float  # 0-1, cosine similarity
    change_magnitude: float = field(init=False)  # 1.0 - similarity

    # Classification
    change_type: "ChangeType" = field(default=ChangeType.UNKNOWN)
    is_fallback: bool = False
    
    # Spatial information (2D)
    centroid_before: Optional[Tuple[float, float]] = None
    centroid_after: Optional[Tuple[float, float]] = None
    displacement: float = 0.0
    velocity: float = 0.0
    
    # Spatial information (3D) - NEW for 3D CIS
    centroid_3d_before: Optional[np.ndarray] = None  # [x, y, z] in mm
    centroid_3d_after: Optional[np.ndarray] = None   # [x, y, z] in mm
    velocity_3d: Optional[np.ndarray] = None         # [vx, vy, vz] in mm/s
    
    # Bounding boxes
    bounding_box_before: Optional[List[float]] = None
    bounding_box_after: Optional[List[float]] = None
    
    # Scene context
    scene_before: Optional[str] = None
    scene_after: Optional[str] = None
    location_before: Optional[str] = None
    location_after: Optional[str] = None
    
    # Embeddings
    embedding_before: Optional[np.ndarray] = None
    embedding_after: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate and compute derived fields"""
        self.change_magnitude = 1.0 - self.similarity_score
        
        if self.similarity_score < 0 or self.similarity_score > 1:
            raise ValueError(f"similarity_score must be in [0, 1], got {self.similarity_score}")

        # Ensure change_type is a ChangeType enum for downstream code
        if not isinstance(self.change_type, ChangeType):
            try:
                self.change_type = ChangeType(self.change_type)  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover - defensive casting
                raise ValueError(
                    f"change_type must be a ChangeType enum value, got {self.change_type}"
                ) from exc
    
    def to_dict(self) -> dict:
        """Convert to serializable dictionary"""
        return {
            "entity_id": self.entity_id,
            "timestamp_before": self.timestamp_before,
            "timestamp_after": self.timestamp_after,
            "frame_before": self.frame_before,
            "frame_after": self.frame_after,
            "description_before": self.description_before,
            "description_after": self.description_after,
            "similarity_score": float(self.similarity_score),
            "change_magnitude": float(self.change_magnitude),
            "displacement": float(self.displacement),
            "velocity": float(self.velocity),
            "change_type": self.change_type.value,
            "is_fallback": self.is_fallback,
        }


@dataclass
class TemporalWindow:
    """
    Adaptive time window containing related state changes.
    
    Used for event composition - groups related state changes
    that should be described as a single event.
    """
    
    start_time: float
    end_time: float
    
    # Content
    state_changes: List[StateChange] = field(default_factory=list)
    active_entities: Set[str] = field(default_factory=set)
    causal_links: List["CausalLink"] = field(default_factory=list)
    average_change_magnitude: float = 0.0
    significance_score: float = 0.0
    fallback_generated: bool = False
    
    def add_state_change(self, change: StateChange) -> None:
        """Add state change and track active entity"""
        self.state_changes.append(change)
        self.active_entities.add(change.entity_id)
        self.end_time = max(self.end_time, change.timestamp_after)
        if change.is_fallback:
            self.fallback_generated = True
    
    @property
    def duration(self) -> float:
        """Duration in seconds"""
        return max(0.0, self.end_time - self.start_time)
    
    @property
    def is_significant(self) -> bool:
        """Window has enough changes to warrant composition"""
        if self.fallback_generated:
            return len(self.state_changes) >= 1

        MIN_CHANGES = 2
        return len(self.state_changes) >= MIN_CHANGES and self.significance_score >= 0.5
    
    def top_causal_links(self, limit: int = 5) -> List["CausalLink"]:
        """Get top-scoring causal links"""
        if not self.causal_links:
            return []
        sorted_links = sorted(
            self.causal_links,
            key=lambda link: link.influence_score,
            reverse=True
        )
        return sorted_links[:limit]


@dataclass
class CausalLink:
    """
    Inferred causal relationship between two entities.
    
    Represents: "Entity A's change likely caused Entity B's change"
    """
    
    agent_id: str  # Entity that caused the change
    patient_id: str  # Entity that was affected
    
    agent_change: StateChange
    patient_change: StateChange
    
    # Scoring
    influence_score: float  # 0-1, from CIS formula
    
    # Justification
    features: Dict[str, float] = field(default_factory=dict)
    justification: str = ""
    
    def to_dict(self) -> dict:
        """Convert to serializable dictionary"""
        return {
            "agent_id": self.agent_id,
            "patient_id": self.patient_id,
            "influence_score": float(self.influence_score),
            "features": self.features,
            "justification": self.justification,
        }


@dataclass
class SceneSegment:
    """
    Aggregated scene/room segment with temporal bounds.
    
    Represents a coherent spatial-temporal context
    (e.g., "the kitchen during frames 100-200").
    """
    start_time: float
    end_time: float
    scene_id: str = ""
    
    # Content
    temporal_windows: List[TemporalWindow] = field(default_factory=list)
    object_classes: List[str] = field(default_factory=list)
    entity_ids: List[str] = field(default_factory=list)
    description: str = ""
    
    # Location
    location_id: str = ""
    location_grid_key: str = ""  # e.g., "r1c2"
    location_profile: Optional["LocationProfile"] = None
    
    # Legacy fields for backward compatibility
    frame_number: int = 0
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0
    
    # Optional embedding
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Sync legacy timestamp fields"""
        if self.start_timestamp == 0.0 and self.start_time > 0:
            self.start_timestamp = self.start_time
        if self.end_timestamp == 0.0 and self.end_time > 0:
            self.end_timestamp = self.end_time
    
    def add_window(self, window: TemporalWindow) -> None:
        """Add temporal window to scene"""
        self.temporal_windows.append(window)
        self.end_time = max(self.end_time, window.end_time)
        self.entity_ids.extend(list(window.active_entities))
        self.entity_ids = list(set(self.entity_ids))  # Deduplicate
    
    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class LocationProfile:
    """
    Logical location inferred from dominant scene objects.
    
    Represents an inferred room/area based on spatial signatures.
    """
    
    location_id: str
    signature: str  # e.g., "kitchen|r1c2|indoor"
    label: str  # Human-readable location name
    object_classes: List[str]
    zone_ids: List[str] = field(default_factory=list)
    
    # Associated scenes
    scene_ids: List[str] = field(default_factory=list)


@dataclass
class Event:
    """
    High-level event composed from temporal window.
    
    LLM-generated description of what happened in a window.
    """
    
    event_id: str
    
    # Description
    description: str
    event_type: str  # e.g., "motion", "interaction", "state_change"
    
    # Temporal
    start_timestamp: float
    end_timestamp: float
    start_frame: int
    end_frame: int
    
    # Participants
    involved_entities: List[str]
    causal_links: List[CausalLink] = field(default_factory=list)
    
    # Confidence
    confidence: float = 0.5  # 0-1, composite score
    
    # Scene context
    scene_id: Optional[str] = None
    location_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate event"""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
    
    @property
    def duration(self) -> float:
        return max(0.0, self.end_timestamp - self.start_timestamp)
    
    def to_dict(self) -> dict:
        """Convert to serializable dictionary"""
        return {
            "event_id": self.event_id,
            "description": self.description,
            "event_type": self.event_type,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "duration": self.duration,
            "involved_entities": self.involved_entities,
            "confidence": float(self.confidence),
            "scene_id": self.scene_id,
            "location_id": self.location_id,
        }


@dataclass
class SemanticResult:
    """Output from Phase 2 (Semantic Engine)"""
    
    # Core outputs
    entities: List["SemanticEntity"]  # Consolidated entities
    state_changes: List[StateChange]
    temporal_windows: List[TemporalWindow]
    events: List[Event]
    causal_links: List[CausalLink]
    
    # Scene & location context
    scenes: List[SceneSegment]
    locations: Dict[str, LocationProfile]
    
    # Spatial context (NEW - Phase 2)
    spatial_zones: List = field(default_factory=list)  # List[SpatialZone] from spatial_utils
    
    # Metadata
    total_detections: int = 0
    unique_entities: int = 0
    state_changes_detected: int = 0
    events_composed: int = 0
    processing_time_seconds: float = 0.0
    
    def __post_init__(self):
        """Compute statistics"""
        self.unique_entities = len(self.entities)
        self.state_changes_detected = len(self.state_changes)
        self.events_composed = len(self.events)


# Forward reference - defined in semantic/tracking/entity.py
@dataclass
class SemanticEntity:
    """
    Consolidated entity across entire video (Phase 2).
    
    Aggregates Phase 1 PerceptionEntity observations temporally
    and tracks semantic state changes.
    """
    
    entity_id: str
    object_class: str
    
    # Observations over time
    observations: List["Observation"] = field(default_factory=list)
    
    # Aggregated properties
    first_timestamp: float = 0.0
    last_timestamp: float = 0.0
    first_appearance_time: float = 0.0
    last_appearance_time: float = 0.0
    average_embedding: Optional[np.ndarray] = None
    average_centroid: Tuple[float, float] = (0.5, 0.5)
    average_bbox: Optional[BoundingBox] = None
    frame_width: float = 1920.0
    frame_height: float = 1080.0
    
    # Scene associations
    scene_ids: Set[str] = field(default_factory=set)
    
    # Spatial zone associations (NEW - Phase 2)
    zone_id: Optional[str] = None
    zone_label: Optional[str] = None
    zone_confidence: Optional[float] = None
    neighbor_entity_ids: List[str] = field(default_factory=list)
    
    # 3D tracking (NEW - for 3D CIS)
    centroid_3d_mm: Optional[np.ndarray] = None  # Current 3D position [x, y, z] in mm
    prev_centroid_3d_mm: Optional[np.ndarray] = None  # Previous 3D position
    velocity_3d: Optional[np.ndarray] = None  # 3D velocity [vx, vy, vz] in mm/s
    
    # Descriptions over time (timestamped)
    descriptions: List[Dict] = field(default_factory=list)
    
    # Single description (from Phase 1) - kept for backward compatibility
    description: Optional[str] = None
    
    # State changes for this entity
    state_changes: List[StateChange] = field(default_factory=list)
    
    # Reference to original perception entity
    perception_entity: Optional["PerceptionEntity"] = None
    
    def add_observation(self, observation: "Observation") -> None:
        """Add observation and update temporal bounds"""
        self.observations.append(observation)
        timestamp = observation.timestamp
        if not self.first_timestamp or timestamp < self.first_timestamp:
            self.first_timestamp = timestamp
        if not self.last_timestamp or timestamp > self.last_timestamp:
            self.last_timestamp = timestamp
    
    def get_timeline(self) -> List["Observation"]:
        """Get observations in chronological order"""
        return sorted(self.observations, key=lambda obs: obs.timestamp)
