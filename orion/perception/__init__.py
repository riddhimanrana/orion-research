"""
Perception Engine Package
=========================

Phase 1 of the Orion pipeline: Detection, Embedding, Tracking, Description.

Modules:
    - types: Data structures (Observation, PerceptionEntity, etc.)
    - config: Configuration for perception components
    - observer: Frame sampling and YOLO detection
    - embedder: CLIP embedding generation
    - tracker: HDBSCAN clustering into entities
    - describer: FastVLM description generation
    - engine: High-level orchestration

Author: Orion Research Team
Date: October 2025
"""

from .types import (
    ObjectClass,
    SpatialZone,
    BoundingBox,
    Observation,
    PerceptionEntity,
    PerceptionResult,
)

from .config import (
    DetectionConfig,
    EmbeddingConfig,
    DescriptionConfig,
    PerceptionConfig,
    get_fast_config,
    get_balanced_config,
    get_accurate_config,
)

__all__ = [
    # Types
    "ObjectClass",
    "SpatialZone",
    "BoundingBox",
    "Observation",
    "PerceptionEntity",
    "PerceptionResult",
    # Config
    "DetectionConfig",
    "EmbeddingConfig",
    "DescriptionConfig",
    "PerceptionConfig",
    "get_fast_config",
    "get_balanced_config",
    "get_accurate_config",
]
