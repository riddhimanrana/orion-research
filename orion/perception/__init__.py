"""
Perception Engine Package
=========================

Phase 1 of the Orion pipeline: Detection, Embedding, Tracking, Description + 3D Perception.

Modules:
    - types: Data structures (Observation, PerceptionEntity, Hand, EntityState3D, etc.)
    - config: Configuration for perception components
    - observer: Frame sampling and YOLO detection
    - embedder: CLIP embedding generation
    - tracker: HDBSCAN clustering into entities
    - describer: FastVLM description generation
    - depth: Monocular depth estimation (MiDaS/ZoeDepth)
    - hand_tracking: MediaPipe hand tracking with 3D projection
    - camera_intrinsics: 3D backprojection utilities
    - occlusion: Depth-based occlusion detection
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

from .depth import DepthEstimator
from .engine import PerceptionEngine

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
    # Modules
    "DepthEstimator",
    "Perception3DEngine",
    "PerceptionEngine",
    "camera_intrinsics",
    "PerceptionConfig",
    "get_fast_config",
    "get_balanced_config",
    "get_accurate_config",
]
