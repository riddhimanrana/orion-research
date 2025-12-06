"""
Perception Engine Package
=========================

Phase 1 of the Orion pipeline: detection, embeddings, tracking, descriptions, and optional 3D perception.

This package now focuses purely on the perception stage. Graph/scene memory utilities moved to
``orion.graph`` so the folder surface is limited to the active runtime components listed below:

Modules:
    - types: Data structures (Observation, PerceptionEntity, Hand, EntityState3D, etc.)
    - config: Configuration for perception components
    - observer: Frame sampling and YOLO/GroundingDINO detection
    - embedder: CLIP embedding generation
    - tracker: HDBSCAN clustering into entities
    - describer: FastVLM description generation
    - depth: Monocular depth estimation helpers
    - camera_intrinsics: 3D backprojection utilities
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
    CameraIntrinsics,
    EntityState,
)

from .config import (
    DetectionConfig,
    EmbeddingConfig,
    DescriptionConfig,
    PerceptionConfig,
    DepthConfig,
    HandTrackingConfig,
    OcclusionConfig,
    CameraConfig,
    ClassCorrectionConfig,
    get_fast_config,
    get_balanced_config,
    get_accurate_config,
)

from .depth import DepthEstimator
from .engine import PerceptionEngine

# Try to import 3D perception components
try:
    from .perception_3d import Perception3DEngine
    PERCEPTION_3D_AVAILABLE = True
except ImportError:
    Perception3DEngine = None
    PERCEPTION_3D_AVAILABLE = False

__all__ = [
    # Types
    "ObjectClass",
    "SpatialZone",
    "BoundingBox",
    "Observation",
    "PerceptionEntity",
    "PerceptionResult",
    "CameraIntrinsics",
    "EntityState",
    # Config
    "DetectionConfig",
    "EmbeddingConfig",
    "DescriptionConfig",
    "DepthConfig",
    "HandTrackingConfig",
    "OcclusionConfig",
    "CameraConfig",
    "ClassCorrectionConfig",
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
