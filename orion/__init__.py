"""Orion research toolkit package."""

import os

# Suppress fork-related warnings from HuggingFace tokenizers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("orion-research")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

# Perception module exports
from orion.perception import (
    PerceptionConfig,
    ObjectClass,
    BoundingBox,
    Observation,
    PerceptionEntity,
    PerceptionResult,
)

from orion.perception.engine import PerceptionEngine
from orion.perception.perception_3d import Perception3DEngine

# Semantic module exports
from orion.semantic import (
    SemanticConfig,
    ChangeType,
    StateChange,
    TemporalWindow,
    CausalLink,
    SceneSegment,
    LocationProfile,
    Event,
    SemanticEntity,
    SemanticResult,
)

from orion.semantic.engine import SemanticEngine, run_semantic

# Pipeline exports
# Note: Old pipeline.py archived. Use core_pipeline for new code.
# from orion.pipeline import VideoPipeline

__all__ = [
    # Version
    "__version__",
    # Perception
    "PerceptionConfig",
    "ObjectClass", 
    "BoundingBox",
    "Observation",
    "PerceptionEntity",
    "PerceptionResult",
    "Perception3DEngine",
    "PerceptionEngine",
    # Semantic
    "SemanticConfig",
    "ChangeType",
    "StateChange",
    "TemporalWindow",
    "CausalLink",
    "SceneSegment",
    "LocationProfile",
    "Event",
    "SemanticEntity",
    "SemanticResult",
    "SemanticEngine",
    "run_semantic",
    # Pipeline
    # "VideoPipeline",  # Use orion.core_pipeline instead
]
