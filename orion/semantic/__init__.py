"""
Semantic Understanding Engine
==============================

Phase 2: Semantic Uplift Pipeline

Transforms perception results into semantic understanding:
- Entity tracking and consolidation
- State change detection
- Scene assembly and location inference
- Temporal windowing
- Causal influence scoring (CIS)
- Event composition via LLM

Author: Orion Research Team
Date: October 2025
"""

# Type exports
from orion.semantic.types import (
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

# Config exports
from orion.semantic.config import (
    StateChangeConfig,
    TemporalWindowConfig,
    EventCompositionConfig,
    CausalConfig,
    SemanticConfig,
)

# Engine export (minimal – deprecated component modules archived)
from orion.semantic.engine import SemanticEngine, run_semantic


from .types import (
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

from .config import (
    StateChangeConfig,
    TemporalWindowConfig,
    EventCompositionConfig,
    CausalConfig,
    SemanticConfig,
    get_fast_semantic_config,
    get_balanced_semantic_config,
    get_accurate_semantic_config,
)

__all__ = [
    # Types
    "ChangeType",
    "StateChange",
    "TemporalWindow",
    "CausalLink",
    "SceneSegment",
    "LocationProfile",
    "Event",
    "SemanticEntity",
    "SemanticResult",
    # Config
    "StateChangeConfig",
    "TemporalWindowConfig",
    "EventCompositionConfig",
    "CausalConfig",
    "SemanticConfig",
    "get_fast_semantic_config",
    "get_balanced_semantic_config",
    "get_accurate_semantic_config",
    # Engine
    "SemanticEngine",
    "run_semantic",
]
