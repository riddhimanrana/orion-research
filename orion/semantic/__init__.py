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

# Engine component exports
from orion.semantic.entity_tracker import SemanticEntityTracker
from orion.semantic.state_detector import StateChangeDetector
from orion.semantic.scene_assembler import SceneAssembler
from orion.semantic.temporal_windows import TemporalWindowManager
from orion.semantic.causal_scorer import CausalInfluenceScorer
from orion.semantic.event_composer import EventComposer
from orion.graph.builder import GraphBuilder
from orion.semantic.engine import SemanticEngine, run_semantic

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
    # Components
    "SemanticEntityTracker",
    "StateChangeDetector",
    "SceneAssembler",
    "TemporalWindowManager",
    "CausalInfluenceScorer",
    "EventComposer",
    "GraphBuilder",
    # Engine
    "SemanticEngine",
    "run_semantic",
]


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
]
