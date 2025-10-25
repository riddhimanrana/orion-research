"""Compatibility shim for legacy causal inference imports."""

from __future__ import annotations

import warnings

from orion.semantic.causal import (  # noqa: F401
    AgentCandidate,
    CausalConfig,
    CausalInferenceEngine,
    CausalLink,
    StateChange,
    cosine_similarity,
)

__all__ = [
    "CausalInferenceEngine",
    "CausalConfig",
    "AgentCandidate",
    "StateChange",
    "CausalLink",
    "cosine_similarity",
]

warnings.warn(
    "`orion.causal_inference` is deprecated; import from `orion.semantic.causal` instead.",
    DeprecationWarning,
    stacklevel=2,
)
