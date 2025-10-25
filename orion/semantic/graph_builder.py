"""Deprecated shim for ``orion.semantic.graph_builder``."""

from __future__ import annotations

import warnings

from orion.graph.builder import GraphBuilder

__all__ = ["GraphBuilder"]


warnings.warn(
    "`orion.semantic.graph_builder` is deprecated; import GraphBuilder from "
    "`orion.graph.builder` instead.",
    DeprecationWarning,
    stacklevel=2,
)

