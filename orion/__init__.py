"""Orion research toolkit package."""

import os
import warnings

# Suppress fork-related warnings from HuggingFace tokenizers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Suppress sklearn FutureWarnings about force_all_finite (from HDBSCAN dependency)
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)

# Suppress Google API Python version warning (we're aware and will upgrade later)
warnings.filterwarnings("ignore", message=".*Python version.*stop supporting.*", category=FutureWarning)

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
    "PerceptionEngine",
]