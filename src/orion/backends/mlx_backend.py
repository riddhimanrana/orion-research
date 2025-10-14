"""Deprecated MLX backend shim.

The project now runs FastVLM exclusively via PyTorch. This module remains only to guard
against stale imports and will redirect callers to the shared torch implementation.
"""

from __future__ import annotations

import logging
from typing import Any

from ..runtime import BackendName
from .torch_fastvlm import FastVLMTorchWrapper

logger = logging.getLogger(__name__)

BACKEND_NAME: BackendName = "mlx"


def ensure_dependencies() -> None:
    """Legacy hook retained for compatibility."""
    logger.warning("MLX backend requested; Orion uses the torch implementation instead.")


def load(*args: Any, **kwargs: Any) -> FastVLMTorchWrapper:
    """Return the unified torch FastVLM wrapper while logging the redirection."""
    ensure_dependencies()
    return FastVLMTorchWrapper(*args, **kwargs)
