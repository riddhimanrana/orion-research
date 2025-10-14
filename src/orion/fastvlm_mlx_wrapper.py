"""Deprecated compatibility shim for the legacy MLX FastVLM wrapper."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class FastVLMMLXWrapper:  # pragma: no cover - retained for backward compatibility only
    """Placeholder wrapper that guides users toward the unified PyTorch backend."""

    def __init__(self, *_: Any, **__: Any) -> None:
        logger.warning(
            "FastVLMMLXWrapper is deprecated. Orion now uses the apple/FastVLM-0.5B Hugging Face "
            "checkpoint across platforms; please remove MLX-specific configuration."
        )
        raise RuntimeError(
            "The MLX-based FastVLM implementation has been removed. Use the Hugging Face torch "
            "pipeline instead."
        )


__all__ = ["FastVLMMLXWrapper"]
