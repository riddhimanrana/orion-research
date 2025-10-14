"""PyTorch-native backend entrypoints for Orion."""

from __future__ import annotations

import importlib.util
from typing import Any

from ..runtime import BackendName
from .torch_fastvlm import FastVLMTorchWrapper

BACKEND_NAME: BackendName = "torch"


def ensure_dependencies() -> None:
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError(
            "PyTorch backend requested but the 'torch' package is not installed. Install PyTorch to proceed."
        )


def load(*args: Any, **kwargs: Any) -> FastVLMTorchWrapper:
    """Instantiate the FastVLM torch wrapper after confirming dependencies."""
    ensure_dependencies()
    return FastVLMTorchWrapper(*args, **kwargs)
