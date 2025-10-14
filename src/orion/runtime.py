"""Runtime backend selection utilities."""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Literal, Optional

BackendName = Literal["torch"]
_AUTO = "auto"
_SUPPORTED: set[str] = {"torch"}

logger = logging.getLogger(__name__)


def _module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _normalize_backend_name(name: str) -> str:
    lowered = name.lower()
    if lowered == "mlx":
        return "torch"
    if lowered in _SUPPORTED:
        return lowered
    raise ValueError(f"Unsupported backend preference '{name}'.")


def is_backend_available(backend: str) -> bool:
    """Check whether the given backend can be imported on this machine."""
    normalized = _normalize_backend_name(backend)
    if normalized == "torch":
        return _module_exists("torch")
    return False


def select_backend(preferred: Optional[str] = None) -> BackendName:
    """Resolve the backend to use based on user preference and environment."""
    raw_preference = preferred if preferred is not None else os.getenv("ORION_RUNTIME", _AUTO)
    choice = raw_preference.lower()

    if choice not in {_AUTO, "auto"}:
        normalized = _normalize_backend_name(choice)
        if normalized == "torch" and not is_backend_available("torch"):
            raise RuntimeError(
                "Requested backend 'torch' is not available. Install PyTorch to continue."
            )
        if choice == "mlx":
            logger.info("'mlx' preference detected; routing to the unified PyTorch backend.")
        return "torch"

    if is_backend_available("torch"):
        return "torch"

    raise RuntimeError("No supported backend found. Install PyTorch to run Orion's perception pipeline.")


def set_active_backend(backend: BackendName) -> None:
    """Record the active backend for other components to read."""
    os.environ["ORION_RUNTIME_BACKEND"] = backend


def get_active_backend(default: Optional[BackendName] = None) -> Optional[BackendName]:
    """Retrieve the backend chosen for the current session."""
    value = os.getenv("ORION_RUNTIME_BACKEND")
    if value is None:
        return default
    try:
        normalized = _normalize_backend_name(value)
    except ValueError:
        return default
    return normalized  # type: ignore[return-value]
