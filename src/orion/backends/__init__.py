"""Backends exposed by Orion."""

from .torch_fastvlm import DEFAULT_MODEL_ID, FastVLMTorchWrapper

__all__ = ["FastVLMTorchWrapper", "DEFAULT_MODEL_ID"]
