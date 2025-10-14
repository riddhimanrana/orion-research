"""Expose LLaVA models within the Orion namespace."""

import sys

from .model import LlavaLlamaForCausalLM, LlavaQwen2ForCausalLM

# Maintain backwards compatibility for modules expecting a top-level "llava" package.
sys.modules.setdefault("llava", sys.modules[__name__])
