"""FastVLM inference wrapper for Orion.

Uses MLX backend for Apple Silicon (M1/M2/M3) and official Apple FastVLM model.

Author: Orion Research Team
"""

from __future__ import annotations

import logging
import platform
from typing import Iterable, List, Optional, Sequence, Union
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, Image.Image]


def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (M1/M2/M3/etc)."""
    return platform.system() == "Darwin" and platform.processor() == "arm"


class FastVLMTorchWrapper:
    """Unified wrapper for FastVLM using MLX backend only."""

    def __init__(
        self,
        model_source: Optional[str] = None,
        *,
        device: Optional[str] = None,
        conv_mode: str = "qwen_2",
        force_backend: Optional[str] = None,
    ) -> None:
        """Initialize FastVLM with MLX backend only.
        Args:
            model_source: Path to local model. Defaults to Apple official model.
            device: Ignored (MLX always uses Metal).
            conv_mode: Conversation template mode (default: "qwen_2")
            force_backend: Ignored (always MLX).
        """
        logger.info("Using MLX backend for FastVLM on Apple Silicon")
        from .mlx_fastvlm import FastVLMMLXWrapper

        self._backend = FastVLMMLXWrapper(model_source=model_source)

    def generate_description(
        self,
        image: ImageInput,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
        top_p: Optional[float] = None,
        num_beams: int = 1,
    ) -> str:
        """Generate description for a single image."""
        return self._backend.generate_description(
            image=image,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
        )

    def batch_generate(
        self,
        images: Sequence[ImageInput],
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
        top_p: Optional[float] = None,
        num_beams: int = 1,
    ) -> List[str]:
        """Generate descriptions for multiple images with the same prompt."""
        return self._backend.batch_generate(
            images=images,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
        )

    def describe_many(self, prompts: Iterable[tuple[ImageInput, str]]) -> List[str]:
        """Generate descriptions for multiple image-prompt pairs."""
        return self._backend.describe_many(prompts)


__all__ = [
    "FastVLMTorchWrapper",
]
