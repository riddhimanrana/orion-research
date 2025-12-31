"""FastVLM inference wrapper for Orion using PyTorch.

This backend uses the transformers library to run FastVLM on any hardware
supported by PyTorch, including CUDA, MPS (Apple Silicon), and CPU.

Author: Orion Research Team
"""

from __future__ import annotations

import logging
import platform
from typing import Iterable, List, Optional, Sequence, Union
from pathlib import Path

from PIL import Image
import torch

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, Image.Image]

try:
    from transformers import AutoModelForCausalLM, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available. Install: pip install transformers")

class FastVLMTorchWrapper:
    """PyTorch-based wrapper for FastVLM inference."""

    def __init__(
        self,
        model_source: Optional[str] = None,
        *,
        device: Optional[str] = None,
        conv_mode: str = "qwen_2",
    ) -> None:
        """Initialize FastVLM with PyTorch backend.

        Args:
            model_source: Path to local model or HuggingFace repo.
            device: Device to use (e.g., "cuda", "mps", "cpu"). Auto-detected if None.
            conv_mode: Conversation template mode.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required for the torch backend. Install with: pip install transformers")

        # Use local model path if available, otherwise fallback to HF
        if model_source is None:
            local_path = Path(__file__).parent.parent.parent / "models" / "fastvlm-0.5b"
            if local_path.exists():
                self.model_source = str(local_path)
                logger.info(f"Using local FastVLM model at {self.model_source}")
            else:
                self.model_source = "apple/fastvlm-0.5b"
                logger.info("Using HuggingFace FastVLM model")
        else:
            self.model_source = model_source
        
        self.device = device or self._detect_device()
        self.conv_mode = conv_mode

        logger.info(f"Initializing FastVLM with torch backend on device: {self.device}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_source,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(self.model_source, trust_remote_code=True)
        self.model.eval()
        logger.info(f"✓ FastVLM model loaded on {self.device}")

    def _detect_device(self) -> str:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

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
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Format prompt manually since FastVLM doesn't have a chat template
        prompt_text = f"USER: <image>\n{prompt}\nASSISTANT:"
        
        inputs = self.processor(text=prompt_text, images=image, return_tensors="pt").to(self.device)

        generation_output = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            do_sample=temperature > 0,
        )

        generated_text = self.processor.batch_decode(generation_output, skip_special_tokens=True)[0]
        
        # The output includes the prompt, so we need to remove it.
        # This is a common pattern with vision-language models.
        cleaned_text = generated_text.split("ASSISTANT:")[-1].strip()
        return cleaned_text

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
        results: List[str] = []
        for image in images:
            result = self.generate_description(
                image,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
            )
            results.append(result)
        return results

    def describe_many(self, prompts: Iterable[tuple[ImageInput, str]]) -> List[str]:
        """Generate descriptions for multiple image-prompt pairs."""
        results: List[str] = []
        for image, prompt in prompts:
            result = self.generate_description(image, prompt)
            results.append(result)
        return results


__all__ = [
    "FastVLMTorchWrapper",
]
