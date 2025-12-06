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
    from transformers import AutoModelForCausalLM, AutoTokenizer
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

        # Use local model path if available, otherwise fallback to Hugging Face
        if model_source is None:
            local_path = Path(__file__).parent.parent.parent / "models" / "fastvlm-0.5b"
            if local_path.exists():
                self.model_source = str(local_path)
                logger.info(f"Using local FastVLM model at {self.model_source}")
            else:
                self.model_source = "apple/FastVLM-0.5B"
                logger.info("Using Hugging Face FastVLM model (apple/FastVLM-0.5B)")
        else:
            self.model_source = model_source
        
        self.device = device or self._detect_device()
        self.conv_mode = conv_mode

        logger.info(f"Initializing FastVLM with torch backend on device: {self.device}")

        # Allow remote download when local weights are absent
        local_files_only = Path(self.model_source).exists()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_source,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_source,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            local_files_only=local_files_only,
        ).to(self.device)

        # FastVLM exposes its own processor via the vision tower
        self.image_processor = self.model.get_vision_tower().image_processor
        self.image_token_index = -200  # FastVLM convention
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

        # Build chat template with explicit <image> placeholder
        messages = [
            {"role": "user", "content": f"<image>\n{prompt}"}
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        try:
            pre_text, post_text = rendered.split("<image>", 1)
        except ValueError:
            raise RuntimeError("FastVLM prompt rendering did not include <image> token")

        pre_ids = self.tokenizer(
            pre_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids
        post_ids = self.tokenizer(
            post_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids

        img_tok = torch.tensor([[self.image_token_index]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)

        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device, dtype=self.model.dtype)

        generation_output = self.model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=pixel_values,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            do_sample=temperature > 0,
        )

        return self.tokenizer.decode(generation_output[0], skip_special_tokens=True)

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
