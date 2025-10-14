"""PyTorch implementation of FastVLM inference.

This module mirrors Apple's reference approach for running FastVLM models
via Hugging Face Transformers. It provides a wrapper with the same interface
used by the MLX implementation so the perception engine can dynamically select
between runtimes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

IMAGE_TOKEN_INDEX = -200
DEFAULT_MODEL_ID = "apple/FastVLM-0.5B"

ImageInput = Union[str, Path, Image.Image]


def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")
    return Image.open(path).convert("RGB")


class FastVLMTorchWrapper:
    """Wrapper around Hugging Face FastVLM for PyTorch runtimes."""

    def __init__(
        self,
        model_source: Optional[str] = None,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        resolved_source = model_source or DEFAULT_MODEL_ID
        self.model_source = str(resolved_source)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device) if device is not None else _detect_device()
        if dtype is None:
            if self.device.type == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32
        self.dtype = dtype

        logger.info("Loading FastVLM model (%s) on %s", self.model_source, self.device)
        load_kwargs = {"trust_remote_code": trust_remote_code}
        if self.cache_dir is not None:
            load_kwargs["cache_dir"] = str(self.cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_source,
            **load_kwargs,
        )
        model_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": "auto" if device is None else None,
            "trust_remote_code": trust_remote_code,
        }
        if self.cache_dir is not None:
            model_kwargs["cache_dir"] = str(self.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_source,
            **model_kwargs,
        )
        if device is not None:
            self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

        # The vision tower exposes its own processor for pixel normalization.
        self.image_processor = self.model.get_vision_tower().image_processor
        logger.info("FastVLM model ready")

    # ------------------------------------------------------------------
    # Prompt / token helpers
    # ------------------------------------------------------------------
    def _build_inputs(self, prompt: str) -> torch.Tensor:
        messages = [{"role": "user", "content": f"<image>\n{prompt}"}]
        rendered = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        if "<image>" not in rendered:
            raise ValueError("Chat template output did not contain <image> placeholder")
        prefix, suffix = rendered.split("<image>", 1)

        pre_ids = self.tokenizer(
            prefix,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids
        post_ids = self.tokenizer(
            suffix,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids

        image_token = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, image_token, post_ids], dim=1)
        return input_ids.to(self.device)

    def _prepare_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(input_ids, device=self.device)

    def _prepare_pixel_values(self, image: ImageInput) -> torch.Tensor:
        pil_image = _ensure_image(image)
        pixel_values = self.image_processor(images=pil_image, return_tensors="pt")[
            "pixel_values"
        ]
        return pixel_values.to(self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_description(
        self,
        image: ImageInput,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.2,
        top_p: Optional[float] = None,
    ) -> str:
        input_ids = self._build_inputs(prompt)
        attention_mask = self._prepare_attention_mask(input_ids)
        pixel_values = self._prepare_pixel_values(image)

        generation_kwargs = {
            "inputs": input_ids,
            "attention_mask": attention_mask,
            "images": pixel_values,
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
        }
        if top_p is not None:
            generation_kwargs["top_p"] = top_p

        with torch.no_grad():
            output_ids = self.model.generate(**generation_kwargs)

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def batch_generate(
        self,
        images: Sequence[ImageInput],
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.2,
        top_p: Optional[float] = None,
    ) -> List[str]:
        outputs: List[str] = []
        for image in images:
            outputs.append(
                self.generate_description(
                    image,
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            )
        return outputs

    def describe_many(self, prompts: Iterable[tuple[ImageInput, str]]) -> List[str]:
        results: List[str] = []
        for image, prompt in prompts:
            results.append(self.generate_description(image, prompt))
        return results


__all__ = ["FastVLMTorchWrapper", "DEFAULT_MODEL_ID", "IMAGE_TOKEN_INDEX"]
