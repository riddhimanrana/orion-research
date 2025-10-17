#!/usr/bin/env python3
"""CLI helper to validate the FastVLM Torch wrapper in isolation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for development
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from PIL import Image

from orion.backends.torch_fastvlm import FastVLMTorchWrapper


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FastVLM wrapper on a single image.")
    parser.add_argument("image", type=Path, help="Path to the image file to describe.")
    parser.add_argument("prompt", type=str, help="Prompt to provide to FastVLM.")
    parser.add_argument(
        "--model",
        dest="model_source",
        type=str,
        default=None,
        help="Optional override for the FastVLM checkpoint directory or Hugging Face model id.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override (e.g. 'mps', 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate (default: 256).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p nucleus sampling value (optional).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    print(f"🚀 Initializing FastVLM...")
    wrapper = FastVLMTorchWrapper(model_source=args.model_source, device=args.device)
    
    print(f"📷 Processing image: {args.image}")
    print(f"💬 Prompt: {args.prompt}")
    
    description = wrapper.generate_description(
        args.image,
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    print(f"\n📝 Result:\n{description}")


if __name__ == "__main__":
    main()
