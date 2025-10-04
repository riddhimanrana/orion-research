"""
FastVLM Wrapper for Orion Research Project
Uses custom fine-tuned FastVLM-0.5B model (4-bit quantized, caption-optimized)

This module provides a clean interface to the FastVLM model for generating
rich descriptions of video frames in the perception engine.

Model Details:
- Base: FastVLM-0.5B (Qwen2-0.5B language model)
- Fine-tuned: Caption generation
- Quantized: 4-bit (group_size=64)
- Vision encoder: FastViTHD (mobileclip_l_1024) - 3072-dim features
- Location: models/fastvlm-0.5b-captions/
"""

import logging
import torch
import os
from pathlib import Path
from PIL import Image
from typing import Optional, Union, List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available. Install with: pip install transformers>=4.36.0")


class FastVLMModel:
    """
    Wrapper for custom fine-tuned FastVLM-0.5B model.
    
    Handles model loading, image preprocessing, and inference.
    Uses local fine-tuned model from models/fastvlm-0.5b-captions/
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize FastVLM model.
        
        Args:
            model_path: Path to local model directory (default: models/fastvlm-0.5b-captions)
            device: Device to load model on ('cuda', 'mps', 'cpu')
            dtype: Data type for model (default: auto-detect based on device)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for FastVLM. "
                "Install with: pip install transformers>=4.36.0 safetensors"
            )
        
        # Set default model path to local fine-tuned model
        if model_path is None:
            # Try to find the model relative to this file
            current_dir = Path(__file__).parent.parent
            model_path = str(current_dir / "models" / "fastvlm-0.5b-captions")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at: {model_path}\n"
                f"Please ensure your fine-tuned model is in models/fastvlm-0.5b-captions/"
            )
        
        self.model_path = model_path
        self.IMAGE_TOKEN_INDEX = 151646  # From config.json - specific to this model
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
        # Set dtype based on device
        # Note: Model is 4-bit quantized, so dtype mainly affects intermediate computations
        if dtype is None:
            if device in ["cuda", "mps"]:
                dtype = torch.float16
            else:
                dtype = torch.float32
        self.dtype = dtype
        
        logger.info(f"Initializing FastVLM model from: {model_path}")
        logger.info(f"Device: {device}, dtype: {dtype}")
        logger.info("Loading 4-bit quantized model (fine-tuned for captions)")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load processor (includes image preprocessing)
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            logger.info("Loaded image processor")
        except Exception as e:
            logger.warning(f"Could not load processor: {e}. Using manual preprocessing.")
            self.processor = None
        
        # Load model (4-bit quantized)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            # low_cpu_mem_usage=True,  # Helpful for quantized models
        )
        
        # Move to device if not using device_map
        if device != "cuda":
            self.model = self.model.to(device)
        
        self.model.eval()
        logger.info("FastVLM model loaded successfully")
        logger.info(f"Model size: 0.5B parameters (4-bit quantized)")
        logger.info(f"Vision encoder: FastViTHD (3072-dim features)")
    
    def generate_description(
        self,
        image: Union[Image.Image, np.ndarray, str],
        prompt: str = "Describe this image in detail.",
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        do_sample: bool = True,
    ) -> str:
        """
        Generate a description for an image.
        
        Args:
            image: PIL Image, numpy array, or path to image file
            prompt: Text prompt for the model
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            do_sample: Whether to use sampling (vs greedy decoding)
        
        Returns:
            Generated description text
        """
        # Convert image to PIL if needed
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Build chat messages
        messages = [
            {"role": "user", "content": f"<image>\n{prompt}"}
        ]
        
        # Render to string (not tokens) so we can place <image> exactly
        rendered = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Split around the <image> token
        if "<image>" not in rendered:
            logger.warning("No <image> token found in rendered prompt, adding manually")
            rendered = f"<image>\n{rendered}"
        
        pre, post = rendered.split("<image>", 1)
        
        # Tokenize the text *around* the image token (no extra specials!)
        pre_ids = self.tokenizer(
            pre,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids
        
        post_ids = self.tokenizer(
            post,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids
        
        # Splice in the IMAGE token id (-200) at the placeholder position
        img_tok = torch.tensor([[self.IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        
        # Preprocess image via the model's own processor
        px = self.model.get_vision_tower().image_processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"]
        px = px.to(self.device, dtype=self.dtype)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=px,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        # The output typically contains the full conversation, we want just the answer
        if "assistant" in generated_text.lower():
            # Split on common assistant markers
            for marker in ["assistant\n", "assistant:", "Assistant\n", "Assistant:"]:
                if marker in generated_text:
                    generated_text = generated_text.split(marker)[-1].strip()
                    break
        
        return generated_text.strip()
    
    def batch_generate(
        self,
        images: list,
        prompts: Optional[list] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        do_sample: bool = True,
    ) -> list:
        """
        Generate descriptions for multiple images.
        
        Note: Currently processes images sequentially. Batch processing
        could be added for better performance.
        
        Args:
            images: List of PIL Images, numpy arrays, or paths
            prompts: List of prompts (one per image), or None for default
            max_new_tokens: Maximum tokens to generate per image
            temperature: Sampling temperature
            do_sample: Whether to use sampling
        
        Returns:
            List of generated descriptions
        """
        if prompts is None:
            prompts = ["Describe this image in detail."] * len(images)
        
        if len(prompts) != len(images):
            raise ValueError(f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})")
        
        results = []
        for image, prompt in zip(images, prompts):
            try:
                description = self.generate_description(
                    image=image,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                )
                results.append(description)
            except Exception as e:
                logger.error(f"Error generating description: {e}")
                results.append(f"[Error: {str(e)}]")
        
        return results
    
    def __call__(self, image, prompt: str = "Describe this image in detail.", **kwargs):
        """Allow model to be called directly like a function."""
        return self.generate_description(image, prompt, **kwargs)


def load_fastvlm(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> FastVLMModel:
    """
    Convenience function to load FastVLM model.
    
    Args:
        model_path: Path to local model directory (default: models/fastvlm-0.5b-captions)
        device: Device to load on ('cuda', 'mps', 'cpu')
        dtype: Data type for model
    
    Returns:
        Loaded FastVLMModel instance
    """
    return FastVLMModel(model_path=model_path, device=device, dtype=dtype)


# Example usage
if __name__ == "__main__":
    import sys
    
    # Test the model
    print("="*80)
    print("FastVLM Model Test - Custom Fine-tuned 0.5B (4-bit quantized)")
    print("="*80)
    print("\nLoading FastVLM model from: models/fastvlm-0.5b-captions/")
    
    try:
        model = load_fastvlm()
        print("✓ Model loaded successfully!")
        print(f"  Device: {model.device}")
        print(f"  dtype: {model.dtype}")
        print(f"  Image token index: {model.IMAGE_TOKEN_INDEX}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        sys.exit(1)
    
    # Test with an image if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe this image in detail."
        
        print(f"\n{'='*80}")
        print(f"Generating description for: {image_path}")
        print(f"Prompt: {prompt}")
        print(f"{'='*80}\n")
        
        try:
            description = model(image_path, prompt)
            print(f"Description:\n{description}")
            print(f"\n{'='*80}")
        except Exception as e:
            print(f"✗ Error generating description: {e}")
    else:
        print("\n" + "="*80)
        print("Model ready for use!")
        print("="*80)
        print("\nUsage: python fastvlm_wrapper.py <image_path> [prompt]")
        print("\nExample:")
        print("  python fastvlm_wrapper.py data/examples/example1.jpg")
        print("  python fastvlm_wrapper.py image.jpg 'What objects are in this image?'")

