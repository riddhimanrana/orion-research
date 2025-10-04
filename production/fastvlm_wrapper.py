"""
FastVLM Wrapper for Orion Research Project
Uses official Apple FastVLM-0.5B from HuggingFace Hub

This module provides a clean interface to the FastVLM model for generating
rich descriptions of video frames in the perception engine.
"""

import logging
import torch
from PIL import Image
from typing import Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available. Install with: pip install transformers")


class FastVLMModel:
    """
    Wrapper for Apple FastVLM-0.5B model.
    
    Handles model loading, image preprocessing, and inference.
    """
    
    def __init__(
        self,
        model_id: str = "apple/FastVLM-0.5B",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize FastVLM model.
        
        Args:
            model_id: HuggingFace model ID
            device: Device to load model on ('cuda', 'mps', 'cpu')
            dtype: Data type for model (default: float16 on GPU, float32 on CPU)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for FastVLM. "
                "Install with: pip install transformers>=4.36.0"
            )
        
        self.model_id = model_id
        self.IMAGE_TOKEN_INDEX = -200  # Special token for image placeholder
        
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
        if dtype is None:
            if device in ["cuda", "mps"]:
                dtype = torch.float16
            else:
                dtype = torch.float32
        self.dtype = dtype
        
        logger.info(f"Initializing FastVLM model: {model_id}")
        logger.info(f"Device: {device}, dtype: {dtype}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        
        # Move to device if not using device_map
        if device != "cuda":
            self.model = self.model.to(device)
        
        self.model.eval()
        logger.info("FastVLM model loaded successfully")
    
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
    model_id: str = "apple/FastVLM-0.5B",
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> FastVLMModel:
    """
    Convenience function to load FastVLM model.
    
    Args:
        model_id: HuggingFace model ID
        device: Device to load on ('cuda', 'mps', 'cpu')
        dtype: Data type for model
    
    Returns:
        Loaded FastVLMModel instance
    """
    return FastVLMModel(model_id=model_id, device=device, dtype=dtype)


# Example usage
if __name__ == "__main__":
    import sys
    
    # Test the model
    print("Loading FastVLM model...")
    model = load_fastvlm()
    
    # Test with an image if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe this image in detail."
        
        print(f"\nGenerating description for: {image_path}")
        print(f"Prompt: {prompt}")
        
        description = model(image_path, prompt)
        print(f"\nDescription:\n{description}")
    else:
        print("\nModel loaded successfully!")
        print("Usage: python fastvlm_wrapper.py <image_path> [prompt]")
