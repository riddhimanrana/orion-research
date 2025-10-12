"""
FastVLM MLX Wrapper - Uses mlx-vlm for Apple Silicon optimized inference

This wrapper is designed for FastVLM models that have been:
1. Converted to MLX format using the mlx-vlm converter
2. Quantized with MLX's 4-bit quantization
3. Have vision encoder exported to CoreML (.mlpackage)

Requirements:
- mlx-vlm library (patched with FastVLM support)
- Apple Silicon Mac (M1/M2/M3/M4)
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Optional, Union
from PIL import Image
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Fix tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger = logging.getLogger(__name__)


class FastVLMMLXWrapper:
    """
    Wrapper for FastVLM models using mlx-vlm for Apple Silicon.
    
    This uses MLX (Apple's ML framework) instead of PyTorch for:
    - Native Apple Silicon optimization
    - Efficient 4-bit quantization
    - CoreML vision encoder acceleration
    """
    
    def __init__(self, model_path: str):
        """
        Initialize FastVLM with MLX.
        
        Args:
            model_path: Path to MLX-converted FastVLM model directory
                       Should contain: model.safetensors, fastvithd.mlpackage, config.json
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        
        # Check for required files
        required_files = ['model.safetensors', 'config.json']
        for file in required_files:
            if not (self.model_path / file).exists():
                raise ValueError(f"Missing required file: {file} in {model_path}")
        
        # Check for CoreML vision encoder
        mlpackage = list(self.model_path.glob("*.mlpackage"))
        if not mlpackage:
            raise ValueError(f"No .mlpackage (CoreML vision encoder) found in {model_path}")
        
        logger.info(f"Loading FastVLM from {model_path}")
        
        # Import mlx_vlm (must be installed with FastVLM patch)
        try:
            from mlx_vlm import load, generate, apply_chat_template
            self.load_fn = load
            self.generate_fn = generate
            self.apply_chat_template_fn = apply_chat_template
        except ImportError as e:
            raise ImportError(
                "mlx-vlm not found or not patched with FastVLM support!\n"
                "Please install it from the patched version:\n"
                "cd ml-fastvlm/model_export\n"
                "Follow README.md instructions to install mlx-vlm with FastVLM patch"
            ) from e
        
        # Load model and processor (suppress warnings)
        logger.info("Loading model with mlx-vlm...")
        
        # Suppress specific warnings during model loading
        import warnings
        import sys
        import io
        
        # Redirect stderr to suppress C++ extension warnings
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*slow image processor.*')
                warnings.filterwarnings('ignore', message='.*Torch version.*')
                warnings.filterwarnings('ignore', message='.*beta version.*')
                self.model, self.processor = self.load_fn(str(self.model_path))
        finally:
            # Restore stderr
            sys.stderr = old_stderr
        
        # Get config from model
        self.config = self.model.config
        logger.info("✅ FastVLM loaded successfully with MLX!")
    
    def generate_description(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.0
    ) -> str:
        """
        Generate a description for an image.
        
        Args:
            image: Image as file path, PIL Image, or numpy array
            prompt: Text prompt/question about the image
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy/deterministic)
        
        Returns:
            Generated description text
        """
        temp_file = None
        try:
            # Convert image to file path format expected by mlx-vlm
            # mlx-vlm's generate function expects a path string, not pre-loaded image data
            if isinstance(image, (str, Path)):
                image_path = str(image)
            elif isinstance(image, Image.Image):
                # Save temporarily for mlx-vlm
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                image.save(temp_file.name)
                image_path = temp_file.name
                temp_file.close()
            elif isinstance(image, np.ndarray):
                # Convert numpy to PIL, then save
                pil_image = Image.fromarray(image.astype('uint8'))
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                pil_image.save(temp_file.name)
                image_path = temp_file.name
                temp_file.close()
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Apply chat template to wrap prompt in conversation format
            # This is CRITICAL - mlx-vlm requires proper formatting with image tokens
            formatted_prompt = self.apply_chat_template_fn(
                processor=self.processor,
                config=self.config,
                prompt=prompt,
                num_images=1  # Single image
            )
            
            # Generate description (suppress deprecation warnings)
            # Pass image path directly - mlx-vlm will handle loading and preprocessing
            import warnings
            import sys
            import io
            
            # Redirect stderr to suppress MLX deprecation warnings
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*get_peak_memory.*')
                    output = self.generate_fn(
                        model=self.model,
                        processor=self.processor,
                        image=image_path,  # Pass path, not loaded data
                        prompt=formatted_prompt,  # Use formatted prompt
                        max_tokens=max_tokens,
                        temp=temperature,
                        verbose=False  # Set to True for debugging
                    )
            finally:
                # Restore stderr
                sys.stderr = old_stderr
            
            return output
            
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            raise
        finally:
            # Clean up temporary file if created
            if temp_file is not None:
                try:
                    import os
                    os.unlink(temp_file.name)
                except:
                    pass
    
    def batch_generate(
        self,
        images: list,
        prompts: list[str],
        max_tokens: int = 100,
        temperature: float = 0.0
    ) -> list[str]:
        """
        Generate descriptions for multiple images.
        
        Args:
            images: List of images (paths, PIL Images, or numpy arrays)
            prompts: List of prompts (one per image)
            max_tokens: Maximum tokens to generate per image
            temperature: Sampling temperature
        
        Returns:
            List of generated descriptions
        """
        if len(images) != len(prompts):
            raise ValueError("Number of images must match number of prompts")
        
        descriptions = []
        for image, prompt in zip(images, prompts):
            desc = self.generate_description(
                image=image,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            descriptions.append(desc)
        
        return descriptions


def main():
    """Test the MLX wrapper"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test FastVLM MLX wrapper")
    parser.add_argument("--model-path", required=True, help="Path to MLX-converted FastVLM model")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--prompt", default="Describe this image.", help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    
    args = parser.parse_args()
    
    # Initialize wrapper
    print(f"Loading FastVLM from {args.model_path}...")
    wrapper = FastVLMMLXWrapper(args.model_path)
    
    # Generate description
    print(f"\nGenerating description for {args.image}")
    print(f"Prompt: {args.prompt}\n")
    
    description = wrapper.generate_description(
        image=args.image,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    print("=" * 60)
    print("DESCRIPTION:")
    print("=" * 60)
    print(description)
    print("=" * 60)


if __name__ == "__main__":
    main()
