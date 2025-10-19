"""
Multimodal Vision Embedding Wrapper
Uses OpenAI's CLIP for multimodal (vision + text) embeddings
EmbeddingGemma is text-only, so we use CLIP for vision+text understanding
"""

import logging
from typing import Optional, Union
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)

try:
    from transformers import CLIPModel, CLIPProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available. Install: pip install transformers")


class EmbeddingGemmaVision:
    """
    CLIP wrapper for visual embeddings with optional text conditioning
    
    This provides better semantic understanding than ResNet50 by:
    1. Multimodal embeddings (vision + language)
    2. Better semantic similarity (not just visual similarity)
    3. Can detect mismatches between image and text
    
    Note: Despite the name, this uses CLIP not EmbeddingGemma because
    EmbeddingGemma is text-only. CLIP is designed for vision+text.
    
    Usage:
        model = EmbeddingGemmaVision()
        
        # Vision only
        embedding = model.encode_image(pil_image)
        
        # Vision + text (better for verifying YOLO classifications)
        embedding = model.encode_multimodal(pil_image, "a bottle")
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None
    ):
        """
        Initialize CLIP model
        
        Args:
            model_name: HuggingFace model name (CLIP variant)
            device: Device to use (None = auto-detect)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required. Install: pip install transformers")
        
        self.model_name = model_name
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading CLIP: {model_name}")
        logger.info(f"Device: {self.device}")
        
        # Load model and processor
        try:
            self.model = CLIPModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
            ).to(self.device)
            
            self.processor = CLIPProcessor.from_pretrained(model_name)
            
            self.model.eval()
            logger.info("✓ CLIP loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            raise
    
    def encode_image(
        self,
        image: Union[Image.Image, np.ndarray],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode image to embedding (vision only)
        
        Args:
            image: PIL Image or numpy array (H, W, C)
            normalize: Whether to L2 normalize the embedding
        
        Returns:
            Embedding vector (numpy array)
        """
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Process image - CLIP processor expects images parameter
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding using vision encoder
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        # Convert to numpy
        embedding = image_features.cpu().numpy()[0]
        
        # Normalize if requested
        if normalize:
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def encode_multimodal(
        self,
        image: Union[Image.Image, np.ndarray],
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode image + text to multimodal embedding
        
        This creates an embedding by combining both image and text features.
        For CLIP, we can compute the similarity between image and text embeddings.
        
        Args:
            image: PIL Image or numpy array
            text: Text to condition on (e.g., YOLO class name)
            normalize: Whether to L2 normalize
        
        Returns:
            Combined image+text embedding vector
        """
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Process both image and text with CLIP
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get both image and text features
        with torch.no_grad():
            image_features = self.model.get_image_features(pixel_values=inputs['pixel_values'])
            text_features = self.model.get_text_features(input_ids=inputs['input_ids'])
            
            # Combine features (average)
            embedding = (image_features + text_features) / 2.0
        
        # Convert to numpy
        embedding = embedding.cpu().numpy()[0]
        
        # Normalize if requested
        if normalize:
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def encode_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Encode text to embedding (for comparison with image embeddings)
        
        Args:
            text: Text to encode
            normalize: Whether to L2 normalize
        
        Returns:
            Text embedding vector
        """
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        
        embedding = text_features.cpu().numpy()[0]
        
        if normalize:
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def verify_classification(
        self,
        image: Union[Image.Image, np.ndarray],
        class_name: str,
        threshold: float = 0.7
    ) -> dict:
        """
        Verify if an image matches a class name
        
        This compares the image embedding with the class name embedding
        to check if they're semantically similar.
        
        Args:
            image: Image to verify
            class_name: Class name from YOLO
            threshold: Similarity threshold (0-1)
        
        Returns:
            dict with 'is_match', 'similarity', 'confidence'
        """
        # Get embeddings
        img_emb = self.encode_image(image, normalize=True)
        text_emb = self.encode_text(f"a {class_name}", normalize=True)
        
        # Compute cosine similarity
        similarity = float(np.dot(img_emb, text_emb))
        
        return {
            'is_match': similarity >= threshold,
            'similarity': similarity,
            'confidence': 'high' if similarity > 0.8 else 'medium' if similarity > 0.6 else 'low',
            'class_name': class_name
        }
    
    def get_embedding_dim(self) -> int:
        """Return embedding dimension"""
        # EmbeddingGemma typically uses 2048 or 768 dim
        # We'll do a quick check
        test_image = Image.new('RGB', (224, 224), color='white')
        test_emb = self.encode_image(test_image)
        return len(test_emb)
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Singleton instance for efficiency
_embedding_gemma_instance: Optional[EmbeddingGemmaVision] = None


def get_embedding_gemma(model_name: str = "openai/clip-vit-base-patch32") -> EmbeddingGemmaVision:
    """
    Get or create singleton CLIP instance (for multimodal vision+text embeddings)
    
    Args:
        model_name: Model to use (default: CLIP ViT-B/32)
    
    Returns:
        EmbeddingGemmaVision instance (actually using CLIP)
    """
    global _embedding_gemma_instance
    
    if _embedding_gemma_instance is None:
        _embedding_gemma_instance = EmbeddingGemmaVision(model_name=model_name)
    
    return _embedding_gemma_instance
