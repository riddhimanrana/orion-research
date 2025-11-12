"""
Lightweight appearance feature extractor for Re-ID.

Supports CLIP (fast, semantic-rich) for real-time tracking.
FastVLM can be added later for even better cross-scene Re-ID.
"""

import numpy as np
import torch
from typing import List, Optional
import cv2
from PIL import Image  # NEW: Required for image processor


class AppearanceExtractor:
    """
    Extract appearance embeddings for object re-identification.
    
    Uses CLIP by default (fast, semantic, 512-dim).
    Can be upgraded to FastVLM for better cross-scene Re-ID.
    
    Performance: ~2ms per crop on M1 Mac
    """
    
    def __init__(
        self,
        model_name: str = "clip",  # "clip" or "fastvlm"
        device: str = "mps",  # M1 Mac acceleration
        cache_size: int = 1000,  # Cache embeddings for speed
    ):
        self.model_name = model_name
        self.device = device
        self.cache = {}  # crop_hash -> embedding
        self.cache_size = cache_size
        self.fallback_mode = False
        
        if model_name == "clip":
            self._init_clip()
        elif model_name == "fastvlm":
            self._init_fastvlm()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _init_clip(self):
        """Initialize CLIP model."""
        try:
            from transformers import CLIPModel, CLIPProcessor
            
            self.model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.model.eval()
            
            print(f"  ✓ CLIP loaded for Re-ID ({self.device})")
        except Exception as e:
            print(f"  ⚠️  CLIP loading failed: {e}")
            self.model = None
    
    def _init_fastvlm(self):
        """Initialize FastVLM model (MLX-optimized for M1)."""
        try:
            import mlx.core as mx
            from mlx_vlm import load
            
            model_path = "models/fastvlm-0.5b-mlx"
            self.model, self.processor = load(model_path)
            
            print(f"  ✓ FastVLM loaded for Re-ID (MLX-optimized)")
            self.fallback_mode = False
        except Exception as e:
            print(f"  ⚠️  FastVLM loading failed: {e}")
            print(f"  → Falling back to CLIP...")
            self.fallback_mode = True
            self._init_clip()  # Fallback to CLIP
    
    def extract_batch(
        self, crops: List[np.ndarray], use_cache: bool = True
    ) -> List[np.ndarray]:
        """
        Extract appearance embeddings for batch of crops.
        
        Args:
            crops: List of RGB crops (H, W, 3) uint8
            use_cache: Whether to use cached embeddings
        
        Returns:
            List of embeddings (normalized vectors)
        """
        if self.model is None:
            # Return dummy embeddings if model failed to load
            return [np.zeros(512) for _ in crops]
        
        embeddings = []
        
        for crop in crops:
            # Check cache
            crop_hash = self._hash_crop(crop) if use_cache else None
            if crop_hash and crop_hash in self.cache:
                embeddings.append(self.cache[crop_hash])
                continue
            
            # Extract embedding
            if self.model_name == "clip" or self.fallback_mode:
                emb = self._extract_clip(crop)
            elif self.model_name == "fastvlm":
                emb = self._extract_fastvlm(crop)
            else:
                emb = np.zeros(512)  # Fallback
            
            # Normalize
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            
            # Cache
            if crop_hash:
                self.cache[crop_hash] = emb
                if len(self.cache) > self.cache_size:
                    # Simple FIFO eviction
                    self.cache.pop(next(iter(self.cache)))
            
            embeddings.append(emb)
        
        return embeddings
    
    def _extract_clip(self, crop):
        """Extract CLIP embedding."""
        from typing import Any
        
        # Resize and normalize
        crop_resized = cv2.resize(crop, (224, 224))
        crop_pil = Image.fromarray(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))
        
        # Process - use PIL image for processor
        inputs = self.processor(images=crop_pil, return_tensors="pt")  # type: ignore
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)  # type: ignore
        
        # Convert to numpy
        emb: Any = outputs.cpu().numpy()[0]
        return emb
    
    def _extract_fastvlm(self, crop):
        """Extract FastVLM embedding.
        
        NOTE: FastVLM processor expects (images, text) pairs for VLM tasks.
        For pure image embedding extraction, we need to adapt this or use CLIP instead.
        """
        # FastVLM is designed for VLM (vision-language) tasks, not pure embeddings
        # Fallback to CLIP for appearance extraction
        if not hasattr(self, 'clip_fallback_initialized'):
            print("  ⚠️  FastVLM requires text input for VLM tasks")
            print("  → Using CLIP for appearance extraction")
            self._init_clip()
            self.clip_fallback_initialized = True
        
        return self._extract_clip(crop)
    
    def _hash_crop(self, crop: np.ndarray) -> int:
        """Fast hash of crop for caching."""
        # Use shape + center pixels as hash
        h, w = crop.shape[:2]
        center = crop[h//2-2:h//2+2, w//2-2:w//2+2]
        return hash((crop.shape, tuple(center.flatten())))
    
    def extract_single(self, crop: np.ndarray) -> np.ndarray:
        """Extract embedding for single crop."""
        return self.extract_batch([crop])[0]
    
    def get_statistics(self) -> dict:
        """Get extractor statistics."""
        return {
            'model': self.model_name,
            'cache_size': len(self.cache),
            'device': self.device,
        }
