"""
Visual Embedder
===============

Generates CLIP embeddings for detected objects.

Responsibilities:
- Convert cropped regions to CLIP embeddings
- Support multimodal conditioning (image + text)
- Normalize embeddings to unit length
- Batch processing for efficiency

Author: Orion Research Team
Date: October 2025
"""

import logging
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from orion.perception.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class VisualEmbedder:
    """
    Generates visual embeddings using CLIP.
    
    Supports both vision-only and multimodal (vision + text) embedding.
    """
    
    def __init__(
        self,
        clip_model,
        config: EmbeddingConfig,
    ):
        """
        Initialize embedder.
        
        Args:
            clip_model: CLIP model instance from ModelManager
            config: Embedding configuration
        """
        self.clip = clip_model
        self.config = config
        
        mode = "multimodal" if config.use_text_conditioning else "vision-only"
        logger.debug(
            f"VisualEmbedder initialized: dim={config.embedding_dim}, "
            f"mode={mode}, batch_size={config.batch_size}"
        )
    
    def embed_detections(self, detections: List[dict]) -> List[dict]:
        """
        Add embeddings to detections.
        
        Args:
            detections: List of detection dicts from FrameObserver
            
        Returns:
            Same detections with 'embedding' field added
        """
        logger.info("="*80)
        logger.info("PHASE 1B: VISUAL EMBEDDING GENERATION")
        logger.info("="*80)
        
        logger.info(f"Generating {self.config.embedding_dim}-dim CLIP embeddings...")
        logger.info(f"  Mode: {'Multimodal (vision + text)' if self.config.use_text_conditioning else 'Vision only'}")
        logger.info(f"  Processing {len(detections)} detections...")
        
        # Process in batches for efficiency
        batch_size = self.config.batch_size
        total_batches = (len(detections) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(detections))
            batch = detections[start_idx:end_idx]
            
            # Generate embeddings for batch
            embeddings = self._embed_batch(batch)
            
            # Add embeddings to detections
            for detection, embedding in zip(batch, embeddings):
                detection["embedding"] = embedding
            
            if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                logger.info(f"  Processed {end_idx}/{len(detections)} detections")
        
        logger.info(f"✓ Generated {len(detections)} embeddings")
        logger.info("="*80 + "\n")
        
        return detections
    
    def _embed_batch(self, batch: List[dict]) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of detections.
        
        Args:
            batch: List of detection dicts
            
        Returns:
            List of embedding arrays
        """
        embeddings = []
        
        for detection in batch:
            crop = detection["crop"]
            class_name = detection.get("object_class")
            
            embedding = self.embed_crop(crop, class_name)
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_crop(
        self,
        crop: np.ndarray,
        class_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate embedding for a single crop.
        
        Args:
            crop: Cropped image region (BGR format)
            class_name: Optional class name for text conditioning
            
        Returns:
            Normalized embedding vector
        """
        # Convert BGR to RGB
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_crop)
        
        # Generate embedding
        if self.config.use_text_conditioning and class_name:
            # Multimodal: condition on YOLO class
            # This helps catch misclassifications - if image doesn't match class,
            # embedding will be different from other instances of that class
            embedding = self.clip.encode_multimodal(
                pil_image,
                f"a {class_name}",
                normalize=True,
            )
        else:
            # Vision only
            embedding = self.clip.encode_image(
                pil_image,
                normalize=True,
            )
        
        # Ensure normalized
        norm = np.linalg.norm(embedding)
        if norm > 0 and abs(norm - 1.0) > 1e-3:
            embedding = embedding / norm
        
        return embedding
