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
    """Generates visual embeddings using selected backend (CLIP or DINO).

    When backend=='clip': retains original CLIP behavior (optionally multimodal).
    When backend=='dino': uses DINO for instance-level embeddings, optionally also
    generating CLIP embeddings for label verification (stored separately).
    """

    def __init__(
        self,
        clip_model,
        config: EmbeddingConfig,
        dino_model: Optional[object] = None,
    ):
        self.clip = clip_model
        self.config = config
        backend = config.backend
        mode = "multimodal" if (backend == "clip" and config.use_text_conditioning) else "vision-only"
        logger.debug(
            f"VisualEmbedder initialized: backend={backend}, dim={config.embedding_dim}, "
            f"mode={mode}, batch_size={config.batch_size}, device={config.device}"
        )
        # DINO model instantiation with device selection
        if backend == "dino":
            # If dino_model is already provided, use it; else, create with device
            if dino_model is not None:
                self.dino = dino_model
            else:
                from orion.backends.dino_backend import DINOBackend
                # Map "auto" to best available
                device = config.device
                import torch
                if device == "auto":
                    if torch.cuda.is_available():
                        device = "cuda"
                    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = "cpu"
                self.dino = DINOBackend(device=device)
    
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
        
        logger.info(f"Generating {self.config.embedding_dim}-dim {self.config.backend.upper()} embeddings...")
        if self.config.backend == "clip":
            logger.info(f"  Mode: {'Multimodal (vision + text)' if self.config.use_text_conditioning else 'Vision only'}")
        else:
            logger.info("  Mode: Vision only (DINO does not support text conditioning)")
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
        backend = self.config.backend
        if backend == "clip":
            embeddings = []
            for detection in batch:
                crop = detection["crop"]
                class_name = detection.get("object_class")
                embedding = self._embed_clip(crop, class_name)
                embeddings.append(embedding)
            return embeddings
        elif backend == "dino":
            crops = [cv2.cvtColor(det["crop"], cv2.COLOR_BGR2RGB) for det in batch]
            embeddings = self.dino.encode_images_batch(crops, normalize=True)
            # Optionally produce auxiliary CLIP embedding for semantic verification
            if self.clip is not None:
                for detection in batch:
                    class_name = detection.get("object_class")
                    detection["clip_embedding"] = self._embed_clip(detection["crop"], class_name)
            return embeddings
        else:
            raise ValueError(f"Unsupported embedding backend: {backend}")
    
    def _embed_clip(self, crop: np.ndarray, class_name: Optional[str]) -> np.ndarray:
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_crop)
        if self.config.use_text_conditioning and class_name:
            embedding = self.clip.encode_multimodal(
                pil_image,
                f"a {class_name}",
                normalize=True,
            )
        else:
            embedding = self.clip.encode_image(
                pil_image,
                normalize=True,
            )
        norm = np.linalg.norm(embedding)
        if norm > 0 and abs(norm - 1.0) > 1e-3:
            embedding = embedding / norm
        return embedding

    def _embed_dino(self, crop: np.ndarray) -> np.ndarray:
        if self.dino is None:
            raise RuntimeError("DINO model requested but not provided to VisualEmbedder")
        # DINO backend expects numpy array; provide RGB array directly
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        embedding = self.dino.encode_image(
            rgb_crop,
            normalize=True,
        )
        norm = np.linalg.norm(embedding)
        if norm > 0 and abs(norm - 1.0) > 1e-3:
            embedding = embedding / norm
        return embedding
