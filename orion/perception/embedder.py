"""
Visual Embedder
===============

Generates embeddings for detected objects (Re-ID backbone).

Supports multiple backends:
- V-JEPA2 (default): 3D-aware, video-native, best for Re-ID
- DINOv2: Public, 2D-only, fast
- DINOv3: Gated access, 3D-aware (if weights available)

Responsibilities:
- Convert cropped regions to embeddings
- Normalize embeddings to unit length
- Batch processing for efficiency
- Backend selection based on config

Author: Orion Research Team
Date: October 2025 (updated January 2026)
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from orion.perception.config import EmbeddingConfig
from orion.utils.profiling import profile

logger = logging.getLogger(__name__)


class VisualEmbedder:
    """Generates visual Re-ID embeddings using configurable backend.

    Supports V-JEPA2 (default), DINOv2 (public), DINOv3 (gated).
    """

    def __init__(
        self,
        clip_model=None,  # Unused; kept for backward compat signature
        config: Optional[EmbeddingConfig] = None,
    ):
        """Initialize visual embedder with configurable backend.
        
        Args:
            clip_model: Unused, kept for backward compatibility
            config: EmbeddingConfig specifying backend and hyperparameters
        """
        self.config = config or EmbeddingConfig()
        self.backend = None
        
        # Resolve device
        import torch
        device = self.config.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = device
        
        # Initialize backend based on config
        self._init_backend(device)
        logger.info(
            f"VisualEmbedder initialized: {self.config.backend} "
            f"(device={device}, dim={self.config.embedding_dim})"
        )
    
    def _init_backend(self, device: str):
        """Initialize embedding backend based on config.backend."""
        if self.config.backend == "vjepa2":
            from orion.backends.vjepa2_backend import VJepa2Embedder
            self.backend = VJepa2Embedder(
                model_name=self.config.model,
                device=device,
            )
            logger.debug("Initialized V-JEPA2 backend (3D-aware video embeddings)")
        
        elif self.config.backend == "dinov2":
            from orion.backends.dino_backend import DINOEmbedder
            self.backend = DINOEmbedder(
                model_name="facebook/dinov2-base",
                device=device,
            )
            logger.debug("Initialized DINOv2 backend (public, visual-only)")
        
        elif self.config.backend == "dinov3":
            from orion.backends.dino_backend import DINOEmbedder
            if not self.config.dinov3_weights_dir:
                raise ValueError(
                    "backend='dinov3' requires dinov3_weights_dir. "
                    "Download from: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/"
                )
            self.backend = DINOEmbedder(
                local_weights_dir=self.config.dinov3_weights_dir,
                device=device,
            )
            logger.debug("Initialized DINOv3 backend (gated access, visual-only)")
        
        else:
            raise ValueError(
                f"Unknown embedding backend: {self.config.backend}. "
                f"Valid options: vjepa2, dinov2, dinov3"
            )
        
        if self.backend is None:
            raise RuntimeError("Failed to initialize embedding backend")

    
    @profile("embedder_embed_detections")
    def embed_detections(self, detections: List[dict]) -> List[dict]:
        """
        Add V-JEPA2 embeddings to detections.
        
        Args:
            detections: List of detection dicts from FrameObserver
            
        Returns:
            Same detections with 'embedding' field added
        """
        logger.info("=" * 80)
        logger.info("PHASE 1B: VISUAL EMBEDDING GENERATION (V-JEPA2)")
        logger.info("=" * 80)
        
        logger.info(f"Generating {self.config.embedding_dim}-dim V-JEPA2 embeddings...")
        logger.info(f"  Processing {len(detections)} detections...")
        
        # Optionally cluster detections per frame before embedding extraction
        if self.config.use_cluster_embeddings:
            detections = self._apply_clustering(detections)

        # Process in batches
        batch_size = self.config.batch_size
        total_batches = (len(detections) + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(detections))
            batch = detections[start_idx:end_idx]
            embeddings = self._embed_batch(batch)
            for detection, embedding in zip(batch, embeddings):
                detection["embedding"] = embedding
            if (batch_idx + 1) % 5 == 0 or batch_idx == total_batches - 1:
                logger.info(f"  Processed {end_idx}/{len(detections)} detections")
        
        logger.info(f"✓ Generated {len(detections)} V-JEPA2 embeddings")
        logger.info("=" * 80 + "\n")
        
        return detections
    
    @profile("embedder_embed_batch")
    def _embed_batch(self, batch: List[dict]) -> List[np.ndarray]:
        """Embed a batch of detection crops using the configured backend.
        
        Args:
            batch: List of detection dicts with 'crop' field
            
        Returns:
            List of normalized embedding vectors
        """
        crops = []
        valid_indices = []
        
        for i, detection in enumerate(batch):
            if 'crop' in detection and detection['crop'] is not None:
                crops.append(detection['crop'])
                valid_indices.append(i)
        
        if not crops:
            # Return zero embeddings if no valid crops
            return [np.zeros(self.config.embedding_dim, dtype=np.float32) for _ in batch]
        
        # V-JEPA2 specific path (supports batch and single methods)
        if self.config.backend == "vjepa2":
            embeddings = []
            for crop in crops:
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                try:
                    embedding_tensor = self.backend.embed_single_image(rgb_crop)
                    embedding = embedding_tensor.numpy().flatten().astype(np.float32)
                    # Normalize
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"V-JEPA2 embedding failed: {e}")
                    embeddings.append(np.zeros(self.config.embedding_dim, dtype=np.float32))
        
        # DINOv2/DINOv3 path (use batch encoding if available)
        else:
            try:
                if hasattr(self.backend, 'encode_images_batch') and len(crops) > 1:
                    # Use batch encoding for efficiency
                    embeddings = self.backend.encode_images_batch(crops, normalize=True)
                else:
                    # Fall back to single encoding
                    embeddings = [self.backend.encode_image(crop, normalize=True) for crop in crops]
                
                # Ensure all embeddings are properly normalized
                embeddings = [
                    (e / np.linalg.norm(e)) if np.linalg.norm(e) > 0 else e
                    for e in embeddings
                ]
            except Exception as e:
                logger.warning(f"{self.config.backend} batch embedding failed: {e}")
                embeddings = [np.zeros(self.config.embedding_dim, dtype=np.float32) for _ in crops]
        
        # Reconstruct full batch with zero embeddings for invalid crops
        result = []
        emb_idx = 0
        for i in range(len(batch)):
            if i in valid_indices:
                result.append(embeddings[emb_idx])
                emb_idx += 1
            else:
                result.append(np.zeros(self.config.embedding_dim, dtype=np.float32))
        
        return result

    # -------------------------------
    # Clustering logic (IoU-based)
    # -------------------------------
    def _apply_clustering(self, detections: List[dict]) -> List[dict]:
        """Merge overlapping detections per frame to reduce embedding cost."""
        thresh = self.config.cluster_similarity_threshold
        clustered: List[dict] = []
        by_frame: dict[int, List[dict]] = {}
        for det in detections:
            by_frame.setdefault(int(det.get("frame_number", 0)), []).append(det)
        for fidx, frame_dets in by_frame.items():
            clusters: List[List[dict]] = []
            for det in frame_dets:
                bb = det.get("bounding_box") or det.get("bbox") or det.get("bbox_2d")
                if bb is None:
                    clusters.append([det])
                    continue
                x1, y1, x2, y2 = bb.x1, bb.y1, bb.x2, bb.y2
                placed = False
                for cluster in clusters:
                    ob = cluster[0].get("bounding_box") or cluster[0].get("bbox") or cluster[0].get("bbox_2d")
                    if ob is None:
                        continue
                    ox1, oy1, ox2, oy2 = ob.x1, ob.y1, ob.x2, ob.y2
                    inter_x1 = max(x1, ox1)
                    inter_y1 = max(y1, oy1)
                    inter_x2 = min(x2, ox2)
                    inter_y2 = min(y2, oy2)
                    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                    area = (x2 - x1) * (y2 - y1)
                    oarea = (ox2 - ox1) * (oy2 - oy1)
                    iou = inter / (area + oarea - inter + 1e-8)
                    if iou >= thresh:
                        cluster.append(det)
                        placed = True
                        break
                if not placed:
                    clusters.append([det])
            # For each cluster create representative detection (first) and mark members
            for cluster in clusters:
                rep = cluster[0]
                rep["cluster_size"] = len(cluster)
                for member in cluster:
                    member["cluster_rep_id"] = id(rep)
                clustered.extend(cluster)
        if clustered:
            avg_size = np.mean([c.get("cluster_size", 1) for c in clustered if "cluster_size" in c])
            logger.info(f"  Clustering reduced duplicates (avg cluster size={avg_size:.2f})")
        return clustered
