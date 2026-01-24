"""
Visual Embedder
===============

Generates V-JEPA2 embeddings for detected objects (Re-ID backbone).

V-JEPA2 is the canonical Re-ID backbone for Orion v2. It provides 3D-aware
video embeddings that handle viewpoint changes better than 2D encoders.

CLIP is used *separately* for candidate-label scoring (open-vocab), not Re-ID.

Responsibilities:
- Convert cropped regions to V-JEPA2 embeddings
- Normalize embeddings to unit length
- Batch processing for efficiency

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
    """Generates visual Re-ID embeddings using configurable backend (V-JEPA2, DINOv2, DINOv3)."""

    def __init__(
        self,
        clip_model=None,  # Unused; kept for backward compat signature
        config: Optional[EmbeddingConfig] = None,
    ):
        self.config = config or EmbeddingConfig()
        self.embedder = None
        
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
        
        # Factory pattern for backend selection
        if self.config.backend == "vjepa2":
            from orion.backends.vjepa2_backend import VJepa2Embedder
            self.embedder = VJepa2Embedder(
                model_name=self.config.model,
                device=device,
            )
        elif self.config.backend == "dinov2":
            from orion.backends.dino_backend import DINOEmbedder
            self.embedder = DINOEmbedder(
                model_name="facebook/dinov2-base",
                device=device,
            )
        elif self.config.backend == "dinov3":
            from orion.backends.dino_backend import DINOEmbedder
            try:
                self.embedder = DINOEmbedder(
                    model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
                    local_weights_dir=self.config.dinov3_weights or None,
                    device=device,
                )
            except Exception as e:
                logger.warning(f"Failed to load DINOv3 ({e}). Falling back to DINOv2.", exc_info=True)
                self.embedder = DINOEmbedder(
                    model_name="facebook/dinov2-base",
                    device=device,
                )
        else:
            raise ValueError(f"Unsupported embedding backend: {self.config.backend}")
        
        logger.info(f"VisualEmbedder initialized: {self.config.backend.upper()} (device={device}, dim={self.config.embedding_dim})")
    
    @profile("embedder_embed_detections")
    def embed_detections(self, detections: List[dict]) -> List[dict]:
        """
        Add embeddings to detections using configured backend.
        
        Args:
            detections: List of detection dicts from FrameObserver
            
        Returns:
            Same detections with 'embedding' field added
        """
        logger.info("=" * 80)
        logger.info(f"PHASE 1B: VISUAL EMBEDDING GENERATION ({self.config.backend.upper()})")
        logger.info("=" * 80)
        
        logger.info(f"Generating {self.config.embedding_dim}-dim {self.config.backend.upper()} embeddings...")
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
        
        logger.info(f"✓ Generated {len(detections)} {self.config.backend.upper()} embeddings")
        logger.info("=" * 80 + "\n")
        
        return detections
    
    @profile("embedder_embed_batch")
    def _embed_batch(self, batch: List[dict]) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of detections using configured backend.
        """
        # Prepare crops
        crops: List[np.ndarray] = []
        valid_indices: List[int] = []
        embeddings: List[np.ndarray] = [None] * len(batch)

        for i, detection in enumerate(batch):
            crop = detection.get("crop")
            if crop is None or crop.size == 0:
                embeddings[i] = np.zeros((self.config.embedding_dim,), dtype=np.float32)
                continue
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            valid_indices.append(i)

        if not crops:
            return embeddings

        # Run batch inference
        try:
            if hasattr(self.embedder, "encode_images_batch"):
                # DINO backend (returns list of np.ndarray)
                batch_embs = self.embedder.encode_images_batch(crops, normalize=True)
            elif hasattr(self.embedder, "embed_batch"):
                # V-JEPA2 backend (returns list of torch.Tensor)
                batch_tensors = self.embedder.embed_batch(crops, batch_size=len(crops))
                batch_embs = [t.cpu().numpy().flatten() for t in batch_tensors]
            else:
                # Fallback for backends without batch support
                batch_embs = [self.embedder.encode_image(c, normalize=True) for c in crops]

            for idx, emb in zip(valid_indices, batch_embs):
                embeddings[idx] = emb.astype(np.float32)

        except Exception as e:
            logger.warning(f"{self.config.backend.upper()} batch embedding failed: {e}")
            for idx in valid_indices:
                embeddings[idx] = np.zeros((self.config.embedding_dim,), dtype=np.float32)

        return embeddings

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
