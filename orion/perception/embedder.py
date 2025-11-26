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
from orion.utils.profiling import profile

logger = logging.getLogger(__name__)


class VisualEmbedder:
    """Generates visual embeddings using selected backend (CLIP / DINO / DINOv3).

    Backends:
        - clip: CLIP embeddings (optionally conditioned on class text)
        - dino: legacy DINO image embeddings (instance-level)
        - dinov3: video-capable DINOv3 (stub: spatial map simulated, pooled per bbox)

    Cluster Embeddings:
        If `config.use_cluster_embeddings` is True, detections are merged per frame
        by IoU >= cluster_similarity_threshold and only a single embedding is computed
        per cluster. All detections in the cluster share that embedding reference.
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
        if backend in {"dino", "dinov3"}:
            # If model supplied reuse, else create
            if dino_model is not None:
                self.dino = dino_model
            else:
                from orion.backends.dino_backend import DINOEmbedder
                device = config.device
                import torch
                if device == "auto":
                    if torch.cuda.is_available():
                        device = "cuda"
                    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = "cpu"
                # Choose default model name
                model_name = "facebook/dinov2-base" if backend == "dino" else "facebook/dinov2-base"  # placeholder for dinov3 local
                self.dino = DINOEmbedder(model_name=model_name, device=device)
    
    @profile("embedder_embed_detections")
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
        
        # Optionally cluster detections per frame before embedding extraction
        if self.config.use_cluster_embeddings:
            detections = self._apply_clustering(detections)

        # Backend-specific embedding strategy
        if self.config.backend == "dinov3":
            self._embed_dinov3_video(detections)
        else:
            # Process in batches for efficiency for clip/dino
            batch_size = self.config.batch_size
            total_batches = (len(detections) + batch_size - 1) // batch_size
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(detections))
                batch = detections[start_idx:end_idx]
                embeddings = self._embed_batch(batch)
                for detection, embedding in zip(batch, embeddings):
                    detection["embedding"] = embedding
                if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                    logger.info(f"  Processed {end_idx}/{len(detections)} detections")
        
        logger.info(f"✓ Generated {len(detections)} embeddings")
        logger.info("="*80 + "\n")
        
        return detections
    
    @profile("embedder_embed_batch")
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
        elif backend == "dinov3":
            # Should not be called; handled by _embed_dinov3_video
            return [np.zeros((self.config.embedding_dim,), dtype=np.float32) for _ in batch]
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

    # -------------------------------
    # Clustering logic (IoU-based)
    # -------------------------------
    def _apply_clustering(self, detections: List[dict]) -> List[dict]:
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
                    clusters.append([det]); continue
                x1,y1,x2,y2 = bb.x1, bb.y1, bb.x2, bb.y2
                placed = False
                for cluster in clusters:
                    # IoU with first box of cluster
                    ob = cluster[0].get("bounding_box") or cluster[0].get("bbox") or cluster[0].get("bbox_2d")
                    ox1,oy1,ox2,oy2 = ob.x1, ob.y1, ob.x2, ob.y2
                    inter_x1 = max(x1, ox1)
                    inter_y1 = max(y1, oy1)
                    inter_x2 = min(x2, ox2)
                    inter_y2 = min(y2, oy2)
                    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                    area = (x2 - x1) * (y2 - y1)
                    oarea = (ox2 - ox1) * (oy2 - oy1)
                    iou = inter / (area + oarea - inter + 1e-8)
                    if iou >= thresh:
                        cluster.append(det); placed = True; break
                if not placed:
                    clusters.append([det])
            # For each cluster create representative detection (first) and mark members
            for cluster in clusters:
                rep = cluster[0]
                rep["cluster_size"] = len(cluster)
                for member in cluster:
                    member["cluster_rep_id"] = id(rep)
                clustered.extend(cluster)
        logger.info(f"  Clustering reduced duplicates (avg cluster size={np.mean([c['cluster_size'] for c in clustered if 'cluster_size' in c]) if clustered else 1:.2f})")
        return clustered

    # ---------------------------------
    # DINOv3 video embedding extraction
    # ---------------------------------
    def _embed_dinov3_video(self, detections: List[dict]) -> None:
        # Group by frame; for each frame extract feature map once
        by_frame: dict[int, List[dict]] = {}
        for det in detections:
            by_frame.setdefault(int(det.get("frame_number", 0)), []).append(det)
        for fidx, frame_dets in by_frame.items():
            # Assume all share same original frame reference via 'frame_width/height'
            if not frame_dets:
                continue
            # We don't have original full frame cached; fallback: use crop of first detection
            # In full implementation pass original frame
            frame_proxy = frame_dets[0]["crop"]
            feature_map = self.dino.extract_frame_features(frame_proxy)
            H = frame_dets[0].get("frame_height", frame_proxy.shape[0])
            W = frame_dets[0].get("frame_width", frame_proxy.shape[1])
            for det in frame_dets:
                bb = det.get("bounding_box") or det.get("bbox") or det.get("bbox_2d")
                if bb is None:
                    emb = feature_map.mean(axis=(0,1))
                else:
                    emb = self.dino.pool_region(feature_map, (bb.x1, bb.y1, bb.x2, bb.y2), (H,W))
                det["embedding"] = emb
                if self.clip is not None:
                    class_name = det.get("object_class")
                    det["clip_embedding"] = self._embed_clip(det["crop"], class_name)

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
