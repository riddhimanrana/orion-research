"""
Appearance Embedder for Re-ID
==============================

Extracts CNN features from detection crops for person/object re-identification.
Supports multiple backends: CLIP, FastVLM, or general CNN embeddings.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class AppearanceEmbedder:
    """Extract appearance embeddings from detection crops"""
    
    def __init__(
        self,
        backend: str = "clip",
        model_name: str = "ViT-B/32",
        device: str = "mps",
        embedding_dim: int = 512
    ):
        """
        Initialize appearance embedder.
        
        Args:
            backend: "clip", "fastvlm", or "resnet"
            model_name: Model identifier (e.g., "ViT-B/32" for CLIP)
            device: "mps", "cuda", or "cpu"
            embedding_dim: Output embedding dimension
        """
        self.backend = backend
        self.model_name = model_name
        self.device = device
        self.embedding_dim = embedding_dim
        self.model = None
        self.preprocess = None
        
        logger.info(f"Initializing AppearanceEmbedder ({backend}/{model_name})")
        self._load_model()
    
    def _load_model(self):
        """Lazy load embedding model based on backend"""
        try:
            if self.backend == "clip":
                self._load_clip()
            elif self.backend == "fastvlm":
                self._load_fastvlm()
            elif self.backend == "resnet":
                self._load_resnet()
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
            
            logger.info(f"✅ {self.backend} model loaded")
        except Exception as e:
            logger.warning(f"Failed to load {self.backend}: {e}")
            logger.info("Falling back to dummy embedder")
            self.model = None
    
    def _load_clip(self):
        """Load OpenAI CLIP model"""
        try:
            import clip
            device_map = "cuda" if self.device == "cuda" else "cpu"
            self.model, self.preprocess = clip.load(self.model_name, device=device_map)
            self.model.eval()
        except ImportError:
            raise RuntimeError("CLIP not installed. Install with: pip install openai-clip")
    
    def _load_fastvlm(self):
        """Load FastVLM model from local repository"""
        try:
            # Attempt to load from existing FastVLM in models/
            from pathlib import Path
            fastvlm_path = Path("mlx-vlm")
            
            if not fastvlm_path.exists():
                logger.warning("FastVLM not found at mlx-vlm/")
                raise RuntimeError("FastVLM path not found")
            
            # For now, use a simple feature extractor as placeholder
            # Full FastVLM integration would go here
            logger.info("FastVLM integration: using placeholder")
            self.model = "fastvlm_placeholder"
            self.embedding_dim = 512
        except Exception as e:
            raise RuntimeError(f"FastVLM loading failed: {e}")
    
    def _load_resnet(self):
        """Load ResNet50 for appearance features"""
        try:
            import torchvision.models as models
            import torch
            
            # Load pretrained ResNet50
            self.model = models.resnet50(pretrained=True)
            # Remove classification head, keep features
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.to(self.device)
            self.model.eval()
            self.embedding_dim = 2048
        except ImportError:
            raise RuntimeError("PyTorch/torchvision not installed")
    
    def extract(
        self,
        frame: np.ndarray,
        bbox: dict,
        pad: float = 0.1
    ) -> np.ndarray:
        """
        Extract appearance embedding from detection crop.
        
        Args:
            frame: Input frame (BGR)
            bbox: Detection bbox {"x1", "y1", "x2", "y2"} in normalized coords
            pad: Padding around bbox (0.1 = 10% larger crop)
            
        Returns:
            embedding: Appearance vector (embedding_dim,) float32
        """
        if self.model is None:
            # Return dummy embedding if model not loaded
            return self._dummy_embedding()
        
        h, w = frame.shape[:2]
        
        # Convert normalized to pixel coords
        x1_px = int(bbox["x1"] * w)
        y1_px = int(bbox["y1"] * h)
        x2_px = int(bbox["x2"] * w)
        y2_px = int(bbox["y2"] * h)
        
        # Add padding
        box_w = x2_px - x1_px
        box_h = y2_px - y1_px
        x1_px = max(0, int(x1_px - pad * box_w))
        y1_px = max(0, int(y1_px - pad * box_h))
        x2_px = min(w, int(x2_px + pad * box_w))
        y2_px = min(h, int(y2_px + pad * box_h))
        
        # Crop
        crop = frame[y1_px:y2_px, x1_px:x2_px].copy()
        
        if crop.size == 0:
            return self._dummy_embedding()
        
        # Extract embedding based on backend
        if self.backend == "clip":
            return self._extract_clip(crop)
        elif self.backend == "fastvlm":
            return self._extract_fastvlm(crop)
        elif self.backend == "resnet":
            return self._extract_resnet(crop)
        else:
            return self._dummy_embedding()
    
    def _extract_clip(self, crop: np.ndarray) -> np.ndarray:
        """Extract CLIP embedding"""
        try:
            import torch
            
            # Preprocess and normalize
            image_tensor = self.preprocess(crop).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
                embedding = embedding.cpu().numpy().flatten().astype(np.float32)
            
            return embedding
        except Exception as e:
            logger.warning(f"CLIP extraction failed: {e}")
            return self._dummy_embedding()
    
    def _extract_fastvlm(self, crop: np.ndarray) -> np.ndarray:
        """Extract FastVLM embedding (placeholder)"""
        # Placeholder: for now just use simple color histogram
        # Full FastVLM would encode semantic understanding of the crop
        hist_b = cv2.calcHist([crop], [0], None, [64], [0, 256]).flatten()
        hist_g = cv2.calcHist([crop], [1], None, [64], [0, 256]).flatten()
        hist_r = cv2.calcHist([crop], [2], None, [64], [0, 256]).flatten()
        
        embedding = np.concatenate([hist_b, hist_g, hist_r]).astype(np.float32)
        # Normalize to 512-dim
        if len(embedding) < 512:
            embedding = np.pad(embedding, (0, 512 - len(embedding)), mode='constant')
        else:
            embedding = embedding[:512]
        
        embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
        return embedding
    
    def _extract_resnet(self, crop: np.ndarray) -> np.ndarray:
        """Extract ResNet50 embedding"""
        try:
            import torch
            import torchvision.transforms as transforms
            
            # Preprocess
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
            
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            from PIL import Image
            image_tensor = transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(image_tensor)
                embedding = embedding.squeeze().cpu().numpy().astype(np.float32)
            
            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
            return embedding
        except Exception as e:
            logger.warning(f"ResNet extraction failed: {e}")
            return self._dummy_embedding()
    
    def _dummy_embedding(self) -> np.ndarray:
        """Return dummy embedding when model not available"""
        return np.random.randn(self.embedding_dim).astype(np.float32)
    
    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1, emb2: Embedding vectors
            
        Returns:
            similarity: Score in [-1, 1], higher = more similar
        """
        # Ensure normalized
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-6)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-6)
        
        return float(np.dot(emb1, emb2))


class EmbeddingMatcher:
    """Match detections across frames using embeddings + spatial proximity"""
    
    def __init__(
        self,
        similarity_threshold: float = 0.5,
        spatial_threshold: float = 0.3
    ):
        """
        Initialize matcher.
        
        Args:
            similarity_threshold: Min cosine similarity for match (0-1)
            spatial_threshold: Max 3D distance for match (meters)
        """
        self.similarity_threshold = similarity_threshold
        self.spatial_threshold = spatial_threshold
    
    def match(
        self,
        prev_dets: List[dict],
        curr_dets: List[dict]
    ) -> List[Tuple[int, int]]:
        """
        Match detections between frames.
        
        Args:
            prev_dets: Detections from previous frame
            curr_dets: Detections from current frame
            
        Returns:
            matches: List of (prev_idx, curr_idx) pairs
        """
        matches = []
        
        if not prev_dets or not curr_dets:
            return matches
        
        # Compute similarity matrix
        sim_matrix = np.zeros((len(prev_dets), len(curr_dets)))
        
        for i, prev_det in enumerate(prev_dets):
            for j, curr_det in enumerate(curr_dets):
                # Class must match
                if prev_det.get("class") != curr_det.get("class"):
                    continue
                
                # Embedding similarity
                if "embedding" in prev_det and "embedding" in curr_det:
                    emb_sim = float(np.dot(
                        prev_det["embedding"],
                        curr_det["embedding"]
                    ))
                else:
                    emb_sim = 0.5  # Neutral if no embeddings
                
                # Spatial distance (3D centroid)
                if "bbox_3d" in prev_det and "bbox_3d" in curr_det:
                    prev_pos = np.array(prev_det["bbox_3d"]["centroid_3d"])
                    curr_pos = np.array(curr_det["bbox_3d"]["centroid_3d"])
                    distance = float(np.linalg.norm(prev_pos - curr_pos))
                    
                    # Distance penalty: closer = higher score
                    spatial_score = 1.0 / (1.0 + distance / self.spatial_threshold)
                else:
                    spatial_score = 0.5  # Neutral if no 3D
                
                # Combined score (weighted average)
                combined_score = 0.7 * emb_sim + 0.3 * spatial_score
                sim_matrix[i, j] = combined_score
        
        # Hungarian matching (greedy for simplicity)
        used_j = set()
        for i in range(len(prev_dets)):
            best_j = None
            best_score = self.similarity_threshold
            
            for j in range(len(curr_dets)):
                if j not in used_j and sim_matrix[i, j] > best_score:
                    best_j = j
                    best_score = sim_matrix[i, j]
            
            if best_j is not None:
                matches.append((i, best_j))
                used_j.add(best_j)
        
        return matches
