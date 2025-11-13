"""
Advanced Re-ID System with OSNet and FastVLM embeddings
Provides robust cross-scene re-identification
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import cv2


@dataclass
class TrackGallery:
    """Visual gallery for a track (stores multiple exemplars)"""
    track_id: int
    embeddings: List[np.ndarray]  # List of embeddings (512D or 768D)
    timestamps: List[float]
    max_exemplars: int = 16
    
    def add_embedding(self, emb: np.ndarray, timestamp: float):
        """Add new embedding to gallery"""
        self.embeddings.append(emb)
        self.timestamps.append(timestamp)
        
        # Keep only recent exemplars
        if len(self.embeddings) > self.max_exemplars:
            self.embeddings.pop(0)
            self.timestamps.pop(0)
    
    def get_mean_embedding(self) -> np.ndarray:
        """Get average embedding"""
        if not self.embeddings:
            return np.zeros(512)
        return np.mean(self.embeddings, axis=0)
    
    def compute_similarity(self, query_emb: np.ndarray) -> float:
        """Compute cosine similarity with query"""
        if not self.embeddings:
            return 0.0
        
        # Compare with mean embedding
        mean_emb = self.get_mean_embedding()
        
        # Cosine similarity
        dot = np.dot(mean_emb, query_emb)
        norm_a = np.linalg.norm(mean_emb)
        norm_b = np.linalg.norm(query_emb)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)


class AdvancedReID:
    """
    Advanced Re-ID system combining:
    - OSNet (lightweight person Re-ID)
    - CLIP embeddings (general objects)
    - FastVLM embeddings (semantic-rich)
    """
    
    def __init__(self, 
                 embedding_dim: int = 512,
                 use_osnet: bool = False,  # Optional - requires torchreid
                 use_fastvlm: bool = True,
                 device: str = "mps"):
        """
        Args:
            embedding_dim: Embedding dimension (512 for CLIP/OSNet, 768 for FastVLM)
            use_osnet: Use OSNet for person Re-ID
            use_fastvlm: Use FastVLM for semantic embeddings
            device: 'mps', 'cuda', or 'cpu'
        """
        self.embedding_dim = embedding_dim
        self.device = device
        self.use_osnet = use_osnet
        self.use_fastvlm = use_fastvlm
        
        # Track galleries
        self.galleries: Dict[int, TrackGallery] = {}
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load Re-ID models"""
        # CLIP (always available, fast)
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_available = True
            print("✓ CLIP loaded for Re-ID")
        except:
            self.clip_available = False
            print("⚠️  CLIP not available")
        
        # OSNet (optional, for person Re-ID)
        if self.use_osnet:
            try:
                import torchreid
                self.osnet = torchreid.models.build_model(
                    name='osnet_x0_25',
                    num_classes=1000,
                    pretrained=True
                )
                self.osnet.eval()
                self.osnet.to(self.device)
                self.osnet_available = True
                print("✓ OSNet loaded for person Re-ID")
            except:
                self.osnet_available = False
                print("⚠️  OSNet not available (install: pip install torchreid)")
        else:
            self.osnet_available = False
        
        # FastVLM (optional, semantic-rich)
        if self.use_fastvlm:
            try:
                from orion.managers.model_manager import ModelManager
                mm = ModelManager.get_instance()
                self.fastvlm = mm.get_fastvlm_model()
                self.fastvlm_available = True
                print("✓ FastVLM loaded for semantic Re-ID")
            except:
                self.fastvlm_available = False
                print("⚠️  FastVLM not available")
        else:
            self.fastvlm_available = False
    
    def extract_embedding(self, 
                          image_crop: np.ndarray,
                          class_name: str = "object") -> np.ndarray:
        """
        Extract Re-ID embedding from image crop
        
        Args:
            image_crop: RGB image crop (H, W, 3)
            class_name: Object class (used to select best model)
            
        Returns:
            Embedding vector (512D or 768D)
        """
        # Person Re-ID: use OSNet if available
        if class_name.lower() in {"person", "man", "woman", "child"} and self.osnet_available:
            return self._extract_osnet_embedding(image_crop)
        
        # Semantic Re-ID: use FastVLM if available
        elif self.fastvlm_available:
            return self._extract_fastvlm_embedding(image_crop)
        
        # Fallback: CLIP
        elif self.clip_available:
            return self._extract_clip_embedding(image_crop)
        
        else:
            # No model available
            return np.random.randn(self.embedding_dim)
    
    def _extract_clip_embedding(self, image_crop: np.ndarray) -> np.ndarray:
        """Extract CLIP embedding"""
        # Preprocess
        from PIL import Image
        pil_image = Image.fromarray(image_crop)
        image_tensor = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.clip_model.encode_image(image_tensor)
            features = features.cpu().numpy().flatten()
        
        # Normalize
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    def _extract_osnet_embedding(self, image_crop: np.ndarray) -> np.ndarray:
        """Extract OSNet embedding"""
        # Resize to 256x128 (OSNet input size)
        resized = cv2.resize(image_crop, (128, 256))
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (resized / 255.0 - mean) / std
        
        # To tensor
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.osnet(tensor)
            features = features.cpu().numpy().flatten()
        
        # Normalize
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    def _extract_fastvlm_embedding(self, image_crop: np.ndarray) -> np.ndarray:
        """Extract FastVLM visual embedding"""
        try:
            # FastVLM returns semantic-rich embeddings
            from orion.managers.model_manager import ModelManager
            mm = ModelManager.get_instance()
            embedding = mm.get_visual_embedding(image_crop)
            return embedding
        except:
            # Fallback to CLIP
            return self._extract_clip_embedding(image_crop)
    
    def update_gallery(self, track_id: int, embedding: np.ndarray, timestamp: float):
        """Update track gallery with new embedding"""
        if track_id not in self.galleries:
            self.galleries[track_id] = TrackGallery(track_id=track_id, embeddings=[], timestamps=[])
        
        self.galleries[track_id].add_embedding(embedding, timestamp)
    
    def match_against_gallery(self, 
                              query_embedding: np.ndarray,
                              threshold: float = 0.7) -> Optional[Tuple[int, float]]:
        """
        Match query embedding against all galleries
        
        Args:
            query_embedding: Query embedding
            threshold: Similarity threshold
            
        Returns:
            (track_id, similarity) if match found, None otherwise
        """
        best_match = None
        best_score = -1.0
        
        for track_id, gallery in self.galleries.items():
            similarity = gallery.compute_similarity(query_embedding)
            
            if similarity > best_score:
                best_score = similarity
                best_match = track_id
        
        if best_score >= threshold:
            return (best_match, best_score)
        
        return None
    
    def get_gallery_summary(self) -> Dict:
        """Get summary of all galleries"""
        return {
            "num_tracks": len(self.galleries),
            "total_embeddings": sum(len(g.embeddings) for g in self.galleries.values()),
            "tracks": {
                tid: {
                    "num_embeddings": len(g.embeddings),
                    "mean_embedding": g.get_mean_embedding().tolist()
                }
                for tid, g in self.galleries.items()
            }
        }
