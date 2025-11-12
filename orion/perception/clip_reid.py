#!/usr/bin/env python3
"""
CLIP Embedding Extractor for Re-ID
===================================

Extracts visual embeddings from detected objects for re-identification.
Uses CLIP ViT-B/32 model for robust feature extraction.

Author: Orion Research
Date: November 11, 2025
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Optional
import clip


class CLIPReIDExtractor:
    """
    CLIP-based embedding extractor for object re-identification
    
    Features:
    - Fast ViT-B/32 model (~50ms for batch of 10)
    - 512-D embeddings
    - Robust to viewpoint/lighting changes
    - Normalized for cosine similarity
    """
    
    def __init__(self, device: str = 'mps', model_name: str = 'ViT-B/32'):
        """
        Args:
            device: 'mps', 'cuda', or 'cpu'
            model_name: CLIP model variant
        """
        self.device = device
        
        print(f"  Loading CLIP {model_name} for re-ID...")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        
        print(f"    ✓ CLIP re-ID ready on {device}")
        print(f"    Model: {model_name} (512-D embeddings)")
    
    def extract(self, 
                frame: np.ndarray,
                detections: List[Dict],
                min_size: int = 32) -> List[np.ndarray]:
        """
        Extract CLIP embeddings for all detections
        
        Args:
            frame: RGB frame (H, W, 3)
            detections: List of detection dicts with 'bbox' key
            min_size: Min bbox size to extract (skip tiny objects)
        
        Returns:
            List of 512-D normalized embeddings (one per detection)
        """
        if not detections:
            return []
        
        embeddings = []
        crops = []
        valid_indices = []
        
        # Extract crops
        for i, det in enumerate(detections):
            bbox = det['bbox']  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, bbox)
            
            # Clip to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            # Skip tiny boxes
            if (x2 - x1) < min_size or (y2 - y1) < min_size:
                embeddings.append(None)
                continue
            
            # Crop object
            crop = frame[y1:y2, x1:x2]
            
            # Convert BGR to RGB if needed
            if crop.shape[2] == 3 and np.mean(crop[:, :, 0]) < np.mean(crop[:, :, 2]):
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            crops.append(crop)
            valid_indices.append(i)
        
        if not crops:
            return [None] * len(detections)
        
        # Batch process crops
        with torch.no_grad():
            # Preprocess crops - CLIP expects PIL Images
            from PIL import Image
            images = []
            for crop in crops:
                # Convert numpy array to PIL Image
                crop_pil = Image.fromarray(crop)
                # Apply CLIP preprocessing
                crop_tensor = self.preprocess(crop_pil).unsqueeze(0)
                images.append(crop_tensor)
            
            # Stack and move to device
            images = torch.cat(images, dim=0).to(self.device)
            
            # Extract features
            features = self.model.encode_image(images)
            
            # Normalize for cosine similarity
            features = F.normalize(features, dim=-1)
            
            # Convert to numpy
            features_np = features.cpu().numpy()
        
        # Fill in embeddings
        result = [None] * len(detections)
        for idx, feat in zip(valid_indices, features_np):
            result[idx] = feat
        
        return result
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            emb1, emb2: 512-D embeddings
        
        Returns:
            Similarity in [0, 1] (higher = more similar)
        """
        if emb1 is None or emb2 is None:
            return 0.0
        
        similarity = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
        )
        
        # Clip to [0, 1]
        return float(np.clip(similarity, 0.0, 1.0))
    
    def batch_similarity(self, 
                        query_emb: np.ndarray, 
                        gallery_embs: List[np.ndarray]) -> np.ndarray:
        """
        Compute similarity between query and gallery of embeddings
        
        Args:
            query_emb: Single 512-D embedding
            gallery_embs: List of 512-D embeddings
        
        Returns:
            Array of similarities
        """
        if query_emb is None:
            return np.zeros(len(gallery_embs))
        
        similarities = []
        for gallery_emb in gallery_embs:
            if gallery_emb is None:
                similarities.append(0.0)
            else:
                sim = np.dot(query_emb, gallery_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(gallery_emb) + 1e-8
                )
                similarities.append(float(np.clip(sim, 0.0, 1.0)))
        
        return np.array(similarities)
