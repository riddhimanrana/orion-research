"""
Rich Semantic Captioning for Orion
==================================

Provides detailed object and scene descriptions beyond basic COCO classes:
- Object attributes: "black laptop", "red hardcover book", "wooden desk"
- Scene understanding: "modern kitchen", "home office", "living room"
- Spatial context: "book on table", "laptop near mouse"

Methods:
1. FastVLM: Natural language captions (best quality, slower)
2. CLIP with custom labels: Match against vocabulary (fast)
3. Hybrid: CLIP for filtering + FastVLM for details

Author: Orion Research Team
Date: November 10, 2025
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import cv2

# Try to import FastVLM
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mlx-vlm"))
    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_image
    FASTVLM_AVAILABLE = True
except ImportError:
    FASTVLM_AVAILABLE = False
    print("⚠️  FastVLM not available - install mlx-vlm")


@dataclass
class RichCaption:
    """Rich semantic caption for an object or scene"""
    object_id: int
    basic_class: str  # COCO class ("keyboard")
    detailed_desc: str  # Rich description ("black mechanical keyboard")
    attributes: List[str]  # ["black", "mechanical"]
    confidence: float
    method: str  # "fastvlm", "clip", "hybrid"


class RichSemanticCaptioner:
    """
    Generate rich semantic captions for objects and scenes
    """
    
    def __init__(self, 
                 use_fastvlm: bool = True,
                 fastvlm_model: str = "mlx-community/Qwen2.5-VL-0.5B-Instruct-4bit",
                 custom_labels: Optional[List[str]] = None):
        """
        Args:
            use_fastvlm: Use FastVLM for natural language captions
            fastvlm_model: MLX-VLM model to use
            custom_labels: Custom vocabulary for CLIP matching
        """
        self.use_fastvlm = use_fastvlm and FASTVLM_AVAILABLE
        self.fastvlm_model = None
        self.fastvlm_processor = None
        
        # Load FastVLM if requested
        if self.use_fastvlm:
            try:
                print(f"Loading FastVLM model: {fastvlm_model}")
                self.fastvlm_model, self.fastvlm_processor = load(fastvlm_model)
                print("✓ FastVLM loaded successfully")
            except Exception as e:
                print(f"⚠️  Could not load FastVLM: {e}")
                print("   Falling back to CLIP")
                self.use_fastvlm = False
        
        # Custom vocabulary for CLIP matching
        self.custom_labels = custom_labels or self._default_vocabulary()
        
        # Object attribute templates
        self.color_attrs = [
            "black", "white", "red", "blue", "green", "yellow", 
            "brown", "gray", "silver", "wooden", "metal"
        ]
        
        self.material_attrs = [
            "wooden", "metal", "plastic", "glass", "leather",
            "fabric", "ceramic", "paper"
        ]
        
        self.size_attrs = [
            "small", "large", "tiny", "big", "compact", "full-size"
        ]
        
        # Scene/room types
        self.room_types = [
            "living room", "bedroom", "kitchen", "office", "bathroom",
            "dining room", "hallway", "garage", "study", "workspace",
            "home office", "modern kitchen", "cozy bedroom"
        ]
    
    def _default_vocabulary(self) -> List[str]:
        """Default rich vocabulary"""
        return [
            # Common objects with attributes
            "black laptop", "silver laptop", "white laptop",
            "mechanical keyboard", "wireless keyboard", "black keyboard",
            "computer mouse", "wireless mouse", "black mouse",
            "hardcover book", "paperback book", "textbook", "novel",
            "water bottle", "coffee mug", "glass cup",
            "smartphone", "tablet", "monitor", "screen",
            
            # Furniture
            "wooden desk", "office desk", "computer desk",
            "office chair", "desk chair", "swivel chair",
            "bookshelf", "cabinet", "drawer",
            "dining table", "coffee table", "side table",
            
            # Room/scene types
            "modern office", "home office", "workspace",
            "living room", "bedroom", "kitchen",
            "dining room", "study room", "hallway",
            
            # Actions/states
            "open laptop", "closed laptop",
            "book on table", "laptop on desk",
            "person sitting", "person standing",
        ]
    
    def caption_object(self, 
                      crop: np.ndarray,
                      basic_class: str,
                      object_id: int,
                      use_clip_fallback: bool = True) -> RichCaption:
        """
        Generate rich caption for a single object crop
        
        Args:
            crop: RGB image crop of object
            basic_class: COCO class name
            object_id: Object ID
            use_clip_fallback: Use CLIP if FastVLM fails
            
        Returns:
            RichCaption with detailed description
        """
        if self.use_fastvlm and self.fastvlm_model is not None:
            try:
                # Use FastVLM for natural language caption
                caption = self._caption_with_fastvlm(crop, basic_class)
                return RichCaption(
                    object_id=object_id,
                    basic_class=basic_class,
                    detailed_desc=caption,
                    attributes=self._extract_attributes(caption),
                    confidence=0.9,
                    method="fastvlm"
                )
            except Exception as e:
                print(f"⚠️  FastVLM failed: {e}")
                if not use_clip_fallback:
                    return self._fallback_caption(object_id, basic_class)
        
        # Fallback to CLIP matching
        return self._caption_with_clip(crop, basic_class, object_id)
    
    def _caption_with_fastvlm(self, crop: np.ndarray, basic_class: str) -> str:
        """Generate caption using FastVLM"""
        # Convert numpy to PIL for FastVLM
        from PIL import Image
        pil_image = Image.fromarray(crop)
        
        # Prompt for detailed description
        prompt = f"Describe this {basic_class} in detail. Include color, material, and distinctive features. Be concise (1 sentence)."
        
        # Generate caption
        response = generate(
            self.fastvlm_model,
            self.fastvlm_processor,
            pil_image,
            prompt,
            max_tokens=50,
            temperature=0.3
        )
        
        return response.strip()
    
    def _caption_with_clip(self, 
                          crop: np.ndarray, 
                          basic_class: str,
                          object_id: int) -> RichCaption:
        """Generate caption using CLIP matching against vocabulary"""
        # Match against custom labels that include this class
        relevant_labels = [
            label for label in self.custom_labels 
            if basic_class.lower() in label.lower()
        ]
        
        if not relevant_labels:
            return self._fallback_caption(object_id, basic_class)
        
        # Use CLIP to find best match
        # (Would integrate with ModelManager.clip here)
        # For now, return enhanced version
        
        # Simple heuristic: pick most specific label
        best_label = max(relevant_labels, key=len)
        
        return RichCaption(
            object_id=object_id,
            basic_class=basic_class,
            detailed_desc=best_label,
            attributes=best_label.replace(basic_class, "").strip().split(),
            confidence=0.7,
            method="clip"
        )
    
    def _fallback_caption(self, object_id: int, basic_class: str) -> RichCaption:
        """Fallback to basic COCO class"""
        return RichCaption(
            object_id=object_id,
            basic_class=basic_class,
            detailed_desc=basic_class,
            attributes=[],
            confidence=0.5,
            method="fallback"
        )
    
    def _extract_attributes(self, caption: str) -> List[str]:
        """Extract attributes from caption"""
        caption_lower = caption.lower()
        attributes = []
        
        # Extract colors
        for color in self.color_attrs:
            if color in caption_lower:
                attributes.append(color)
        
        # Extract materials
        for material in self.material_attrs:
            if material in caption_lower:
                attributes.append(material)
        
        # Extract sizes
        for size in self.size_attrs:
            if size in caption_lower:
                attributes.append(size)
        
        return attributes
    
    def classify_scene(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Classify the scene/room type
        
        Args:
            frame: Full RGB frame
            
        Returns:
            (room_type, confidence)
        """
        if self.use_fastvlm and self.fastvlm_model is not None:
            try:
                from PIL import Image
                pil_image = Image.fromarray(frame)
                
                prompt = "What type of room or space is this? Answer in 2-3 words (e.g., 'modern office', 'living room', 'kitchen')."
                
                response = generate(
                    self.fastvlm_model,
                    self.fastvlm_processor,
                    pil_image,
                    prompt,
                    max_tokens=10,
                    temperature=0.1
                )
                
                return response.strip(), 0.8
            
            except Exception as e:
                print(f"⚠️  Scene classification failed: {e}")
        
        # Fallback: simple heuristic (would use CLIP in practice)
        return "indoor space", 0.5
    
    def batch_caption(self, 
                     crops: List[np.ndarray],
                     classes: List[str],
                     object_ids: List[int]) -> List[RichCaption]:
        """
        Batch caption multiple objects
        
        Args:
            crops: List of RGB crops
            classes: List of COCO classes
            object_ids: List of object IDs
            
        Returns:
            List of RichCaptions
        """
        captions = []
        for crop, cls, oid in zip(crops, classes, object_ids):
            caption = self.caption_object(crop, cls, oid)
            captions.append(caption)
        
        return captions
    
    def supports_fastvlm(self) -> bool:
        """Check if FastVLM is available"""
        return self.use_fastvlm and self.fastvlm_model is not None


# Convenience function
def create_captioner(use_fastvlm: bool = True) -> RichSemanticCaptioner:
    """Create a RichSemanticCaptioner instance"""
    return RichSemanticCaptioner(use_fastvlm=use_fastvlm)
