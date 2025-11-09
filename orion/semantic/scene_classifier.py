"""
Scene Classification Module

Classifies scenes into types (bedroom, kitchen, office, outdoor, etc.)
Uses CLIP embeddings to determine scene context.

Author: Orion Research Team
Date: November 2025
"""

import numpy as np
from enum import Enum
from typing import Tuple, Optional
from dataclasses import dataclass


class SceneType(Enum):
    """Scene type classifications"""
    # Indoor scenes
    BEDROOM = "bedroom"
    KITCHEN = "kitchen"
    LIVING_ROOM = "living_room"
    OFFICE = "office"
    BATHROOM = "bathroom"
    HALLWAY = "hallway"
    DINING_ROOM = "dining_room"
    
    # Outdoor scenes
    STREET = "street"
    PARK = "park"
    OUTDOOR = "outdoor"
    
    # Special
    MOVING = "moving"
    UNKNOWN = "unknown"


@dataclass
class SceneClassification:
    """Result of scene classification"""
    scene_type: SceneType
    confidence: float
    is_indoor: bool
    is_outdoor: bool
    is_moving: bool


class SceneClassifier:
    """
    Classify scenes using CLIP embeddings and text prompts.
    """
    
    def __init__(self, clip_model=None):
        """
        Initialize scene classifier.
        
        Args:
            clip_model: CLIP model for embeddings (optional, will load if None)
        """
        self.clip_model = clip_model
        
        # Scene prompts for CLIP
        self.scene_prompts = {
            SceneType.BEDROOM: [
                "a bedroom with a bed",
                "a bedroom interior",
                "sleeping room",
            ],
            SceneType.KITCHEN: [
                "a kitchen with appliances",
                "a kitchen interior",
                "cooking area",
            ],
            SceneType.LIVING_ROOM: [
                "a living room with furniture",
                "a living room interior",
                "sitting room",
            ],
            SceneType.OFFICE: [
                "an office with a desk",
                "an office interior",
                "workspace",
            ],
            SceneType.BATHROOM: [
                "a bathroom with sink",
                "a bathroom interior",
                "toilet room",
            ],
            SceneType.HALLWAY: [
                "a hallway corridor",
                "an indoor corridor",
                "passage way",
            ],
            SceneType.DINING_ROOM: [
                "a dining room with table",
                "a dining area",
                "eating space",
            ],
            SceneType.STREET: [
                "a city street",
                "urban street view",
                "outdoor street",
            ],
            SceneType.PARK: [
                "a park with trees",
                "outdoor park",
                "green space",
            ],
            SceneType.OUTDOOR: [
                "an outdoor scene",
                "outside view",
                "exterior space",
            ],
        }
        
        # Cache text embeddings
        self._text_embeddings = None
        self._embedding_to_scene = None
    
    def _ensure_text_embeddings(self):
        """Compute and cache text embeddings for all scene prompts."""
        if self._text_embeddings is not None:
            return
        
        if self.clip_model is None:
            from orion.managers.model_manager import ModelManager
            self.clip_model = ModelManager.get_instance().clip
        
        embeddings = []
        embedding_to_scene = []
        
        for scene_type, prompts in self.scene_prompts.items():
            for prompt in prompts:
                emb = self.clip_model.encode_text(prompt, normalize=True)
                embeddings.append(emb)
                embedding_to_scene.append(scene_type)
        
        self._text_embeddings = np.array(embeddings)
        self._embedding_to_scene = embedding_to_scene
    
    def classify(
        self,
        frame: np.ndarray,
        return_top_k: int = 3
    ) -> Tuple[SceneType, float]:
        """
        Classify scene type from video frame.
        
        Args:
            frame: RGB video frame (H, W, 3)
            return_top_k: Number of top predictions to consider
        
        Returns:
            Tuple of (scene_type, confidence)
        """
        self._ensure_text_embeddings()
        
        # Extract image embedding
        img_embedding = self.clip_model.encode_image(frame, normalize=True)
        
        # Compute similarities
        similarities = np.dot(self._text_embeddings, img_embedding)
        
        # Get top-k matches
        top_k_indices = np.argsort(similarities)[-return_top_k:][::-1]
        
        # Vote by scene type
        scene_votes = {}
        for idx in top_k_indices:
            scene_type = self._embedding_to_scene[idx]
            score = float(similarities[idx])
            
            if scene_type not in scene_votes:
                scene_votes[scene_type] = []
            scene_votes[scene_type].append(score)
        
        # Aggregate votes (mean similarity)
        scene_scores = {
            scene: np.mean(scores)
            for scene, scores in scene_votes.items()
        }
        
        # Best scene
        best_scene = max(scene_scores.items(), key=lambda x: x[1])
        scene_type, confidence = best_scene
        
        return scene_type, confidence
    
    def classify_detailed(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None
    ) -> SceneClassification:
        """
        Detailed scene classification with context.
        
        Args:
            frame: Current RGB frame
            prev_frame: Previous frame (for motion detection)
        
        Returns:
            SceneClassification with full details
        """
        # Get scene type
        scene_type, confidence = self.classify(frame)
        
        # Determine indoor/outdoor
        indoor_types = {
            SceneType.BEDROOM, SceneType.KITCHEN, SceneType.LIVING_ROOM,
            SceneType.OFFICE, SceneType.BATHROOM, SceneType.HALLWAY,
            SceneType.DINING_ROOM
        }
        outdoor_types = {
            SceneType.STREET, SceneType.PARK, SceneType.OUTDOOR
        }
        
        is_indoor = scene_type in indoor_types
        is_outdoor = scene_type in outdoor_types
        
        # Check if moving (camera motion)
        is_moving = False
        if prev_frame is not None:
            motion_mag = self._compute_camera_motion(frame, prev_frame)
            is_moving = motion_mag > 0.3  # threshold
        
        return SceneClassification(
            scene_type=scene_type,
            confidence=confidence,
            is_indoor=is_indoor,
            is_outdoor=is_outdoor,
            is_moving=is_moving
        )
    
    def _compute_camera_motion(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> float:
        """
        Compute camera motion magnitude using optical flow.
        
        Args:
            frame1: Previous frame
            frame2: Current frame
        
        Returns:
            Motion magnitude (0=static, 1=high motion)
        """
        import cv2
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Compute magnitude
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Normalize to [0, 1]
        mean_mag = float(np.mean(mag))
        max_expected_mag = 10.0  # pixels/frame
        
        return min(1.0, mean_mag / max_expected_mag)
    
    def get_scene_context(
        self,
        scene_type: SceneType
    ) -> dict:
        """
        Get expected objects and interactions for a scene type.
        
        Args:
            scene_type: Scene classification
        
        Returns:
            Dictionary with expected_objects, common_interactions, etc.
        """
        scene_contexts = {
            SceneType.KITCHEN: {
                'expected_objects': [
                    'knife', 'oven', 'refrigerator', 'sink', 'cup',
                    'bowl', 'spoon', 'fork', 'plate'
                ],
                'common_interactions': [
                    ('hand', 'knife'), ('hand', 'cup'), ('hand', 'oven'),
                    ('knife', 'food')
                ],
                'boost_multiplier': 1.3,
            },
            SceneType.OFFICE: {
                'expected_objects': [
                    'keyboard', 'mouse', 'monitor', 'laptop', 'phone',
                    'desk', 'chair', 'book', 'pen'
                ],
                'common_interactions': [
                    ('hand', 'keyboard'), ('hand', 'mouse'), ('hand', 'phone')
                ],
                'boost_multiplier': 1.3,
            },
            SceneType.BEDROOM: {
                'expected_objects': [
                    'bed', 'pillow', 'lamp', 'book', 'phone', 'clock'
                ],
                'common_interactions': [
                    ('hand', 'book'), ('hand', 'phone'), ('hand', 'lamp')
                ],
                'boost_multiplier': 1.2,
            },
            SceneType.LIVING_ROOM: {
                'expected_objects': [
                    'tv', 'remote', 'couch', 'table', 'lamp', 'book'
                ],
                'common_interactions': [
                    ('hand', 'remote'), ('hand', 'tv')
                ],
                'boost_multiplier': 1.2,
            },
        }
        
        return scene_contexts.get(
            scene_type,
            {
                'expected_objects': [],
                'common_interactions': [],
                'boost_multiplier': 1.0,
            }
        )
