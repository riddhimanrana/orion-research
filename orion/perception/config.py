"""
Perception Engine Configuration
==============================

Validated configuration for Phase 1 (Perception & Tracking).

Supports preset configs (fast/balanced/accurate) and custom values.
All fields are validated with meaningful error messages.

Author: Orion Research Team
Date: October 2025
"""

from dataclasses import dataclass, field
from typing import Literal

import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    """YOLO11 detection configuration"""
    
    # Model variant
    model: Literal["yolo11n", "yolo11s", "yolo11m", "yolo11x"] = "yolo11x"
    """YOLO model size (n=fastest, x=most accurate)"""
    
    # Detection thresholds
    confidence_threshold: float = 0.25
    """Minimum detection confidence (0-1, lower = more detections)"""
    
    iou_threshold: float = 0.45
    """NMS IoU threshold for overlapping boxes"""
    
    # Filtering
    min_object_size: int = 32
    """Minimum object size in pixels (smaller ignored)"""
    
    # Cropping
    bbox_padding_percent: float = 0.1
    """Padding to add around bounding boxes when cropping (as percentage of box size)"""
    
    def __post_init__(self):
        """Validate detection config"""
        # Model validation
        valid_models = {"yolo11n", "yolo11s", "yolo11m", "yolo11x"}
        if self.model not in valid_models:
            raise ValueError(f"Invalid model: {self.model}. Must be one of {valid_models}")
        
        # Threshold validation
        if not (0 <= self.confidence_threshold <= 1):
            raise ValueError(f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}")
        
        if not (0 <= self.iou_threshold <= 1):
            raise ValueError(f"iou_threshold must be in [0, 1], got {self.iou_threshold}")
        
        # Size validation
        if self.min_object_size < 1:
            raise ValueError(f"min_object_size must be >= 1, got {self.min_object_size}")
        
        # Padding validation
        if self.bbox_padding_percent < 0:
            raise ValueError(f"bbox_padding_percent must be >= 0, got {self.bbox_padding_percent}")
        
        logger.debug(
            f"DetectionConfig validated: model={self.model}, "
            f"conf_thresh={self.confidence_threshold}, "
            f"iou_thresh={self.iou_threshold}"
        )


@dataclass
class EmbeddingConfig:
    """CLIP embedding configuration"""
    
    # Model
    model: str = "openai/clip-vit-base-patch32"
    """CLIP model name from HuggingFace"""
    
    embedding_dim: int = 512
    """Output embedding dimension"""
    
    # Batch processing
    batch_size: int = 32
    """Embeddings per batch (higher = faster but more memory)"""
    
    # Conditioning
    use_text_conditioning: bool = True
    """Condition embeddings on YOLO class labels"""
    
    def __post_init__(self):
        """Validate embedding config"""
        valid_models = {
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",
            "openai/clip-vit-base-patch16",
        }
        if self.model not in valid_models:
            logger.warning(f"Unusual model: {self.model}. May not be recognized.")
        
        valid_dims = {512, 768, 1024}
        if self.embedding_dim not in valid_dims:
            raise ValueError(
                f"embedding_dim must be one of {valid_dims}, got {self.embedding_dim}"
            )
        
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        
        if self.batch_size > 256:
            logger.warning(f"Very large batch_size: {self.batch_size}. May cause OOM.")
        
        logger.debug(
            f"EmbeddingConfig validated: model={self.model}, "
            f"dim={self.embedding_dim}, batch_size={self.batch_size}"
        )


@dataclass
class DescriptionConfig:
    """FastVLM description generation configuration"""
    
    # Generation settings
    max_tokens: int = 200
    """Maximum tokens per description"""
    
    temperature: float = 0.3
    """Sampling temperature (lower = deterministic, higher = creative)"""
    
    # Optimization
    describe_once: bool = True
    """Only describe each entity once (from best frame)"""
    
    # Best frame selection weights
    size_weight: float = 0.5
    """Weight for object size"""
    
    centrality_weight: float = 0.3
    """Weight for proximity to image center"""
    
    confidence_weight: float = 0.2
    """Weight for detection confidence"""

    # Sanity checks
    min_crop_size: int = 32
    """Minimum width/height in pixels required to trust a crop"""

    min_crop_std: float = 5.0
    """Minimum grayscale standard deviation to avoid flat/blank crops"""
    
    def __post_init__(self):
        """Validate description config"""
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        
        if self.max_tokens > 512:
            logger.warning(f"Very large max_tokens: {self.max_tokens}. Descriptions may be verbose.")
        
        if not (0 <= self.temperature <= 2):
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")
        
        # Validate weights sum to ~1.0
        total_weight = self.size_weight + self.centrality_weight + self.confidence_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Frame selection weights sum to {total_weight:.2f}, expected 1.0. "
                f"Rescaling..."
            )
            # Auto-rescale
            if total_weight > 0:
                scale = 1.0 / total_weight
                self.size_weight *= scale
                self.centrality_weight *= scale
                self.confidence_weight *= scale

        if self.min_crop_size < 8:
            raise ValueError(f"min_crop_size must be >= 8 pixels, got {self.min_crop_size}")

        if self.min_crop_std < 0:
            raise ValueError(f"min_crop_std must be >= 0, got {self.min_crop_std}")
        
        logger.debug(
            f"DescriptionConfig validated: max_tokens={self.max_tokens}, "
            f"temperature={self.temperature}, describe_once={self.describe_once}"
        )


@dataclass
class PerceptionConfig:
    """Complete perception engine configuration"""
    
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    description: DescriptionConfig = field(default_factory=DescriptionConfig)
    
    # General settings
    target_fps: float = 4.0
    """Target frames per second for processing"""
    
    use_scene_detection: bool = False
    """Enable intelligent scene change detection for frame sampling"""
    
    def __post_init__(self):
        """Validate perception config"""
        if self.target_fps <= 0:
            raise ValueError(f"target_fps must be > 0, got {self.target_fps}")
        
        logger.debug(
            f"PerceptionConfig validated: target_fps={self.target_fps}, "
            f"scene_detection={self.use_scene_detection}"
        )


# Preset configurations
def get_fast_config() -> PerceptionConfig:
    """
    Fast mode: prioritize speed over accuracy
    
    Use for quick iteration, demos, testing
    """
    return PerceptionConfig(
        detection=DetectionConfig(
            model="yolo11n",
            confidence_threshold=0.4,
        ),
        embedding=EmbeddingConfig(
            embedding_dim=512,
            batch_size=64,
        ),
        description=DescriptionConfig(
            max_tokens=100,
            temperature=0.5,
        ),
        target_fps=2.0,
    )


def get_balanced_config() -> PerceptionConfig:
    """
    Balanced mode: good accuracy with reasonable speed
    
    Recommended for production use
    """
    return PerceptionConfig(
        detection=DetectionConfig(
            model="yolo11m",
            confidence_threshold=0.25,
        ),
        embedding=EmbeddingConfig(
            embedding_dim=512,
            batch_size=32,
        ),
        description=DescriptionConfig(
            max_tokens=200,
            temperature=0.3,
        ),
        target_fps=4.0,
    )


def get_accurate_config() -> PerceptionConfig:
    """
    Accurate mode: maximum quality (slowest)
    
    For research and evaluation
    """
    return PerceptionConfig(
        detection=DetectionConfig(
            model="yolo11x",
            confidence_threshold=0.15,
        ),
        embedding=EmbeddingConfig(
            embedding_dim=1024,
            batch_size=16,
        ),
        description=DescriptionConfig(
            max_tokens=300,
            temperature=0.2,
        ),
        target_fps=8.0,
    )
