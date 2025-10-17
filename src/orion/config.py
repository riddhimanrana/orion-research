"""
Unified Configuration for Orion
================================

Clean, centralized configuration for all components.

Architecture:
    YOLO11x → Detection
    CLIP    → Re-ID embeddings
    FastVLM → Descriptions
    Gemma3  → Q&A

Author: Orion Research Team
Date: October 2025
"""

import logging
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class VideoConfig:
    """Video processing configuration"""
    
    # Frame sampling
    target_fps: float = 4.0
    """Target FPS for processing (lower = faster)"""
    
    # Frame selection (optional scene detection)
    use_scene_detection: bool = False
    """Enable scene change detection for intelligent frame selection"""
    
    scene_detection_threshold: float = 0.15
    """Threshold for scene change detection (lower = more sensitive)"""


@dataclass
class DetectionConfig:
    """YOLO detection configuration"""
    
    # Model
    model: Literal["yolo11n", "yolo11s", "yolo11m", "yolo11x"] = "yolo11x"
    """YOLO model variant (n=fastest, x=most accurate)"""
    
    # Detection thresholds
    confidence_threshold: float = 0.25
    """Minimum confidence for detection (lower = more detections)"""
    
    iou_threshold: float = 0.45
    """NMS IoU threshold for overlapping boxes"""
    
    # Object filtering
    min_object_size: int = 32
    """Minimum object size in pixels (smaller objects ignored)"""
    
    bbox_padding_percent: float = 0.10
    """Padding around bounding boxes when cropping (0.10 = 10%)"""
    
    # Confidence-based entity filtering
    high_confidence_threshold: float = 0.6
    """High confidence detections are trusted"""
    
    low_confidence_threshold: float = 0.4
    """Low confidence detections are skipped"""


@dataclass
class EmbeddingConfig:
    """CLIP embedding configuration"""
    
    # Model
    model: str = "openai/clip-vit-base-patch32"
    """CLIP model to use"""
    
    embedding_dim: int = 512
    """Embedding dimension (ViT-B/32 = 512, ViT-L/14 = 768)"""
    
    # Multimodal settings
    use_text_conditioning: bool = True
    """Condition embeddings on YOLO class (helps detect misclassifications)"""
    
    # Batch processing
    batch_size: int = 32
    """Number of images to embed at once (higher = faster but more memory)"""


@dataclass
class ClusteringConfig:
    """HDBSCAN clustering configuration"""
    
    # Core parameters
    min_cluster_size: int = 3
    """Minimum appearances to form an entity"""
    
    min_samples: int = 1
    """Minimum samples in neighborhood (lower = more aggressive)"""
    
    # Distance metric
    metric: Literal["euclidean", "cosine"] = "euclidean"
    """Distance metric for clustering"""
    
    # CLIP-optimized epsilon
    cluster_selection_epsilon: float = 0.35
    """Max distance to merge clusters (tuned for CLIP's 512-dim space)"""
    
    # State change detection
    state_change_threshold: float = 0.75
    """Cosine similarity threshold for detecting state changes"""
    
    min_state_duration_frames: int = 2
    """State change must persist for this many frames"""


@dataclass
class DescriptionConfig:
    """FastVLM description configuration"""
    
    # Generation settings
    max_tokens: int = 200
    """Maximum tokens per description"""
    
    temperature: float = 0.3
    """Sampling temperature (lower = more deterministic)"""
    
    # Smart description (describe once per entity)
    describe_once: bool = True
    """Only describe each entity once (from best frame)"""
    
    # Best frame selection
    size_weight: float = 0.5
    """Weight for object size in best frame selection"""
    
    centrality_weight: float = 0.3
    """Weight for object centrality (closer to image center)"""
    
    confidence_weight: float = 0.2
    """Weight for detection confidence"""


@dataclass
class PerformanceConfig:
    """Performance optimization settings"""
    
    # Parallel processing
    use_multiprocessing: bool = False
    """Enable multiprocessing for descriptions (experimental)"""
    
    num_workers: int = 2
    """Number of worker processes"""
    
    # Memory management
    clear_cache_after_phase: bool = True
    """Clear GPU cache after each phase"""
    
    # Batch processing
    enable_batching: bool = True
    """Batch process embeddings and descriptions"""


@dataclass
class LoggingConfig:
    """Logging configuration"""
    
    level: int = logging.INFO
    """Logging level"""
    
    show_progress: bool = True
    """Show progress bars"""
    
    verbose: bool = False
    """Verbose output (debug info)"""


@dataclass
class OrionConfig:
    """
    Master configuration for Orion pipeline.
    
    Usage:
        # Use defaults
        config = OrionConfig()
        
        # Customize
        config = OrionConfig(
            detection=DetectionConfig(model="yolo11x"),
            embedding=EmbeddingConfig(use_text_conditioning=True),
            clustering=ClusteringConfig(cluster_selection_epsilon=0.40)
        )
        
        # Access
        print(config.detection.confidence_threshold)  # 0.25
    """
    
    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    description: DescriptionConfig = field(default_factory=DescriptionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Initialize sub-configs if not provided"""
        if not isinstance(self.video, VideoConfig):
            self.video = VideoConfig()
        if not isinstance(self.detection, DetectionConfig):
            self.detection = DetectionConfig()
        if not isinstance(self.embedding, EmbeddingConfig):
            self.embedding = EmbeddingConfig()
        if not isinstance(self.clustering, ClusteringConfig):
            self.clustering = ClusteringConfig()
        if not isinstance(self.description, DescriptionConfig):
            self.description = DescriptionConfig()
        if not isinstance(self.performance, PerformanceConfig):
            self.performance = PerformanceConfig()
        if not isinstance(self.logging, LoggingConfig):
            self.logging = LoggingConfig()


# Preset configurations

def get_fast_config() -> OrionConfig:
    """
    Fast preset: Optimized for speed.
    
    - YOLO11n (fastest detector)
    - Lower resolution
    - Less aggressive clustering
    - Fewer descriptions
    """
    return OrionConfig(
        video=VideoConfig(target_fps=2.0),
        detection=DetectionConfig(
            model="yolo11n",
            confidence_threshold=0.35,
        ),
        clustering=ClusteringConfig(
            min_cluster_size=5,
            cluster_selection_epsilon=0.45,
        ),
        performance=PerformanceConfig(enable_batching=True),
    )


def get_balanced_config() -> OrionConfig:
    """
    Balanced preset: Good speed and accuracy.
    
    - YOLO11m (balanced detector)
    - Standard settings
    - Default clustering
    """
    return OrionConfig(
        detection=DetectionConfig(model="yolo11m"),
        clustering=ClusteringConfig(cluster_selection_epsilon=0.35),
    )


def get_accurate_config() -> OrionConfig:
    """
    Accurate preset: Optimized for accuracy.
    
    - YOLO11x (most accurate detector)
    - Lower confidence threshold
    - More aggressive clustering
    - More detailed descriptions
    """
    return OrionConfig(
        video=VideoConfig(target_fps=5.0),
        detection=DetectionConfig(
            model="yolo11x",
            confidence_threshold=0.20,
            high_confidence_threshold=0.7,
        ),
        clustering=ClusteringConfig(
            min_cluster_size=2,
            cluster_selection_epsilon=0.30,
        ),
        description=DescriptionConfig(max_tokens=300),
    )


# Default configuration
DEFAULT_CONFIG = get_balanced_config()
