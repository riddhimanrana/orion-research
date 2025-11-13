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
from typing import Literal, Optional

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
    
    backend: Literal["clip", "dino", "dinov3"] = "clip"
    """Embedding backend to use ('clip' for CLIP, 'dino' legacy, 'dinov3' for video encoder)."""

    # Cluster / memory efficiency settings
    use_cluster_embeddings: bool = False
    """If True, aggregate overlapping detections per frame into cluster embeddings to reduce memory."""

    cluster_similarity_threshold: float = 0.65
    """IoU threshold (0-1) to merge detections into same cluster before embedding extraction."""

    max_embeddings_per_entity: int = 25
    """Cap number of stored observation embeddings per entity (older ones downsampled)."""

    # Debug / verbosity
    reid_debug: bool = False
    """If True, print detailed pairwise similarity and merge decisions in Re-ID phase."""
    

    # Device selection
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    """Device for embedding model (auto/cuda/mps/cpu)"""

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
        
        if self.backend not in {"clip", "dino", "dinov3"}:
            raise ValueError(f"backend must be 'clip', 'dino', or 'dinov3', got {self.backend}")
        
        if self.backend in {"dino", "dinov3"} and self.use_text_conditioning:
            # DINO variants are vision-only; disable conditioning
            logger.warning("Text conditioning requested with DINO backend; disabling use_text_conditioning.")
            self.use_text_conditioning = False

        if not (0.0 <= self.cluster_similarity_threshold <= 1.0):
            raise ValueError(
                f"cluster_similarity_threshold must be in [0,1], got {self.cluster_similarity_threshold}"
            )
        if self.max_embeddings_per_entity < 1:
            raise ValueError(
                f"max_embeddings_per_entity must be >=1, got {self.max_embeddings_per_entity}"
            )
        
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.batch_size > 256:
            logger.warning(f"Very large batch_size: {self.batch_size}. May cause OOM.")
        valid_devices = {"auto", "cuda", "mps", "cpu"}
        if self.device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}, got {self.device}")
        logger.debug(
            f"EmbeddingConfig validated: model={self.model}, "
            f"dim={self.embedding_dim}, batch_size={self.batch_size}, device={self.device}"
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
    
    # 3D Perception settings
    enable_3d: bool = False
    """Enable 3D perception (depth, 3D coordinates, occlusion)"""
    
    depth_model: Literal["midas", "zoe"] = "midas"
    """Depth estimation model (deprecated; DepthAnythingV2 is now used internally)"""
    
    # TODO: Hand tracking (future implementation with HOT3D dataset)
    # enable_hands: bool = False
    # """Enable hand tracking (requires HOT3D-trained model, not yet implemented)"""
    
    enable_occlusion: bool = False
    """Enable occlusion detection (requires enable_3d=True)"""
    
    # Phase 2: Tracking settings
    enable_tracking: bool = False
    """Enable temporal entity tracking with Bayesian beliefs"""
    
    tracking_max_distance_pixels: float = 150.0
    """Maximum 2D distance for track association (pixels)"""
    
    tracking_max_distance_3d_mm: float = 1500.0
    """Maximum 3D distance for track association (millimeters)"""
    
    tracking_ttl_frames: int = 30
    """Frames before declaring entity disappeared"""
    
    tracking_reid_window_frames: int = 90
    """Window for attempting re-identification after disappearance"""
    
    tracking_class_belief_lr: float = 0.3
    """Learning rate for Bayesian class belief updates"""

    # Tracker backend selection (prototype: 'simple' existing logic or 'tapnet')
    tracker_backend: Literal["simple", "tapnet"] = "simple"
    """Select tracking backend: 'simple' (current centroid/HDBSCAN) or 'tapnet' (point trajectory model)"""

    # TapNet configuration (only used when tracker_backend='tapnet')
    tapnet_checkpoint_path: Optional[str] = None
    """Local path to TapNet / BootsTAPIR causal checkpoint (e.g., models/tapnet/causal_bootstapir_checkpoint.pt). Required for tapnet backend."""

    tapnet_max_points: int = 256
    """Maximum number of query points to track (sampled from detections centroids)."""

    tapnet_resolution: int = 256
    """Inference resolution (square). 256 for speed, 512 for accuracy."""

    tapnet_online_mode: bool = True
    """Use causal/online TapNet variant (per-frame). Set False for full-video batch (higher latency)."""

    tapnet_min_track_length: int = 3
    """Minimum length (frames) before promoting a point track to a persistent entity candidate."""

    tapnet_reid_merge_similarity: float = 0.35
    """Cosine similarity threshold between DINOv3 embeddings of track endpoints to merge trajectories (entity persistence)."""
    
    def __post_init__(self):
        """Validate perception config"""
        if self.target_fps <= 0:
            raise ValueError(f"target_fps must be > 0, got {self.target_fps}")
        
        # Validate 3D perception dependencies
        if self.enable_occlusion and not self.enable_3d:
            logger.warning(
                "enable_occlusion set but enable_3d=False. "
                "Setting enable_3d=True automatically."
            )
            self.enable_3d = True
        
        logger.debug(
            f"PerceptionConfig validated: target_fps={self.target_fps}, "
            f"scene_detection={self.use_scene_detection}, "
            f"3d_enabled={self.enable_3d}"
        )

        # TapNet validation
        if self.tracker_backend == "tapnet":
            if not self.tapnet_checkpoint_path:
                logger.warning(
                    "tracker_backend='tapnet' but tapnet_checkpoint_path not set. "
                    "Tracking will fall back to 'simple'. Set tapnet_checkpoint_path to enable TapNet."
                )
                self.tracker_backend = "simple"
            if self.tapnet_resolution not in (256, 512):
                logger.warning(
                    f"tapnet_resolution={self.tapnet_resolution} unsupported; forcing 256"
                )
                self.tapnet_resolution = 256


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
    Enables: DINO embeddings, SLAM, 3D perception, tracking
    """
    return PerceptionConfig(
        detection=DetectionConfig(
            model="yolo11x",
            confidence_threshold=0.15,
        ),
        embedding=EmbeddingConfig(
            embedding_dim=1024,
            backend="dino",
            batch_size=16,
            device="auto",  # Use GPU if available
        ),
        description=DescriptionConfig(
            max_tokens=300,
            temperature=0.2,
        ),
        target_fps=8.0,
        enable_3d=True,
        depth_model="midas",  # Note: DepthAnythingV2 used internally regardless
        enable_tracking=True,
    )
