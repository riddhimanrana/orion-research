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
from typing import Literal, Optional, List

import logging

from .intrinsics_presets import DEFAULT_INTRINSICS_PRESET, INTRINSICS_PRESETS
from orion.semantic.config import SemanticConfig

logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    """Detection backend configuration (YOLO or GroundingDINO)."""

    backend: Literal["yolo", "groundingdino"] = "yolo"
    """Primary detector to use for frame observations."""

    # YOLO-specific settings
    model: Literal["yolo11n", "yolo11s", "yolo11m", "yolo11x"] = "yolo11x"
    """YOLO model size (n=fastest, x=most accurate)."""

    # Detection thresholds (shared)
    confidence_threshold: float = 0.25
    """Minimum detection confidence (0-1, lower = more detections)."""

    iou_threshold: float = 0.45
    """NMS IoU threshold for overlapping boxes."""

    # Filtering
    min_object_size: int = 32
    """Minimum object size in pixels (smaller ignored)."""

    # Cropping
    bbox_padding_percent: float = 0.1
    """Padding to add around bounding boxes when cropping (percentage of box size)."""

    # GroundingDINO-specific options
    groundingdino_model_id: str = "IDEA-Research/grounding-dino-base"
    """Hugging Face model identifier for GroundingDINO."""

    groundingdino_prompt: str = (
        "person . chair . couch . dining table . tv . laptop . keyboard . monitor . "
        "book . cup . bottle . plant . refrigerator . microwave . oven . sink . bed . "
        "counter . cabinet . dog . cat . backpack . suitcase . chair"
    )
    """Dot-separated open-vocabulary prompt used by GroundingDINO."""

    groundingdino_box_threshold: float = 0.30
    """Box confidence threshold for GroundingDINO post-processing."""

    groundingdino_text_threshold: float = 0.25
    """Text confidence threshold for GroundingDINO token matching."""

    groundingdino_max_detections: int = 100
    """Upper bound on detections returned per frame when using GroundingDINO."""

    def __post_init__(self):
        """Validate detection config."""
        if self.backend not in {"yolo", "groundingdino"}:
            raise ValueError(f"backend must be 'yolo' or 'groundingdino', got {self.backend}")

        # Model validation
        valid_models = {"yolo11n", "yolo11s", "yolo11m", "yolo11x"}
        if self.model not in valid_models:
            raise ValueError(f"Invalid model: {self.model}. Must be one of {valid_models}")

        # Threshold validation
        if not (0 <= self.confidence_threshold <= 1):
            raise ValueError(f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}")

        if not (0 <= self.iou_threshold <= 1):
            raise ValueError(f"iou_threshold must be in [0, 1], got {self.iou_threshold}")

        if not (0 <= self.groundingdino_box_threshold <= 1):
            raise ValueError(
                f"groundingdino_box_threshold must be in [0, 1], got {self.groundingdino_box_threshold}"
            )

        if not (0 <= self.groundingdino_text_threshold <= 1):
            raise ValueError(
                f"groundingdino_text_threshold must be in [0, 1], got {self.groundingdino_text_threshold}"
            )

        # Size validation
        if self.min_object_size < 1:
            raise ValueError(f"min_object_size must be >= 1, got {self.min_object_size}")

        # Padding validation
        if self.bbox_padding_percent < 0:
            raise ValueError(f"bbox_padding_percent must be >= 0, got {self.bbox_padding_percent}")

        if self.backend == "groundingdino":
            if not self.groundingdino_prompt.strip():
                raise ValueError("groundingdino_prompt cannot be empty when backend='groundingdino'")
            if self.groundingdino_max_detections < 1:
                raise ValueError("groundingdino_max_detections must be >= 1")

        logger.debug(
            f"DetectionConfig validated: backend={self.backend}, model={self.model}, "
            f"conf_thresh={self.confidence_threshold}, iou_thresh={self.iou_threshold}"
        )

    def grounding_categories(self) -> list[str]:
        """Return normalized category list for GroundingDINO prompts."""
        return [token.strip() for token in self.groundingdino_prompt.split('.') if token.strip()]


@dataclass
class SegmentationConfig:
    """Instance segmentation refinement configuration (SAM)."""

    enabled: bool = False
    """Enable SAM-based mask refinement after detection."""

    model_type: Literal["vit_h", "vit_l", "vit_b"] = "vit_h"
    """SAM backbone variant to load."""

    checkpoint_path: Optional[str] = None
    """Override path to SAM checkpoint (.pth). Defaults to models/weights/sam_<model>.pth."""

    device: Literal["auto", "cuda", "cpu"] = "auto"
    """Device for SAM inference (auto selects CUDA when available)."""

    mask_threshold: float = 0.5
    """Probability threshold applied to SAM logits before binarizing masks."""

    stability_score_threshold: float = 0.85
    """Minimum stability score accepted from SAM outputs."""

    min_mask_area: int = 400
    """Minimum number of pixels required for a mask to be retained."""

    batch_size: int = 16
    """Maximum number of bounding boxes to refine per SAM call."""

    refine_bounding_box: bool = True
    """Replace detection bboxes with tight masks bounds when True."""

    apply_mask_to_crops: bool = True
    """Zero out background pixels in crops using the predicted mask."""

    def __post_init__(self):
        if self.mask_threshold <= 0 or self.mask_threshold >= 1:
            raise ValueError("mask_threshold must be in (0,1)")
        if self.stability_score_threshold <= 0 or self.stability_score_threshold > 1:
            raise ValueError("stability_score_threshold must be in (0,1]")
        if self.min_mask_area < 1:
            raise ValueError("min_mask_area must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be 'auto', 'cuda', or 'cpu'")


@dataclass
class ClassCorrectionConfig:
    """Configuration for CLIP-based class label correction."""

    enabled: bool = True
    min_similarity: float = 0.30
    max_description_tokens: int = 6
    include_detector_label: bool = True
    prompt_template: str = "a photo of {label}"
    extra_labels: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not (0.0 <= self.min_similarity <= 1.0):
            raise ValueError("min_similarity must be within [0,1]")
        if self.max_description_tokens < 1:
            raise ValueError("max_description_tokens must be >= 1")


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

    prompt_template: str = "Describe this object in detail."
    """Prompt template for VLM description generation."""
    
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
class DepthConfig:
    """Depth estimation configuration for Phase 1 3D perception."""

    model_name: Literal["depth_anything_v3"] = "depth_anything_v3"
    model_size: Literal["small", "base", "large"] = "small"
    device: Optional[str] = None  # auto-detect by default
    half_precision: bool = True
    max_depth_mm: float = 10000.0

    def __post_init__(self):
        if self.max_depth_mm <= 0:
            raise ValueError(f"max_depth_mm must be > 0, got {self.max_depth_mm}")
        if self.model_size not in {"small", "base", "large"}:
            raise ValueError(
                f"model_size must be one of ['small', 'base', 'large'], got {self.model_size}"
            )


@dataclass
class HandTrackingConfig:
    """MediaPipe hand tracking configuration."""

    max_num_hands: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    device: Optional[str] = None

    def __post_init__(self):
        if self.max_num_hands < 1:
            raise ValueError("max_num_hands must be >= 1")
        for name, value in (
            ("min_detection_confidence", self.min_detection_confidence),
            ("min_tracking_confidence", self.min_tracking_confidence),
        ):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {value}")


@dataclass
class OcclusionConfig:
    """Occlusion detection configuration."""

    depth_margin_mm: float = 100.0
    occlusion_threshold: float = 0.3

    def __post_init__(self):
        if self.depth_margin_mm < 0:
            raise ValueError("depth_margin_mm must be >= 0")
        if not (0.0 <= self.occlusion_threshold <= 1.0):
            raise ValueError("occlusion_threshold must be in [0,1]")


@dataclass
class CameraConfig:
    """Camera metadata for auto estimating intrinsics."""

    width: int = 1920
    height: int = 1080
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    auto_estimate: bool = True
    intrinsics_preset: Optional[str] = DEFAULT_INTRINSICS_PRESET

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.intrinsics_preset and self.intrinsics_preset not in INTRINSICS_PRESETS:
            raise ValueError(
                f"Unknown intrinsics preset '{self.intrinsics_preset}'. "
                f"Valid presets: {list(INTRINSICS_PRESETS.keys())}"
            )

    def resolve_intrinsics(self, width: int, height: int) -> "CameraIntrinsics":
        """Resolve intrinsics using preset or manual overrides."""
        from .types import CameraIntrinsics

        if self.intrinsics_preset:
            preset = INTRINSICS_PRESETS[self.intrinsics_preset]
            return CameraIntrinsics(
                fx=preset.fx,
                fy=preset.fy,
                cx=preset.cx,
                cy=preset.cy,
                width=width,
                height=height,
            )

        if not self.auto_estimate and all(
            value is not None for value in (self.fx, self.fy, self.cx, self.cy)
        ):
            return CameraIntrinsics(
                fx=float(self.fx),
                fy=float(self.fy),
                cx=float(self.cx),
                cy=float(self.cy),
                width=width,
                height=height,
            )

        return CameraIntrinsics.auto_estimate(width=width, height=height)


@dataclass
class PerceptionConfig:
    """Complete perception engine configuration"""
    
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    description: DescriptionConfig = field(default_factory=DescriptionConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    hand_tracking: HandTrackingConfig = field(default_factory=HandTrackingConfig)
    occlusion: OcclusionConfig = field(default_factory=OcclusionConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    class_correction: ClassCorrectionConfig = field(default_factory=ClassCorrectionConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    
    # General settings
    target_fps: float = 4.0
    """Target frames per second for processing"""
    
    use_scene_detection: bool = False
    """Enable intelligent scene change detection for frame sampling"""
    
    enable_descriptions: bool = True
    """Enable FastVLM-based entity descriptions."""

    # 3D Perception settings
    enable_3d: bool = True
    """Enable 3D perception (depth, 3D coordinates, occlusion)"""
    
    depth_model: Literal["depth_anything_v3", "depth_anything_v2"] = "depth_anything_v3"
    """Depth estimation model (DepthAnything V3 primary, V2 fallback)."""
    
    enable_depth: bool = True
    """Enable depth estimation stage (Phase 1 3D)."""
    
    enable_hands: bool = True
    """Enable hand tracking (MediaPipe)"""
    
    # TODO: Hand tracking (future implementation with HOT3D dataset)
    # enable_hands: bool = False
    # """Enable hand tracking (requires HOT3D-trained model, not yet implemented)"""
    
    enable_occlusion: bool = True
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
    """Path to TapNet checkpoint (e.g. 'models/tapnet/tapir_checkpoint_panning.npy')"""
    
    tapnet_resolution: int = 256
    """Resolution for TapNet tracking (256 or 512)"""

    # Clustering / HDBSCAN tuning
    clustering_min_cluster_size: int = 4
    """Minimum cluster size for HDBSCAN entity grouping (>=2)."""

    clustering_min_samples: int = 1
    """Minimum samples parameter for HDBSCAN (>=1)."""

    clustering_cluster_selection_epsilon: float = 0.25
    """Epsilon for HDBSCAN cluster selection (lower = more clusters)."""

    # Memgraph configuration
    use_memgraph: bool = False
    """Enable Memgraph backend for real-time graph updates"""
    
    memgraph_host: str = "127.0.0.1"
    """Memgraph host address"""
    
    memgraph_port: int = 7687
    """Memgraph port"""

    # Semantic reasoning
    enable_cis: bool = True
    """Compute causal influence scores between tracked entities."""

    cis_max_links: int = 50
    """Limit number of CIS links retained in metrics (avoids bloating results)."""

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
        if self.enable_occlusion and not self.enable_depth:
            logger.warning(
                "enable_occlusion=True requires enable_depth=True. Enabling depth automatically."
            )
            self.enable_depth = True
        
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

        # Clustering parameter validation
        if self.clustering_min_cluster_size < 2:
            logger.warning(
                f"clustering_min_cluster_size={self.clustering_min_cluster_size} too small; forcing 2"
            )
            self.clustering_min_cluster_size = 2
        if self.clustering_min_samples < 1:
            logger.warning(
                f"clustering_min_samples={self.clustering_min_samples} invalid; forcing 1"
            )
            self.clustering_min_samples = 1
        if self.clustering_cluster_selection_epsilon < 0:
            logger.warning(
                f"clustering_cluster_selection_epsilon={self.clustering_cluster_selection_epsilon} < 0; forcing 0.0"
            )
            self.clustering_cluster_selection_epsilon = 0.0

        if self.cis_max_links < 1:
            raise ValueError("cis_max_links must be >= 1 when CIS is enabled")


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
