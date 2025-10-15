"""
Part 1: The Asynchronous Perception Engine
===========================================

This module implements a two-tier asynchronous video processing system that:
1. Intelligently selects interesting frames using scene change detection
2. Performs fast object detection and visual embedding generation (Tier 1)
3. Asynchronously generates rich semantic descriptions (Tier 2)

Models Used:
- Object Detection: YOLO11m (ultralytics) - 80 COCO classes, 20.1M params
- Scene Detection: FastViT-T8 (timm) - Lightweight vision transformer
- Visual Embeddings: OSNet (torchreid/timm) - 512-dim Re-ID feature vectors
  * OSNet is optimized for person/object re-identification across varying scales,
    poses, and lighting conditions - critical for long-term tracking
- Description Generation: FastVLM-0.5B (apple/FastVLM-0.5B via Hugging Face transformers)
All model weights are cached under the repository's ``models/`` directory via the shared
asset manager so runs remain self-contained.

Motion Tracking:
- Velocity estimation via frame-to-frame centroid tracking
- Direction analysis for causal inference (agents moving towards patients)

Author: Orion Research Team
Date: January 2025
"""

import logging
import multiprocessing as mp
import os
import sys
import time
import warnings
from collections.abc import MutableSequence
from dataclasses import asdict, dataclass
from enum import Enum
from multiprocessing import Manager, Process, Queue
from queue import Empty as QueueEmpty
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

try:  # Local import works both as package and script
    from .models import ModelManager as AssetManager
except ImportError:  # pragma: no cover
    from models import ModelManager as AssetManager  # type: ignore

try:
    from .runtime import get_active_backend, select_backend
except ImportError:  # pragma: no cover
    from runtime import get_active_backend, select_backend  # type: ignore

try:
    from .motion_tracker import MotionTracker, MotionData
except ImportError:  # pragma: no cover
    from motion_tracker import MotionTracker, MotionData  # type: ignore

# Set multiprocessing start method to 'spawn' for accelerator compatibility
# CRITICAL: GPU/Metal-backed models often fail with 'fork' on macOS
# 'fork' copies the parent's memory, which can break driver state and tensor cores
# 'spawn' starts fresh processes, allowing models to initialize properly
try:
    mp.set_start_method("spawn", force=True)
    logging.info(
        "Set multiprocessing start method to 'spawn' for accelerator compatibility"
    )
except RuntimeError:
    # Already set
    pass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as SyncEvent
else:  # pragma: no cover
    SyncEvent = Any  # type: ignore[assignment]

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Fix tokenizers parallelism warning (must be set before importing tokenizers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_ASSET_MANAGER: Optional[AssetManager] = None


def _get_asset_manager() -> AssetManager:
    global _ASSET_MANAGER
    if _ASSET_MANAGER is None:
        _ASSET_MANAGER = AssetManager()
    return _ASSET_MANAGER


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================


class DescriptionMode(Enum):
    """Mode for FastVLM description generation"""

    SCENE = "scene"  # One description per frame (shared by all objects)
    OBJECT = "object"  # One description per object (focused on individual object)
    HYBRID = "hybrid"  # Both scene + object descriptions (most comprehensive)


class Config:
    """Configuration parameters for the perception engine"""

    # Video Processing
    TARGET_FPS = 4.0  # Sample frames at 4 FPS
    USE_SCENE_DETECTION = False  # Set to False to process all frames (faster startup)
    SCENE_SIMILARITY_THRESHOLD = (
        0.98  # Threshold for scene change detection (only if USE_SCENE_DETECTION=True)
    )
    FRAME_RESIZE_DIM = 768  # Resize frames to 768x768 for scene detection

    # Object Detection (YOLO)
    YOLO_CONFIDENCE_THRESHOLD = 0.25
    YOLO_IOU_THRESHOLD = 0.45
    YOLO_MAX_DETECTIONS = 100
    MIN_OBJECT_SIZE = 32  # Minimum width/height in pixels
    BBOX_PADDING_PERCENT = 0.10  # 10% padding around bounding box

    # Visual Embedding (OSNet)
    OSNET_INPUT_SIZE = (256, 128)  # Height, Width for ReID models
    EMBEDDING_DIM = 512

    # Multiprocessing
    NUM_WORKERS = 2  # Number of worker processes for descriptions
    QUEUE_MAX_SIZE = 1000
    QUEUE_TIMEOUT = 0.5  # Seconds
    WORKER_SHUTDOWN_TIMEOUT = 30  # Seconds

    # FastVLM Description Generation (Graph-Optimized)
    DESCRIPTION_MODE = DescriptionMode.OBJECT  # SCENE, OBJECT, or HYBRID
    DESCRIPTION_MAX_TOKENS = 200  # Optimized for <200 tokens (graph DB friendly)
    DESCRIPTION_TEMPERATURE = 0.3  # Lower temp for more focused, consistent outputs
    # Note: Prompts are handled by FastVLM wrapper based on mode

    # Performance & Logging
    LOG_LEVEL = logging.DEBUG  # Temporarily DEBUG for troubleshooting
    PROGRESS_BAR = True
    CHECKPOINT_INTERVAL = 100  # Save checkpoint every N frames


# ============================================================================
# LOGGING SETUP
# ============================================================================


def setup_logger(name: str, level: int = Config.LOG_LEVEL) -> logging.Logger:
    """
    Set up a logger with consistent formatting

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger("PerceptionEngine")


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class RichPerceptionObject:
    """
    Represents a detected object with rich visual and semantic information
    """

    timestamp: float  # Video timestamp in seconds
    frame_number: int  # Frame index
    bounding_box: List[int]  # [x1, y1, x2, y2]
    visual_embedding: List[float]  # 512-dim OSNet embedding
    detection_confidence: float  # YOLO confidence score
    object_class: str  # YOLO class label
    crop_size: Tuple[int, int]  # (width, height) of cropped region
    rich_description: Optional[str] = None  # FastVLM generated description
    entity_id: Optional[str] = None  # To be filled in Part 2
    temp_id: Optional[str] = None  # Temporary unique ID
    
    # Motion tracking data (for causal inference)
    centroid: Optional[Tuple[float, float]] = None  # (x, y) center
    velocity: Optional[Tuple[float, float]] = None  # (vx, vy) pixels/second
    speed: Optional[float] = None  # Magnitude of velocity
    direction: Optional[float] = None  # Angle in radians

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    def is_complete(self) -> bool:
        """Check if all required fields are populated"""
        return self.rich_description is not None


# ============================================================================
# MODEL LOADING UTILITIES
# ============================================================================


class PerceptionModelManager:
    """
    Manages loading and caching of all required models
    """

    _instance = None
    _models = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PerceptionModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.asset_manager: AssetManager = _get_asset_manager()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def get_scene_detector(self) -> Optional[Dict[str, Any]]:
        """Load FastViT model for scene change detection"""
        if not Config.USE_SCENE_DETECTION:
            logger.info("Scene detection disabled (USE_SCENE_DETECTION=False)")
            return None

        if "scene_detector" not in self._models:
            try:
                import timm
                from timm.data.config import resolve_model_data_config
                from timm.data.transforms_factory import create_transform

                logger.info("Loading FastViT model for scene detection...")

                # Load lightweight FastViT model
                model = timm.create_model("fastvit_t8.apple_in1k", pretrained=True)
                model = model.to(self.device)
                model.eval()

                # Get the model's data configuration for preprocessing
                data_config = resolve_model_data_config(model)
                transforms = create_transform(**data_config, is_training=False)

                self._models["scene_detector"] = {
                    "model": model,
                    "transforms": transforms,
                }
                logger.info("FastViT model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load FastViT model: {e}")
                logger.warning("Scene detection will be disabled")
                self._models["scene_detector"] = None

        return self._models.get("scene_detector")

    def get_object_detector(self) -> Any:
        """Load YOLO11m model for object detection"""
        if "object_detector" not in self._models:
            try:
                from ultralytics import YOLO

                logger.info("Loading YOLO11m model...")

                asset_dir = self.asset_manager.ensure_asset("yolo11m")
                model_path = asset_dir / "yolo11m.pt"
                if not model_path.exists():
                    raise FileNotFoundError(
                        "YOLO11m weights were not downloaded correctly. "
                        f"Expected to find {model_path}."
                    )

                model = YOLO(str(model_path))

                self._models["object_detector"] = model
                logger.info("✓ YOLO11m model loaded successfully")
                logger.info(f"  Model: {model_path} (20.1M params, 80 COCO classes)")
            except Exception as e:
                logger.error(f"Failed to load YOLO11m model: {e}")
                logger.error("Install with: pip install ultralytics")
                raise RuntimeError("Object detector is required for perception engine")

        result = self._models.get("object_detector")
        if result is None:
            raise RuntimeError("Object detector is not available")
        return result

    def get_embedding_model(self) -> Dict[str, Any]:
        """Load OSNet model for visual embeddings (Re-ID optimized)"""
        if "embedding_model" not in self._models:
            try:
                logger.info("Loading Re-ID model for visual embeddings...")

                # Try multiple OSNet sources in order of preference
                model = None
                model_name = None
                
                # Option 1: Try torchreid (dedicated Re-ID library)
                try:
                    import torchreid
                    logger.info("Attempting to load OSNet from torchreid...")
                    model = torchreid.models.build_model(
                        name='osnet_x1_0',
                        num_classes=1000,  # Will use feature layer
                        pretrained=True,
                        loss='softmax'
                    )
                    model_name = "OSNet (torchreid)"
                    logger.info(f"✓ Loaded {model_name}")
                except ImportError:
                    logger.debug("torchreid not available, trying timm...")
                except Exception as e:
                    logger.debug(f"torchreid failed: {e}")
                
                # Option 2: Try timm (might have OSNet)
                if model is None:
                    try:
                        import timm
                        logger.info("Attempting to load OSNet from timm...")
                        model = timm.create_model(
                            "osnet_x1_0",
                            pretrained=True,
                            num_classes=0  # Feature extraction mode
                        )
                        model_name = "OSNet (timm)"
                        logger.info(f"✓ Loaded {model_name}")
                    except Exception as e:
                        logger.debug(f"timm OSNet failed: {e}")
                
                # Option 3: Try OSNet variants from timm
                if model is None:
                    try:
                        import timm
                        logger.info("Attempting OSNet variants from timm...")
                        # Try different OSNet model names
                        for variant in ['osnet_x0_75', 'osnet_ibn_x1_0']:
                            try:
                                model = timm.create_model(
                                    variant,
                                    pretrained=True,
                                    num_classes=0
                                )
                                model_name = f"OSNet-{variant} (timm)"
                                logger.info(f"✓ Loaded {model_name}")
                                break
                            except:
                                continue
                    except Exception as e:
                        logger.debug(f"timm OSNet variants failed: {e}")
                
                # If all OSNet attempts failed, raise helpful error
                if model is None:
                    error_msg = (
                        "Failed to load OSNet Re-ID model. Please install one of:\n"
                        "  1. torchreid: pip install torchreid\n"
                        "  2. timm with OSNet support: pip install timm>=0.9.0\n"
                        "OSNet is required for robust person/object re-identification."
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                # Move model to device and set to eval
                model = model.to(self.device)
                model.eval()

                # Create preprocessing transforms
                if 'torchreid' in str(type(model)):
                    # torchreid models expect specific preprocessing
                    import torchvision.transforms as T
                    transforms = T.Compose([
                        T.Resize((256, 128)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
                    ])
                else:
                    # Use timm's transforms
                    import timm
                    from timm.data.config import resolve_model_data_config
                    from timm.data.transforms_factory import create_transform
                    data_config = resolve_model_data_config(model)
                    transforms = create_transform(**data_config, is_training=False)

                self._models["embedding_model"] = {
                    "model": model,
                    "transforms": transforms,
                    "model_name": model_name,
                }
                logger.info(f"✓ Re-ID embedding model ready: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise RuntimeError(f"Embedding model is required for perception engine: {e}")

        result = self._models.get("embedding_model")
        if result is None:
            raise RuntimeError("Embedding model is not available")
        return result


# ============================================================================
# MODULE 1: VIDEO INGESTION & FRAME SELECTION
# ============================================================================


class VideoFrameSelector:
    """
    Handles intelligent frame selection using scene change detection
    """

    def __init__(self, model_manager: PerceptionModelManager):
        self.model_manager = model_manager
        self.scene_detector: Optional[Dict[str, Any]] = (
            model_manager.get_scene_detector()
        )
        self.last_scene_embedding: Optional[np.ndarray] = None

    def load_video(self, video_path: str) -> Tuple[cv2.VideoCapture, Dict[str, Any]]:
        """
        Load video and extract metadata

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (VideoCapture object, metadata dict)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Extract metadata
        metadata = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
        }

        logger.info(f"Video loaded: {video_path}")
        logger.info(f"  Resolution: {metadata['width']}x{metadata['height']}")
        logger.info(f"  FPS: {metadata['fps']:.2f}")
        logger.info(f"  Duration: {metadata['duration']:.2f}s")
        logger.info(f"  Total frames: {metadata['total_frames']}")

        return cap, metadata

    def compute_scene_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute scene embedding for frame using FastViT

        Args:
            frame: BGR frame from OpenCV

        Returns:
            Scene embedding vector or None if model unavailable
        """
        if self.scene_detector is None:
            return None

        try:
            # Resize frame
            resized = cv2.resize(
                frame,
                (Config.FRAME_RESIZE_DIM, Config.FRAME_RESIZE_DIM),
                interpolation=cv2.INTER_AREA,
            )

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Apply transforms and add batch dimension
            input_tensor = self.scene_detector["transforms"](pil_image).unsqueeze(0)
            input_tensor = input_tensor.to(self.model_manager.device)

            # Generate embedding
            with torch.no_grad():
                embedding = self.scene_detector["model"](input_tensor)
                embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalize

            return embedding.cpu().numpy()[0]

        except Exception as e:
            logger.warning(f"Failed to compute scene embedding: {e}")
            return None

    def is_interesting_frame(self, frame: np.ndarray) -> bool:
        """
        Determine if frame is interesting enough to process

        Args:
            frame: BGR frame from OpenCV

        Returns:
            True if frame should be processed
        """
        # If no scene detector, process all frames
        if self.scene_detector is None:
            return True

        current_embedding = self.compute_scene_embedding(frame)

        if current_embedding is None:
            return True

        # First frame is always interesting
        if self.last_scene_embedding is None:
            self.last_scene_embedding = current_embedding
            return True

        # Calculate cosine similarity
        similarity = np.dot(current_embedding, self.last_scene_embedding)

        # Frame is interesting if sufficiently different from last processed frame
        if similarity < Config.SCENE_SIMILARITY_THRESHOLD:
            self.last_scene_embedding = current_embedding
            return True

        return False

    def get_sampled_frames(
        self, video_path: str
    ) -> List[Tuple[int, float, np.ndarray]]:
        """
        Get intelligently sampled frames from video

        Args:
            video_path: Path to video file

        Returns:
            List of tuples: (frame_number, timestamp, frame_array)
        """
        cap, metadata = self.load_video(video_path)

        # Calculate frame interval for target FPS
        original_fps = metadata["fps"]
        frame_interval = int(original_fps / Config.TARGET_FPS)

        if frame_interval < 1:
            frame_interval = 1

        logger.info(
            f"Sampling every {frame_interval} frames (target {Config.TARGET_FPS} FPS)"
        )

        selected_frames = []
        frame_count = 0
        processed_count = 0

        pbar = tqdm(
            total=metadata["total_frames"],
            desc="Selecting frames",
            disable=not Config.PROGRESS_BAR,
        )

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Sample at target FPS
                if frame_count % frame_interval == 0:
                    # Check if frame is interesting
                    if self.is_interesting_frame(frame):
                        timestamp = frame_count / original_fps
                        selected_frames.append((frame_count, timestamp, frame.copy()))
                        processed_count += 1

                frame_count += 1
                pbar.update(1)

        finally:
            cap.release()
            pbar.close()

        logger.info(
            f"Selected {len(selected_frames)} interesting frames from {frame_count} total frames"
        )
        logger.info(
            f"Selection ratio: {len(selected_frames)/max(frame_count//frame_interval, 1):.2%}"
        )

        return selected_frames


# ============================================================================
# MODULE 2: TIER 1 - REAL-TIME PROCESSING LOOP
# ============================================================================


class RealTimeObjectProcessor:
    """
    Fast object detection and visual embedding generation (Tier 1)
    """

    def __init__(self, model_manager: PerceptionModelManager):
        self.model_manager = model_manager
        detector = model_manager.get_object_detector()
        if detector is None:  # pragma: no cover - defensive
            raise RuntimeError("Object detector failed to initialize")
        self.object_detector: Any = detector

        embedding_model = model_manager.get_embedding_model()
        if embedding_model is None:  # pragma: no cover - defensive
            raise RuntimeError("Embedding model failed to initialize")
        self.embedding_model: Dict[str, Any] = embedding_model
        self.detection_count = 0
        
        # Initialize motion tracker for causal inference
        self.motion_tracker = MotionTracker(smoothing_window=3)

    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in frame using YOLO

        Args:
            frame: BGR frame from OpenCV

        Returns:
            List of detection dictionaries
        """
        try:
            # Run YOLO inference
            results = self.object_detector(
                frame,
                conf=Config.YOLO_CONFIDENCE_THRESHOLD,
                iou=Config.YOLO_IOU_THRESHOLD,
                max_det=Config.YOLO_MAX_DETECTIONS,
                verbose=False,
            )

            detections = []

            # Parse results
            for result in results:
                boxes = result.boxes

                for i in range(len(boxes)):
                    # Get bounding box in xyxy format
                    bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                    x1, y1, x2, y2 = bbox

                    # Filter by size
                    width = x2 - x1
                    height = y2 - y1

                    if (
                        width < Config.MIN_OBJECT_SIZE
                        or height < Config.MIN_OBJECT_SIZE
                    ):
                        continue

                    detections.append(
                        {
                            "bbox": bbox,
                            "confidence": float(boxes.conf[i]),
                            "class_id": int(boxes.cls[i]),
                            "class_name": result.names[int(boxes.cls[i])],
                        }
                    )

            return detections

        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []

    def crop_object(
        self, frame: np.ndarray, bbox: List[int]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Crop object from frame with padding

        Args:
            frame: BGR frame
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Tuple of (cropped image, crop size)
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Add padding
        padding = Config.BBOX_PADDING_PERCENT
        width = x2 - x1
        height = y2 - y1

        x1_padded = max(0, int(x1 - width * padding))
        y1_padded = max(0, int(y1 - height * padding))
        x2_padded = min(w, int(x2 + width * padding))
        y2_padded = min(h, int(y2 + height * padding))

        crop = frame[y1_padded:y2_padded, x1_padded:x2_padded]
        crop_size = (x2_padded - x1_padded, y2_padded - y1_padded)

        return crop, crop_size

    def generate_visual_embedding(self, crop: np.ndarray) -> np.ndarray:
        """
        Generate visual embedding using embedding model

        Args:
            crop: Cropped object image (BGR)

        Returns:
            Normalized embedding vector
        """
        try:
            # Convert BGR to RGB
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_crop)

            # Apply transforms
            input_tensor = self.embedding_model["transforms"](pil_image).unsqueeze(0)
            input_tensor = input_tensor.to(self.model_manager.device)

            # Generate embedding
            with torch.no_grad():
                embedding = self.embedding_model["model"](input_tensor)
                embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalize

            return embedding.cpu().numpy()[0]

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return zero vector as fallback
            return np.zeros(Config.EMBEDDING_DIM, dtype=np.float32)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        background_queue: Queue,
        shared_perception_log: MutableSequence[Dict[str, Any]],
    ) -> int:
        """
        Process single frame: detect objects, generate embeddings, queue for description

        Queuing strategy depends on DESCRIPTION_MODE:
        - SCENE: Queue frame once with all object indices
        - OBJECT: Queue each object crop separately
        - HYBRID: Queue each object crop with full frame

        Args:
            frame: BGR frame
            frame_number: Frame index
            timestamp: Video timestamp
            background_queue: Queue for description workers
            shared_perception_log: Shared list for storing perception objects

        Returns:
            Number of objects detected
        """
        detections = self.detect_objects(frame)

        if not detections:
            return 0

        # Gather all detections for this frame (for context)
        frame_width = frame.shape[1]
        frame_detections = []
        object_indices = []

        for detection in detections:
            # Crop object
            crop, crop_size = self.crop_object(frame, detection["bbox"])

            # Generate visual embedding
            embedding = self.generate_visual_embedding(crop)

            # Calculate spatial position
            bbox = detection["bbox"]
            center_x = (bbox[0] + bbox[2]) / 2
            if center_x < frame_width / 3:
                position = "left"
            elif center_x < 2 * frame_width / 3:
                position = "center"
            else:
                position = "right"

            # Create perception object
            temp_id = f"det_{self.detection_count:06d}"
            self.detection_count += 1
            
            # Update motion tracking
            motion_data = self.motion_tracker.update(
                temp_id,
                timestamp,
                detection["bbox"]
            )

            perception_obj = RichPerceptionObject(
                timestamp=timestamp,
                frame_number=frame_number,
                bounding_box=detection["bbox"],
                visual_embedding=embedding.tolist(),
                detection_confidence=detection["confidence"],
                object_class=detection["class_name"],
                crop_size=crop_size,
                temp_id=temp_id,
                # Motion data for causal inference
                centroid=motion_data.centroid if motion_data else None,
                velocity=motion_data.velocity if motion_data else None,
                speed=motion_data.speed if motion_data else None,
                direction=motion_data.direction if motion_data else None,
            )

            # Add to shared log (will be updated by workers)
            obj_dict = perception_obj.to_dict()
            shared_perception_log.append(obj_dict)
            obj_index = len(shared_perception_log) - 1
            object_indices.append(obj_index)

            # Build detection info for context
            frame_detections.append(
                {
                    "label": detection["class_name"],
                    "bbox": detection["bbox"],
                    "position": position,
                    "confidence": detection["confidence"],
                }
            )

            # Queue for description generation based on mode
            mode = Config.DESCRIPTION_MODE

            if mode == DescriptionMode.OBJECT:
                # OBJECT MODE: Queue each object crop
                try:
                    background_queue.put(
                        (
                            crop.copy(),
                            obj_index,
                            temp_id,
                            frame_number,
                            detection["class_name"],
                            position,
                            frame_detections.copy(),
                        ),
                        timeout=Config.QUEUE_TIMEOUT,
                    )
                    logger.debug(f"Queued object {temp_id} (index {obj_index})")
                except Exception as e:
                    logger.error(f"Failed to queue {temp_id}: {e}")
                    import traceback

                    logger.error(traceback.format_exc())

            elif mode == DescriptionMode.HYBRID:
                # HYBRID MODE: Queue crop + full frame
                try:
                    background_queue.put(
                        (
                            crop.copy(),
                            obj_index,
                            temp_id,
                            frame_number,
                            detection["class_name"],
                            position,
                            frame.copy(),
                            frame_detections.copy(),
                        ),
                        timeout=Config.QUEUE_TIMEOUT,
                    )
                except Exception:
                    logger.warning(f"Queue full, skipping description for {temp_id}")

        # SCENE MODE: Queue frame once for all objects
        if Config.DESCRIPTION_MODE == DescriptionMode.SCENE and detections:
            try:
                background_queue.put(
                    (frame.copy(), frame_number, frame_detections, object_indices),
                    timeout=Config.QUEUE_TIMEOUT,
                )
            except Exception:
                logger.warning(
                    f"Queue full, skipping scene description for frame {frame_number}"
                )

        return len(detections)


# ============================================================================
# MODULE 3: TIER 2 - ASYNCHRONOUS DESCRIPTION PROCESS
# ============================================================================

# Global FastVLM model instance (loaded once per worker)
_FASTVLM_MODEL: Optional[Any] = None


def load_fastvlm_model() -> Optional[Any]:
    """Load the FastVLM Hugging Face model (shared across runtimes)."""
    global _FASTVLM_MODEL

    if _FASTVLM_MODEL is not None:
        return _FASTVLM_MODEL

    backend = get_active_backend()
    if backend is None:
        backend = select_backend()

    model_source: Optional[str] = (
        os.getenv("ORION_FASTVLM_MODEL")
        or os.getenv("ORION_FASTVLM_TORCH_MODEL")
        or os.getenv("ORION_FASTVLM_TORCH_PATH")
    )

    from .backends.torch_fastvlm import DEFAULT_MODEL_ID, FastVLMTorchWrapper

    asset_manager = _get_asset_manager()
    asset_dir = asset_manager.get_asset_dir("fastvlm-0.5b")

    if model_source is None:
        try:
            asset_manager.ensure_asset("fastvlm-0.5b")
            model_source = str(asset_dir)
        except Exception as asset_exc:  # pragma: no cover - handled below
            logger.warning(
                "FastVLM asset preparation failed locally (%s); falling back to Hugging Face streaming.",
                asset_exc,
            )
            model_source = DEFAULT_MODEL_ID

    logger.info(
        "Loading FastVLM model '%s' via Hugging Face transformers (requested backend: %s)",
        model_source,
        backend,
    )

    try:
        _FASTVLM_MODEL = FastVLMTorchWrapper(
            model_source=model_source,
            cache_dir=asset_dir,
        )
    except Exception as exc:  # pragma: no cover - handled by caller fallback
        logger.error("Failed to load FastVLM model '%s': %s", model_source, exc)
        logger.warning("Falling back to placeholder descriptions")
        import traceback

        logger.debug(traceback.format_exc())
        _FASTVLM_MODEL = None

    return _FASTVLM_MODEL


def generate_rich_description(
    image: np.ndarray,
    mode: DescriptionMode,
    object_class: str = "",
    object_position: str = "",
    all_frame_detections: Optional[List[Dict[str, Any]]] = None,
    use_fastvlm: bool = True,
) -> str:
    """
    Generate rich description using the standard FastVLM Hugging Face model

    Supports three modes:
    - SCENE: Full frame description with all objects
    - OBJECT: Focused object description with contextual awareness
    - HYBRID: Both scene + object (not implemented in this function, handled at worker level)

    Model details:
    - Source: apple/FastVLM-0.5B (Hugging Face)
    - Runtime: PyTorch (CPU, CUDA, or MPS automatically)

    Args:
        image: Full frame (SCENE mode) or object crop (OBJECT mode) - BGR format from OpenCV
        mode: Description mode (SCENE, OBJECT, or HYBRID)
        object_class: Object class for OBJECT mode (e.g., "person", "chair")
        object_position: Spatial position for OBJECT mode (e.g., "center", "left")
        all_frame_detections: List of all YOLO detections in frame with:
            - label: object class
            - bbox: bounding box coordinates
            - position: spatial position (left/center/right)
        use_fastvlm: Whether to use FastVLM or fallback to placeholder

    Returns:
        Rich textual description
    """
    logger.debug(
        f"generate_rich_description called with mode={mode}, object_class={object_class}"
    )

    # Try to load and use FastVLM model
    if use_fastvlm:
        try:
            logger.debug("Loading FastVLM model...")
            model = load_fastvlm_model()
            logger.debug(f"Model loaded: {model is not None}")

            if model is not None:
                logger.debug("Converting image to RGB...")
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                logger.debug(f"Image converted, size: {pil_image.size}")

                if mode == DescriptionMode.SCENE:
                    logger.debug("Building SCENE mode prompt...")
                    # SCENE MODE: Full frame description with all objects
                    # Build context-aware prompt with detections
                    if all_frame_detections:
                        objects = [
                            d.get("label", "object") for d in all_frame_detections[:5]
                        ]
                        object_str = ", ".join(objects)
                        prompt = f"Describe this scene in detail. Objects detected: {object_str}."
                    else:
                        prompt = "Describe this scene in detail, including all visible objects and their arrangement."

                    logger.debug(
                        f"Calling model.generate_description with prompt: {prompt[:50]}..."
                    )
                    description = model.generate_description(
                        image=pil_image,
                        prompt=prompt,
                        max_tokens=Config.DESCRIPTION_MAX_TOKENS,
                        temperature=Config.DESCRIPTION_TEMPERATURE,
                    )
                    logger.debug(
                        f"Got SCENE description: {description[:50] if description else 'None'}..."
                    )

                elif mode == DescriptionMode.OBJECT:
                    logger.debug("Building OBJECT mode prompt...")
                    # OBJECT MODE: Focused object description with context
                    # Build object-focused prompt
                    if all_frame_detections:
                        context_objects = [
                            d.get("label", "object") for d in all_frame_detections[:5]
                        ]
                        context_str = ", ".join(context_objects)
                        prompt = f"Describe this {object_class or 'object'} in detail. It is positioned {object_position or 'in the scene'}. Other objects in the scene: {context_str}."
                    else:
                        prompt = f"Describe this {object_class or 'object'} in detail, including its appearance, color, and characteristics."

                    logger.debug(
                        f"Calling model.generate_description with prompt: {prompt[:50]}..."
                    )
                    description = model.generate_description(
                        image=pil_image,
                        prompt=prompt,
                        max_tokens=150,  # Shorter for object-level
                        temperature=Config.DESCRIPTION_TEMPERATURE,
                    )
                    logger.debug(
                        f"Got OBJECT description: {description[:50] if description else 'None'}..."
                    )

                else:  # HYBRID mode
                    # For hybrid, this function should be called twice (once for scene, once for object)
                    # This shouldn't happen - hybrid is handled at worker level
                    logger.warning(
                        f"HYBRID mode should be handled at worker level, defaulting to OBJECT mode"
                    )
                    return generate_rich_description(
                        image,
                        DescriptionMode.OBJECT,
                        object_class,
                        object_position,
                        all_frame_detections,
                        use_fastvlm,
                    )

                return description

        except Exception as e:
            logger.warning(f"FastVLM generation failed: {e}, using fallback")
            import traceback

            logger.debug(traceback.format_exc())

    # Fallback to placeholder descriptions
    time.sleep(0.1)  # Simulate processing

    if mode == DescriptionMode.SCENE:
        # Build fallback based on detections if available
        if all_frame_detections:
            obj_list = ", ".join(
                [d.get("label", "object") for d in all_frame_detections[:3]]
            )
            return f"Scene containing {obj_list} with typical visual characteristics."
        return "A scene with various objects and visual elements."

    elif mode == DescriptionMode.OBJECT:
        # Object-specific fallback
        if object_class:
            return f"A {object_class} with typical appearance and features, positioned {object_position or 'in the scene'}."
        return "An object with distinct visual characteristics."

    else:  # HYBRID
        return "Scene with objects (hybrid mode fallback)."


def description_worker(
    worker_id: int,
    background_queue: Queue,
    shared_perception_log: MutableSequence[Dict[str, Any]],
    stop_event: SyncEvent,
):
    """
    Worker process for generating rich descriptions with YOLO context

    Supports three modes:
    - SCENE: Processes full frame once per frame, all objects share description
    - OBJECT: Processes object crop, generates unique description per object
    - HYBRID: Generates both scene + object descriptions

    Args:
        worker_id: Worker identifier
        background_queue: Input queue with tuples based on mode:
            - SCENE: (frame, frame_number, all_detections)
            - OBJECT: (crop, obj_index, temp_id, frame_number, object_class, position)
            - HYBRID: (crop, obj_index, temp_id, frame_number, object_class, position, frame, all_detections)
        shared_perception_log: Shared list to update with descriptions
        stop_event: Event to signal worker shutdown
    """
    worker_logger = setup_logger(f"Worker-{worker_id}")
    worker_logger.info(
        f"Description worker {worker_id} started (Mode: {Config.DESCRIPTION_MODE.value})"
    )

    # PRE-LOAD FastVLM model before processing (to avoid delays during queue processing)
    worker_logger.info(f"Worker {worker_id} pre-loading FastVLM model...")
    load_fastvlm_model()
    worker_logger.info(f"Worker {worker_id} model loaded, ready to process")

    processed_count = 0
    mode = Config.DESCRIPTION_MODE

    try:
        worker_logger.info(f"Worker {worker_id} entering processing loop...")
        while not stop_event.is_set():
            try:
                # Get item from queue with timeout
                worker_logger.debug(
                    f"Worker {worker_id} waiting for item from queue..."
                )
                item = background_queue.get(timeout=Config.QUEUE_TIMEOUT)
                worker_logger.debug(
                    f"Worker {worker_id} got item from queue: {type(item)}"
                )

                if item is None:  # Poison pill
                    worker_logger.info(f"Worker {worker_id} received stop signal")
                    break

                if mode == DescriptionMode.SCENE:
                    # SCENE MODE: (frame, frame_number, all_detections, object_indices)
                    frame, frame_number, frame_detections, object_indices = item

                    # Generate scene description once
                    scene_description = generate_rich_description(
                        image=frame,
                        mode=DescriptionMode.SCENE,
                        all_frame_detections=frame_detections,
                    )

                    # Update all objects from this frame with the same scene description
                    # CRITICAL: Manager proxy requires getting, modifying, then re-assigning
                    for obj_idx in object_indices:
                        obj_data = shared_perception_log[obj_idx]
                        obj_data["rich_description"] = scene_description
                        shared_perception_log[obj_idx] = obj_data

                    processed_count += len(object_indices)

                elif mode == DescriptionMode.OBJECT:
                    # OBJECT MODE: (crop, obj_index, temp_id, frame_number, object_class, position, frame_detections)
                    (
                        crop,
                        obj_index,
                        temp_id,
                        frame_number,
                        object_class,
                        position,
                        frame_detections,
                    ) = item

                    worker_logger.debug(
                        f"Worker {worker_id} unpacked item: obj_index={obj_index}, temp_id={temp_id}"
                    )

                    # Generate object-focused description
                    worker_logger.debug(
                        f"Worker {worker_id} calling generate_rich_description for {temp_id}"
                    )
                    object_description = generate_rich_description(
                        image=crop,
                        mode=DescriptionMode.OBJECT,
                        object_class=object_class,
                        object_position=position,
                        all_frame_detections=frame_detections,
                    )
                    worker_logger.debug(
                        f"Worker {worker_id} got description for {temp_id}: {object_description[:50] if object_description else 'None'}..."
                    )

                    # Update this specific object
                    # CRITICAL: Manager proxy requires getting, modifying, then re-assigning
                    obj_data = shared_perception_log[obj_index]
                    obj_data["rich_description"] = object_description
                    shared_perception_log[obj_index] = obj_data
                    worker_logger.debug(
                        f"Worker {worker_id} updated shared_perception_log[{obj_index}]"
                    )

                    processed_count += 1
                    worker_logger.debug(
                        f"Worker {worker_id} processed_count now: {processed_count}"
                    )

                elif mode == DescriptionMode.HYBRID:
                    # HYBRID MODE: Generate both scene + object descriptions
                    (
                        crop,
                        obj_index,
                        temp_id,
                        frame_number,
                        object_class,
                        position,
                        frame,
                        frame_detections,
                    ) = item

                    # Generate scene description (if not already done for this frame)
                    # Note: In production, you'd want to cache scene descriptions per frame
                    scene_description = generate_rich_description(
                        image=frame,
                        mode=DescriptionMode.SCENE,
                        all_frame_detections=frame_detections,
                    )

                    # Generate object description
                    object_description = generate_rich_description(
                        image=crop,
                        mode=DescriptionMode.OBJECT,
                        object_class=object_class,
                        object_position=position,
                        all_frame_detections=frame_detections,
                    )

                    # Store both descriptions
                    # CRITICAL: Manager proxy requires getting, modifying, then re-assigning
                    obj_data = shared_perception_log[obj_index]
                    obj_data["rich_description"] = object_description
                    obj_data["scene_description"] = scene_description
                    shared_perception_log[obj_index] = obj_data

                    processed_count += 1

                if processed_count % 10 == 0:
                    worker_logger.debug(
                        f"Worker {worker_id} processed {processed_count} items"
                    )

            except QueueEmpty:
                continue
            except Exception as e:
                worker_logger.error(f"Worker {worker_id} error: {e}")
                import traceback

                worker_logger.error(traceback.format_exc())

    finally:
        worker_logger.info(
            f"Worker {worker_id} shutting down. Processed {processed_count} items."
        )
        worker_logger.info(
            f"Worker {worker_id} stop_event.is_set() = {stop_event.is_set()}"
        )


# ============================================================================
# MAIN PERCEPTION ENGINE
# ============================================================================


def run_perception_engine(video_path: str) -> List[Dict[str, Any]]:
    """
    Main perception engine function - orchestrates the entire pipeline

    Args:
        video_path: Path to video file

    Returns:
        List of RichPerceptionObject dictionaries, fully populated
    """
    logger.info("=" * 80)
    logger.info("STARTING ASYNCHRONOUS PERCEPTION ENGINE")
    logger.info("=" * 80)

    start_time = time.time()

    # Initialize model manager
    model_manager = PerceptionModelManager()

    # Initialize components
    frame_selector = VideoFrameSelector(model_manager)
    object_processor = RealTimeObjectProcessor(model_manager)

    # Create multiprocessing infrastructure
    manager = Manager()
    shared_perception_log = cast(MutableSequence[Dict[str, Any]], manager.list())
    background_queue = Queue(maxsize=Config.QUEUE_MAX_SIZE)
    stop_event = mp.Event()

    # Start worker processes
    workers = []
    for i in range(Config.NUM_WORKERS):
        worker = Process(
            target=description_worker,
            args=(i, background_queue, shared_perception_log, stop_event),
        )
        worker.start()
        workers.append(worker)

    logger.info(f"Started {Config.NUM_WORKERS} description workers")

    # Phase 1: Frame selection and Tier 1 processing
    logger.info("\n" + "-" * 80)
    logger.info("PHASE 1: INTELLIGENT FRAME SELECTION & TIER 1 PROCESSING")
    logger.info("-" * 80)

    selected_frames = frame_selector.get_sampled_frames(video_path)

    total_detections = 0

    logger.info("\nProcessing selected frames...")
    for frame_number, timestamp, frame in tqdm(
        selected_frames,
        desc="Detecting & embedding objects",
        disable=not Config.PROGRESS_BAR,
    ):
        num_detections = object_processor.process_frame(
            frame, frame_number, timestamp, background_queue, shared_perception_log
        )
        total_detections += num_detections

    logger.info(f"\nTier 1 complete: {total_detections} objects detected")

    # Phase 2: Wait for Tier 2 (description generation) to complete
    logger.info("\n" + "-" * 80)
    logger.info("PHASE 2: TIER 2 ASYNCHRONOUS DESCRIPTION GENERATION")
    logger.info("-" * 80)

    # Send poison pills to workers
    for _ in range(Config.NUM_WORKERS):
        background_queue.put(None)

    logger.info("Waiting for description workers to complete...")

    # Wait for all workers with timeout
    for i, worker in enumerate(workers):
        worker.join(timeout=Config.WORKER_SHUTDOWN_TIMEOUT)
        if worker.is_alive():
            logger.warning(f"Worker {i} did not terminate gracefully, forcing shutdown")
            worker.terminate()

    # Convert shared list to regular list of dicts
    perception_log = list(shared_perception_log)

    # Verify completeness
    complete_count = sum(
        1 for obj in perception_log if obj["rich_description"] is not None
    )

    logger.info(
        f"\nDescription generation complete: {complete_count}/{len(perception_log)} objects"
    )

    if complete_count < len(perception_log):
        logger.warning(
            f"{len(perception_log) - complete_count} objects missing descriptions"
        )

    # Final statistics
    elapsed_time = time.time() - start_time

    logger.info("\n" + "=" * 80)
    logger.info("PERCEPTION ENGINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total frames processed: {len(selected_frames)}")
    logger.info(f"Total objects detected: {len(perception_log)}")
    logger.info(f"Complete perception objects: {complete_count}")
    logger.info(f"Total time: {elapsed_time:.2f}s")
    logger.info(f"Processing rate: {len(selected_frames)/elapsed_time:.2f} frames/sec")
    logger.info("=" * 80 + "\n")

    return perception_log


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == "__main__":
    # Example usage
    import json

    # Configure paths
    VIDEO_PATH = "/Users/riddhiman.rana/Desktop/Coding/Orion/orion-research/data/examples/sample_video.mp4"
    OUTPUT_PATH = "/Users/riddhiman.rana/Desktop/Coding/Orion/orion-research/data/perception_log.json"

    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        logger.error(f"Video file not found: {VIDEO_PATH}")
        logger.info("Please update VIDEO_PATH to point to a valid video file")
        sys.exit(1)

    try:
        # Run perception engine
        perception_log = run_perception_engine(VIDEO_PATH)

        # Save results
        output_dir = os.path.dirname(OUTPUT_PATH)
        os.makedirs(output_dir, exist_ok=True)

        with open(OUTPUT_PATH, "w") as f:
            json.dump(perception_log, f, indent=2)

        logger.info(f"Perception log saved to: {OUTPUT_PATH}")

        # Print sample results
        if perception_log:
            logger.info("\nSample perception object:")
            logger.info(json.dumps(perception_log[0], indent=2))

    except Exception as e:
        logger.error(f"Perception engine failed: {e}", exc_info=True)
        sys.exit(1)
