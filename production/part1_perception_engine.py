"""
Part 1: The Asynchronous Perception Engine
===========================================

This module implements a two-tier asynchronous video processing system that:
1. Intelligently selects interesting frames using scene change detection
2. Performs fast object detection and visual embedding generation (Tier 1)
3. Asynchronously generates rich semantic descriptions (Tier 2)

Author: Orion Research Team
Date: October 3, 2025
"""

import os
import sys
import time
import logging
import multiprocessing as mp
from multiprocessing import Queue, Process, Manager
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import warnings

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

class Config:
    """Configuration parameters for the perception engine"""
    
    # Video Processing
    TARGET_FPS = 4.0  # Sample frames at 4 FPS
    SCENE_SIMILARITY_THRESHOLD = 0.98  # Threshold for scene change detection
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
    
    # FastVLM Description Generation
    DESCRIPTION_MAX_TOKENS = 150
    DESCRIPTION_TEMPERATURE = 0.7
    DESCRIPTION_PROMPT = (
        "Describe this object in detail, including its appearance, state, "
        "actions, and any notable features. Be specific and concise."
    )
    
    # Performance & Logging
    LOG_LEVEL = logging.INFO
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
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

logger = setup_logger('PerceptionEngine')


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def is_complete(self) -> bool:
        """Check if all required fields are populated"""
        return self.rich_description is not None


# ============================================================================
# MODEL LOADING UTILITIES
# ============================================================================

class ModelManager:
    """
    Manages loading and caching of all required models
    """
    
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def get_scene_detector(self):
        """Load FastViT model for scene change detection"""
        if 'scene_detector' not in self._models:
            try:
                import timm
                logger.info("Loading FastViT model for scene detection...")
                
                # Load lightweight FastViT model
                model = timm.create_model('fastvit_t8.apple_in1k', pretrained=True)
                model = model.to(self.device)
                model.eval()
                
                # Get the model's data configuration for preprocessing
                data_config = timm.data.resolve_model_data_config(model)
                transforms = timm.data.create_transform(**data_config, is_training=False)
                
                self._models['scene_detector'] = {
                    'model': model,
                    'transforms': transforms
                }
                logger.info("FastViT model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load FastViT model: {e}")
                logger.warning("Scene detection will be disabled")
                self._models['scene_detector'] = None
        
        return self._models.get('scene_detector')
    
    def get_object_detector(self):
        """Load YOLOv11 model for object detection"""
        if 'object_detector' not in self._models:
            try:
                from ultralytics import YOLO
                logger.info("Loading YOLOv11m model...")
                
                model = YOLO('yolov11m.pt')  # Will auto-download if not present
                
                self._models['object_detector'] = model
                logger.info("YOLOv11m model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                raise RuntimeError("Object detector is required for perception engine")
        
        return self._models.get('object_detector')
    
    def get_embedding_model(self):
        """Load OSNet model for visual embeddings"""
        if 'embedding_model' not in self._models:
            try:
                import timm
                logger.info("Loading OSNet model for visual embeddings...")
                
                # Use a ResNet-based feature extractor as OSNet alternative
                # OSNet might not be directly available in timm, so we use resnet50
                model = timm.create_model('resnet50', pretrained=True, num_classes=0)
                model = model.to(self.device)
                model.eval()
                
                # Create preprocessing transforms
                data_config = timm.data.resolve_model_data_config(model)
                transforms = timm.data.create_transform(**data_config, is_training=False)
                
                self._models['embedding_model'] = {
                    'model': model,
                    'transforms': transforms
                }
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise RuntimeError("Embedding model is required for perception engine")
        
        return self._models.get('embedding_model')


# ============================================================================
# MODULE 1: VIDEO INGESTION & FRAME SELECTION
# ============================================================================

class VideoFrameSelector:
    """
    Handles intelligent frame selection using scene change detection
    """
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.scene_detector = model_manager.get_scene_detector()
        self.last_scene_embedding = None
        
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
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
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
            resized = cv2.resize(frame, (Config.FRAME_RESIZE_DIM, Config.FRAME_RESIZE_DIM),
                               interpolation=cv2.INTER_AREA)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Apply transforms and add batch dimension
            input_tensor = self.scene_detector['transforms'](pil_image).unsqueeze(0)
            input_tensor = input_tensor.to(self.model_manager.device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.scene_detector['model'](input_tensor)
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
    
    def get_sampled_frames(self, video_path: str) -> List[Tuple[int, float, np.ndarray]]:
        """
        Get intelligently sampled frames from video
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of tuples: (frame_number, timestamp, frame_array)
        """
        cap, metadata = self.load_video(video_path)
        
        # Calculate frame interval for target FPS
        original_fps = metadata['fps']
        frame_interval = int(original_fps / Config.TARGET_FPS)
        
        if frame_interval < 1:
            frame_interval = 1
        
        logger.info(f"Sampling every {frame_interval} frames (target {Config.TARGET_FPS} FPS)")
        
        selected_frames = []
        frame_count = 0
        processed_count = 0
        
        pbar = tqdm(total=metadata['total_frames'], desc="Selecting frames",
                   disable=not Config.PROGRESS_BAR)
        
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
        
        logger.info(f"Selected {len(selected_frames)} interesting frames from {frame_count} total frames")
        logger.info(f"Selection ratio: {len(selected_frames)/max(frame_count//frame_interval, 1):.2%}")
        
        return selected_frames


# ============================================================================
# MODULE 2: TIER 1 - REAL-TIME PROCESSING LOOP
# ============================================================================

class RealTimeObjectProcessor:
    """
    Fast object detection and visual embedding generation (Tier 1)
    """
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.object_detector = model_manager.get_object_detector()
        self.embedding_model = model_manager.get_embedding_model()
        self.detection_count = 0
    
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
                verbose=False
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
                    
                    if width < Config.MIN_OBJECT_SIZE or height < Config.MIN_OBJECT_SIZE:
                        continue
                    
                    detections.append({
                        'bbox': bbox,
                        'confidence': float(boxes.conf[i]),
                        'class_id': int(boxes.cls[i]),
                        'class_name': result.names[int(boxes.cls[i])]
                    })
            
            return detections
        
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    def crop_object(self, frame: np.ndarray, bbox: List[int]) -> Tuple[np.ndarray, Tuple[int, int]]:
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
            input_tensor = self.embedding_model['transforms'](pil_image).unsqueeze(0)
            input_tensor = input_tensor.to(self.model_manager.device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.embedding_model['model'](input_tensor)
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
        shared_perception_log: List
    ) -> int:
        """
        Process single frame: detect objects, generate embeddings, queue for description
        
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
        
        for detection in detections:
            # Crop object
            crop, crop_size = self.crop_object(frame, detection['bbox'])
            
            # Generate visual embedding
            embedding = self.generate_visual_embedding(crop)
            
            # Create perception object
            temp_id = f"det_{self.detection_count:06d}"
            self.detection_count += 1
            
            perception_obj = RichPerceptionObject(
                timestamp=timestamp,
                frame_number=frame_number,
                bounding_box=detection['bbox'],
                visual_embedding=embedding.tolist(),
                detection_confidence=detection['confidence'],
                object_class=detection['class_name'],
                crop_size=crop_size,
                temp_id=temp_id
            )
            
            # Add to shared log (will be updated by workers)
            obj_dict = perception_obj.to_dict()
            shared_perception_log.append(obj_dict)
            obj_index = len(shared_perception_log) - 1
            
            # Queue for description generation
            try:
                background_queue.put(
                    (crop.copy(), obj_index, temp_id),
                    timeout=Config.QUEUE_TIMEOUT
                )
            except:
                logger.warning(f"Queue full, skipping description for {temp_id}")
        
        return len(detections)


# ============================================================================
# MODULE 3: TIER 2 - ASYNCHRONOUS DESCRIPTION PROCESS
# ============================================================================

def generate_rich_description(image: np.ndarray, object_class: str = "") -> str:
    """
    PLACEHOLDER: Generate rich description using FastVLM
    
    In the full implementation, this would:
    1. Load the FastVLM model from ml-fastvlm directory
    2. Prepare the image and prompt
    3. Run inference with the model
    4. Return the generated description
    
    Args:
        image: Cropped object image (BGR)
        object_class: Object class from YOLO
        
    Returns:
        Rich textual description
    """
    # PLACEHOLDER IMPLEMENTATION
    # In production, this would use the FastVLM model from ml-fastvlm/
    
    # Simulate processing time
    time.sleep(0.1)
    
    # Generate placeholder description based on object class
    descriptions = {
        'person': 'A person wearing casual clothing, standing upright with neutral posture.',
        'car': 'A four-wheeled vehicle with standard automotive features, stationary.',
        'dog': 'A medium-sized dog with brown fur, alert and attentive.',
        'cat': 'A small feline with sleek fur, sitting in a relaxed position.',
        'default': f'An object of type {object_class} with distinct visual features.'
    }
    
    return descriptions.get(object_class, descriptions['default'])


def description_worker(
    worker_id: int,
    background_queue: Queue,
    shared_perception_log: List,
    stop_event: mp.Event
):
    """
    Worker process for generating rich descriptions
    
    Args:
        worker_id: Worker identifier
        background_queue: Input queue with (image, index, temp_id) tuples
        shared_perception_log: Shared list to update with descriptions
        stop_event: Event to signal worker shutdown
    """
    worker_logger = setup_logger(f'Worker-{worker_id}')
    worker_logger.info(f"Description worker {worker_id} started")
    
    processed_count = 0
    
    try:
        while not stop_event.is_set():
            try:
                # Get item from queue with timeout
                item = background_queue.get(timeout=Config.QUEUE_TIMEOUT)
                
                if item is None:  # Poison pill
                    worker_logger.info(f"Worker {worker_id} received stop signal")
                    break
                
                crop, obj_index, temp_id = item
                
                # Get object class for context-aware description
                object_class = shared_perception_log[obj_index].get('object_class', '')
                
                # Generate description
                description = generate_rich_description(crop, object_class)
                
                # Update shared perception log
                shared_perception_log[obj_index]['rich_description'] = description
                
                processed_count += 1
                
                if processed_count % 10 == 0:
                    worker_logger.debug(f"Worker {worker_id} processed {processed_count} objects")
            
            except mp.queues.Empty:
                continue
            except Exception as e:
                worker_logger.error(f"Worker {worker_id} error: {e}")
    
    finally:
        worker_logger.info(f"Worker {worker_id} shutting down. Processed {processed_count} objects.")


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
    logger.info("="*80)
    logger.info("STARTING ASYNCHRONOUS PERCEPTION ENGINE")
    logger.info("="*80)
    
    start_time = time.time()
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # Initialize components
    frame_selector = VideoFrameSelector(model_manager)
    object_processor = RealTimeObjectProcessor(model_manager)
    
    # Create multiprocessing infrastructure
    manager = Manager()
    shared_perception_log = manager.list()
    background_queue = Queue(maxsize=Config.QUEUE_MAX_SIZE)
    stop_event = mp.Event()
    
    # Start worker processes
    workers = []
    for i in range(Config.NUM_WORKERS):
        worker = Process(
            target=description_worker,
            args=(i, background_queue, shared_perception_log, stop_event)
        )
        worker.start()
        workers.append(worker)
    
    logger.info(f"Started {Config.NUM_WORKERS} description workers")
    
    # Phase 1: Frame selection and Tier 1 processing
    logger.info("\n" + "-"*80)
    logger.info("PHASE 1: INTELLIGENT FRAME SELECTION & TIER 1 PROCESSING")
    logger.info("-"*80)
    
    selected_frames = frame_selector.get_sampled_frames(video_path)
    
    total_detections = 0
    
    logger.info("\nProcessing selected frames...")
    for frame_number, timestamp, frame in tqdm(selected_frames, 
                                               desc="Detecting & embedding objects",
                                               disable=not Config.PROGRESS_BAR):
        num_detections = object_processor.process_frame(
            frame, frame_number, timestamp,
            background_queue, shared_perception_log
        )
        total_detections += num_detections
    
    logger.info(f"\nTier 1 complete: {total_detections} objects detected")
    
    # Phase 2: Wait for Tier 2 (description generation) to complete
    logger.info("\n" + "-"*80)
    logger.info("PHASE 2: TIER 2 ASYNCHRONOUS DESCRIPTION GENERATION")
    logger.info("-"*80)
    
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
    complete_count = sum(1 for obj in perception_log if obj['rich_description'] is not None)
    
    logger.info(f"\nDescription generation complete: {complete_count}/{len(perception_log)} objects")
    
    if complete_count < len(perception_log):
        logger.warning(f"{len(perception_log) - complete_count} objects missing descriptions")
    
    # Final statistics
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("PERCEPTION ENGINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Total frames processed: {len(selected_frames)}")
    logger.info(f"Total objects detected: {len(perception_log)}")
    logger.info(f"Complete perception objects: {complete_count}")
    logger.info(f"Total time: {elapsed_time:.2f}s")
    logger.info(f"Processing rate: {len(selected_frames)/elapsed_time:.2f} frames/sec")
    logger.info("="*80 + "\n")
    
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
        
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(perception_log, f, indent=2)
        
        logger.info(f"Perception log saved to: {OUTPUT_PATH}")
        
        # Print sample results
        if perception_log:
            logger.info("\nSample perception object:")
            logger.info(json.dumps(perception_log[0], indent=2))
    
    except Exception as e:
        logger.error(f"Perception engine failed: {e}", exc_info=True)
        sys.exit(1)
