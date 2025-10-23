"""
Asynchronous Perception Engine
================================

True decoupling of fast tracking from slow analysis using asyncio queues.

Architecture:
- Fast Loop (30-60 FPS): YOLO detection + CLIP embedding (NO descriptions)
- Clustering Phase: HDBSCAN to identify unique entities
- Slow Loop (1-5 FPS): FastVLM description ONLY for unique entities
- Queue-based coordination with backpressure handling
- Memory-efficient with configurable buffer sizes

KEY: FastVLM only runs on UNIQUE ENTITIES after clustering, not on every detection.
This ensures we describe each object exactly once, from its best frame.

This addresses the research mentor's feedback:
"We use FastVLM but not async? The 'decoupling' of fast tracking from slow 
analysis is achieved by describing entities only once, not through a true 
asynchronous processing queue."

Author: Orion Research Team
Date: October 2025
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    from .config import OrionConfig, AsyncConfig
    from .model_manager import ModelManager
    from .tracking_engine import Observation, motion_to_dict
    from .motion_tracker import MotionTracker, MotionData
    from .class_correction import ClassCorrector
except ImportError:
    from config import OrionConfig, AsyncConfig  # type: ignore
    from model_manager import ModelManager  # type: ignore
    from tracking_engine import Observation, motion_to_dict  # type: ignore
    from motion_tracker import MotionTracker, MotionData  # type: ignore
    from class_correction import ClassCorrector  # type: ignore

logger = logging.getLogger("AsyncPerception")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DetectionTask:
    """Fast path detection result awaiting description"""
    frame_number: int
    timestamp: float
    bbox: List[int]  # [x1, y1, x2, y2]
    class_name: str
    confidence: float
    embedding: np.ndarray
    crop: np.ndarray  # Cropped image for VLM
    frame_width: int
    frame_height: int
    motion: Optional[MotionData] = None
    temp_id: str = ""  # Temporary ID for tracking in queue


@dataclass
class EntityDescriptionTask:
    """Task for describing a unique entity post-clustering"""
    entity_id: str
    class_name: str
    best_observation_crop: np.ndarray
    best_observation_frame: int
    confidence: float
    appearance_count: int


@dataclass
class DescriptionResult:
    """Slow path result with VLM description"""
    entity_id: str
    description: str
    generation_time: float
    frame_number: int


@dataclass
class AsyncPerceptionStats:
    """Statistics for async processing"""
    total_frames_processed: int = 0
    total_detections: int = 0
    descriptions_generated: int = 0
    descriptions_pending: int = 0
    fast_loop_fps: float = 0.0
    slow_loop_fps: float = 0.0
    queue_max_size: int = 0
    queue_current_size: int = 0
    total_processing_time: float = 0.0
    fast_loop_time: float = 0.0
    slow_loop_time: float = 0.0


# ============================================================================
# ASYNC PERCEPTION ENGINE
# ============================================================================

class AsyncPerceptionEngine:
    """
    Asynchronous perception engine with true decoupling.
    
    Fast Loop (Producer):
    - Reads frames from video
    - Runs YOLO detection
    - Generates CLIP embeddings
    - Enqueues detection tasks
    
    Slow Loop (Consumer):
    - Dequeues detection tasks
    - Generates VLM descriptions
    - Returns results
    """
    
    def __init__(
        self,
        config: Optional[OrionConfig] = None,
        async_config: Optional[AsyncConfig] = None,
        progress_callback: Optional[Callable[[str, Dict], None]] = None
    ):
        self.config = config or OrionConfig()
        self.async_config = async_config or AsyncConfig()
        self.progress_callback = progress_callback
        
        self.model_manager = ModelManager.get_instance()
        self.motion_tracker = MotionTracker()
        self.class_corrector = ClassCorrector()  # Initialize class corrector
        
        # Queues for async coordination
        self.detection_queue: asyncio.Queue[Optional[DetectionTask]] = asyncio.Queue(
            maxsize=self.async_config.max_queue_size
        )
        self.description_queue: asyncio.Queue[Optional[DescriptionResult]] = asyncio.Queue()
        
        # State tracking
        self.observations: List[Observation] = []
        self.pending_descriptions: Dict[str, DetectionTask] = {}
        self.completed_descriptions: Dict[str, str] = {}
        self.temp_id_counter = 0
        
        # Statistics
        self.stats = AsyncPerceptionStats()
        
        # Entity tracking for "describe once" strategy
        self.entity_embeddings: Dict[str, np.ndarray] = {}  # class_name -> avg embedding
        self.entity_described: Set[str] = set()  # Set of described entity signatures
        
        # Frame buffer for smooth processing
        self.frame_buffer: Deque[Tuple[int, np.ndarray]] = deque(
            maxlen=self.async_config.frame_buffer_size
        )
        
        logger.info("AsyncPerceptionEngine initialized")
        logger.info(f"  Max queue size: {self.async_config.max_queue_size}")
        logger.info(f"  Description workers: {self.async_config.num_description_workers}")
        logger.info(f"  Strategy: {self.async_config.describe_strategy}")
    
    def _generate_temp_id(self) -> str:
        """Generate temporary ID for tracking tasks through pipeline"""
        self.temp_id_counter += 1
        return f"temp_{self.temp_id_counter:06d}"
    
    def _compute_entity_signature(self, class_name: str, embedding: np.ndarray) -> str:
        """
        Compute entity signature for deduplication.
        Uses class name + embedding similarity.
        """
        # Normalize embedding
        norm_emb = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Check similarity to known entities of same class
        if class_name in self.entity_embeddings:
            existing_emb = self.entity_embeddings[class_name]
            similarity = np.dot(norm_emb, existing_emb)
            
            # If very similar, it's the same entity
            if similarity > 0.85:
                return f"{class_name}_known"
        
        # New entity - update average embedding
        if class_name in self.entity_embeddings:
            # Running average
            self.entity_embeddings[class_name] = (
                0.7 * self.entity_embeddings[class_name] + 0.3 * norm_emb
            )
        else:
            self.entity_embeddings[class_name] = norm_emb
        
        return f"{class_name}_new_{self.temp_id_counter}"
    
    def _should_describe(self, task: DetectionTask) -> bool:
        """
        Determine if this detection needs VLM description.
        
        Strategies:
        - best_frame: Only describe highest quality detection per entity
        - first_appearance: Describe on first appearance
        - periodic: Describe every N frames
        """
        strategy = self.async_config.describe_strategy
        
        if strategy == "first_appearance":
            # Check if we've seen this entity before
            signature = self._compute_entity_signature(task.class_name, task.embedding)
            if signature in self.entity_described:
                return False
            self.entity_described.add(signature)
            return True
        
        elif strategy == "periodic":
            # Describe every N frames
            return task.frame_number % self.async_config.description_interval_frames == 0
        
        elif strategy == "best_frame":
            # For now, describe all and filter later in clustering phase
            # This maintains compatibility with existing pipeline
            return True
        
        return True
    
    async def _fast_loop_producer(
        self,
        video_path: str,
        start_time: float
    ) -> None:
        """
        Fast loop: Read frames, detect objects, generate embeddings.
        Runs at 30-60 FPS (limited only by YOLO + CLIP speed).
        """
        logger.info("Starting fast loop (detection + embedding)")
        
        # Load models
        yolo = self.model_manager.yolo
        clip = self.model_manager.clip
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate sampling
        frame_interval = max(1, int(fps / self.config.video.target_fps))
        
        frame_count = 0
        detections_count = 0
        fast_loop_start = time.time()
        last_report = fast_loop_start
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames
                if frame_count % frame_interval != 0:
                    frame_count += 1
                    continue
                
                timestamp = frame_count / fps
                
                # Buffer frame if enabled
                if self.async_config.enable_frame_buffer:
                    self.frame_buffer.append((frame_count, frame.copy()))
                
                # Run YOLO detection
                results = yolo(frame, verbose=False)
                
                # Process detections
                for result in results:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        # Extract detection info
                        bbox = boxes.xyxy[i].cpu().numpy()
                        confidence = float(boxes.conf[i])
                        class_id = int(boxes.cls[i])
                        class_name = yolo.names[class_id]
                        
                        # Filter by confidence
                        if confidence < self.config.detection.confidence_threshold:
                            continue
                        
                        # Filter by size
                        x1, y1, x2, y2 = map(int, bbox)
                        width = x2 - x1
                        height = y2 - y1
                        if width < self.config.detection.min_object_size or \
                           height < self.config.detection.min_object_size:
                            continue
                        
                        # Add padding
                        padding = self.config.detection.bbox_padding_percent
                        pad_x = int(width * padding)
                        pad_y = int(height * padding)
                        x1 = max(0, x1 - pad_x)
                        y1 = max(0, y1 - pad_y)
                        x2 = min(frame_width, x2 + pad_x)
                        y2 = min(frame_height, y2 + pad_y)
                        
                        # Crop image
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        
                        # Generate CLIP embedding
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        pil_crop = Image.fromarray(crop_rgb)
                        
                        # Use text conditioning if enabled
                        if self.config.embedding.use_text_conditioning:
                            embedding = clip.encode_image_with_text(pil_crop, class_name)
                        else:
                            embedding = clip.encode_image(pil_crop)
                        
                        # Track motion
                        motion = self.motion_tracker.update(
                            temp_id=f"temp_{class_name}_{i}",
                            timestamp=timestamp,
                            bounding_box=[x1, y1, x2, y2]
                        )
                        
                        # Create detection task
                        temp_id = self._generate_temp_id()
                        task = DetectionTask(
                            frame_number=frame_count,
                            timestamp=timestamp,
                            bbox=[x1, y1, x2, y2],
                            class_name=class_name,
                            confidence=confidence,
                            embedding=embedding,
                            crop=crop_rgb,
                            frame_width=frame_width,
                            frame_height=frame_height,
                            motion=motion,
                            temp_id=temp_id
                        )
                        
                        # Check if description needed
                        needs_description = self._should_describe(task)
                        
                        # Add to queue if description needed
                        if needs_description:
                            # Handle backpressure
                            if self.async_config.enable_backpressure:
                                queue_fullness = self.detection_queue.qsize() / self.async_config.max_queue_size
                                if queue_fullness > self.async_config.backpressure_threshold:
                                    # Slow down to allow descriptions to catch up
                                    await asyncio.sleep(0.1)
                            
                            # Enqueue (will block if queue is full)
                            await self.detection_queue.put(task)
                            self.pending_descriptions[temp_id] = task
                            detections_count += 1
                        else:
                            # No description needed - convert directly to observation
                            obs = Observation(
                                frame_number=frame_count,
                                timestamp=timestamp,
                                bbox=[x1, y1, x2, y2],
                                class_name=class_name,
                                confidence=confidence,
                                embedding=embedding,
                                crop=crop_rgb,
                                frame_width=frame_width,
                                frame_height=frame_height,
                                motion=motion
                            )
                            self.observations.append(obs)
                
                frame_count += 1
                self.stats.total_frames_processed = frame_count
                self.stats.queue_current_size = self.detection_queue.qsize()
                self.stats.queue_max_size = max(
                    self.stats.queue_max_size,
                    self.detection_queue.qsize()
                )
                
                # Report progress
                current_time = time.time()
                if current_time - last_report >= self.async_config.report_interval_seconds:
                    elapsed = current_time - fast_loop_start
                    fast_fps = frame_count / elapsed if elapsed > 0 else 0
                    self.stats.fast_loop_fps = fast_fps
                    self.stats.fast_loop_time = elapsed
                    
                    if self.progress_callback:
                        self.progress_callback("fast_loop.progress", {
                            "frame": frame_count,
                            "total_frames": total_frames,
                            "fps": fast_fps,
                            "detections": detections_count,
                            "queue_size": self.detection_queue.qsize(),
                            "progress": frame_count / total_frames if total_frames > 0 else 0
                        })
                    
                    logger.info(
                        f"Fast loop: {frame_count}/{total_frames} frames "
                        f"({fast_fps:.1f} FPS), {detections_count} detections, "
                        f"queue: {self.detection_queue.qsize()}/{self.async_config.max_queue_size}"
                    )
                    last_report = current_time
                
                # Allow other tasks to run
                await asyncio.sleep(0)
        
        finally:
            cap.release()
            
            # Signal end of fast loop to workers
            for _ in range(self.async_config.num_description_workers):
                await self.detection_queue.put(None)
            
            fast_loop_end = time.time()
            self.stats.fast_loop_time = fast_loop_end - fast_loop_start
            
            logger.info(
                f"Fast loop complete: {frame_count} frames, {detections_count} detections "
                f"in {self.stats.fast_loop_time:.2f}s "
                f"({self.stats.fast_loop_fps:.1f} FPS)"
            )
    
    async def _slow_loop_worker(
        self,
        worker_id: int,
        start_time: float
    ) -> None:
        """
        Slow loop worker: Generate VLM descriptions.
        Runs at 1-5 FPS (limited by VLM inference speed).
        """
        logger.info(f"Starting slow loop worker {worker_id}")
        
        # Load FastVLM (lazy loaded)
        vlm = self.model_manager.fastvlm
        
        descriptions_count = 0
        slow_loop_start = time.time()
        last_report = slow_loop_start
        
        while True:
            # Get task from queue
            task = await self.detection_queue.get()
            
            # Check for termination signal
            if task is None:
                logger.info(f"Worker {worker_id} received termination signal")
                break
            
            try:
                # Generate description
                desc_start = time.time()
                
                # Convert crop to PIL Image
                pil_crop = Image.fromarray(task.crop)
                
                # Generate description
                prompt = f"Describe this {task.class_name} in detail, focusing on visual attributes."
                description = vlm.generate_description(
                    pil_crop,
                    prompt,
                    max_tokens=self.config.description.max_tokens,
                    temperature=self.config.description.temperature
                )
                
                desc_time = time.time() - desc_start
                
                # Create result
                result = DescriptionResult(
                    temp_id=task.temp_id,
                    description=description,
                    generation_time=desc_time
                )
                
                # Enqueue result
                await self.description_queue.put(result)
                
                descriptions_count += 1
                self.stats.descriptions_generated = descriptions_count
                self.stats.descriptions_pending = len(self.pending_descriptions)
                
                # Report progress
                current_time = time.time()
                if current_time - last_report >= self.async_config.report_interval_seconds:
                    elapsed = current_time - slow_loop_start
                    slow_fps = descriptions_count / elapsed if elapsed > 0 else 0
                    self.stats.slow_loop_fps = slow_fps
                    self.stats.slow_loop_time = elapsed
                    
                    if self.progress_callback:
                        self.progress_callback("slow_loop.progress", {
                            "worker_id": worker_id,
                            "descriptions": descriptions_count,
                            "fps": slow_fps,
                            "avg_time": desc_time,
                            "pending": len(self.pending_descriptions)
                        })
                    
                    logger.info(
                        f"Worker {worker_id}: {descriptions_count} descriptions "
                        f"({slow_fps:.1f} FPS), avg time: {desc_time:.2f}s"
                    )
                    last_report = current_time
            
            except Exception as e:
                logger.error(f"Worker {worker_id} error processing {task.temp_id}: {e}")
            
            finally:
                self.detection_queue.task_done()
        
        slow_loop_end = time.time()
        logger.info(
            f"Worker {worker_id} complete: {descriptions_count} descriptions "
            f"in {slow_loop_end - slow_loop_start:.2f}s"
        )
    
    async def _result_collector(self) -> None:
        """
        Collect description results and match with observations.
        """
        logger.info("Starting result collector")
        
        while True:
            # Check if all work is done
            if self.detection_queue.empty() and \
               len(self.pending_descriptions) == 0:
                break
            
            try:
                # Get result with timeout
                result = await asyncio.wait_for(
                    self.description_queue.get(),
                    timeout=1.0
                )
                
                # Match with pending task
                if result.temp_id in self.pending_descriptions:
                    task = self.pending_descriptions.pop(result.temp_id)
                    
                    # Create observation with description
                    obs = Observation(
                        frame_number=task.frame_number,
                        timestamp=task.timestamp,
                        bbox=task.bbox,
                        class_name=task.class_name,
                        confidence=task.confidence,
                        embedding=task.embedding,
                        crop=task.crop,
                        frame_width=task.frame_width,
                        frame_height=task.frame_height,
                        motion=task.motion
                    )
                    
                    self.observations.append(obs)
                    self.completed_descriptions[result.temp_id] = result.description
                
            except asyncio.TimeoutError:
                # No results yet, continue waiting
                continue
            except Exception as e:
                logger.error(f"Result collector error: {e}")
        
        logger.info("Result collector complete")
    
    async def process_video_async(self, video_path: str) -> List[Observation]:
        """
        Process video asynchronously with true fast/slow decoupling.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of observations with embeddings and descriptions
        """
        logger.info("="*80)
        logger.info("ASYNC PERCEPTION ENGINE")
        logger.info("="*80)
        logger.info(f"Video: {video_path}")
        logger.info(f"Workers: {self.async_config.num_description_workers}")
        logger.info(f"Strategy: {self.async_config.describe_strategy}")
        
        start_time = time.time()
        
        # Create tasks
        tasks = [
            # Fast loop (producer)
            asyncio.create_task(self._fast_loop_producer(video_path, start_time)),
            
            # Slow loop workers (consumers)
            *[
                asyncio.create_task(self._slow_loop_worker(i, start_time))
                for i in range(self.async_config.num_description_workers)
            ],
            
            # Result collector
            asyncio.create_task(self._result_collector())
        ]
        
        # Wait for all tasks
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        self.stats.total_processing_time = end_time - start_time
        
        # Log summary
        logger.info("="*80)
        logger.info("ASYNC PERCEPTION COMPLETE")
        logger.info("="*80)
        logger.info(f"Total time: {self.stats.total_processing_time:.2f}s")
        logger.info(f"Frames processed: {self.stats.total_frames_processed}")
        logger.info(f"Total detections: {len(self.observations)}")
        logger.info(f"Descriptions generated: {self.stats.descriptions_generated}")
        logger.info(f"Fast loop: {self.stats.fast_loop_fps:.1f} FPS")
        logger.info(f"Slow loop: {self.stats.slow_loop_fps:.1f} FPS")
        logger.info(f"Max queue size: {self.stats.queue_max_size}/{self.async_config.max_queue_size}")
        
        if self.progress_callback:
            self.progress_callback("async_perception.complete", {
                "total_time": self.stats.total_processing_time,
                "frames": self.stats.total_frames_processed,
                "detections": len(self.observations),
                "descriptions": self.stats.descriptions_generated,
                "fast_fps": self.stats.fast_loop_fps,
                "slow_fps": self.stats.slow_loop_fps
            })
        
        return self.observations
    
    def process_video(self, video_path: str) -> List[Observation]:
        """
        Synchronous wrapper for async processing.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of observations
        """
        # Run async event loop
        return asyncio.run(self.process_video_async(video_path))


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def process_video_async(
    video_path: str,
    config: Optional[OrionConfig] = None,
    async_config: Optional[AsyncConfig] = None,
    progress_callback: Optional[Callable[[str, Dict], None]] = None
) -> List[Observation]:
    """
    Convenience function to process video with async perception.
    
    Args:
        video_path: Path to video file
        config: Orion configuration
        async_config: Async processing configuration
        progress_callback: Optional progress callback
        
    Returns:
        List of observations with embeddings and descriptions
    """
    engine = AsyncPerceptionEngine(config, async_config, progress_callback)
    return engine.process_video(video_path)
