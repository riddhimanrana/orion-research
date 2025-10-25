"""
Perception Engine
=================

High-level orchestration of the perception pipeline.

Orchestrates the complete Phase 1 flow:
1. Frame observation & detection (observer)
2. Visual embedding generation (embedder)
3. Entity clustering & tracking (tracker)
4. Entity description (describer)

Author: Orion Research Team
Date: October 2025
"""

import logging
import time
from typing import List, Optional, Tuple

from orion.perception.types import Observation, PerceptionEntity, PerceptionResult, ObjectClass
from orion.perception.config import PerceptionConfig
from orion.perception.observer import FrameObserver
from orion.perception.embedder import VisualEmbedder
from orion.perception.tracker import EntityTracker
from orion.perception.describer import EntityDescriber

from orion.managers.model_manager import ModelManager

logger = logging.getLogger(__name__)


class PerceptionEngine:
    """
    Complete perception pipeline orchestrator.
    
    Manages the full Phase 1 pipeline from video to described entities.
    """
    
    def __init__(
        self,
        config: Optional[PerceptionConfig] = None,
        verbose: bool = False,
    ):
        """
        Initialize perception engine.
        
        Args:
            config: Perception configuration (uses defaults if None)
            verbose: Enable verbose logging
        """
        self.config = config or PerceptionConfig()
        self.verbose = verbose
        
        if verbose:
            logging.getLogger("orion.perception").setLevel(logging.DEBUG)
        
        # Model manager (lazy loading)
        self.model_manager = ModelManager.get_instance()
        
        # Pipeline components (initialized lazily)
        self.observer: Optional[FrameObserver] = None
        self.embedder: Optional[VisualEmbedder] = None
        self.tracker: Optional[EntityTracker] = None
        self.describer: Optional[EntityDescriber] = None
        
        logger.info("PerceptionEngine initialized")
        logger.info(f"  Detection: {self.config.detection.model}")
        logger.info(f"  Embedding: {self.config.embedding.embedding_dim}-dim")
        logger.info(f"  Target FPS: {self.config.target_fps}")
    
    def process_video(self, video_path: str) -> PerceptionResult:
        """
        Process video through complete perception pipeline.
        
        Args:
            video_path: Path to video file
            
        Returns:
            PerceptionResult with entities and observations
        """
        logger.info("\n" + "="*80)
        logger.info("PERCEPTION ENGINE - PHASE 1")
        logger.info("="*80)
        
        start_time = time.time()
        
        # Initialize pipeline components
        self._initialize_components()
        
        # Step 1: Observe & detect
        detections = self.observer.process_video(video_path)
        
        # Step 2: Embed detections
        detections = self.embedder.embed_detections(detections)
        
        # Step 3: Convert detections to Observation objects
        observations = self._detections_to_observations(detections)
        
        # Step 4: Cluster into entities
        entities = self.tracker.cluster_observations(observations)
        
        # Step 5: Describe entities
        entities = self.describer.describe_entities(entities)
        
        # Get video metadata (approximate from observations)
        total_frames = max([obs.frame_number for obs in observations]) if observations else 0
        fps = self.config.target_fps
        duration = max([obs.timestamp for obs in observations]) if observations else 0.0
        
        # Build result
        result = PerceptionResult(
            entities=entities,
            raw_observations=observations,
            video_path=video_path,
            total_frames=total_frames,
            fps=fps,
            duration_seconds=duration,
            processing_time_seconds=time.time() - start_time,
        )
        
        elapsed = time.time() - start_time
        
        logger.info("\n" + "="*80)
        logger.info("PERCEPTION COMPLETE")
        logger.info("="*80)
        logger.info(f"  Total detections: {result.total_detections}")
        logger.info(f"  Unique entities: {result.unique_entities}")
        logger.info(f"  Processing time: {elapsed:.2f}s")
        logger.info("="*80 + "\n")
        
        return result
    
    def _initialize_components(self):
        """Initialize all pipeline components with models."""
        logger.info("Loading models...")
        
        # YOLO
        yolo = self.model_manager.yolo
        logger.info("  ✓ YOLO11x loaded")
        
        # CLIP
        clip = self.model_manager.clip
        logger.info(f"  ✓ CLIP loaded ({self.config.embedding.embedding_dim}-dim)")
        
        # FastVLM
        vlm = self.model_manager.fastvlm
        logger.info("  ✓ FastVLM loaded")
        
        # Create components
        self.observer = FrameObserver(
            yolo_model=yolo,
            config=self.config.detection,
            target_fps=self.config.target_fps,
            show_progress=True,
        )
        
        self.embedder = VisualEmbedder(
            clip_model=clip,
            config=self.config.embedding,
        )
        
        self.tracker = EntityTracker(
            config=self.config,
        )
        
        self.describer = EntityDescriber(
            vlm_model=vlm,
            config=self.config.description,
        )
        
        logger.info("✓ All components initialized\n")
    
    def _detections_to_observations(self, detections: List[dict]) -> List[Observation]:
        """
        Convert detection dicts to Observation objects.
        
        Args:
            detections: List of detection dicts
            
        Returns:
            List of Observation objects
        """
        observations = []
        
        for i, det in enumerate(detections):
            # Map class name to ObjectClass enum
            try:
                object_class = ObjectClass(det["object_class"])
            except ValueError:
                object_class = ObjectClass.UNKNOWN
            
            obs = Observation(
                bounding_box=det["bounding_box"],
                centroid=det["centroid"],
                object_class=object_class,
                confidence=det["confidence"],
                visual_embedding=det["embedding"],
                frame_number=det["frame_number"],
                timestamp=det["timestamp"],
                temp_id=f"obs_{i}",
                image_patch=det.get("crop"),
                spatial_zone=det.get("spatial_zone"),
                raw_yolo_class=det.get("object_class"),
                frame_width=det.get("frame_width"),
                frame_height=det.get("frame_height"),
            )
            
            observations.append(obs)
        
        return observations


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def run_perception(
    video_path: str,
    config: Optional[PerceptionConfig] = None,
    verbose: bool = False,
) -> PerceptionResult:
    """
    Convenience function to run perception pipeline.
    
    Args:
        video_path: Path to video file
        config: Optional perception configuration
        verbose: Enable verbose logging
        
    Returns:
        PerceptionResult with entities and observations
        
    Example:
        >>> from orion.perception.engine import run_perception
        >>> result = run_perception("video.mp4")
        >>> print(f"Found {result.unique_entities} entities")
    """
    engine = PerceptionEngine(config=config, verbose=verbose)
    return engine.process_video(video_path)
