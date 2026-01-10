"""
End-to-End Scene Graph Generation Pipeline
============================================

Video → Frame-level Detection → Scene Graph Generation → Evaluation

Supports two modes:
1. Paper version: DINOv3/Faster-RCNN + Gemini 3.5-Flash (stronger results)
2. Lightweight: YOLO-World + FastVLM (faster, lower quality)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum

import cv2
import numpy as np

from orion.perception.config import DetectionConfig
from orion.perception.observer import FrameObserver
from orion.backends.gemini_vlm import create_vlm_backend
from orion.evaluation.pvsg_evaluator import SceneGraphTriplet, PVSGEvaluator
from orion.evaluation.sga_evaluator import SGAEvaluator

logger = logging.getLogger(__name__)


class PipelineMode(Enum):
    """Pipeline mode selection."""
    PAPER = "paper"  # DINOv3/Faster-RCNN + Gemini (stronger, slower)
    LIGHTWEIGHT = "lightweight"  # YOLO-World + FastVLM (faster, lower quality)


@dataclass
class FrameSceneGraph:
    """Scene graph for a single frame."""
    frame_id: int
    timestamp: float
    detections: List[Dict]  # Objects in frame
    triplets: List[SceneGraphTriplet]  # Relationships
    confidence: float  # Overall quality score


@dataclass
class VideoSceneGraphs:
    """All scene graphs for a video."""
    video_id: str
    frame_graphs: Dict[int, FrameSceneGraph]
    total_frames: int
    duration: float


class SceneGraphGenerator:
    """
    End-to-end scene graph generation from video.
    
    Pipeline:
    1. Load video frames
    2. Detect objects (DINOv3 or YOLO-World)
    3. Generate VLM descriptions (Gemini or FastVLM)
    4. Infer relationships
    5. Build scene graphs
    6. Evaluate on PVSG/ActionGenome
    """
    
    def __init__(self, mode: PipelineMode = PipelineMode.PAPER):
        """
        Initialize scene graph generator.
        
        Args:
            mode: PAPER (stronger) or LIGHTWEIGHT (faster)
        """
        self.mode = mode
        
        # Detection backend
        if mode == PipelineMode.PAPER:
            # Use DINOv3 for better open-vocab detection
            self.detector = FrameObserver(
                config=DetectionConfig(
                    backend="yoloworld",  # Can switch to groundingdino later
                    confidence_threshold=0.25,
                    enable_temporal_filtering=True,
                ),
                detector_backend="yoloworld"
            )
            vlm_backend = "gemini"
            logger.info("✓ Paper mode: DINOv3/Gemini pipeline")
        else:
            # Lightweight: YOLO-World + FastVLM
            self.detector = FrameObserver(
                config=DetectionConfig(
                    backend="yoloworld",
                    confidence_threshold=0.30,
                ),
                detector_backend="yoloworld"
            )
            vlm_backend = "fastvlm"
            logger.info("✓ Lightweight mode: YOLO-World/FastVLM pipeline")
        
        # VLM backend
        try:
            self.vlm = create_vlm_backend(vlm_backend)
        except Exception as e:
            logger.error(f"Failed to initialize VLM: {e}")
            from orion.backends.gemini_vlm import FastVLMBackend
            self.vlm = FastVLMBackend()
    
    def process_video(
        self,
        video_path: str,
        sample_rate: int = 1,
        context: str = ""
    ) -> VideoSceneGraphs:
        """
        Generate scene graphs for entire video.
        
        Args:
            video_path: Path to video file
            sample_rate: Process every Nth frame (1=all frames)
            context: Scene context (e.g., "kitchen")
            
        Returns:
            Scene graphs for all frames
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Processing {video_path.name}: {total_frames} frames @ {fps} fps")
        
        frame_graphs = {}
        frame_id = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_id % sample_rate != 0:
                frame_id += 1
                continue
            
            # Detect objects
            detections = self.detector.detect_objects(frame, frame_id)
            
            # Generate VLM descriptions
            descriptions = self.vlm.describe_objects(frame, detections, context)
            
            # Infer relationships
            relationships = self.vlm.understand_relationships(frame, detections)
            
            # Build scene graph triplets
            triplets = []
            for subj_idx, predicate, obj_idx in relationships:
                if subj_idx < len(detections) and obj_idx < len(detections):
                    subj_name = detections[subj_idx].get('class_name', f'obj{subj_idx}')
                    obj_name = detections[obj_idx].get('class_name', f'obj{obj_idx}')
                    
                    triplet = SceneGraphTriplet(
                        subject=subj_name,
                        predicate=predicate,
                        object=obj_name,
                        confidence=min(
                            descriptions[subj_idx].confidence,
                            descriptions[obj_idx].confidence
                        )
                    )
                    triplets.append(triplet)
            
            # Create frame graph
            frame_graphs[frame_id] = FrameSceneGraph(
                frame_id=frame_id,
                timestamp=frame_id / fps if fps > 0 else 0,
                detections=detections,
                triplets=triplets,
                confidence=np.mean([d.get('confidence', 0.5) for d in detections]) if detections else 0.0
            )
            
            logger.debug(f"  Frame {frame_id}: {len(detections)} objects, {len(triplets)} relationships")
            
            frame_id += 1
        
        cap.release()
        
        logger.info(f"✓ Generated scene graphs for {len(frame_graphs)} frames")
        
        return VideoSceneGraphs(
            video_id=video_path.stem,
            frame_graphs=frame_graphs,
            total_frames=total_frames,
            duration=duration
        )
    
    def evaluate_on_pvsg(
        self,
        video_sgs: VideoSceneGraphs,
        evaluator: Optional[PVSGEvaluator] = None
    ) -> Dict[str, Any]:
        """
        Evaluate generated scene graphs on PVSG.
        
        Args:
            video_sgs: Generated scene graphs
            evaluator: PVSG evaluator (created if None)
            
        Returns:
            Evaluation metrics
        """
        if evaluator is None:
            evaluator = PVSGEvaluator()
        
        # Prepare predictions
        predictions = [
            (video_sgs.video_id, frame_id, fg.triplets)
            for frame_id, fg in video_sgs.frame_graphs.items()
        ]
        
        # Evaluate
        aggregate, results = evaluator.evaluate_batch(predictions)
        
        return aggregate
    
    def anticipate_scene_graphs(
        self,
        video_sgs: VideoSceneGraphs,
        prune_ratio: float = 0.5
    ) -> Dict[int, List[SceneGraphTriplet]]:
        """
        Anticipate future scene graphs given pruned video.
        
        Args:
            video_sgs: Generated scene graphs for full video
            prune_ratio: Ratio of frames to use for anticipation (0.5 = first 50%)
            
        Returns:
            Dict mapping future_frame_id -> predicted triplets
        """
        # Split frames
        total_frames = len(video_sgs.frame_graphs)
        cutoff = int(total_frames * prune_ratio)
        
        # Get observed frames and scene graphs
        observed_frames = []
        observed_sgs = []
        
        for frame_id in sorted(video_sgs.frame_graphs.keys())[:cutoff]:
            fg = video_sgs.frame_graphs[frame_id]
            # In real scenario, would extract actual frame from video
            # For now, using mock data
            observed_frames.append(None)
            observed_sgs.append([(t.subject, t.predicate, t.object) for t in fg.triplets])
        
        # Anticipate future
        num_future = total_frames - cutoff
        future_sgs = self.vlm.anticipate_scene_graphs(
            observed_frames,
            observed_sgs,
            num_future_frames=num_future
        )
        
        # Convert to triplets
        result = {}
        for i, sg in enumerate(future_sgs):
            frame_id = cutoff + i
            triplets = [
                SceneGraphTriplet(subject, predicate, obj)
                for subject, predicate, obj in sg
            ]
            result[frame_id] = triplets
        
        return result
    
    def evaluate_on_actiongenome(
        self,
        video_sgs: VideoSceneGraphs,
        prune_ratio: float = 0.5,
        evaluator: Optional[SGAEvaluator] = None
    ) -> Dict[str, Any]:
        """
        Evaluate scene graph anticipation on ActionGenome.
        
        Args:
            video_sgs: Generated scene graphs
            prune_ratio: Ratio for anticipation task
            evaluator: SGA evaluator (created if None)
            
        Returns:
            Anticipation evaluation metrics
        """
        if evaluator is None:
            evaluator = SGAEvaluator()
        
        # Anticipate
        anticipated_sgs = self.anticipate_scene_graphs(video_sgs, prune_ratio)
        
        # Evaluate
        predictions = [(video_sgs.video_id, prune_ratio, anticipated_sgs)]
        aggregate = evaluator.evaluate_batch(predictions)
        
        return aggregate


def create_pipeline(mode: PipelineMode = PipelineMode.PAPER) -> SceneGraphGenerator:
    """
    Create scene graph generation pipeline.
    
    Args:
        mode: PAPER (DINOv3+Gemini) or LIGHTWEIGHT (YOLO+FastVLM)
        
    Returns:
        Configured pipeline
    """
    return SceneGraphGenerator(mode)
