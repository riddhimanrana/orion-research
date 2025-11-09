"""
3D Perception Engine: Depth + Hands + 3D Coordinates
====================================================

Integrates depth estimation, hand tracking, and 3D backprojection
with YOLO detection pipeline for 3D perception capabilities.
"""

import time
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np

from .types import (
    EntityState3D,
    Hand,
    CameraIntrinsics,
    Perception3DResult,
    VisibilityState,
)
from .depth import DepthEstimator
from .hand_tracking import HandTracker
from .occlusion import OcclusionDetector
from .camera_intrinsics import backproject_bbox

logger = logging.getLogger(__name__)


class Perception3DEngine:
    """
    Phase 1 Perception Engine: Depth + Hands + 3D.
    
    Takes YOLO detections and enhances them with:
    - Monocular depth estimation (MiDaS/ZoeDepth)
    - Hand tracking with 3D landmarks (MediaPipe)
    - 3D backprojection (camera intrinsics)
    - Occlusion detection (depth-based)
    """
    
    def __init__(
        self,
        enable_depth: bool = True,
        enable_hands: bool = True,
        enable_occlusion: bool = True,
        depth_model: str = "zoe",
        device: Optional[str] = None,
    ):
        """
        Initialize Phase 1 engine.
        
        Args:
            enable_depth: Enable depth estimation
            enable_hands: Enable hand tracking
            enable_occlusion: Enable occlusion detection
            depth_model: "zoe" or "midas"
            device: Device for inference (None for auto-detect)
        """
        self.enable_depth = enable_depth
        self.enable_hands = enable_hands
        self.enable_occlusion = enable_occlusion
        
        # Initialize depth estimator
        self.depth_estimator = None
        if enable_depth:
            logger.info("[Phase1] Initializing depth estimator...")
            self.depth_estimator = DepthEstimator(model_name=depth_model, device=device)
        
        # Initialize hand tracker
        self.hand_tracker = None
        if enable_hands:
            logger.info("[Phase1] Initializing hand tracker...")
            try:
                self.hand_tracker = HandTracker()
            except RuntimeError as e:
                logger.warning(f"[Phase1] Hand tracking disabled: {e}")
                self.hand_tracker = None
        
        # Initialize occlusion detector
        self.occlusion_detector = None
        if enable_occlusion:
            logger.info("[Phase1] Initializing occlusion detector...")
            self.occlusion_detector = OcclusionDetector()
        
        logger.info("[Phase1] ✓ Phase 1 engine initialized")
    
    def process_frame(
        self,
        frame: np.ndarray,
        yolo_detections: List[Dict],
        frame_number: int,
        timestamp: float,
        camera_intrinsics: Optional[CameraIntrinsics] = None,
    ) -> Perception3DResult:
        """
        Process a frame with Phase 1 perception.
        
        Args:
            frame: RGB frame (H, W, 3)
            yolo_detections: YOLO detections with keys:
                - bbox: (x1, y1, x2, y2)
                - class: class label
                - confidence: detection confidence
                - entity_id: unique entity ID
            frame_number: Frame number
            timestamp: Timestamp in seconds
            camera_intrinsics: Camera calibration (auto-estimated if None)
            
        Returns:
            Perception3DResult with 3D-enhanced entities and hands
        """
        start_time = time.time()
        
        height, width = frame.shape[:2]
        
        # Auto-estimate camera intrinsics if not provided
        if camera_intrinsics is None:
            camera_intrinsics = CameraIntrinsics.auto_estimate(width, height)
        
        # Step 1: Depth estimation
        depth_map = None
        if self.depth_estimator is not None:
            depth_map, _ = self.depth_estimator.estimate(frame)
        
        # Step 2: Hand tracking
        # TODO: Implement hand tracking with HOT3D dataset (CVPR 2025)
        # For now, disabled due to egocentric view challenges with standard MediaPipe
        # Future: Train custom hand detector on HOT3D or use specialized models
        hands = []
        # if self.hand_tracker is not None and depth_map is not None:
        #     hands = self.hand_tracker.detect(frame, depth_map, camera_intrinsics)
        
        # Step 3: Enhance YOLO detections with 3D information
        entities = []
        for detection in yolo_detections:
            bbox_2d = detection['bbox']
            
            # Compute centroid
            x1, y1, x2, y2 = bbox_2d
            centroid_2d = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            
            # Backproject to 3D if depth available
            if depth_map is not None:
                bbox_3d_info = backproject_bbox(bbox_2d, depth_map, camera_intrinsics)
                centroid_3d = bbox_3d_info['centroid_3d']
                bbox_3d_mm = (bbox_3d_info['bbox_3d_min'], bbox_3d_info['bbox_3d_max'])
                depth_mean = bbox_3d_info['depth_mean']
                depth_variance = bbox_3d_info['depth_variance']
            else:
                centroid_3d = None
                bbox_3d_mm = None
                depth_mean = None
                depth_variance = None
            
            # Create EntityState3D
            entity = EntityState3D(
                entity_id=detection['entity_id'],
                frame_number=frame_number,
                timestamp=timestamp,
                class_label=detection['class'],
                class_confidence=detection['confidence'],
                bbox_2d_px=tuple(int(c) for c in bbox_2d),
                centroid_2d_px=centroid_2d,
                centroid_3d_mm=centroid_3d,
                bbox_3d_mm=bbox_3d_mm,
                depth_mean_mm=depth_mean,
                depth_variance_mm2=depth_variance,
            )
            entities.append(entity)
        
        # Step 4: Occlusion detection
        if self.occlusion_detector is not None and depth_map is not None:
            self.occlusion_detector.detect_occlusions(entities, hands, depth_map)
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        return Perception3DResult(
            frame_number=frame_number,
            timestamp=timestamp,
            entities=entities,
            hands=hands,
            depth_map=depth_map,
            camera_intrinsics=camera_intrinsics,
            processing_time_ms=processing_time,
        )
