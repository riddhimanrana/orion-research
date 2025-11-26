"""
3D Perception Engine
====================

Handles depth estimation, SLAM, and 3D lifting of entities.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from orion.perception.types import Perception3DResult, EntityState3D, Hand, CameraIntrinsics, VisibilityState
from orion.perception.depth import DepthEstimator
from orion.perception.camera_intrinsics import backproject_point
from orion.slam.slam_engine import SLAMEngine, SLAMConfig

logger = logging.getLogger(__name__)

class Perception3DEngine:
    def __init__(
        self,
        enable_depth: bool = True,
        enable_hands: bool = False,
        enable_occlusion: bool = False,
        enable_slam: bool = True,
    ):
        self.enable_depth = enable_depth
        self.enable_hands = enable_hands
        self.enable_occlusion = enable_occlusion
        self.enable_slam = enable_slam
        
        self.depth_estimator = None
        if enable_depth:
            try:
                self.depth_estimator = DepthEstimator(model_name="depth_anything_v3", model_size="small")
                logger.info("Perception3DEngine: DepthEstimator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize DepthEstimator: {e}")
        
        self.slam_engine = None
        if enable_slam:
            try:
                self.slam_engine = SLAMEngine(SLAMConfig(enable_slam=True, use_depth=enable_depth))
                logger.info("Perception3DEngine: SLAMEngine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize SLAMEngine: {e}")

    def process_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        frame_number: int,
        timestamp: float
    ) -> Perception3DResult:
        
        depth_map = None
        if self.depth_estimator:
            depth_map, _ = self.depth_estimator.estimate(frame)
            
        camera_pose = None
        if self.slam_engine:
            camera_pose = self.slam_engine.process_frame(frame, depth_map, timestamp=timestamp)
            
        # Lift detections to 3D
        entities_3d = []
        if depth_map is not None:
            h, w = depth_map.shape
            intrinsics = CameraIntrinsics.auto_estimate(w, h)
            
            for det in detections:
                bbox = det.get('bbox') # (x1, y1, x2, y2)
                if bbox:
                    x1, y1, x2, y2 = [int(c) for c in bbox]
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    
                    # Clamp to image bounds
                    cx = max(0, min(w - 1, cx))
                    cy = max(0, min(h - 1, cy))
                    
                    # Sample depth at centroid
                    z = float(depth_map[cy, cx])
                    
                    # Backproject
                    centroid_3d = backproject_point(float(cx), float(cy), z, intrinsics)
                    
                    entity_3d = EntityState3D(
                        entity_id=det.get('entity_id', 'unknown'),
                        frame_number=frame_number,
                        timestamp=timestamp,
                        class_label=det.get('class', 'unknown'),
                        class_confidence=det.get('confidence', 0.0),
                        bbox_2d_px=(x1, y1, x2, y2),
                        centroid_2d_px=(float(cx), float(cy)),
                        centroid_3d_mm=centroid_3d,
                        depth_mean_mm=z,
                        visibility_state=VisibilityState.FULLY_VISIBLE
                    )
                    entities_3d.append(entity_3d)

        return Perception3DResult(
            frame_number=frame_number,
            timestamp=timestamp,
            entities=entities_3d,
            hands=[],
            depth_map=depth_map,
            camera_pose=camera_pose
        )
