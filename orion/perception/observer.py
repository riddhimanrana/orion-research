"""
Frame Observer & Object Detector
=================================

Handles video frame sampling and YOLO object detection.

Responsibilities:
- Sample frames from video at target FPS
- Run YOLO11x detection
- Filter detections by confidence and size
- Extract bounding boxes and class labels

Author: Orion Research Team
Date: October 2025
"""

import logging
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

from orion.perception.types import ObjectClass, BoundingBox
from orion.perception.config import DetectionConfig

logger = logging.getLogger(__name__)


class FrameObserver:
    """
    Observes video frames and detects objects using YOLO.
    
    This is the first stage of perception - raw detection without
    tracking or identity.
    """
    
    def __init__(
        self,
        yolo_model,
        config: DetectionConfig,
        target_fps: float = 4.0,
        show_progress: bool = True,
    ):
        """
        Initialize observer.
        
        Args:
            yolo_model: YOLO model instance from ModelManager
            config: Detection configuration
            target_fps: Target frames per second for processing
            show_progress: Show progress bar
        """
        self.yolo = yolo_model
        self.config = config
        self.target_fps = target_fps
        self.show_progress = show_progress
        
        logger.debug(
            f"FrameObserver initialized: model={config.model}, "
            f"conf_thresh={config.confidence_threshold}, target_fps={target_fps}"
        )
    
    def process_video(self, video_path: str) -> List[Dict]:
        """
        Process video and detect objects in sampled frames.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of raw detections with frame metadata
        """
        logger.info("="*80)
        logger.info("PHASE 1A: FRAME OBSERVATION & DETECTION")
        logger.info("="*80)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        logger.info(f"Video: {video_path}")
        logger.info(f"  FPS: {video_fps:.2f}, Frames: {total_frames}, Duration: {duration:.2f}s")
        logger.info(f"  Resolution: {frame_width}x{frame_height}")
        
        # Calculate frame sampling interval
        frame_interval = max(1, int(video_fps / self.target_fps))
        expected_samples = total_frames // frame_interval
        
        logger.info(f"Sampling every {frame_interval} frames (target {self.target_fps} FPS)")
        logger.info(f"Expected ~{expected_samples} sampled frames")
        
        # Process frames
        detections = []
        frame_count = 0
        sampled_count = 0
        
        pbar = tqdm(
            total=total_frames,
            desc="Detecting objects",
            disable=not self.show_progress,
        )
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames at target FPS
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / video_fps if video_fps > 0 else 0.0
                    
                    # Run detection
                    frame_detections = self.detect_objects(
                        frame=frame,
                        frame_number=frame_count,
                        timestamp=timestamp,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                    
                    detections.extend(frame_detections)
                    sampled_count += 1
                
                frame_count += 1
                pbar.update(1)
        
        finally:
            cap.release()
            pbar.close()
        
        logger.info(f"\n✓ Detected {len(detections)} objects across {sampled_count} sampled frames")
        logger.info(f"  Average: {len(detections) / max(sampled_count, 1):.1f} detections/frame")
        logger.info("="*80 + "\n")
        
        return detections
    
    def detect_objects(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        frame_width: int,
        frame_height: int,
    ) -> List[Dict]:
        """
        Detect objects in a single frame.
        
        Args:
            frame: Frame image (BGR format)
            frame_number: Frame index in video
            timestamp: Timestamp in seconds
            frame_width: Video frame width
            frame_height: Video frame height
            
        Returns:
            List of detection dictionaries
        """
        # Run YOLO
        results = self.yolo(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            verbose=False,
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Extract bbox coordinates
                bbox_xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                x1, y1, x2, y2 = bbox_xyxy
                
                # Filter by minimum size
                width = x2 - x1
                height = y2 - y1
                if width < self.config.min_object_size or height < self.config.min_object_size:
                    continue
                
                # Extract detection info
                confidence = float(boxes.conf[i])
                class_id = int(boxes.cls[i])
                class_name = result.names[class_id]
                
                # Create bounding box
                bbox = BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
                
                # Compute centroid
                centroid = bbox.center
                
                # Crop object region with padding
                crop, padded_bbox = self._crop_with_padding(frame, bbox)
                
                # Determine spatial zone
                spatial_zone = self._compute_spatial_zone(centroid, frame_width, frame_height)
                
                detection = {
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "bounding_box": bbox,
                    "centroid": centroid,
                    "object_class": class_name,
                    "class_id": class_id,
                    "confidence": confidence,
                    "crop": crop,
                    "padded_bbox": padded_bbox,
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                    "spatial_zone": spatial_zone,
                }
                
                detections.append(detection)
        
        return detections
    
    def _crop_with_padding(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
    ) -> Tuple[np.ndarray, BoundingBox]:
        """
        Crop object region with padding.
        
        Args:
            frame: Frame image
            bbox: Bounding box
            
        Returns:
            Tuple of (cropped image, padded bounding box)
        """
        h, w = frame.shape[:2]
        
        # Calculate padding
        padding = self.config.bbox_padding_percent
        width = bbox.width
        height = bbox.height
        
        # Apply padding
        x1_padded = max(0, int(bbox.x1 - width * padding))
        y1_padded = max(0, int(bbox.y1 - height * padding))
        x2_padded = min(w, int(bbox.x2 + width * padding))
        y2_padded = min(h, int(bbox.y2 + height * padding))
        
        # Crop
        crop = frame[y1_padded:y2_padded, x1_padded:x2_padded]
        padded_bbox = BoundingBox(
            x1=float(x1_padded),
            y1=float(y1_padded),
            x2=float(x2_padded),
            y2=float(y2_padded),
        )
        
        return crop, padded_bbox
    
    def _compute_spatial_zone(
        self,
        centroid: Tuple[float, float],
        frame_width: int,
        frame_height: int,
    ) -> str:
        """
        Compute coarse spatial zone for object centroid.
        
        Divides frame into 3x3 grid: top/middle/bottom x left/center/right
        
        Args:
            centroid: (x, y) centroid
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            Spatial zone string (e.g., "center", "top_left")
        """
        cx, cy = centroid
        
        # Horizontal zones
        if cx < frame_width / 3:
            h_zone = "left"
        elif cx < 2 * frame_width / 3:
            h_zone = "center"
        else:
            h_zone = "right"
        
        # Vertical zones
        if cy < frame_height / 3:
            v_zone = "top"
        elif cy < 2 * frame_height / 3:
            v_zone = "middle"
        else:
            v_zone = "bottom"
        
        # Combine
        if h_zone == "center" and v_zone == "middle":
            return "center"
        elif h_zone == "center":
            return v_zone
        elif v_zone == "middle":
            return h_zone
        else:
            return f"{v_zone}_{h_zone}"
