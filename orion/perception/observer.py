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
from typing import Any, Dict, List, Tuple, Optional, Literal

import cv2
import numpy as np
from tqdm import tqdm

from orion.perception.types import ObjectClass, BoundingBox
from orion.perception.config import DetectionConfig
from orion.utils.profiling import profile

# Check if 3D perception is available
try:
    from orion.perception.perception_3d import Perception3DEngine
    PERCEPTION_3D_AVAILABLE = True
except ImportError:
    Perception3DEngine = None
    PERCEPTION_3D_AVAILABLE = False

logger = logging.getLogger(__name__)


class FrameObserver:
    """
    Observes video frames and detects objects using YOLO.
    
    This is the first stage of perception - raw detection without
    tracking or identity.
    """
    
    def __init__(
        self,
        config: DetectionConfig,
        detector_backend: Literal["yolo", "groundingdino"] = "yolo",
        yolo_model: Optional[Any] = None,
        grounding_dino: Optional[Any] = None,
        target_fps: float = 2.0,
        show_progress: bool = True,
        enable_3d: bool = False,
        depth_model: str = "midas",
        enable_occlusion: bool = False,
    ):
        """
        Initialize observer.
        
        Args:
            detector_backend: Detection backend identifier
            yolo_model: YOLO model instance from ModelManager
            grounding_dino: GroundingDINO wrapper when backend='groundingdino'
            config: Detection configuration
            target_fps: Target frames per second for processing
            show_progress: Show progress bar
            enable_3d: Enable 3D perception (depth, occlusion)
            depth_model: Depth model to use ("zoe" or "midas")
            enable_occlusion: Enable occlusion detection (requires enable_3d=True)
        """
        self.detector_backend = detector_backend
        self.yolo = yolo_model if detector_backend == "yolo" else None
        self.grounding_dino = grounding_dino if detector_backend == "groundingdino" else None
        self.yoloworld = None if detector_backend != "yoloworld" else None

        if self.detector_backend == "yolo" and self.yolo is None:
            raise ValueError("YOLO backend selected but yolo_model was not provided")
        if self.detector_backend == "groundingdino" and self.grounding_dino is None:
            raise ValueError("GroundingDINO backend selected but wrapper was not provided")

        # Detector class registry (used by downstream trackers)
        if self.detector_backend == "yolo" and hasattr(self.yolo, "names"):
            self.detector_classes: List[str] = list(self.yolo.names.values())
        else:
            self.detector_classes = list(self.config.grounding_categories())
            if self.detector_backend == "yoloworld" and self.yoloworld is None:
                from orion.perception.yoloworld_detector import YOLOWorldDetector
                self.yoloworld = YOLOWorldDetector(
                    model_path=None,
                    confidence_threshold=self.config.confidence_threshold,
                    device="mps",
                    batch_size=8,
                    categories=None,
                )
        self._grounding_label_map = {
            label.lower(): idx for idx, label in enumerate(self.detector_classes)
        }
        
        # Initialize 3D perception engine if requested
        self.perception_engine = None
        if enable_3d and PERCEPTION_3D_AVAILABLE:
            try:
                self.perception_engine = Perception3DEngine(
                    enable_depth=True,
                    enable_hands=False,  # Hand tracking disabled (future work with HOT3D)
                    enable_occlusion=enable_occlusion,
                    enable_slam=True, # Enable SLAM by default if 3D is enabled
                )
                logger.info("  ✓ Perception3DEngine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize 3D perception: {e}")
                self.perception_engine = None
        elif enable_3d:
            logger.warning("3D perception requested but not available")
        
        logger.debug(
            "FrameObserver initialized: backend=%s, target_fps=%s, 3d_enabled=%s",
            self.detector_backend,
            target_fps,
            self.perception_engine is not None,
        )

    def update_camera_intrinsics(self, intrinsics: "CameraIntrinsics"):
        """Passes camera intrinsics update to the 3D perception engine."""
        if self.perception_engine:
            self.perception_engine.update_camera_intrinsics(intrinsics)
    
    @profile("observer_process_video")
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

        # === FIX: Detect and handle portrait video rotation ===
        needs_rotation = frame_height > frame_width
        if needs_rotation:
            logger.info("Portrait video detected. Frames will be rotated for processing.")
            # Swap width and height for processing logic
            frame_width, frame_height = frame_height, frame_width
        # === END FIX ===
        
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
                
                # === FIX: Apply rotation if needed ===
                if needs_rotation:
                    # Rotate frame 90 degrees counter-clockwise
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # === END FIX ===

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

    def _run_detection_backend(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Dispatch to the configured detection backend."""
        if self.detector_backend == "groundingdino":
            return self._detect_with_groundingdino(frame)
        elif self.detector_backend == "yoloworld":
            return self._detect_with_yoloworld(frame)
        else:
            return self._detect_with_yolo(frame)

    def _detect_with_yoloworld(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.yoloworld is None:
            return []
        # Optionally pass categories for open-vocab detection
        return self.yoloworld.detect_frame(frame)

    @profile("observer_detect_with_yolo")
    def _detect_with_yolo(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO inference and normalize outputs."""
        if self.yolo is None:
            return []

        results = self.yolo(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                bbox_xyxy = boxes.xyxy[i].cpu().numpy().astype(float).tolist()
                detections.append(
                    {
                        "bbox": bbox_xyxy,
                        "confidence": float(boxes.conf[i]),
                        "class_id": int(boxes.cls[i]),
                        "class_name": result.names[int(boxes.cls[i])],
                    }
                )
        return detections

    def _detect_with_groundingdino(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run GroundingDINO inference and normalize outputs."""
        if self.grounding_dino is None:
            return []

        raw_detections = self.grounding_dino.detect(
            frame_bgr=frame,
            prompt=self.config.groundingdino_prompt,
            box_threshold=self.config.groundingdino_box_threshold,
            text_threshold=self.config.groundingdino_text_threshold,
            max_detections=self.config.groundingdino_max_detections,
        )

        normalized: List[Dict[str, Any]] = []
        for det in raw_detections:
            label = det.get("label", "object").strip() or "object"
            class_id = self._register_grounding_label(label)
            normalized.append(
                {
                    "bbox": det.get("bbox", [0, 0, 0, 0]),
                    "confidence": det.get("confidence", 0.0),
                    "class_id": class_id,
                    "class_name": label,
                }
            )
        return normalized

    def _register_grounding_label(self, label: str) -> int:
        """Register unseen GroundingDINO labels for downstream components."""
        normalized = label.lower()
        if normalized not in self._grounding_label_map:
            self._grounding_label_map[normalized] = len(self.detector_classes)
            self.detector_classes.append(label)
        return self._grounding_label_map[normalized]
    
    @profile("observer_detect_objects")
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
        detection_candidates = self._run_detection_backend(frame)
        
        # Run 3D perception if enabled
        perception_3d = None
        if self.perception_engine is not None:
            # Convert YOLO results to format expected by PerceptionEngine
            yolo_detections = []
            for idx, detection in enumerate(detection_candidates):
                x1, y1, x2, y2 = [int(v) for v in detection["bbox"]]
                width = x2 - x1
                height = y2 - y1
                if width < self.config.min_object_size or height < self.config.min_object_size:
                    continue
                yolo_detections.append({
                    'entity_id': f'det_{frame_number}_{idx}',
                    'class': detection["class_name"],
                    'confidence': detection["confidence"],
                    'bbox': (x1, y1, x2, y2),
                })
            
            # Process with 3D perception
            try:
                perception_3d = self.perception_engine.process_frame(
                    frame, yolo_detections, frame_number, timestamp
                )
            except Exception as e:
                logger.warning(f"3D perception failed on frame {frame_number}: {e}")
                perception_3d = None
        
        detections = []
        
        for detection_source in detection_candidates:
            x1, y1, x2, y2 = [int(v) for v in detection_source["bbox"]]

            width = x2 - x1
            height = y2 - y1
            if width < self.config.min_object_size or height < self.config.min_object_size:
                continue

            confidence = float(detection_source["confidence"])
            class_id = int(detection_source.get("class_id", -1))
            class_name = detection_source["class_name"]

            bbox = BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
            centroid = bbox.center
            crop, padded_bbox = self._crop_with_padding(frame, bbox)
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
            # Ensure YOLO-World detections have description and object_class fields
            if self.detector_backend == "yoloworld":
                detection["description"] = class_name
                detection["object_class"] = class_name

            if perception_3d is not None and perception_3d.entities:
                for entity_3d in perception_3d.entities:
                    if entity_3d.class_label == class_name:
                        e_x1, e_y1, e_x2, e_y2 = entity_3d.bbox_2d_px
                        if abs(e_x1 - x1) < 20 and abs(e_y1 - y1) < 20:
                            detection["depth_mm"] = entity_3d.depth_mean_mm
                            detection["centroid_3d_mm"] = entity_3d.centroid_3d_mm
                            detection["visibility_state"] = entity_3d.visibility_state.value
                            detection["occlusion_ratio"] = entity_3d.occlusion_ratio
                            detection["occluded_by"] = entity_3d.occluded_by
                            break

            if perception_3d is not None and perception_3d.hands:
                detection["hands_detected"] = len(perception_3d.hands)
                detection["hands"] = [h.to_dict() for h in perception_3d.hands]

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
