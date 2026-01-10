"""
Advanced Detection Module - Detectron2 Integration
Provides instance segmentation and better detection quality for accurate mode
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
import cv2


class Detectron2Detector:
    """Detectron2-based detector with instance segmentation"""
    
    def __init__(self, config_path: str = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
                 confidence_threshold: float = 0.3,
                 device: str = "mps"):
        """
        Initialize Detectron2 detector
        
        Args:
            config_path: Detectron2 config (Mask R-CNN by default)
            confidence_threshold: Detection threshold (0.3 for accurate mode)
            device: 'mps' for Apple Silicon, 'cuda' for GPU, 'cpu' otherwise
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.cfg = None
        
        # Lazy loading - only import if Detectron2 available
        try:
            from detectron2 import model_zoo
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file(config_path))
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
            
            # Device configuration
            # NOTE: Detectron2 on MPS is EXTREMELY slow (280s per frame!)
            # Force CPU which is 10-100x faster than broken MPS backend
            if device == "mps":
                print("⚠️  Detectron2 MPS is extremely slow - forcing CPU mode")
                self.cfg.MODEL.DEVICE = "cpu"
            elif device == "cuda" and torch.cuda.is_available():
                self.cfg.MODEL.DEVICE = "cuda"
            else:
                self.cfg.MODEL.DEVICE = "cpu"
            
            self.model = DefaultPredictor(self.cfg)
            self.available = True
            print(f"✓ Detectron2 loaded (Mask R-CNN, device={self.cfg.MODEL.DEVICE})")
            
        except ImportError:
            self.available = False
            print("⚠️  Detectron2 not available - install with: pip install detectron2")
            print("    Falling back to YOLO detection")
    
    def detect(self, image: np.ndarray) -> dict:
        """
        Run detection on image
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            dict with:
                - boxes: (N, 4) xyxy format
                - scores: (N,)
                - classes: (N,)
                - masks: (N, H, W) binary masks
        """
        if not self.available or self.model is None:
            return {"boxes": [], "scores": [], "classes": [], "masks": []}
        
        # Convert RGB to BGR for Detectron2
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Run inference
        outputs = self.model(bgr_image)
        instances = outputs["instances"].to("cpu")
        
        # Extract predictions
        boxes = instances.pred_boxes.tensor.numpy()  # (N, 4) xyxy
        scores = instances.scores.numpy()  # (N,)
        classes = instances.pred_classes.numpy()  # (N,)
        
        # Extract masks if available
        if instances.has("pred_masks"):
            masks = instances.pred_masks.numpy()  # (N, H, W)
        else:
            masks = None
        
        return {
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
            "masks": masks,
            "num_detections": len(boxes)
        }
    
    def get_class_name(self, class_id: int) -> str:
        """Get COCO class name from ID"""
        if not self.available:
            return "unknown"
        
        from detectron2.data import MetadataCatalog
        metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        return metadata.thing_classes[class_id]


class HybridDetector:
    """
    Hybrid detector that can switch between YOLO, YOLO-seg, and Detectron2
    
    Modes:
        - 'yolo': Fast bbox detection (no masks)
        - 'yolo-seg': Fast detection with instance masks (RECOMMENDED)
        - 'detectron2': High quality but very slow on M1 (NOT RECOMMENDED)
    """
    
    def __init__(self, mode: str = "yolo", yolo_model: str = "yolo11n", use_segmentation: bool = False):
        """
        Args:
            mode: 'yolo', 'yolo-seg', or 'detectron2'
            yolo_model: YOLO variant (e.g., 'yolo11n', 'yolo11m', 'yolo11x')
            use_segmentation: If True and mode='yolo', uses YOLO-seg variant
        """
        self.mode = mode
        self.use_segmentation = use_segmentation
        
        # Auto-switch to yolo-seg if segmentation requested
        if mode == "yolo" and use_segmentation:
            self.mode = "yolo-seg"
        
        if mode == "detectron2":
            self.detectron2 = Detectron2Detector()
            if not self.detectron2.available:
                print("⚠️  Detectron2 unavailable, falling back to YOLO")
                self.mode = "yolo"
        
        if self.mode in ["yolo", "yolo-seg"]:
            from orion.managers.model_manager import ModelManager
            
            # Determine model name
            if self.mode == "yolo-seg":
                # Use segmentation variant
                model_name = f"{yolo_model}-seg"
                print(f"Loading YOLO segmentation model: {model_name}")
            else:
                model_name = yolo_model
            
            # Load via ModelManager or directly
            self.model_manager = ModelManager.get_instance()
            
            # Check if ModelManager supports seg models, otherwise load directly
            if self.mode == "yolo-seg":
                try:
                    from ultralytics import YOLO
                    self.yolo = YOLO(f"{model_name}.pt")
                    print(f"✓ Using YOLO-seg detector: {model_name}.pt")
                except Exception as e:
                    print(f"⚠️  Could not load YOLO-seg: {e}")
                    print(f"   Falling back to regular YOLO (no masks)")
                    self.mode = "yolo"
                    self.model_manager.yolo_model_name = yolo_model
                    self.yolo = self.model_manager.yolo
            else:
                self.model_manager.yolo_model_name = yolo_model
                self.yolo = self.model_manager.yolo  # Property, not method
                print(f"✓ Using YOLO detector: {yolo_model}")
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.35) -> dict:
        """
        Run detection
        
        Returns unified format:
            - boxes: (N, 4) xyxy
            - scores: (N,)
            - classes: (N,) class IDs
            - class_names: list[str]
            - masks: Optional (N, H, W) for Detectron2 or YOLO-seg
            - num_detections: int
        """
        if self.mode == "detectron2":
            result = self.detectron2.detect(image)
            result["class_names"] = [self.detectron2.get_class_name(int(c)) for c in result["classes"]]
            return result
        
        else:  # YOLO or YOLO-seg
            results = self.yolo(image, conf=conf_threshold, verbose=False)
            
            if len(results) == 0 or len(results[0].boxes) == 0:
                return {
                    "boxes": np.array([]),
                    "scores": np.array([]),
                    "classes": np.array([]),
                    "class_names": [],
                    "masks": None,
                    "num_detections": 0
                }
            
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            class_names = [results[0].names[int(c)] for c in classes]
            
            # Extract masks if YOLO-seg
            masks = None
            if self.mode == "yolo-seg" and hasattr(results[0], 'masks') and results[0].masks is not None:
                # YOLO-seg masks are (N, H, W) in results[0].masks.data
                masks = results[0].masks.data.cpu().numpy()
                print(f"  ✓ Extracted {len(masks)} instance masks from YOLO-seg")
            
            return {
                "boxes": boxes,
                "scores": scores,
                "classes": classes,
                "class_names": class_names,
                "masks": masks,
                "num_detections": len(boxes)
            }
    
    def supports_masks(self) -> bool:
        """Check if current detector provides instance masks"""
        return self.mode in ["detectron2", "yolo-seg"]
