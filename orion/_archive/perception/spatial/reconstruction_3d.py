"""
3D Reconstruction Module
========================

Converts 2D detections + depth maps to 3D coordinates using camera intrinsics.

Key concepts:
- Camera intrinsics: focal length, principal point, image size
- Depth map: per-pixel depth (relative or metric)
- 3D projection: pixel (u,v,d) → world point (x,y,z)
- Detection bbox → 3D bbox (8 corners + centroid)
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class CameraIntrinsics:
    """Camera calibration parameters for 3D projection"""
    
    # Common device intrinsics (approximate)
    PRESETS = {
        "iphone_12_pro": {
            "fx": 2632, "fy": 2640, "cx": 1920, "cy": 1440,
            "img_width": 3840, "img_height": 2880,
            "description": "iPhone 12 Pro wide camera"
        },
        "macbook_air_m1": {
            "fx": 1920, "fy": 1920, "cx": 640, "cy": 360,
            "img_width": 1280, "img_height": 720,
            "description": "MacBook Air M1 built-in camera (approx)"
        },
        "generic_640": {
            "fx": 500, "fy": 500, "cx": 320, "cy": 240,
            "img_width": 640, "img_height": 480,
            "description": "Generic VGA camera"
        },
        "generic_720p": {
            "fx": 800, "fy": 800, "cx": 640, "cy": 360,
            "img_width": 1280, "img_height": 720,
            "description": "Generic 720p camera"
        }
    }
    
    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        img_width: int,
        img_height: int,
        distortion_coeffs: Optional[np.ndarray] = None
    ):
        """
        Initialize camera intrinsics.
        
        Args:
            fx, fy: Focal lengths (pixels)
            cx, cy: Principal point (pixels)
            img_width, img_height: Image dimensions
            distortion_coeffs: Optional k1, k2, p1, p2, k3
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.img_width = img_width
        self.img_height = img_height
        self.distortion_coeffs = distortion_coeffs or np.zeros(5)
        
        # Intrinsic matrix K
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Inverse for back-projection
        self.K_inv = np.linalg.inv(self.K)
    
    @classmethod
    def from_preset(cls, preset_name: str) -> "CameraIntrinsics":
        """Load camera from preset"""
        if preset_name not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(cls.PRESETS.keys())}")
        
        params = cls.PRESETS[preset_name]
        logger.info(f"Loaded camera preset: {preset_name} - {params['description']}")
        return cls(
            fx=params["fx"],
            fy=params["fy"],
            cx=params["cx"],
            cy=params["cy"],
            img_width=params["img_width"],
            img_height=params["img_height"]
        )
    
    def pixel_to_3d(
        self,
        u: float,
        v: float,
        depth: float
    ) -> np.ndarray:
        """
        Convert pixel (u, v) with depth to 3D point.
        
        Args:
            u, v: Pixel coordinates
            depth: Depth value (meters)
            
        Returns:
            xyz: 3D point [x, y, z] in camera frame
        """
        # Normalized coordinates
        x_norm = (u - self.cx) / self.fx
        y_norm = (v - self.cy) / self.fy
        
        # 3D point
        x = x_norm * depth
        y = y_norm * depth
        z = depth
        
        return np.array([x, y, z], dtype=np.float32)
    
    def bbox2d_to_3d(
        self,
        bbox_2d: Dict,
        depth_map: np.ndarray
    ) -> Dict:
        """
        Convert 2D bounding box to 3D bounding box using depth map.
        
        Args:
            bbox_2d: {"x1": normalized, "y1": normalized, "x2": normalized, "y2": normalized}
            depth_map: Depth map (H, W) in meters
            
        Returns:
            bbox_3d: {
                "x1_2d", "y1_2d", "x2_2d", "y2_2d": 2D bbox in pixels
                "corners_3d": List of 8 corners [x,y,z]
                "centroid_3d": Center point [x,y,z]
                "depth_min", "depth_max", "depth_mean": Depth stats
                "width_3d", "height_3d", "depth_3d": 3D dimensions
            }
        """
        h, w = depth_map.shape
        
        # Convert normalized coords to pixels
        x1_px = int(bbox_2d["x1"] * w)
        y1_px = int(bbox_2d["y1"] * h)
        x2_px = int(bbox_2d["x2"] * w)
        y2_px = int(bbox_2d["y2"] * h)
        
        # Clamp to image bounds
        x1_px = max(0, min(x1_px, w - 1))
        y1_px = max(0, min(y1_px, h - 1))
        x2_px = max(0, min(x2_px, w - 1))
        y2_px = max(0, min(y2_px, h - 1))
        
        # Get depth statistics for bbox region
        bbox_depth = depth_map[y1_px:y2_px+1, x1_px:x2_px+1]
        depth_min = float(np.min(bbox_depth))
        depth_max = float(np.max(bbox_depth))
        depth_mean = float(np.mean(bbox_depth))
        
        # Project 4 corners of 2D bbox to 3D (using mean depth)
        depth_use = depth_mean
        
        corners_2d = [
            (x1_px, y1_px),  # top-left
            (x2_px, y1_px),  # top-right
            (x2_px, y2_px),  # bottom-right
            (x1_px, y2_px),  # bottom-left
        ]
        
        corners_3d = []
        for u, v in corners_2d:
            point_3d = self.pixel_to_3d(u, v, depth_use)
            corners_3d.append(point_3d)
        
        # Centroid (average of corners)
        centroid_3d = np.mean(corners_3d, axis=0)
        
        # 3D dimensions
        width_3d = float(np.linalg.norm(corners_3d[1] - corners_3d[0]))
        height_3d = float(np.linalg.norm(corners_3d[2] - corners_3d[1]))
        depth_3d = depth_max - depth_min
        
        return {
            "x1_2d": x1_px,
            "y1_2d": y1_px,
            "x2_2d": x2_px,
            "y2_2d": y2_px,
            "corners_3d": corners_3d,
            "centroid_3d": centroid_3d.tolist(),
            "depth_min": depth_min,
            "depth_max": depth_max,
            "depth_mean": depth_mean,
            "width_3d": width_3d,
            "height_3d": height_3d,
            "depth_3d": float(depth_3d)
        }


class DetectionMerger:
    """Merge nearby same-class detections to fix fragmentation"""
    
    def __init__(self, distance_threshold: float = 0.1):
        """
        Initialize merger.
        
        Args:
            distance_threshold: Merge boxes if IoU > threshold (0-1)
        """
        self.distance_threshold = distance_threshold
    
    def compute_iou(self, box1: Dict, box2: Dict) -> float:
        """Compute Intersection over Union for two 2D boxes"""
        x1_min, y1_min = box1["x1"], box1["y1"]
        x1_max, y1_max = box1["x2"], box1["y2"]
        
        x2_min, y2_min = box2["x1"], box2["y1"]
        x2_max, y2_max = box2["x2"], box2["y2"]
        
        # Intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max < x_min or y_max < y_min:
            return 0.0
        
        inter_area = (x_max - x_min) * (y_max - y_min)
        
        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def merge_detections(
        self,
        detections: List[Dict]
    ) -> List[Dict]:
        """
        Merge nearby same-class detections.
        
        Args:
            detections: List of detection dicts with keys:
                       {"x1", "y1", "x2", "y2", "class", "confidence", ...}
        
        Returns:
            merged_detections: List with nearby boxes merged
        """
        if not detections:
            return []
        
        # Group by class
        by_class = {}
        for det in detections:
            cls = det.get("class")
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(det)
        
        merged = []
        
        # Process each class separately
        for cls, boxes in by_class.items():
            used = set()
            
            for i, box1 in enumerate(boxes):
                if i in used:
                    continue
                
                # Find all boxes to merge with this one
                to_merge = [box1]
                used.add(i)
                
                for j, box2 in enumerate(boxes[i+1:], start=i+1):
                    if j in used:
                        continue
                    
                    iou = self.compute_iou(box1, box2)
                    if iou > self.distance_threshold:
                        to_merge.append(box2)
                        used.add(j)
                
                # Merge boxes: average coordinates, max confidence
                if len(to_merge) > 1:
                    merged_box = self._merge_boxes(to_merge)
                else:
                    merged_box = to_merge[0].copy()
                
                merged.append(merged_box)
        
        return merged
    
    def _merge_boxes(self, boxes: List[Dict]) -> Dict:
        """Merge multiple boxes into one"""
        x1_vals = [b["x1"] for b in boxes]
        y1_vals = [b["y1"] for b in boxes]
        x2_vals = [b["x2"] for b in boxes]
        y2_vals = [b["y2"] for b in boxes]
        confs = [b.get("confidence", 1.0) for b in boxes]
        
        merged = boxes[0].copy()
        merged["x1"] = min(x1_vals)
        merged["y1"] = min(y1_vals)
        merged["x2"] = max(x2_vals)
        merged["y2"] = max(y2_vals)
        merged["confidence"] = max(confs)  # Highest confidence
        merged["merged_count"] = len(boxes)
        
        return merged
