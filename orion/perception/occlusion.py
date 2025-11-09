"""
Occlusion detection using depth maps.
"""

import time
from typing import List, Optional
import numpy as np

from .types import EntityState3D, Hand, VisibilityState, OcclusionInfo


class OcclusionDetector:
    """
    Detect occlusions using depth information.
    
    Uses depth map to identify when entities are occluded by:
    - Other entities (depth comparison)
    - Hands (geometric overlap + depth)
    """
    
    def __init__(
        self,
        occlusion_threshold: float = 0.3,
        depth_margin_mm: float = 100.0,
    ):
        """
        Initialize occlusion detector.
        
        Args:
            occlusion_threshold: Ratio of pixels occluded to mark as OCCLUDED
            depth_margin_mm: Depth difference threshold for occlusion (mm)
        """
        self.occlusion_threshold = occlusion_threshold
        self.depth_margin_mm = depth_margin_mm
        
        print(f"[OcclusionDetector] threshold={occlusion_threshold}, margin={depth_margin_mm}mm")
    
    def detect_occlusions(
        self,
        entities: List[EntityState3D],
        hands: List[Hand],
        depth_map: np.ndarray
    ) -> List[OcclusionInfo]:
        """
        Detect occlusions for all entities.
        
        Args:
            entities: List of detected entities
            hands: List of detected hands
            depth_map: Depth map (H, W) in millimeters
            
        Returns:
            List of OcclusionInfo for each entity
        """
        start_time = time.time()
        
        occlusion_infos = []
        
        for entity in entities:
            occlusion_info = self._check_entity_occlusion(
                entity, entities, hands, depth_map
            )
            occlusion_infos.append(occlusion_info)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return occlusion_infos
    
    def _check_entity_occlusion(
        self,
        entity: EntityState3D,
        all_entities: List[EntityState3D],
        hands: List[Hand],
        depth_map: np.ndarray
    ) -> OcclusionInfo:
        """
        Check if a single entity is occluded.
        
        Args:
            entity: Entity to check
            all_entities: All entities in frame
            hands: All hands in frame
            depth_map: Depth map
            
        Returns:
            OcclusionInfo with occlusion status
        """
        # Skip if entity has no valid depth
        if entity.depth_mean_mm is None or entity.depth_mean_mm <= 0:
            return OcclusionInfo(
                entity_id=entity.entity_id,
                occlusion_ratio=0.0,
                visibility_state=VisibilityState.UNKNOWN,
                occluded_by=None,
            )
        
        x1, y1, x2, y2 = entity.bbox_2d_px
        
        # Clamp to image bounds
        h, w = depth_map.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)
        
        # Extract depth region
        entity_region = depth_map[y1:y2, x1:x2]
        
        # Filter valid depths
        valid_mask = (entity_region > 0) & np.isfinite(entity_region)
        valid_depths = entity_region[valid_mask]
        
        if len(valid_depths) == 0:
            return OcclusionInfo(
                entity_id=entity.entity_id,
                occlusion_ratio=0.0,
                visibility_state=VisibilityState.UNKNOWN,
                occluded_by=None,
            )
        
        # Count pixels significantly in front of entity mean depth
        occluded_mask = valid_depths < (entity.depth_mean_mm - self.depth_margin_mm)
        occlusion_ratio = float(np.sum(occluded_mask)) / len(valid_depths)
        
        # Determine visibility state
        if occlusion_ratio < 0.1:
            visibility_state = VisibilityState.FULLY_VISIBLE
            occluded_by = None
        elif occlusion_ratio < self.occlusion_threshold:
            visibility_state = VisibilityState.PARTIALLY_OCCLUDED
            occluded_by = self._find_occluder(entity, all_entities, hands)
        else:
            visibility_state = VisibilityState.PARTIALLY_OCCLUDED
            occluded_by = self._find_occluder(entity, all_entities, hands)
        
        return OcclusionInfo(
            entity_id=entity.entity_id,
            occlusion_ratio=occlusion_ratio,
            visibility_state=visibility_state,
            occluded_by=occluded_by,
        )
    
    def _find_occluder(
        self,
        entity: EntityState3D,
        all_entities: List[EntityState3D],
        hands: List[Hand]
    ) -> Optional[str]:
        """
        Find what is occluding the entity.
        
        Args:
            entity: Entity being occluded
            all_entities: All entities
            hands: All hands
            
        Returns:
            ID of occluding object ("hand" or entity_id) or None
        """
        entity_bbox = entity.bbox_2d_px
        entity_depth = entity.depth_mean_mm
        
        if entity_depth is None:
            return None
        
        # Check if hands are occluding (hands in front + overlapping bbox)
        for hand in hands:
            # Get hand bounding box from landmarks
            landmarks_2d = hand.landmarks_2d
            if len(landmarks_2d) == 0:
                continue
            
            xs = [lm[0] for lm in landmarks_2d]
            ys = [lm[1] for lm in landmarks_2d]
            
            # Convert normalized to pixels (assuming 1920x1080 for now)
            # TODO: Pass actual image dimensions
            hand_x1 = min(xs) * 1920
            hand_y1 = min(ys) * 1080
            hand_x2 = max(xs) * 1920
            hand_y2 = max(ys) * 1080
            
            # Check overlap
            if self._boxes_overlap(entity_bbox, (hand_x1, hand_y1, hand_x2, hand_y2)):
                # Check if hand is in front
                hand_depth = hand.palm_center_3d[2]  # Z coordinate
                if hand_depth < entity_depth - self.depth_margin_mm:
                    return "hand"
        
        # Check other entities
        for other in all_entities:
            if other.entity_id == entity.entity_id:
                continue
            
            if other.depth_mean_mm is None:
                continue
            
            # Check if other entity overlaps and is in front
            if self._boxes_overlap(entity_bbox, other.bbox_2d_px):
                if other.depth_mean_mm < entity_depth - self.depth_margin_mm:
                    return other.entity_id
        
        return None
    
    def _boxes_overlap(
        self,
        box1: tuple,
        box2: tuple,
        iou_threshold: float = 0.1
    ) -> bool:
        """
        Check if two bounding boxes overlap.
        
        Args:
            box1: (x1, y1, x2, y2)
            box2: (x1, y1, x2, y2)
            iou_threshold: Minimum IoU to consider overlap
            
        Returns:
            True if boxes overlap significantly
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Compute intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return False
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Compute union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        
        return iou >= iou_threshold
