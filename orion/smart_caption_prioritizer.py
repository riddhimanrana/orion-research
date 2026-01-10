"""
Smart Caption Prioritization

Only captions the MOST IMPORTANT entities:
- Hand-held objects (egocentric view)
- Zone transition events (room changes)
- Long-tracked objects (furniture, landmarks)
- Novel object classes (first time seen)

Goal: 10-15 high-value captions for 60s video, not 100!
"""

from typing import List, Optional, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CaptionPriority:
    """Priority scoring for caption decisions"""
    entity_id: int
    class_name: str
    priority_score: float
    reason: str  # Why this entity is important
    
    # Priority factors
    is_hand_held: bool = False
    is_zone_transition: bool = False
    is_long_tracked: bool = False
    is_novel_class: bool = False
    is_in_center_frame: bool = False


class SmartCaptionPrioritizer:
    """
    Intelligently prioritizes which entities to caption
    
    Strategy:
    - HIGHEST: Hand-held objects (close to camera, moving with camera)
    - HIGH: Zone transitions (object changes rooms)
    - MEDIUM: Long-tracked landmarks (furniture, TV, etc.)
    - LOW: Static background objects
    
    Target: 10-15 captions for 60s video
    """
    
    def __init__(
        self,
        target_captions: int = 12,  # Conservative target
        hand_zone_threshold_mm: float = 1000.0,  # 1m from camera = hand-held
        center_threshold: float = 0.3  # Within 30% of center = focused
    ):
        self.target_captions = target_captions
        self.hand_zone_threshold_mm = hand_zone_threshold_mm
        self.center_threshold = center_threshold
        
        # Track what we've seen
        self.captioned_entities: Set[int] = set()
        self.seen_classes: Set[str] = set()
        self.zone_transitions: dict[int, list[int]] = {}  # entity_id -> [zone_ids]
        
        # Priority classes (always caption these)
        self.priority_classes = {
            'book', 'cell phone', 'laptop', 'remote', 'bottle', 
            'cup', 'keys', 'wallet', 'bag'  # Hand-held objects
        }
        
        # Landmark classes (caption once per video)
        self.landmark_classes = {
            'tv', 'couch', 'bed', 'refrigerator', 'oven', 
            'dining table', 'desk', 'chair'
        }
    
    def should_caption(
        self,
        entity_id: int,
        class_name: str,
        depth_mm: float,
        bbox_center: tuple,  # (x, y) normalized 0-1
        zone_id: Optional[int],
        tracking_frames: int,
        frame_width: int,
        frame_height: int
    ) -> Optional[CaptionPriority]:
        """
        Decide if entity should be captioned
        
        Returns:
            CaptionPriority if should caption, None otherwise
        """
        # Skip if already captioned
        if entity_id in self.captioned_entities:
            return None
        
        # Skip if budget exhausted
        if len(self.captioned_entities) >= self.target_captions:
            return None
        
        priority_score = 0.0
        reasons = []
        
        # Factor 1: Hand-held detection (HIGHEST PRIORITY)
        is_hand_held = depth_mm < self.hand_zone_threshold_mm and depth_mm > 0
        if is_hand_held:
            priority_score += 100.0
            reasons.append("HAND-HELD (egocentric)")
        
        # Factor 2: Center frame (user is focusing on this)
        cx, cy = bbox_center
        frame_cx, frame_cy = 0.5, 0.5
        distance_from_center = ((cx - frame_cx)**2 + (cy - frame_cy)**2) ** 0.5
        is_in_center = distance_from_center < self.center_threshold
        
        if is_in_center:
            priority_score += 30.0
            reasons.append("CENTER_FOCUS")
        
        # Factor 3: Priority class (common hand-held items)
        if class_name in self.priority_classes:
            priority_score += 50.0
            reasons.append("PRIORITY_CLASS")
        
        # Factor 4: Zone transition (room change)
        is_zone_transition = False
        if entity_id in self.zone_transitions and zone_id is not None:
            prev_zones = self.zone_transitions[entity_id]
            if zone_id not in prev_zones:
                is_zone_transition = True
                priority_score += 40.0
                reasons.append("ZONE_TRANSITION")
                self.zone_transitions[entity_id].append(zone_id)
        elif zone_id is not None:
            self.zone_transitions[entity_id] = [zone_id]
        
        # Factor 5: Novel class (first time seeing this type)
        is_novel_class = class_name not in self.seen_classes
        if is_novel_class:
            if class_name in self.landmark_classes:
                # Landmarks: Caption once
                priority_score += 20.0
                reasons.append("LANDMARK")
                self.seen_classes.add(class_name)
            elif class_name in self.priority_classes:
                # Priority items: Always caption
                priority_score += 25.0
                reasons.append("NOVEL_PRIORITY")
                self.seen_classes.add(class_name)
        
        # Factor 6: Long tracking (well-established object)
        is_long_tracked = tracking_frames > 30  # >1 second at 30fps
        if is_long_tracked and class_name in self.landmark_classes:
            priority_score += 15.0
            reasons.append("LANDMARK_STABLE")
        
        # Decision threshold
        CAPTION_THRESHOLD = 25.0  # Minimum score to caption
        
        if priority_score >= CAPTION_THRESHOLD:
            return CaptionPriority(
                entity_id=entity_id,
                class_name=class_name,
                priority_score=priority_score,
                reason=" + ".join(reasons),
                is_hand_held=is_hand_held,
                is_zone_transition=is_zone_transition,
                is_long_tracked=is_long_tracked,
                is_novel_class=is_novel_class,
                is_in_center_frame=is_in_center
            )
        
        return None
    
    def mark_captioned(self, entity_id: int):
        """Mark entity as captioned"""
        self.captioned_entities.add(entity_id)
        logger.info(f"Captioned entity {entity_id} ({len(self.captioned_entities)}/{self.target_captions})")
    
    def get_statistics(self) -> dict:
        """Get prioritization statistics"""
        return {
            'total_captioned': len(self.captioned_entities),
            'target_captions': self.target_captions,
            'budget_used': len(self.captioned_entities) / self.target_captions,
            'novel_classes_seen': len(self.seen_classes),
            'zone_transitions_tracked': len(self.zone_transitions)
        }
    
    def get_priority_summary(self) -> str:
        """Get human-readable summary"""
        stats = self.get_statistics()
        return (
            f"Smart Captioning: {stats['total_captioned']}/{stats['target_captions']} "
            f"({stats['budget_used']*100:.0f}% budget used)"
        )
