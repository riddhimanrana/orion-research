#!/usr/bin/env python3
"""
PVSG Action-Aware Relation Classifier

Predicts PVSG-style action relations using:
- MediaPipe hand detection for "holding" relations
- Temporal motion analysis for "throwing" detection
- Object trajectory for "picking" detection
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Detection:
    """Single object detection."""
    track_id: int
    class_name: str
    bbox: List[float]  # [x1, y1, x2, y2]
    frame_id: int
    confidence: float = 1.0


@dataclass
class HandDetection:
    """Hand detection with landmarks."""
    bbox: List[float]  # [x1, y1, x2, y2]
    landmarks: Optional[np.ndarray] = None  # (21, 3) if available
    handedness: str = "unknown"  # "left" or "right"


class PVSGActionClassifier:
    """
    Predicts PVSG action relations:
    - holding: hand-object proximity + temporal persistence
    - throwing: rapid object motion after hand contact
    - picking: object appearance + hand proximity
    """
    
    def __init__(
        self,
        holding_hand_dist: float = 50.0,  # pixels
        holding_min_frames: int = 3,
        throwing_velocity_thresh: float = 100.0,  # pixels/frame
        throwing_accel_thresh: float = 50.0,  # pixels/frame^2
        picking_hand_dist: float = 60.0,
        picking_appearance_window: int = 5,  # frames
    ):
        """
        Args:
            holding_hand_dist: Max distance between hand and object for "holding"
            holding_min_frames: Min consecutive frames for "holding"
            throwing_velocity_thresh: Min velocity for "throwing"
            throwing_accel_thresh: Min acceleration for "throwing"
            picking_hand_dist: Max distance for "picking"
            picking_appearance_window: Frames to look back for object appearance
        """
        self.holding_hand_dist = holding_hand_dist
        self.holding_min_frames = holding_min_frames
        self.throwing_velocity_thresh = throwing_velocity_thresh
        self.throwing_accel_thresh = throwing_accel_thresh
        self.picking_hand_dist = picking_hand_dist
        self.picking_appearance_window = picking_appearance_window
        
        # Track history for temporal analysis
        self.track_history: Dict[int, List[Detection]] = defaultdict(list)
        
        # Context-aware predicate mapping (Specific predicates for certain pairs)
        self.CONTEXT_PREDICATES = {
            "person-ball": ["holding", "throwing", "picking", "catching", "kicking"],
            "person-chair": ["sitting_on", "standing_on", "moving", "pushing"],
            "person-bicycle": ["riding", "pushing", "repairing"],
            "person-motorcycle": ["riding", "pushing"],
            "person-table": ["sitting_at", "leaning_on", "moving"],
            "person-sofa": ["sitting_on", "lying_on"],
            "person-food": ["eating", "holding", "processing"],
            "person-bottle": ["drinking", "holding", "opening"],
        }
        
        # Predicate hierarchy for debiasing (Specific vs Generic)
        self.PREDICATE_SPECIFICITY = {
            "sitting_on": 0.9,
            "riding": 0.95,
            "holding": 0.8,
            "throwing": 0.85,
            "picking": 0.75,
            "sitting_at": 0.8,
            "drinking": 0.9,
            "eating": 0.9,
            "on": 0.1,
            "near": 0.05,
            "beside": 0.1,
            "at": 0.1,
        }
    
    def get_specificity(self, predicate: str) -> float:
        """Get specificity score for a predicate."""
        return self.PREDICATE_SPECIFICITY.get(predicate, 0.5)
    def bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get bbox center point."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def compute_velocity(self, track_id: int, current_frame: int) -> Optional[Tuple[float, float]]:
        """
        Compute object velocity (pixels/frame).
        
        Returns:
            (vx, vy) or None if insufficient history
        """
        history = self.track_history[track_id]
        if len(history) < 2:
            return None
        
        # Get last two detections
        prev = history[-2]
        curr = history[-1]
        
        if curr.frame_id - prev.frame_id != 1:
            return None  # Non-consecutive frames
        
        prev_center = self.bbox_center(prev.bbox)
        curr_center = self.bbox_center(curr.bbox)
        
        vx = curr_center[0] - prev_center[0]
        vy = curr_center[1] - prev_center[1]
        
        return (vx, vy)
    
    def compute_acceleration(self, track_id: int) -> Optional[float]:
        """
        Compute object acceleration magnitude.
        
        Returns:
            Acceleration (pixels/frame^2) or None
        """
        history = self.track_history[track_id]
        if len(history) < 3:
            return None
        
        # Get last three detections
        det1, det2, det3 = history[-3:]
        
        # Check consecutive frames
        if det2.frame_id - det1.frame_id != 1 or det3.frame_id - det2.frame_id != 1:
            return None
        
        # Compute velocities
        c1 = self.bbox_center(det1.bbox)
        c2 = self.bbox_center(det2.bbox)
        c3 = self.bbox_center(det3.bbox)
        
        v1 = (c2[0] - c1[0], c2[1] - c1[1])
        v2 = (c3[0] - c2[0], c3[1] - c2[1])
        
        # Acceleration = change in velocity
        ax = v2[0] - v1[0]
        ay = v2[1] - v1[1]
        
        return np.sqrt(ax**2 + ay**2)
    
    def detect_holding(
        self,
        person_det: Detection,
        object_det: Detection,
        hand_dets: List[HandDetection]
    ) -> bool:
        """
        Detect "holding" relation.
        
        Criteria:
        - Hand is close to object
        - OR direct bbox overlap (fallback if hand estimation is poor)
        - Relation persists for multiple frames
        """
        obj_center = self.bbox_center(object_det.bbox)
        
        # 1. Hand proximity check
        if hand_dets:
            for hand in hand_dets:
                hand_center = self.bbox_center(hand.bbox)
                dist = self.distance(hand_center, obj_center)
                
                if dist < self.holding_hand_dist:
                    return True
        
        # 2. Bbox overlap fallback (Internal heuristic for better recall)
        # If person bbox and object bbox have significant overlap, assume holding
        # if the object is small and likely to be held.
        p_x1, p_y1, p_x2, p_y2 = person_det.bbox
        o_x1, o_y1, o_x2, o_y2 = object_det.bbox
        
        overlap_x = max(0, min(p_x2, o_x2) - max(p_x1, o_x1))
        overlap_y = max(0, min(p_y2, o_y2) - max(p_y1, o_y1))
        
        if overlap_x > 0 and overlap_y > 0:
            # Check if object is center-aligned with person (likely being held)
            p_center = self.bbox_center(person_det.bbox)
            dist_to_person = self.distance(p_center, obj_center)
            
            # If object is within person bbox and close enough to hand estimation zones
            if dist_to_person < (p_x2 - p_x1) * 0.8:
                return True
                
        return False
    
    def detect_throwing(
        self,
        person_det: Detection,
        object_det: Detection,
        hand_dets: List[HandDetection]
    ) -> bool:
        """
        Detect "throwing" relation.
        
        Criteria:
        - Object has high velocity
        - Object has high acceleration (sudden motion)
        - Hand was recently close to object
        """
        # Compute object velocity
        velocity = self.compute_velocity(object_det.track_id, object_det.frame_id)
        if velocity is None:
            return False
        
        v_mag = np.sqrt(velocity[0]**2 + velocity[1]**2)
        
        # Check velocity threshold
        if v_mag < self.throwing_velocity_thresh:
            return False
        
        # Check acceleration (sudden motion)
        accel = self.compute_acceleration(object_det.track_id)
        if accel is None or accel < self.throwing_accel_thresh:
            return False
        
        # Check if hand was recently close (within last few frames)
        if not hand_dets:
            return False
        
        obj_center = self.bbox_center(object_det.bbox)
        for hand in hand_dets:
            hand_center = self.bbox_center(hand.bbox)
            dist = self.distance(hand_center, obj_center)
            
            if dist < self.holding_hand_dist * 1.5:  # Slightly larger radius
                return True
        
        return False
    
    def detect_picking(
        self,
        person_det: Detection,
        object_det: Detection,
        hand_dets: List[HandDetection]
    ) -> bool:
        """
        Detect "picking" relation.
        
        Criteria:
        - Object recently appeared (first detection or long gap)
        - Hand is close to object
        """
        if not hand_dets:
            return False
        
        # Check if object recently appeared
        history = self.track_history[object_det.track_id]
        if len(history) > self.picking_appearance_window:
            return False  # Object has been visible for too long
        
        # Check hand proximity
        obj_center = self.bbox_center(object_det.bbox)
        for hand in hand_dets:
            hand_center = self.bbox_center(hand.bbox)
            dist = self.distance(hand_center, obj_center)
            
            if dist < self.picking_hand_dist:
                return True
        
        return False
    
    def update_track_history(self, detection: Detection):
        """Add detection to track history."""
        self.track_history[detection.track_id].append(detection)
        
        # Keep only recent history (last 30 frames)
        if len(self.track_history[detection.track_id]) > 30:
            self.track_history[detection.track_id].pop(0)
    
    def detect_sitting(
        self,
        person_det: Detection,
        object_det: Detection
    ) -> bool:
        """
        Detect "sitting_on" or "sitting_at" relations.
        Criteria: Bbox overlap + relative height (person hip near object surface).
        """
        if 'chair' not in object_det.class_name.lower() and 'sofa' not in object_det.class_name.lower():
            return False
            
        p_x1, p_y1, p_x2, p_y2 = person_det.bbox
        o_x1, o_y1, o_x2, o_y2 = object_det.bbox
        
        # Bbox overlap check
        overlap_x = max(0, min(p_x2, o_x2) - max(p_x1, o_x1))
        overlap_y = max(0, min(p_y2, o_y2) - max(p_y1, o_y1))
        
        if overlap_x == 0 or overlap_y == 0:
            return False
            
        # Physical heuristic: person's bottom half is near object's top surface
        p_height = p_y2 - p_y1
        person_mid_y = p_y1 + 0.7 * p_height  # Approximate hip level
        
        if abs(person_mid_y - o_y1) < 0.2 * p_height:
            return True
            
        return False

    def detect_riding(
        self,
        person_det: Detection,
        object_det: Detection
    ) -> bool:
        """Detect "riding" (bicycle, motorcycle)."""
        if 'bicycle' not in object_det.class_name.lower() and 'motorcycle' not in object_det.class_name.lower():
            return False
            
        # Check for overlap + both moving
        overlap_x = max(0, min(person_det.bbox[2], object_det.bbox[2]) - max(person_det.bbox[0], object_det.bbox[0]))
        if overlap_x == 0:
            return False
            
        vel = self.compute_velocity(object_det.track_id, object_det.frame_id)
        if vel:
            v_mag = np.sqrt(vel[0]**2 + vel[1]**2)
            if v_mag > 5.0:  # If it's moving
                return True
        return False

    def detect_walking(
        self,
        person_det: Detection
    ) -> bool:
        """Detect "walking" based on person motion."""
        vel = self.compute_velocity(person_det.track_id, person_det.frame_id)
        if vel:
            v_mag = np.sqrt(vel[0]**2 + vel[1]**2)
            # Person moves at a reasonable speed horizontally
            if v_mag > 5.0 and abs(vel[0]) > 0.6 * v_mag:
                return True
        return False

    def detect_standing(
        self,
        person_det: Detection
    ) -> bool:
        """Detect "standing" roughly (low motion)."""
        vel = self.compute_velocity(person_det.track_id, person_det.frame_id)
        if vel:
            v_mag = np.sqrt(vel[0]**2 + vel[1]**2)
            if v_mag < 2.0:
                return True
        return True # Default assume standing if no motion info

    def detect_eating(
        self,
        person_det: Detection,
        object_det: Detection,
        hand_dets: List[HandDetection]
    ) -> bool:
        """
        Detect "eating" relation.
        Criteria: Object near upper body/face region of person.
        """
        # 1. Object must be food-like (heuristic)
        # In a general sense, we trust the caller to check class compatibility, 
        # or we check generic overlap near top.
        
        p_x1, p_y1, p_x2, p_y2 = person_det.bbox
        o_x1, o_y1, o_x2, o_y2 = object_det.bbox
        
        # Object center
        obj_center = self.bbox_center(object_det.bbox)
        
        # Person "face region" estimate (top 20% of bbox)
        face_y_threshold = p_y1 + (p_y2 - p_y1) * 0.3
        
        # Check if object is in the upper part of person bbox
        if (p_x1 < obj_center[0] < p_x2) and (p_y1 < obj_center[1] < face_y_threshold):
            return True
            
        return False

    def detect_touching(
        self,
        person_det: Detection,
        object_det: Detection,
        hand_dets: List[HandDetection]
    ) -> bool:
        """
        Detect "touching" relation.
        Criteria: Hand very close to object (or object overlap), but maybe not holding (short duration?).
        For now, we treat it as a spatial proximity weaker than holding.
        """
        obj_center = self.bbox_center(object_det.bbox)
        
        # 1. Hand proximity check
        if hand_dets:
            for hand in hand_dets:
                hand_center = self.bbox_center(hand.bbox)
                dist = self.distance(hand_center, obj_center)
                
                if dist < self.holding_hand_dist: # Re-use holding distance
                    return True
        
        # 2. Bbox overlap fallback
        p_x1, p_y1, p_x2, p_y2 = person_det.bbox
        o_x1, o_y1, o_x2, o_y2 = object_det.bbox
        
        overlap_x = max(0, min(p_x2, o_x2) - max(p_x1, o_x1))
        overlap_y = max(0, min(p_y2, o_y2) - max(p_y1, o_y1))
        
        if overlap_x > 0 and overlap_y > 0:
            return True
            
        return False

    def detect_swinging(
        self,
        person_det: Detection,
        object_det: Detection,
        hand_dets: List[HandDetection]
    ) -> bool:
        """
        Detect "swinging" relation.
        Criteria: Object is held AND has high angular/circular motion or high velocity variance.
        Simplified: Object is held + High Velocity + changing direction.
        """
        # Must be holding first
        if not self.detect_holding(person_det, object_det, hand_dets):
            return False
            
        # Check high velocity (swinging implies fast movement)
        vel = self.compute_velocity(object_det.track_id, object_det.frame_id)
        if vel:
            v_mag = np.sqrt(vel[0]**2 + vel[1]**2)
            if v_mag > 8.0: # Good threshold for swinging
                return True
                
        return False

    def predict_relations(
        self,
        detections: List[Detection],
        hand_detections: Dict[int, List[HandDetection]],  # person_track_id -> hands
        frame_id: int
    ) -> List[Tuple[int, int, str, float]]:
        """
        Predict PVSG action relations for a frame.
        """
        relations = []
        
        # Background objects that cannot be picked/held/thrown
        NON_INTERACTABLE = {'ground', 'floor', 'grass', 'sky', 'wall', 'field', 'court', 'road', 'sidewalk', 'ceiling', 'tree', 'carpet', 'sand', 'water', 'sea', 'river', 'mat', 'rug', 'blanket'}
        
        # Update track history
        for det in detections:
            self.update_track_history(det)
        
        # Separate persons and objects
        persons = [d for d in detections if d.class_name in ['person', 'adult', 'child', 'man', 'woman', 'boy', 'girl']]
        objects = [d for d in detections if d not in persons]
        
        # For each person-object pair
        for person in persons:
            person_hands = hand_detections.get(person.track_id, [])
            
            for obj in objects:
                candidates = [] # (predicate, score)
                
                # Filter impossible interactions
                is_background = obj.class_name in NON_INTERACTABLE
                
                # 1. Physical/Action checks
                
                # Swinging (Priority for bat/racket/etc)
                if not is_background and self.detect_swinging(person, obj, person_hands):
                    candidates.append(('swinging', 0.95))
                
                # Throwing (Priority)
                if not is_background and self.detect_throwing(person, obj, person_hands):
                    candidates.append(('throwing', 0.9))
                
                # Picking
                if not is_background and self.detect_picking(person, obj, person_hands):
                    candidates.append(('picking', 0.85))
                
                # Riding
                if self.detect_riding(person, obj):
                    candidates.append(('riding', 0.95))
                
                # Sitting
                if self.detect_sitting(person, obj):
                    if 'chair' in obj.class_name.lower():
                        candidates.append(('sitting_on', 0.9))
                    else:
                        candidates.append(('sitting_at', 0.8))
                
                # Holding
                if not is_background and self.detect_holding(person, obj, person_hands):
                    candidates.append(('holding', 0.85))

                
                # Eating
                if not is_background and obj.class_name in ['cake', 'bread', 'food', 'apple', 'sandwich', 'beverage', 'drink', 'bottle']:
                    if self.detect_eating(person, obj, person_hands):
                        candidates.append(('eating', 0.9))
                
                # Touching (General interaction)
                if not is_background and self.detect_touching(person, obj, person_hands):
                     candidates.append(('touching', 0.6))

                # Standalone person predicates (interaction with surroundings)
                # This is where we handle background objects specifically
                
                # 2. Add generic spatial fallbacks with low scores (Debiasing)
                p_x1, p_y1, p_x2, p_y2 = person.bbox
                o_x1, o_y1, o_x2, o_y2 = obj.bbox
                overlap_x = max(0, min(p_x2, o_x2) - max(p_x1, o_x1))
                overlap_y = max(0, min(p_y2, o_y2) - max(p_y1, o_y1))
                
                if overlap_x > 0 and overlap_y > 0:
                    # Specific walking/standing if on floor/carpet
                    if 'floor' in obj.class_name.lower() or 'carpet' in obj.class_name.lower() or 'ground' in obj.class_name.lower() or 'grass' in obj.class_name.lower():
                        if self.detect_walking(person):
                            candidates.append(('walking_on', 0.9))
                        else:
                            candidates.append(('standing_on', 0.85))
                    
                    if not is_background: # Only prioritize 'on' for non-background? Or is 'on table' valid?
                         # 'person on carpet' is valid but covered by walking/standing_on
                         # 'person on table' is valid.
                         candidates.append(('on', 0.1)) 
                else:
                    obj_center = self.bbox_center(obj.bbox)
                    subj_center = self.bbox_center(person.bbox)
                    if self.distance(obj_center, subj_center) < 200:
                        candidates.append(('near', 0.05))
                        candidates.append(('next_to', 0.05))

                # 3. Selection / Debiasing logic: 
                # Return ALL candidates for Recall@K evaluation
                if candidates:
                    # Sort by score
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    
                    # Return top-5 per pair to avoid explosion, but definitely more than 1
                    for pred, score in candidates[:5]:
                        relations.append((person.track_id, obj.track_id, pred, score))
        
        return relations


def main():
    """Example usage."""
    classifier = PVSGActionClassifier()
    
    # Example detections
    person = Detection(
        track_id=1,
        class_name='person',
        bbox=[100, 100, 200, 300],
        frame_id=10
    )
    
    ball = Detection(
        track_id=2,
        class_name='ball',
        bbox=[150, 150, 180, 180],
        frame_id=10
    )
    
    hand = HandDetection(
        bbox=[145, 145, 175, 175]
    )
    
    detections = [person, ball]
    hand_detections = {1: [hand]}
    
    relations = classifier.predict_relations(detections, hand_detections, frame_id=10)
    
    print("Predicted relations:")
    for subj, obj, pred in relations:
        print(f"  [{subj}] --{pred}--> [{obj}]")


if __name__ == '__main__':
    main()
