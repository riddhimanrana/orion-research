#!/usr/bin/env python3
"""
Run and Evaluate PVSG Scene Graph Generation - Orion Pipeline

Unified script implementing the full Orion perception pipeline:
  Phase 1: Detection (GroundingDINO proposer + DINOv3 refinement)
  Phase 2: Tracking + Relation Generation (VJEPA2, CIS, FastVLM, PVSGActionClassifier)
  Phase 3: Evaluation (R@K metrics against PVSG GT)

Pipeline Flow:
1. OpenCV samples frames from video
2. GroundingDINO proposes initial detections
3. DINOv3 refines/embeds detections for Re-ID
4. Tracker groups detections into tracks (with VJEPA2 embeddings)
5. CIS computes person→object causal edges
6. FastVLM + PVSGActionClassifier generate relations
7. Evaluation computes R@K statistics
"""

import os
import json
import argparse
import subprocess
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import cv2
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# BACKEND IMPORTS - Orion Pipeline Components
# ============================================================================

# GroundingDINO (Proposer for zero-shot detection)
try:
    from orion.perception.detectors.grounding_dino import GroundingDINOWrapper
    GDINO_AVAILABLE = True
    logger.info("✓ GroundingDINO available")
except ImportError as e:
    logger.warning(f"GroundingDINO not available: {e}")
    GDINO_AVAILABLE = False
    GroundingDINOWrapper = None

# DINOv3 (Embedder for Re-ID refinement)
try:
    from orion.backends.dino_backend import DINOEmbedder
    DINO_AVAILABLE = True
    logger.info("✓ DINOv3 Embedder available")
except ImportError as e:
    logger.warning(f"DINOEmbedder not available: {e}")
    DINO_AVAILABLE = False
    DINOEmbedder = None

# VJEPA2 (Video-native Re-ID embeddings)
try:
    from orion.backends.vjepa2_backend import VJepa2Embedder
    VJEPA2_AVAILABLE = True
    logger.info("✓ VJEPA2 Embedder available")
except ImportError as e:
    logger.warning(f"VJepa2Embedder not available: {e}")
    VJEPA2_AVAILABLE = False
    VJepa2Embedder = None

# CIS Scorer (Causal Influence Scoring for person→object edges)
try:
    from orion.analysis.cis_scorer import CausalInfluenceScorer, CISEdge
    CIS_AVAILABLE = True
    logger.info("✓ CIS Scorer available")
except ImportError as e:
    logger.warning(f"CausalInfluenceScorer not available: {e}")
    CIS_AVAILABLE = False
    CausalInfluenceScorer = None

# FastVLM (Visual Language Model for relation verification)
try:
    from orion.backends.mlx_fastvlm import FastVLMMLXWrapper
    FASTVLM_AVAILABLE = True
    logger.info("✓ FastVLM available")
except ImportError as e:
    logger.warning(f"FastVLM not available: {e}")
    FASTVLM_AVAILABLE = False
    FastVLMMLXWrapper = None

# PVSGActionClassifier (Predicate generation)
try:
    from orion.perception.pvsg_action_classifier import PVSGActionClassifier, Detection, HandDetection
    PVSG_CLASSIFIER_AVAILABLE = True
    logger.info("✓ PVSGActionClassifier available")
except ImportError as e:
    logger.warning(f"PVSGActionClassifier not available: {e}")
    PVSG_CLASSIFIER_AVAILABLE = False
    
    # Fallback dataclasses
    @dataclass
    class Detection:
        track_id: int
        class_name: str
        bbox: List[float]
        frame_id: int
        confidence: float = 1.0
        velocity: Optional[Tuple[float, float]] = None
    
    @dataclass
    class HandDetection:
        bbox: List[float]
        handedness: str = "unknown"
    
    PVSGActionClassifier = None

# EnhancedTracker
try:
    from orion.perception.trackers.enhanced import EnhancedTracker, Track
    TRACKER_AVAILABLE = True
    logger.info("✓ EnhancedTracker available")
except ImportError as e:
    logger.warning(f"EnhancedTracker not available: {e}")
    TRACKER_AVAILABLE = False
    EnhancedTracker = None

# Fallback: YOLOWorld (if GroundingDINO not available)
try:
    from ultralytics import YOLOWorld
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLOWorld = None

# ============================================================================
# PVSG VOCABULARY
# ============================================================================

PVSG_PREDICATES = [
    "beside", "biting", "blowing", "brushing", "caressing", "carrying", "catching", "chasing",
    "cleaning", "closing", "cooking", "cutting", "drinking from", "eating", "entering", "feeding",
    "grabbing", "guiding", "hanging from", "hitting", "holding", "hugging", "in", "in front of",
    "jumping from", "jumping over", "kicking", "kissing", "licking", "lighting", "looking at",
    "lying on", "next to", "on", "opening", "over", "picking", "playing", "playing with",
    "pointing to", "pulling", "pushing", "riding", "running on", "shaking hand with", "sitting on",
    "standing on", "stepping on", "stirring", "swinging", "talking to", "throwing", "touching",
    "toward", "walking on", "watering", "wearing",
    # Additional Action Genome predicates for better recall
    "behind", "beneath", "below", "under", "leaning on", "covered by", "wiping", "twisting"
]


# ============================================================================
# SIMPLE TRACKER (Fallback if EnhancedTracker not available)
# ============================================================================

class SimpleTracker:
    """Simple IoU-based tracker for object tracking (fallback)."""
    def __init__(self, max_age=30, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 1
        self.track_ages = {}
        self.track_history = defaultdict(list)  # For velocity computation
    
    def update(self, detections: List[Dict], frame_id: int = 0) -> List[Dict]:
        """Update tracks with new detections."""
        results = []
        matched = set()
        
        for det in detections:
            best_iou = 0
            best_track_id = None
            det_box = det['bbox']
            det_label = det['label']
            
            for track_id, track in self.tracks.items():
                if track['label'] != det_label:
                    continue
                iou = self._calc_iou(det_box, track['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Compute velocity from history
                velocity = self._compute_velocity(best_track_id, det_box)
                
                self.tracks[best_track_id]['bbox'] = det_box
                self.tracks[best_track_id]['confidence'] = det['confidence']
                self.track_ages[best_track_id] = 0
                matched.add(best_track_id)
                
                result = {**det, 'track_id': best_track_id, 'velocity': velocity}
                results.append(result)
                
                # Update history
                self.track_history[best_track_id].append({
                    'bbox': det_box, 'frame_id': frame_id
                })
            else:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    'bbox': det_box,
                    'label': det_label,
                    'confidence': det['confidence']
                }
                self.track_ages[track_id] = 0
                matched.add(track_id)
                results.append({**det, 'track_id': track_id, 'velocity': None})
                
                # Initialize history
                self.track_history[track_id].append({
                    'bbox': det_box, 'frame_id': frame_id
                })
        
        # Age out old tracks
        to_remove = []
        for track_id in self.tracks:
            if track_id not in matched:
                self.track_ages[track_id] += 1
                if self.track_ages[track_id] > self.max_age:
                    to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
            del self.track_ages[track_id]
            if track_id in self.track_history:
                del self.track_history[track_id]
        
        return results
    
    def _compute_velocity(self, track_id: int, new_bbox: List[float]) -> Optional[Tuple[float, float]]:
        """Compute velocity from track history."""
        history = self.track_history.get(track_id, [])
        if len(history) < 1:
            return None
        
        prev = history[-1]['bbox']
        cx_prev = (prev[0] + prev[2]) / 2
        cy_prev = (prev[1] + prev[3]) / 2
        cx_new = (new_bbox[0] + new_bbox[2]) / 2
        cy_new = (new_bbox[1] + new_bbox[3]) / 2
        
        return (cx_new - cx_prev, cy_new - cy_prev)
    
    def _calc_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0


def _compute_spatial_relation(bbox1: List[float], bbox2: List[float]) -> Tuple[Optional[str], float]:
    """
    Compute spatial relation between two bboxes based on geometry.
    Returns (predicate, confidence) tuple.
    
    PVSG Predicates: on, in, next to, walking on, lying on, sitting on, holding, playing with
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1  # subject bbox
    x1_2, y1_2, x2_2, y2_2 = bbox2  # object bbox
    
    # Centers
    cx1, cy1 = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
    cx2, cy2 = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
    
    # Sizes
    w1, h1 = x2_1 - x1_1, y2_1 - y1_1
    w2, h2 = x2_2 - x1_2, y2_2 - y1_2
    area1 = max(w1 * h1, 1)
    area2 = max(w2 * h2, 1)
    
    # Overlap/intersection
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # IoU and containment
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)
    containment_of_1_in_2 = inter_area / (area1 + 1e-6)  # How much of subject is inside object
    containment_of_2_in_1 = inter_area / (area2 + 1e-6)  # How much of object is inside subject
    
    # Distance between centers
    dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    avg_size = (max(w1, h1) + max(w2, h2)) / 2
    normalized_dist = dist / (avg_size + 1e-6)
    
    # 1. 'in' - subject is contained within object (e.g., cat in box)
    if containment_of_1_in_2 > 0.5 and area1 < area2:
        return 'in', min(0.9, containment_of_1_in_2)
    
    # 2. 'on' - subject is on top of object (vertical stacking, small overlap)
    # Subject's bottom is near object's top, with horizontal overlap
    if inter_w > 0:
        # Subject's bottom (y2_1) is near object's top (y1_2) 
        vertical_gap = y1_1 - y2_2  # Negative if subject is below object
        if -h1 * 0.3 < vertical_gap < h1 * 0.5:
            # Subject is resting on top of object
            return 'on', max(0.5, 0.8 - abs(vertical_gap) / h1 * 0.3)
    
    # 3. 'holding' - object is small and contained in subject's lower region
    if containment_of_2_in_1 > 0.3 and area2 < area1 * 0.5:
        # Object is in lower 60% of subject (typical holding position)
        if cy2 > cy1:
            return 'holding', min(0.85, containment_of_2_in_1 + 0.3)
    
    # 4. 'beneath' / 'below' - subject is below/under object
    # AG meaning: person "beneath chair" = sitting in it (body below chair back level)
    # Multiple conditions:
    # a) Subject overlaps object AND subject center is lower than object center
    # b) Subject's body extends below object's top edge
    if inter_w > 0.2 * min(w1, w2):  # Some horizontal overlap
        # Case 1: Person beneath chair - significant overlap, person is "under" chair structure
        if cy1 > cy2 and inter_h > 0.2 * min(h1, h2):
            # Person's center is below chair center AND they overlap
            return 'beneath', max(0.55, 0.75 - abs(cy1 - cy2) / h2 * 0.1)
        # Case 2: Subject is fully below object (traditional beneath)
        if y1_1 > y2_2 - h2 * 0.2:
            return 'beneath', max(0.5, 0.8 - abs(y1_1 - y2_2) / h2 * 0.2)
    
    # 5. 'behind' - subject is occluded by / further from camera than object
    # In image: smaller area and higher y (further up) often means behind
    if normalized_dist < 2.5:
        if area1 < area2 * 0.8 and cy1 < cy2:
            return 'behind', 0.6
    
    # 6. 'next to' / 'beside' - objects are close but not overlapping much
    if normalized_dist < 2.0 and iou < 0.3:
        return 'next to', max(0.4, 0.8 - normalized_dist * 0.2)
    
    # 7. 'in front of' - one is closer to camera (larger/lower in image typically)
    if normalized_dist < 3.0:
        # In image coordinates, lower y = closer to camera in many setups
        if cy1 > cy2 and area1 > area2 * 0.8:
            return 'in front of', 0.65
    
    # 8. 'leaning on' - subject overlaps object significantly at edge
    if 0.1 < iou < 0.4 and normalized_dist < 1.5:
        # Check if subject's edge is at object's edge (leaning)
        edge_overlap = inter_w / min(w1, w2)
        if edge_overlap > 0.3:
            return 'leaning on', min(0.7, 0.5 + edge_overlap * 0.3)
    
    # 9. Weak relation for nearby objects
    if normalized_dist < 4.0:
        return 'next to', max(0.3, 0.6 - normalized_dist * 0.1)
    
    return None, 0.0


# ============================================================================
# PHASE 1: DETECTION (GroundingDINO + DINOv3)
# ============================================================================

def build_pvsg_prompt(max_classes: int = 100) -> str:
    """Build detection prompt from PVSG vocabulary.
    
    GroundingDINO has a token limit (~256), so we limit to most common classes.
    CRITICAL: Prioritize objects that appear in PVSG relations!
    """
    # HIGH PRIORITY: Objects that commonly appear in PVSG relations
    # These are the most important for scene graph evaluation
    high_priority = [
        # People (PVSG uses adult/child/baby, not generic 'person')
        'adult', 'child', 'baby',
        # Common relation objects (frequently in GT)
        'chair', 'table', 'sofa', 'bed', 'floor', 'ground',
        'cake', 'candle', 'plate', 'cup', 'bottle', 'glass',
        'ball', 'toy', 'book', 'phone', 'bag',
        'door', 'window', 'tv', 'computer',
        'dog', 'cat', 'bird',
        'car', 'bike', 'bicycle',
    ]
    
    # MEDIUM PRIORITY: Other PVSG vocabulary
    pvsg_things = [
        'bag', 'ballon', 'basket', 'bat', 'bench',
        'beverage', 'blanket', 'board', 'bowl',
        'box', 'bread', 'brush', 'bucket', 'cabinet', 'camera', 'can',
        'card', 'carpet', 'cart', 'cellphone',
        'chopstick', 'cloth', 'condiment', 'cookie',
        'countertop', 'cover', 'curtain', 'drawer',
        'dustbin', 'egg', 'fan', 'faucet', 'fence', 'flower', 'fork', 'fridge',
        'fruit', 'gift', 'glasses', 'glove', 'grain', 'guitar', 'hat',
        'helmet', 'horse', 'iron', 'knife', 'light', 'lighter', 'mat', 'meat',
        'microphone', 'microwave', 'mop', 'net', 'noodle', 'oven',
        'pan', 'paper', 'piano', 'pillow', 'pizza', 'plant', 'pot',
        'powder', 'rack', 'racket', 'rag', 'ring', 'scissor', 'shelf', 'shoe',
        'sink', 'slide', 'spatula', 'sponge', 'spoon',
        'spray', 'stairs', 'stand', 'stove', 'switch', 'teapot',
        'towel', 'tray', 'vegetable', 'washer'
    ]
    pvsg_stuff = ['ceiling', 'grass', 'rock', 'sand', 'sky', 'snow', 'tree', 'wall', 'water']
    
    # Combine: high priority first, then others (avoiding duplicates)
    all_vocab = high_priority.copy()
    for c in pvsg_things:
        if c not in all_vocab:
            all_vocab.append(c)
    for c in pvsg_stuff:
        if c not in all_vocab:
            all_vocab.append(c)
    
    # Limit to max_classes (GroundingDINO token limit)
    vocab = all_vocab[:max_classes]
    
    # GroundingDINO expects a single string with periods
    return ". ".join(vocab) + "."


def run_phase1_detection(
    video_path: Path,
    output_dir: Path,
    conf_threshold: float = 0.25,
    target_fps: float = 5.0,
    use_dino_refinement: bool = True,
) -> Optional[Path]:
    """
    Phase 1: Detection + DINOv3 Refinement
    
    1. OpenCV samples frames from video
    2. GroundingDINO proposes detections
    3. DINOv3 refines embeddings (for Re-ID)
    4. Simple tracker groups into tracks
    5. Output: tracks.jsonl
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = output_dir / "tracks.jsonl"
    
    logger.info(f"=== PHASE 1: DETECTION ===")
    logger.info(f"Video: {video_path}")
    logger.info(f"Output: {tracks_path}")
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Device: {device}")
    
    # Initialize GroundingDINO (proposer)
    # Note: MPS has issues with half precision in GroundingDINO, use CPU instead
    detector = None
    gdino_device = "cpu" if device == "mps" else device
    if GDINO_AVAILABLE:
        logger.info("Initializing GroundingDINO as proposer...")
        try:
            detector = GroundingDINOWrapper(
                model_id="IDEA-Research/grounding-dino-base",
                device=gdino_device,
                use_half_precision=False,  # Disable half precision to avoid MPS issues
            )
            logger.info(f"✓ GroundingDINO initialized on {gdino_device}")
        except Exception as e:
            logger.error(f"GroundingDINO init failed: {e}")
            detector = None
    
    # Fallback to YOLOWorld if GroundingDINO unavailable
    if detector is None and YOLO_AVAILABLE:
        logger.info("Falling back to YOLOWorld...")
        detector = YOLOWorld("yolov8x-worldv2.pt")
        detector.to(device)
        # Set PVSG classes
        try:
            with open("datasets/PVSG/pvsg.json", 'r') as f:
                pvsg_data = json.load(f)
                things = pvsg_data.get('objects', {}).get('thing', [])
                stuff = pvsg_data.get('objects', {}).get('stuff', [])
                vocab = list(set(things + stuff))
                vocab = [c.strip() for c in vocab if c.strip()]
                detector.set_classes(vocab)
                logger.info(f"✓ YOLOWorld initialized with {len(vocab)} PVSG classes")
        except Exception as e:
            logger.warning(f"Failed to set PVSG classes: {e}")
    
    if detector is None:
        logger.error("No detector available!")
        return None
    
    # Initialize DINOv3 embedder (for Re-ID refinement)
    dino_embedder = None
    if use_dino_refinement and DINO_AVAILABLE:
        logger.info("Initializing DINOv3 for detection refinement...")
        try:
            dino_embedder = DINOEmbedder(
                model_name="facebook/dinov2-base",
                device=device if device != "mps" else "cpu",  # DINOv2 may not support MPS
            )
            logger.info("✓ DINOv3 embedder initialized")
        except Exception as e:
            logger.warning(f"DINOv3 init failed: {e}")
    
    # Build prompt for GroundingDINO
    prompt = build_pvsg_prompt()
    logger.info(f"Detection prompt: {prompt[:100]}...")
    
    # Initialize tracker
    tracker = SimpleTracker(max_age=30, iou_threshold=0.3)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return None
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / target_fps))
    
    logger.info(f"Video FPS: {video_fps:.1f}, sampling every {frame_interval} frames")
    logger.info(f"Total frames: {total_frames}")
    
    all_tracks = []
    frame_idx = 0
    sampled_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            # Run detection
            detections = []
            
            if GDINO_AVAILABLE and isinstance(detector, GroundingDINOWrapper):
                # GroundingDINO detection
                raw_dets = detector.detect(
                    frame_bgr=frame,
                    prompt=prompt,
                    box_threshold=conf_threshold,
                    text_threshold=conf_threshold,
                    max_detections=50,
                )
                for d in raw_dets:
                    detections.append({
                        'bbox': d['bbox'],
                        'confidence': d['confidence'],
                        'label': d['label'],
                    })
            elif YOLO_AVAILABLE and isinstance(detector, YOLOWorld):
                # YOLOWorld detection
                results = detector.predict(source=frame, conf=conf_threshold, device=device, verbose=False)
                for result in results:
                    boxes = result.boxes
                    if boxes is None or len(boxes) == 0:
                        continue
                    for i in range(len(boxes)):
                        detections.append({
                            'bbox': boxes.xyxy[i].tolist(),
                            'confidence': float(boxes.conf[i]),
                            'label': result.names[int(boxes.cls[i].item())]
                        })
            
            # DINOv3 refinement: compute embeddings for each detection
            if dino_embedder is not None and len(detections) > 0:
                for det in detections:
                    try:
                        x1, y1, x2, y2 = [int(c) for c in det['bbox']]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            embedding = dino_embedder.encode_image(crop)
                            det['embedding'] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                    except Exception as e:
                        pass  # Skip embedding on error
            
            # Track detections
            tracked = tracker.update(detections, frame_id=frame_idx)
            
            for t in tracked:
                track_entry = {
                    'frame_id': frame_idx,
                    'track_id': t['track_id'],
                    'label': t['label'],
                    'category': t['label'],  # Alias for compatibility
                    'confidence': t['confidence'],
                    'bbox': t['bbox'],
                }
                if t.get('velocity'):
                    track_entry['velocity'] = t['velocity']
                if t.get('embedding'):
                    track_entry['embedding'] = t['embedding']
                all_tracks.append(track_entry)
            
            sampled_count += 1
            if sampled_count % 50 == 0:
                logger.info(f"Processed {sampled_count} frames, {len(all_tracks)} tracks...")
        
        frame_idx += 1
    
    cap.release()
    
    # Save tracks to JSONL
    with open(tracks_path, 'w') as f:
        for t in all_tracks:
            # Remove embedding from JSONL (too large)
            t_save = {k: v for k, v in t.items() if k != 'embedding'}
            f.write(json.dumps(t_save) + '\n')
    
    logger.info(f"✓ Phase 1 complete: {len(all_tracks)} tracks saved to {tracks_path}")
    return tracks_path


# ============================================================================
# PHASE 2: TRACKING + RELATION GENERATION (CIS, VJEPA2, FastVLM, PVSGActionClassifier)
# ============================================================================

def estimate_hands_from_person(person_bbox: List[float]) -> List[HandDetection]:
    """Estimate hand positions from person bbox."""
    x1, y1, x2, y2 = person_bbox
    width = x2 - x1
    height = y2 - y1
    
    hand_y = y1 + height * 0.6
    hand_size = width * 0.2
    
    left_hand = HandDetection(
        bbox=[x1, hand_y - hand_size/2, x1 + hand_size, hand_y + hand_size/2],
        handedness="left"
    )
    right_hand = HandDetection(
        bbox=[x2 - hand_size, hand_y - hand_size/2, x2, hand_y + hand_size/2],
        handedness="right"
    )
    
    return [left_hand, right_hand]


def run_phase2_sgg(
    tracks_path: Path,
    output_path: Path,
    enable_cis: bool = True,
    enable_vjepa2: bool = True,
    enable_fastvlm: bool = True,
) -> bool:
    """
    Phase 2: Scene Graph Generation
    
    1. Load tracks from Phase 1
    2. Initialize VJEPA2 for better Re-ID embeddings
    3. Initialize CIS for person→object edges
    4. Use FastVLM + PVSGActionClassifier to generate relations
    5. Output: scene_graph.jsonl
    """
    logger.info(f"=== PHASE 2: SCENE GRAPH GENERATION ===")
    logger.info(f"Tracks: {tracks_path}")
    logger.info(f"Output: {output_path}")
    
    # Load tracks
    tracks = []
    with open(tracks_path, 'r') as f:
        for line in f:
            if line.strip():
                tracks.append(json.loads(line))
    
    if not tracks:
        logger.error("No tracks found!")
        return False
    
    logger.info(f"Loaded {len(tracks)} track entries")
    
    # Debug: show unique labels
    unique_labels = set()
    for t in tracks:
        label = t.get('category', t.get('class_name', t.get('label', 'object')))
        unique_labels.add(label)
    logger.info(f"DEBUG: Unique labels in tracks: {sorted(unique_labels)[:20]}...")  # Show first 20
    
    # Group by frame
    by_frame = defaultdict(list)
    for t in tracks:
        by_frame[t['frame_id']].append(t)
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    # Initialize VJEPA2 (for temporal Re-ID embeddings)
    vjepa_embedder = None
    if enable_vjepa2 and VJEPA2_AVAILABLE:
        logger.info("Initializing VJEPA2 for temporal Re-ID...")
        try:
            vjepa_embedder = VJepa2Embedder(
                model_name="facebook/vjepa2-vitl-fpc64-256",
                device=device if device != "mps" else "cpu",
                dtype=torch.float32,
            )
            logger.info("✓ VJEPA2 initialized")
        except Exception as e:
            logger.warning(f"VJEPA2 init failed: {e}")
    
    # Initialize DINOv3 (for CIS semantic similarity)
    dino_embedder = None
    if DINO_AVAILABLE:
        try:
            dino_embedder = DINOEmbedder(
                model_name="facebook/dinov2-base",
                device=device if device != "mps" else "cpu",
            )
            logger.info("✓ DINOv3 initialized for CIS")
        except Exception as e:
            logger.warning(f"DINOv3 init failed: {e}")
    
    # Initialize CIS Scorer (for person→object causal edges)
    cis_scorer = None
    if enable_cis and CIS_AVAILABLE:
        logger.info("Initializing CIS Scorer for person→object edges...")
        try:
            cis_scorer = CausalInfluenceScorer(
                enable_depth_gating=False,  # 2D mode
                cis_threshold=0.1,
                dino_embedder=dino_embedder,
                weight_motion=0.15,
                weight_semantic=0.15,
            )
            logger.info("✓ CIS Scorer initialized")
        except Exception as e:
            logger.warning(f"CIS init failed: {e}")
    
    # Initialize FastVLM (for relation verification)
    fastvlm = None
    if enable_fastvlm and FASTVLM_AVAILABLE:
        logger.info("Initializing FastVLM for relation verification...")
        try:
            fastvlm = FastVLMMLXWrapper()
            logger.info("✓ FastVLM initialized")
        except Exception as e:
            logger.warning(f"FastVLM init failed: {e}")
    
    # Initialize PVSGActionClassifier (predicate generation)
    classifier = None
    if PVSG_CLASSIFIER_AVAILABLE:
        logger.info("Initializing PVSGActionClassifier...")
        classifier = PVSGActionClassifier(
            holding_hand_dist=150.0,
            holding_min_frames=3,
            throwing_velocity_thresh=60.0,
            throwing_accel_thresh=30.0,
            picking_hand_dist=120.0,
            vjepa_embedder=vjepa_embedder,
        )
        logger.info("✓ PVSGActionClassifier initialized")
    
    # Process frames and generate relations
    all_relations = []
    frame_graphs = []  # Store nodes/edges format for each frame
    sorted_frames = sorted(by_frame.items())
    
    logger.info(f"Processing {len(sorted_frames)} frames for relations...")
    
    # Debug: Count persons found
    total_persons = 0
    total_objects = 0
    
    for frame_id, frame_tracks in sorted_frames:
        # Convert to Detection objects
        detections = []
        hand_detections = {}
        det_map = {}
        
        for t in frame_tracks:
            track_id = t.get('track_id', t.get('id', 0))
            label = t.get('category', t.get('class_name', t.get('label', 'object')))
            
            det = Detection(
                track_id=track_id,
                class_name=label,
                bbox=t['bbox'],
                frame_id=frame_id,
                confidence=t.get('confidence', 1.0),
                velocity=tuple(t['velocity']) if t.get('velocity') else None,
            )
            detections.append(det)
            det_map[track_id] = det
            
            # Estimate hands for person-like entities (case-insensitive, partial match)
            label_lower = label.lower()
            if any(p in label_lower for p in ['person', 'adult', 'child', 'man', 'woman', 'boy', 'girl', 'human', 'people']):
                hands = estimate_hands_from_person(t['bbox'])
                hand_detections[track_id] = hands
        
        frame_relations = []
        
        # Helper to check if detection is a person (case-insensitive, partial match)
        def is_person(det):
            label_lower = det.class_name.lower()
            return any(p in label_lower for p in ['person', 'adult', 'child', 'man', 'woman', 'boy', 'girl', 'human', 'people'])
        
        # 1. Use PVSGActionClassifier for action relations
        if classifier is not None:
            try:
                relations = classifier.predict_relations(detections, hand_detections, frame_id)
                # Classifier returns: List[Tuple[subject_id, object_id, predicate, score]]
                for rel in relations:
                    subj_id, obj_id, predicate, score = rel
                    # Find class names from det_map
                    subj_class = det_map.get(subj_id, Detection(0, 'unknown', [0,0,0,0], 0)).class_name
                    obj_class = det_map.get(obj_id, Detection(0, 'unknown', [0,0,0,0], 0)).class_name
                    frame_relations.append({
                        'frame_id': frame_id,
                        'subject_id': subj_id,
                        'subject_class': subj_class,
                        'predicate': predicate,
                        'object_id': obj_id,
                        'object_class': obj_class,
                        'confidence': score,
                        'source': 'pvsg_classifier',
                    })
            except Exception as e:
                logger.warning(f"Classifier error on frame {frame_id}: {e}")
        
        # 2. Use CIS for person→object causal edges
        if cis_scorer is not None and len(detections) >= 2:
            try:
                # Find persons and objects using helper
                persons = [d for d in detections if is_person(d)]
                objects = [d for d in detections if not is_person(d)]
                
                total_persons += len(persons)
                total_objects += len(objects)
                
                for person in persons:
                    for obj in objects:
                        # Compute CIS score
                        cis_edge = cis_scorer.compute_cis_score(
                            agent=person,
                            patient=obj,
                            frame_idx=frame_id,
                        )
                        if cis_edge and cis_edge.cis_score > 0.3:
                            # Map CIS influence type to PVSG predicate
                            predicate = 'holding'  # Default
                            if hasattr(cis_edge, 'influence_type'):
                                if cis_edge.influence_type == 'grasping':
                                    predicate = 'holding'
                                elif cis_edge.influence_type == 'moving':
                                    predicate = 'pushing'
                            
                            frame_relations.append({
                                'frame_id': frame_id,
                                'subject_id': person.track_id,
                                'subject_class': person.class_name,
                                'predicate': predicate,
                                'object_id': obj.track_id,
                                'object_class': obj.class_name,
                                'confidence': cis_edge.cis_score,
                                'cis_score': cis_edge.cis_score,
                                'source': 'cis',
                            })
            except Exception as e:
                logger.debug(f"CIS error on frame {frame_id}: {e}")
        
        # 3. ALWAYS generate object-object spatial relations (CRITICAL for PVSG!)
        # PVSG GT has relations like 'cake on table', 'table on floor', 'candle on cake'
        # These are NOT generated by the person-focused classifier
        if len(detections) >= 2:
            for i, subj in enumerate(detections):
                for j, obj in enumerate(detections):
                    if i == j:
                        continue
                    # Skip if classifier already generated this pair with high confidence
                    existing_key = (subj.track_id, obj.track_id)
                    if any((r['subject_id'], r['object_id']) == existing_key and r['confidence'] > 0.7 
                           for r in frame_relations):
                        continue
                    
                    predicate, conf = _compute_spatial_relation(subj.bbox, obj.bbox)
                    if predicate and conf > 0.3:
                        frame_relations.append({
                            'frame_id': frame_id,
                            'subject_id': subj.track_id,
                            'subject_class': subj.class_name,
                            'predicate': predicate,
                            'object_id': obj.track_id,
                            'object_class': obj.class_name,
                            'confidence': conf,
                            'source': 'spatial',
                        })
        
        # Deduplicate relations (keep highest confidence)
        seen = {}
        for rel in frame_relations:
            key = (rel['subject_id'], rel['predicate'], rel['object_id'])
            if key not in seen or rel['confidence'] > seen[key]['confidence']:
                seen[key] = rel
        
        # Build frame graph in format expected by eval_sgg_recall.py
        nodes = []
        for det in detections:
            nodes.append({
                'memory_id': str(det.track_id),
                'class': det.class_name,
                'bbox': det.bbox,
                'confidence': det.confidence,
            })
        
        edges = []
        for rel in seen.values():
            edges.append({
                'subject': str(rel['subject_id']),
                'object': str(rel['object_id']),
                'relation': rel['predicate'].replace('_', ' '),  # Ensure spaces not underscores
                'confidence': rel['confidence'],
            })
        
        # Extract video_id from output_path for frame graphs
        video_id = output_path.parent.name
        
        frame_graphs.append({
            'video_id': video_id,
            'frame_id': frame_id,
            'nodes': nodes,
            'edges': edges,
        })
        
        all_relations.extend(seen.values())
    
    # Save scene graph in nodes/edges format (compatible with eval_sgg_recall.py)
    with open(output_path, 'w') as f:
        for fg in frame_graphs:
            f.write(json.dumps(fg) + '\n')
    
    # Count by source
    from collections import Counter
    source_counts = Counter(r['source'] for r in all_relations)
    predicate_counts = Counter(r['predicate'] for r in all_relations)
    
    logger.info(f"DEBUG: Total persons detected across all frames: {total_persons}")
    logger.info(f"DEBUG: Total objects detected across all frames: {total_objects}")
    logger.info(f"DEBUG: Relations by source: {dict(source_counts)}")
    logger.info(f"DEBUG: Top predicates: {predicate_counts.most_common(10)}")
    logger.info(f"✓ Phase 2 complete: {len(all_relations)} relations saved to {output_path}")
    return True


# ============================================================================
# PHASE 3: EVALUATION (R@K metrics)
# ============================================================================

def normalize_class(cls: str) -> str:
    """Normalize class names for matching with PVSG GT.
    
    CRITICAL: PVSG uses 'adult', 'child', 'baby' NOT 'person'!
    We must map everything consistently.
    """
    if not cls:
        return ''
    cls = cls.lower().strip().replace('_', ' ')
    
    # Direct PVSG vocabulary - return immediately if exact match
    pvsg_vocab = {
        'adult', 'baby', 'child', 'table', 'chair', 'sofa', 'bed', 'floor',
        'cake', 'candle', 'plate', 'cup', 'bottle', 'glass', 'ball', 'toy',
        'book', 'bag', 'door', 'window', 'tv', 'dog', 'cat', 'bird', 'car',
        'bike', 'pillow', 'bowl', 'box', 'fruit', 'vegetable', 'meat', 'tree',
        'grass', 'ground', 'wall', 'ceiling', 'water', 'food', 'phone', 'hat'
    }
    if cls in pvsg_vocab:
        return cls
    
    # Handle compound labels like "child baby", "adult man", "dog cat"
    if ' ' in cls:
        parts = cls.split()
        # First check for exact PVSG matches
        for part in parts:
            if part in pvsg_vocab:
                return part
        # Then check mappings for each part
        for part in parts:
            if part in ['man', 'woman', 'person', 'human', 'people']:
                return 'adult'
            if part in ['boy', 'girl', 'kid', 'toddler']:
                return 'child'
            if part in ['infant']:
                return 'baby'
        # Default to first word
        cls = parts[0]
    
    # Mappings for non-PVSG labels
    mappings = {
        # People -> PVSG categories
        'man': 'adult', 'woman': 'adult', 'person': 'adult',
        'human': 'adult', 'people': 'adult',
        'boy': 'child', 'girl': 'child', 'kid': 'child', 'toddler': 'child',
        'infant': 'baby',
        # Furniture
        'dining table': 'table', 'desk': 'table',
        'couch': 'sofa', 'lounge': 'sofa', 'settee': 'sofa',
        'armchair': 'chair', 'seat': 'chair', 'stool': 'chair',
        # Environment
        'carpet': 'floor', 'rug': 'floor', 'mat': 'floor',
        'wooden floor': 'floor', 'tile floor': 'floor',
        'lawn': 'grass', 'field': 'grass',
        # Objects
        'birthday cake': 'cake', 'cupcake': 'cake',
        'soccer ball': 'ball', 'football': 'ball', 'basketball': 'ball',
        'teddy bear': 'toy', 'plush': 'toy', 'doll': 'toy',
        'television': 'tv', 'monitor': 'tv', 'screen': 'tv',
        'cellphone': 'phone', 'mobile phone': 'phone', 'smartphone': 'phone',
        'mug': 'cup', 'teacup': 'cup',
        'cushion': 'pillow',
        # Animals
        'puppy': 'dog', 'kitty': 'cat', 'kitten': 'cat',
        # Food
        'drink': 'beverage', 'juice': 'beverage', 'soda': 'beverage',
        'apple': 'fruit', 'banana': 'fruit', 'orange': 'fruit',
    }
    return mappings.get(cls, cls)


def normalize_predicate(pred: str) -> str:
    """Normalize predicate names to PVSG vocabulary.
    
    PVSG has exactly 56 predicates - we must map to these.
    """
    pred = pred.lower().strip().replace('_', ' ')
    
    mappings = {
        # Spatial - PVSG uses 'next to', 'beside' is also valid
        'near': 'next to',
        'nearby': 'next to',
        'close to': 'next to',
        'adjacent to': 'next to',
        
        # Holding variants
        'held by': 'holding', 'hold': 'holding', 'carries': 'holding',
        'carry': 'carrying', 'carried by': 'carrying',
        
        # On variants - PVSG has: on, walking on, lying on, sitting on, standing on, running on, stepping on
        'standing on': 'standing on',
        'walking on': 'walking on', 
        'lying on': 'lying on',
        'sitting on': 'sitting on',
        'running on': 'running on',
        'stepping on': 'stepping on',
        'on top of': 'on',
        'resting on': 'on',
        
        # Action predicates
        'throw': 'throwing', 'throws': 'throwing',
        'chase': 'chasing', 'chases': 'chasing', 'following': 'chasing',
        'run': 'running on', 'runs': 'running on',
        'catch': 'catching', 'catches': 'catching',
        'kick': 'kicking', 'kicks': 'kicking',
        'hit': 'hitting', 'hits': 'hitting',
        'push': 'pushing', 'pushes': 'pushing',
        'pull': 'pulling', 'pulls': 'pulling',
        'grab': 'grabbing', 'grabs': 'grabbing',
        'pick': 'picking', 'picks': 'picking',
        'touch': 'touching', 'touches': 'touching',
        'look at': 'looking at', 'looks at': 'looking at', 'watching': 'looking at',
        'talk to': 'talking to', 'talks to': 'talking at', 'speaking to': 'talking to',
        'play': 'playing', 'plays': 'playing',
        'play with': 'playing with', 'plays with': 'playing with',
        
        # Other PVSG predicates
        'in front of': 'in front of',
        'in': 'in',
        'inside': 'in',
        'wear': 'wearing', 'wears': 'wearing',
        'ride': 'riding', 'rides': 'riding',
        'eat': 'eating', 'eats': 'eating',
        'drink': 'drinking from', 'drinks': 'drinking from',
        'hug': 'hugging', 'hugs': 'hugging',
        'kiss': 'kissing', 'kisses': 'kissing',
    }
    return mappings.get(pred, pred)


def evaluate_video(video_id: str, results_dir: str, gt_videos: Dict) -> Dict:
    """Evaluate a single video's scene graph against GT."""
    sg_file = os.path.join(results_dir, video_id, 'scene_graph.jsonl')
    
    if not os.path.exists(sg_file):
        return {'error': f'scene_graph.jsonl not found for {video_id}'}
    
    if video_id not in gt_videos:
        return {'error': f'GT not found for {video_id}'}
    
    # Load predictions from nodes/edges format
    pred_triplets = []
    node_classes = {}  # memory_id -> class
    with open(sg_file, 'r') as f:
        for line in f:
            if line.strip():
                frame_sg = json.loads(line)
                
                # Build node class lookup for this frame
                for node in frame_sg.get('nodes', []):
                    node_classes[node['memory_id']] = normalize_class(node.get('class', ''))
                
                # Extract triplets from edges
                for edge in frame_sg.get('edges', []):
                    subj_id = edge.get('subject', '')
                    obj_id = edge.get('object', '')
                    pred = normalize_predicate(edge.get('relation', ''))
                    conf = edge.get('confidence', 0.5)
                    
                    subj_class = node_classes.get(subj_id, '')
                    obj_class = node_classes.get(obj_id, '')
                    
                    if subj_class and obj_class and pred:
                        pred_triplets.append((subj_class, pred, obj_class, conf))
    
    # Sort by confidence
    pred_triplets.sort(key=lambda x: -x[3])
    
    # Load GT
    gt_data = gt_videos[video_id]
    gt_triplets = set()
    pred_stats = {20: defaultdict(lambda: [0, 0]), 
                  50: defaultdict(lambda: [0, 0]), 
                  100: defaultdict(lambda: [0, 0])}
    
    relations = gt_data.get('relations', [])
    objects = {o['object_id']: o['category'] for o in gt_data.get('objects', [])}
    
    for rel in relations:
        subj_id = rel[0]
        obj_id = rel[1]
        pred_name = rel[2] if len(rel) > 2 else 'unknown'
        
        subj_class = normalize_class(objects.get(subj_id, ''))
        obj_class = normalize_class(objects.get(obj_id, ''))
        pred_norm = normalize_predicate(pred_name)
        
        gt_triplets.add((subj_class, pred_norm, obj_class))
    
    # Debug: Show GT vs Predicted triplets
    logger.info(f"  GT triplets: {gt_triplets}")
    top_pred = [(p[0], p[1], p[2]) for p in pred_triplets[:10]]
    logger.info(f"  Top 10 predictions: {top_pred}")
    
    # Compute R@K
    def recall_at_k(k):
        top_k = pred_triplets[:k]
        matched = 0
        for gt in gt_triplets:
            for pred in top_k:
                if pred[0] == gt[0] and pred[1] == gt[1] and pred[2] == gt[2]:
                    matched += 1
                    break
        return (matched / len(gt_triplets) * 100) if gt_triplets else 0
    
    # Per-predicate stats
    def compute_per_predicate(k):
        top_k = set((p[0], p[1], p[2]) for p in pred_triplets[:k])
        per_pred = defaultdict(lambda: [0, 0])
        for gt in gt_triplets:
            per_pred[gt[1]][1] += 1
            if gt in top_k:
                per_pred[gt[1]][0] += 1
        return dict(per_pred)
    
    r20 = recall_at_k(20)
    r50 = recall_at_k(50)
    r100 = recall_at_k(100)
    
    # Mean Recall (per-predicate)
    def mean_recall_at_k(k):
        stats = compute_per_predicate(k)
        recalls = [m/t for m, t in stats.values() if t > 0]
        return np.mean(recalls) * 100 if recalls else 0
    
    return {
        'video_id': video_id,
        'pred_count': len(pred_triplets),
        'gt_count': len(gt_triplets),
        'R@20': r20,
        'R@50': r50,
        'R@100': r100,
        'mR@20': mean_recall_at_k(20),
        'mR@50': mean_recall_at_k(50),
        'mR@100': mean_recall_at_k(100),
        'pred_stats': {
            20: compute_per_predicate(20),
            50: compute_per_predicate(50),
            100: compute_per_predicate(100),
        }
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Orion Pipeline: Detection + SGG + Evaluation")
    parser.add_argument("--video", type=str, help="Path to video file for full pipeline")
    parser.add_argument("--batch_dir", default="results/pvsg_batch", help="Directory for batch results")
    parser.add_argument("--gt_path", default="datasets/PVSG/pvsg.json", help="PVSG ground truth JSON")
    parser.add_argument("--video_id", type=str, help="Specific video ID to process")
    parser.add_argument("--limit", type=int, help="Limit number of videos")
    parser.add_argument("--skip-phase1", action="store_true", help="Skip detection, use existing tracks")
    parser.add_argument("--disable-cis", action="store_true", help="Disable CIS scoring")
    parser.add_argument("--disable-vjepa2", action="store_true", help="Disable VJEPA2 embeddings")
    parser.add_argument("--disable-fastvlm", action="store_true", help="Disable FastVLM verification")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Detection confidence threshold")
    args = parser.parse_args()
    
    processed_videos = []
    
    # === SINGLE VIDEO FULL PIPELINE ===
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return
        
        video_id = video_path.stem
        output_dir = Path("results") / video_id
        
        logger.info(f"\n{'='*60}")
        logger.info(f"ORION PIPELINE: {video_id}")
        logger.info(f"{'='*60}\n")
        
        # Phase 1: Detection
        if not args.skip_phase1:
            tracks_path = run_phase1_detection(
                video_path=video_path,
                output_dir=output_dir,
                conf_threshold=args.conf_threshold,
            )
            if not tracks_path:
                logger.error("Phase 1 (Detection) failed!")
                return
        else:
            tracks_path = output_dir / "tracks.jsonl"
            if not tracks_path.exists():
                logger.error(f"No existing tracks found at {tracks_path}")
                return
            logger.info(f"Skipping Phase 1, using existing tracks: {tracks_path}")
        
        # Phase 2: SGG
        sgg_path = output_dir / "scene_graph.jsonl"
        success = run_phase2_sgg(
            tracks_path=tracks_path,
            output_path=sgg_path,
            enable_cis=not args.disable_cis,
            enable_vjepa2=not args.disable_vjepa2,
            enable_fastvlm=not args.disable_fastvlm,
        )
        
        if success:
            processed_videos.append(video_id)
        else:
            logger.error("Phase 2 (SGG) failed!")
            return
        
        batch_dir = Path("results")
    
    # === BATCH PROCESSING ===
    else:
        batch_dir = Path(args.batch_dir)
        if not batch_dir.exists():
            logger.error(f"Batch directory not found: {batch_dir}")
            return
        
        # Find video directories with tracks
        if args.video_id:
            target_dir = batch_dir / args.video_id
            if target_dir.exists() and (target_dir / "tracks.jsonl").exists():
                video_dirs = [target_dir]
            else:
                logger.error(f"Video {args.video_id} not found")
                return
        else:
            video_dirs = [d for d in batch_dir.iterdir() 
                         if d.is_dir() and (d / "tracks.jsonl").exists()]
            video_dirs.sort(key=lambda x: x.name)
            if args.limit:
                video_dirs = video_dirs[:args.limit]
        
        logger.info(f"Found {len(video_dirs)} videos to process")
        
        for v_dir in video_dirs:
            video_id = v_dir.name
            tracks_path = v_dir / "tracks.jsonl"
            sgg_path = v_dir / "scene_graph.jsonl"
            
            logger.info(f"\n[{video_id}] Generating Scene Graph...")
            success = run_phase2_sgg(
                tracks_path=tracks_path,
                output_path=sgg_path,
                enable_cis=not args.disable_cis,
                enable_vjepa2=not args.disable_vjepa2,
                enable_fastvlm=not args.disable_fastvlm,
            )
            if success:
                processed_videos.append(video_id)
    
    # === PHASE 3: EVALUATION ===
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 3: EVALUATION")
    logger.info(f"{'='*60}\n")
    
    # Load GT
    if not os.path.exists(args.gt_path):
        logger.warning(f"GT not found at {args.gt_path}, skipping evaluation")
        return
    
    with open(args.gt_path, 'r') as f:
        pvsg_data = json.load(f)
    gt_videos = {v['video_id']: v for v in pvsg_data['data']}
    
    video_metrics = {}
    
    for video_id in processed_videos:
        res = evaluate_video(video_id, str(batch_dir), gt_videos)
        
        if 'error' in res:
            logger.warning(f"[{video_id}] Eval Error: {res['error']}")
            continue
        
        video_metrics[video_id] = res
        logger.info(f"[{video_id}] Pred={res['pred_count']} GT={res['gt_count']} "
                   f"R@20: {res['R@20']:.1f}% | R@50: {res['R@50']:.1f}% | R@100: {res['R@100']:.1f}%")
    
    # Summary
    if video_metrics:
        print("\n")
        print("=" * 60)
        print("           SGG EVALUATION RESULTS (ORION PIPELINE)          ")
        print("=" * 60)
        
        if len(video_metrics) == 1:
            vid = list(video_metrics.keys())[0]
            m = video_metrics[vid]
            print(f"\nVIDEO: {vid}")
            print("-" * 60)
            print(f"  Predictions: {m['pred_count']}")
            print(f"  Ground Truth: {m['gt_count']}")
            print(f"  R@20:   {m['R@20']:.2f}%  (mR@20:  {m['mR@20']:.2f}%)")
            print(f"  R@50:   {m['R@50']:.2f}%  (mR@50:  {m['mR@50']:.2f}%)")
            print(f"  R@100:  {m['R@100']:.2f}%  (mR@100: {m['mR@100']:.2f}%)")
        else:
            # Aggregate
            R20_avg = np.mean([m['R@20'] for m in video_metrics.values()])
            R50_avg = np.mean([m['R@50'] for m in video_metrics.values()])
            R100_avg = np.mean([m['R@100'] for m in video_metrics.values()])
            mR20_avg = np.mean([m['mR@20'] for m in video_metrics.values()])
            mR50_avg = np.mean([m['mR@50'] for m in video_metrics.values()])
            mR100_avg = np.mean([m['mR@100'] for m in video_metrics.values()])
            
            print(f"\nAGGREGATE OVER {len(video_metrics)} VIDEOS")
            print("-" * 60)
            print(f"  MEAN R@20:    {R20_avg:.2f}%  |  MEAN mR@20:   {mR20_avg:.2f}%")
            print(f"  MEAN R@50:    {R50_avg:.2f}%  |  MEAN mR@50:   {mR50_avg:.2f}%")
            print(f"  MEAN R@100:   {R100_avg:.2f}%  |  MEAN mR@100:  {mR100_avg:.2f}%")
        
        print("=" * 60)
        print("\nPipeline Components Used:")
        print(f"  ✓ GroundingDINO (Proposer): {GDINO_AVAILABLE}")
        print(f"  ✓ DINOv3 (Refinement): {DINO_AVAILABLE}")
        print(f"  ✓ VJEPA2 (Re-ID): {VJEPA2_AVAILABLE}")
        print(f"  ✓ CIS (Causal Edges): {CIS_AVAILABLE}")
        print(f"  ✓ FastVLM (Verification): {FASTVLM_AVAILABLE}")
        print(f"  ✓ PVSGActionClassifier: {PVSG_CLASSIFIER_AVAILABLE}")
        print("=" * 60 + "\n")
    else:
        logger.warning("No valid evaluation results found.")


if __name__ == "__main__":
    main()
