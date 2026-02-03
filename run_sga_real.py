#!/usr/bin/env python3
"""
Real Scene Graph Anticipation (SGA) Pipeline

This script runs ACTUAL video processing:
1. Load Action Genome videos
2. Run GroundingDINO/YOLO detection on observed frames
3. Track objects with embeddings
4. Generate scene graphs from observed portion
5. Anticipate future relations
6. Evaluate against GT future relations

This is SLOW because it processes real video frames with neural networks.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import time

import numpy as np
import cv2
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# IMPORTS - Try to load Orion components
# ============================================================================

# Detection
try:
    from orion.perception.detectors.grounding_dino import GroundingDINOWrapper
    GDINO_AVAILABLE = True
    logger.info("✓ GroundingDINO available")
except ImportError:
    GDINO_AVAILABLE = False
    GroundingDINOWrapper = None

try:
    from ultralytics import YOLO, YOLOWorld
    YOLO_AVAILABLE = True
    logger.info("✓ YOLO available")
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None
    YOLOWorld = None

# Embeddings
try:
    from orion.backends.dino_backend import DINOEmbedder
    DINO_AVAILABLE = True
except ImportError:
    DINO_AVAILABLE = False
    DINOEmbedder = None

# ============================================================================
# ACTION GENOME VOCABULARY
# ============================================================================

AG_PREDICATES = [
    "looking at", "not looking at", "unsure",
    "above", "beneath", "in front of", "behind",
    "on the side of", "in", "carrying", "covered by",
    "drinking from", "eating", "have it on the back",
    "holding", "leaning on", "lying on", "not contacting",
    "other relationship", "sitting on", "standing on",
    "touching", "twisting", "wearing", "wiping", "writing on"
]

AG_OBJECTS = [
    "person", "bag", "bed", "blanket", "book", "box", "broom",
    "chair", "closet/cabinet", "clothes", "cup/glass/bottle",
    "dish", "door", "doorknob", "doorway", "floor", "food",
    "groceries", "laptop", "light", "medicine", "mirror",
    "paper/notebook", "phone/camera", "picture", "pillow/cushion",
    "refrigerator", "sandwich", "shelf", "shoe", "sofa/couch",
    "table", "television", "towel", "vacuum", "window"
]


# ============================================================================
# SIMPLE TRACKER
# ============================================================================

class SimpleTracker:
    """IoU-based tracker."""
    def __init__(self, max_age=30, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 1
        self.track_ages = {}
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        results = []
        matched = set()
        
        for det in detections:
            best_iou = 0
            best_track_id = None
            det_box = det['bbox']
            det_label = det.get('label', 'object')
            
            for track_id, track in self.tracks.items():
                if track.get('label') != det_label:
                    continue
                iou = self._calc_iou(det_box, track['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                self.tracks[best_track_id]['bbox'] = det_box
                self.track_ages[best_track_id] = 0
                matched.add(best_track_id)
                results.append({**det, 'track_id': best_track_id})
            else:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {'bbox': det_box, 'label': det_label}
                self.track_ages[track_id] = 0
                matched.add(track_id)
                results.append({**det, 'track_id': track_id})
        
        # Age out old tracks
        to_remove = []
        for track_id in self.tracks:
            if track_id not in matched:
                self.track_ages[track_id] = self.track_ages.get(track_id, 0) + 1
                if self.track_ages[track_id] > self.max_age:
                    to_remove.append(track_id)
        for track_id in to_remove:
            del self.tracks[track_id]
            del self.track_ages[track_id]
        
        return results
    
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


# ============================================================================
# SPATIAL RELATION COMPUTATION
# ============================================================================

def compute_spatial_relations(detections: List[Dict]) -> List[Tuple[int, str, int, float]]:
    """Compute spatial relations between detected objects."""
    relations = []
    
    for i, subj in enumerate(detections):
        for j, obj in enumerate(detections):
            if i == j:
                continue
            
            subj_box = subj['bbox']
            obj_box = obj['bbox']
            
            # Centers
            cx1 = (subj_box[0] + subj_box[2]) / 2
            cy1 = (subj_box[1] + subj_box[3]) / 2
            cx2 = (obj_box[0] + obj_box[2]) / 2
            cy2 = (obj_box[1] + obj_box[3]) / 2
            
            # Sizes
            w1 = subj_box[2] - subj_box[0]
            h1 = subj_box[3] - subj_box[1]
            w2 = obj_box[2] - obj_box[0]
            h2 = obj_box[3] - obj_box[1]
            area1 = max(w1 * h1, 1)
            area2 = max(w2 * h2, 1)
            
            # Intersection
            inter_x1 = max(subj_box[0], obj_box[0])
            inter_y1 = max(subj_box[1], obj_box[1])
            inter_x2 = min(subj_box[2], obj_box[2])
            inter_y2 = min(subj_box[3], obj_box[3])
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            
            # Distance
            dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            avg_size = (max(w1, h1) + max(w2, h2)) / 2
            norm_dist = dist / (avg_size + 1e-6)
            
            subj_id = subj['track_id']
            obj_id = obj['track_id']
            
            # Containment
            containment = inter_area / (area1 + 1e-6)
            
            # Determine relation
            if containment > 0.5 and area1 < area2:
                relations.append((subj_id, 'in', obj_id, min(0.9, containment)))
            elif cy1 < cy2 - h2 * 0.3:  # Subject above object
                relations.append((subj_id, 'above', obj_id, 0.7))
            elif cy1 > cy2 + h2 * 0.3:  # Subject below object
                relations.append((subj_id, 'beneath', obj_id, 0.7))
            elif inter_area > 0 and cy1 > cy2:  # On top with overlap
                relations.append((subj_id, 'on', obj_id, 0.8))
            elif norm_dist < 1.5:
                # Touching/near
                if subj.get('label', '').lower() == 'person':
                    relations.append((subj_id, 'touching', obj_id, 0.6))
                else:
                    relations.append((subj_id, 'on the side of', obj_id, 0.5))
    
    return relations


# ============================================================================
# PROCESS SINGLE VIDEO
# ============================================================================

def process_video_sga(
    video_path: str,
    observe_fraction: float,
    detector,
    device: str,
    target_fps: float = 5.0,
    conf_threshold: float = 0.3,
) -> Tuple[Set[Tuple], Set[Tuple], Dict]:
    """
    Process a video for SGA:
    1. Split into observed and future portions
    2. Run detection + tracking on observed frames
    3. Build scene graph from observations
    4. Return observed triplets (for anticipation) and GT structure for eval
    
    Returns:
        observed_triplets: Set of (subject_class, predicate, object_class) from observed frames
        all_detections: Dict with frame_id -> list of detections
        stats: Dict with processing statistics
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return set(), {}, {'error': 'Cannot open video'}
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0 or video_fps <= 0:
        cap.release()
        return set(), {}, {'error': 'Invalid video metadata'}
    
    # Compute split point
    observe_frames = int(total_frames * observe_fraction)
    frame_interval = max(1, int(video_fps / target_fps))
    
    tracker = SimpleTracker()
    observed_triplets = set()
    all_detections = {}
    
    frame_idx = 0
    processed_frames = 0
    
    # Build prompt for detection
    prompt = ". ".join(AG_OBJECTS[:30]) + "."  # Limit for token budget
    
    while frame_idx < observe_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            # Run detection
            detections = []
            
            if GDINO_AVAILABLE and hasattr(detector, 'detect'):
                try:
                    raw_dets = detector.detect(
                        frame_bgr=frame,
                        prompt=prompt,
                        box_threshold=conf_threshold,
                        text_threshold=conf_threshold,
                        max_detections=30,
                    )
                    for d in raw_dets:
                        detections.append({
                            'bbox': d['bbox'],
                            'confidence': d['confidence'],
                            'label': d['label'],
                        })
                except Exception as e:
                    pass
            elif YOLO_AVAILABLE and hasattr(detector, 'predict'):
                try:
                    results = detector.predict(source=frame, conf=conf_threshold, verbose=False)
                    for result in results:
                        if result.boxes is not None:
                            for i in range(len(result.boxes)):
                                detections.append({
                                    'bbox': result.boxes.xyxy[i].tolist(),
                                    'confidence': float(result.boxes.conf[i]),
                                    'label': result.names[int(result.boxes.cls[i].item())]
                                })
                except Exception as e:
                    pass
            
            # Track
            if detections:
                tracked = tracker.update(detections)
                all_detections[frame_idx] = tracked
                
                # Compute relations
                relations = compute_spatial_relations(tracked)
                
                # Build triplets
                det_map = {d['track_id']: d for d in tracked}
                for subj_id, pred, obj_id, conf in relations:
                    subj_class = det_map.get(subj_id, {}).get('label', 'object')
                    obj_class = det_map.get(obj_id, {}).get('label', 'object')
                    observed_triplets.add((subj_class.lower(), pred, obj_class.lower()))
            
            processed_frames += 1
        
        frame_idx += 1
    
    cap.release()
    
    stats = {
        'total_frames': total_frames,
        'observe_frames': observe_frames,
        'processed_frames': processed_frames,
        'unique_triplets': len(observed_triplets),
    }
    
    return observed_triplets, all_detections, stats


# ============================================================================
# LOAD ACTION GENOME GT
# ============================================================================

def load_ag_gt(gt_path: str) -> Dict:
    """Load Action Genome ground truth."""
    with open(gt_path, 'r') as f:
        return json.load(f)


def get_video_gt_triplets(video_data: Dict, start_frame: int, end_frame: int) -> Set[Tuple]:
    """Extract GT triplets for a frame range."""
    triplets = set()
    
    for rel in video_data.get('relations', []):
        frame_id = rel.get('frame_id', 0)
        if start_frame <= frame_id < end_frame:
            subj = rel.get('subject', '').lower()
            pred = rel.get('predicate', '').lower()
            obj = rel.get('object', '').lower()
            if subj and pred and obj:
                triplets.add((subj, pred, obj))
    
    return triplets


# ============================================================================
# ANTICIPATION MODEL
# ============================================================================

def anticipate_future(observed_triplets: Set[Tuple], top_k: int = 50) -> List[Tuple]:
    """
    Anticipation model: Predict future relations based on observed ones.
    
    Simple persistence baseline: Relations seen in observed portion
    are likely to persist into the future.
    
    Returns list of (subject, predicate, object, confidence) tuples.
    """
    predictions = []
    
    # Persistence: observed relations continue
    for triplet in observed_triplets:
        predictions.append((*triplet, 0.8))
    
    # Sort by confidence and return top-k
    predictions.sort(key=lambda x: -x[3])
    return predictions[:top_k]


# ============================================================================
# EVALUATION
# ============================================================================

def compute_recall_at_k(predictions: List[Tuple], gt_triplets: Set[Tuple], k: int) -> float:
    """Compute Recall@K."""
    if not gt_triplets:
        return 0.0
    
    top_k = predictions[:k]
    top_k_set = set((p[0], p[1], p[2]) for p in top_k)
    
    matched = len(gt_triplets & top_k_set)
    return matched / len(gt_triplets) * 100


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Real SGA Pipeline on Action Genome")
    parser.add_argument("--ag_gt", default="data/ag_ground_truth_full.json", help="AG ground truth")
    parser.add_argument("--ag_videos", default="datasets/ActionGenome/videos", help="AG videos directory")
    parser.add_argument("--max_videos", type=int, default=10, help="Max videos to process")
    parser.add_argument("--observe_fraction", type=float, default=0.5, help="Fraction to observe (0.3, 0.5, 0.7)")
    parser.add_argument("--output", default="sga_real_results.json", help="Output results file")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("REAL SGA PIPELINE - Action Genome")
    logger.info("=" * 60)
    logger.info(f"Observe fraction: {args.observe_fraction}")
    logger.info(f"Max videos: {args.max_videos}")
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Device: {device}")
    
    # Initialize detector
    detector = None
    gdino_device = "cpu" if device == "mps" else device  # MPS has issues with GDINO
    
    if GDINO_AVAILABLE:
        logger.info("Initializing GroundingDINO...")
        try:
            detector = GroundingDINOWrapper(
                model_id="IDEA-Research/grounding-dino-base",
                device=gdino_device,
                use_half_precision=False,
            )
            logger.info(f"✓ GroundingDINO initialized on {gdino_device}")
        except Exception as e:
            logger.warning(f"GroundingDINO failed: {e}")
    
    if detector is None and YOLO_AVAILABLE:
        logger.info("Falling back to YOLO...")
        try:
            detector = YOLO("yolo11m.pt")
            detector.to(device)
            logger.info(f"✓ YOLO initialized on {device}")
        except Exception as e:
            logger.error(f"YOLO failed: {e}")
            return
    
    if detector is None:
        logger.error("No detector available!")
        return
    
    # Load GT
    logger.info(f"Loading GT from {args.ag_gt}...")
    ag_gt = load_ag_gt(args.ag_gt)
    logger.info(f"Loaded {len(ag_gt)} videos from GT")
    
    # Find video files
    videos_dir = Path(args.ag_videos)
    if not videos_dir.exists():
        logger.error(f"Videos directory not found: {videos_dir}")
        logger.info("Looking for videos in alternate locations...")
        
        # Try alternate paths
        alt_paths = [
            Path("data/ActionGenome/videos"),
            Path("datasets/ag/videos"),
            Path("data/ag/videos"),
        ]
        for alt in alt_paths:
            if alt.exists():
                videos_dir = alt
                logger.info(f"Found videos at: {videos_dir}")
                break
    
    # Match GT videos to actual files
    video_files = {}
    if videos_dir.exists():
        for ext in ['*.mp4', '*.avi', '*.mkv']:
            for f in videos_dir.rglob(ext):
                video_files[f.stem] = str(f)
    
    logger.info(f"Found {len(video_files)} video files")
    
    # Process videos
    results = []
    video_ids = list(ag_gt.keys())[:args.max_videos]
    
    total_start = time.time()
    
    for idx, video_id in enumerate(video_ids):
        video_path = video_files.get(video_id)
        
        if video_path is None:
            logger.warning(f"[{idx+1}/{len(video_ids)}] {video_id}: Video file not found, skipping")
            continue
        
        logger.info(f"[{idx+1}/{len(video_ids)}] Processing {video_id}...")
        start_time = time.time()
        
        # Process video
        observed_triplets, all_dets, stats = process_video_sga(
            video_path=video_path,
            observe_fraction=args.observe_fraction,
            detector=detector,
            device=device,
        )
        
        if 'error' in stats:
            logger.warning(f"  Error: {stats['error']}")
            continue
        
        # Get GT for future frames
        video_data = ag_gt[video_id]
        total_frames = stats['total_frames']
        observe_frames = stats['observe_frames']
        
        # Build GT triplets from future frames
        gt_future_triplets = set()
        for rel in video_data.get('relations', []):
            frame_id = rel.get('frame_id', 0)
            if frame_id >= observe_frames:
                subj = rel.get('subject', '').lower()
                pred = rel.get('predicate', '').lower()
                obj = rel.get('object', '').lower()
                if subj and pred and obj:
                    gt_future_triplets.add((subj, pred, obj))
        
        # Anticipate
        predictions = anticipate_future(observed_triplets)
        
        # Evaluate
        r10 = compute_recall_at_k(predictions, gt_future_triplets, 10)
        r20 = compute_recall_at_k(predictions, gt_future_triplets, 20)
        r50 = compute_recall_at_k(predictions, gt_future_triplets, 50)
        
        elapsed = time.time() - start_time
        
        result = {
            'video_id': video_id,
            'observed_triplets': len(observed_triplets),
            'gt_future_triplets': len(gt_future_triplets),
            'predictions': len(predictions),
            'R@10': r10,
            'R@20': r20,
            'R@50': r50,
            'processing_time': elapsed,
        }
        results.append(result)
        
        logger.info(f"  Observed: {len(observed_triplets)} triplets | "
                   f"GT Future: {len(gt_future_triplets)} | "
                   f"R@10={r10:.1f}% R@20={r20:.1f}% R@50={r50:.1f}% | "
                   f"{elapsed:.1f}s")
    
    total_elapsed = time.time() - total_start
    
    # Summary
    if results:
        avg_r10 = np.mean([r['R@10'] for r in results])
        avg_r20 = np.mean([r['R@20'] for r in results])
        avg_r50 = np.mean([r['R@50'] for r in results])
        
        print("\n" + "=" * 60)
        print(f"REAL SGA RESULTS (F={args.observe_fraction}, {len(results)} videos)")
        print("=" * 60)
        print(f"  Mean R@10: {avg_r10:.2f}%")
        print(f"  Mean R@20: {avg_r20:.2f}%")
        print(f"  Mean R@50: {avg_r50:.2f}%")
        print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/len(results):.1f}s/video)")
        print("=" * 60)
        
        # Save results
        output_data = {
            'observe_fraction': args.observe_fraction,
            'num_videos': len(results),
            'mean_R@10': avg_r10,
            'mean_R@20': avg_r20,
            'mean_R@50': avg_r50,
            'total_time': total_elapsed,
            'per_video': results,
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    else:
        logger.error("No videos processed successfully!")


if __name__ == "__main__":
    main()
