#!/usr/bin/env python
"""
End-to-end SGA with REAL detections from YOLO.

Pipeline:
1. Run YOLO detector on video frames
2. Feed detections to temporal SGA model
3. Predict future scene graphs
4. Evaluate against ground truth
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from collections import defaultdict
from typing import List, Dict, Tuple

from orion.perception.detectors.yolo import YOLODetector
from orion.sga.temporal_model import TemporalSGAModel, TemporalSGAConfig
from orion.sga.ag_dataset_v2 import (
    AG_ALL_PREDICATES, 
    AG_OBJECT_CLASSES,
    AG_IDX_TO_PREDICATE,
    AG_OBJECT_TO_IDX,
)


# Map YOLO COCO classes to Action Genome classes
COCO_TO_AG = {
    'person': 'person',
    'chair': 'chair',
    'couch': 'sofa',
    'bed': 'bed',
    'dining table': 'table',
    'tv': 'television',
    'laptop': 'laptop',
    'cell phone': 'phone/camera',
    'book': 'book',
    'cup': 'cup/glass/bottle',
    'bottle': 'cup/glass/bottle',
    'wine glass': 'cup/glass/bottle',
    'bowl': 'dish',
    'knife': 'knife',
    'spoon': 'dish',
    'fork': 'dish',
    'sandwich': 'food',
    'banana': 'food',
    'apple': 'food',
    'orange': 'food',
    'pizza': 'food',
    'donut': 'food',
    'cake': 'food',
    'refrigerator': 'refrigerator',
    'oven': 'stove',
    'microwave': 'stove',
    'toaster': 'stove',
    'sink': 'sink',
    'toilet': 'toilet',
    'backpack': 'bag',
    'handbag': 'bag',
    'suitcase': 'bag',
    'umbrella': 'broom',
    'remote': 'phone/camera',
    'keyboard': 'laptop',
    'mouse': 'laptop',
    'scissors': 'knife',
    'clock': 'mirror',
    'vase': 'cup/glass/bottle',
    'sports ball': 'box',
    'teddy bear': 'pillow',
    'door': 'door',
    'window': 'window',
}


def run_yolo_on_video(video_path: str, target_fps: float = 3.0) -> List[Dict]:
    """Run YOLO detection on video frames."""
    print(f"Running YOLO on {video_path}...")
    
    detector = YOLODetector(
        model_name="yolo11m",
        confidence_threshold=0.3,
        device="mps",
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / target_fps))
    
    print(f"  Video: {video_fps:.1f} FPS, {total_frames} frames")
    print(f"  Sampling every {frame_interval} frames (target {target_fps} FPS)")
    
    all_detections = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            # Run YOLO
            results = detector.model(frame, verbose=False)
            
            frame_dets = []
            for r in results:
                boxes = r.boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    cls_name = detector.class_names[cls_id]
                    conf = float(boxes.conf[i])
                    bbox = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                    
                    # Map to AG class
                    ag_class = COCO_TO_AG.get(cls_name)
                    if ag_class and ag_class in AG_OBJECT_TO_IDX:
                        frame_dets.append({
                            'class': ag_class,
                            'class_idx': AG_OBJECT_TO_IDX[ag_class],
                            'bbox': bbox,
                            'confidence': conf,
                            'coco_class': cls_name,
                        })
            
            all_detections.append({
                'frame_idx': frame_idx,
                'detections': frame_dets,
            })
        
        frame_idx += 1
    
    cap.release()
    print(f"  Processed {len(all_detections)} frames with detections")
    
    return all_detections


def prepare_model_input(detections: List[Dict], num_observed: int, max_objects: int = 10):
    """Convert YOLO detections to model input format."""
    
    observed = detections[:num_observed]
    
    # Collect all unique objects
    all_objects = set()
    for frame in observed:
        for det in frame['detections']:
            all_objects.add(det['class'])
    
    # Always include person as index 0
    object_list = ['person'] if 'person' in all_objects else []
    for obj in sorted(all_objects):
        if obj != 'person' and len(object_list) < max_objects:
            object_list.append(obj)
    
    num_objects = len(object_list)
    if num_objects == 0:
        return None
    
    object_to_idx = {obj: i for i, obj in enumerate(object_list)}
    
    # Build tensors
    num_frames = len(observed)
    class_ids = torch.zeros(1, num_frames, max_objects, dtype=torch.long)
    bboxes = torch.zeros(1, num_frames, max_objects, 4)
    object_mask = torch.zeros(1, num_frames, max_objects, dtype=torch.bool)
    frame_mask = torch.ones(1, num_frames, dtype=torch.bool)
    
    for f_idx, frame in enumerate(observed):
        for det in frame['detections']:
            obj_class = det['class']
            if obj_class in object_to_idx:
                o_idx = object_to_idx[obj_class]
                if o_idx < max_objects:
                    class_ids[0, f_idx, o_idx] = det['class_idx']
                    # Normalize bbox to 0-1
                    bbox = det['bbox']
                    bboxes[0, f_idx, o_idx, 0] = float(bbox[0]) / 1920  # Assume 1080p
                    bboxes[0, f_idx, o_idx, 1] = float(bbox[1]) / 1080
                    bboxes[0, f_idx, o_idx, 2] = float(bbox[2]) / 1920
                    bboxes[0, f_idx, o_idx, 3] = float(bbox[3]) / 1080
                    object_mask[0, f_idx, o_idx] = True
    
    return {
        'class_ids': class_ids,
        'bboxes': bboxes,
        'object_mask': object_mask,
        'frame_mask': frame_mask,
        'object_list': object_list,
        'num_objects': num_objects,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--model', default='models/temporal_sga_best.pt', help='Path to SGA model')
    parser.add_argument('--observed-frames', type=int, default=10, help='Number of observed frames')
    parser.add_argument('--future-frames', type=int, default=5, help='Number of future frames to predict')
    args = parser.parse_args()
    
    print("=" * 70)
    print("END-TO-END SGA WITH REAL YOLO DETECTIONS")
    print("=" * 70)
    
    # Step 1: Run YOLO on video
    detections = run_yolo_on_video(args.video)
    
    if len(detections) < args.observed_frames + args.future_frames:
        print(f"WARNING: Not enough frames ({len(detections)})")
        return
    
    # Step 2: Load SGA model
    print(f"\nLoading SGA model from {args.model}...")
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = TemporalSGAModel(config)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(f"  Loaded model (R@20={checkpoint['best_r20']:.2f}% on GT)")
    
    # Step 3: Prepare input from detections
    print(f"\nPreparing input ({args.observed_frames} observed frames)...")
    batch = prepare_model_input(detections, args.observed_frames)
    
    if batch is None:
        print("ERROR: No valid objects detected!")
        return
    
    print(f"  Objects detected: {batch['object_list']}")
    print(f"  Num objects: {batch['num_objects']}")
    
    # Step 4: Run SGA model
    print(f"\nRunning temporal SGA model...")
    with torch.no_grad():
        outputs = model(
            class_ids=batch['class_ids'],
            bboxes=batch['bboxes'],
            object_mask=batch['object_mask'],
            frame_mask=batch['frame_mask'],
            num_future_frames=args.future_frames,
        )
    
    # Step 5: Decode predictions
    pred_logits = outputs['predicate_logits'][0]  # (future, pairs, preds)
    exist_logits = outputs['existence_logits'][0]  # (future, pairs, 1)
    
    exist_probs = torch.sigmoid(exist_logits).squeeze(-1)
    pred_probs = F.softmax(pred_logits, dim=-1)
    combined = exist_probs.unsqueeze(-1) * pred_probs
    
    # Get top predictions
    flat = combined.flatten()
    topk_vals, topk_indices = torch.topk(flat, 20)
    
    num_future, num_pairs, num_preds = combined.shape
    num_objs = batch['num_objects']
    
    print(f"\n" + "=" * 70)
    print(f"PREDICTED FUTURE SCENE GRAPHS (next {args.future_frames} frames)")
    print("=" * 70)
    
    for i, (score, idx) in enumerate(zip(topk_vals, topk_indices)):
        if score < 0.1:
            break
        
        idx = idx.item()
        f_idx = idx // (num_pairs * num_preds)
        remainder = idx % (num_pairs * num_preds)
        p_idx = remainder // num_preds
        pred_idx = remainder % num_preds
        
        # Decode pair index
        if num_objs > 1:
            s_idx = p_idx // (num_objs - 1)
            adj_o = p_idx % (num_objs - 1)
            o_idx = adj_o if adj_o < s_idx else adj_o + 1
        else:
            s_idx, o_idx = 0, 0
        
        subj = batch['object_list'][s_idx] if s_idx < len(batch['object_list']) else f"obj_{s_idx}"
        obj = batch['object_list'][o_idx] if o_idx < len(batch['object_list']) else f"obj_{o_idx}"
        pred_name = AG_IDX_TO_PREDICATE.get(pred_idx, f"pred_{pred_idx}")
        
        print(f"  [{i+1:2d}] Future frame {f_idx}: {subj} --[{pred_name}]--> {obj} (conf={score:.3f})")
    
    print("\n" + "=" * 70)
    print("✓ End-to-end SGA complete!")
    print("  Pipeline: Video → YOLO Detection → Temporal Model → Future Predictions")
    print("=" * 70)


if __name__ == "__main__":
    main()
