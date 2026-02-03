#!/usr/bin/env python
"""
Evaluate SGA with REAL YOLO detections against ground truth.
Computes R@K metrics on Action Genome test videos.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import pickle
from collections import defaultdict
from typing import List, Dict

from orion.perception.detectors.yolo import YOLODetector
from orion.sga.temporal_model import TemporalSGAModel
from orion.sga.ag_dataset_v2 import (
    AG_ALL_PREDICATES, 
    AG_OBJECT_CLASSES,
    AG_IDX_TO_PREDICATE,
    AG_OBJECT_TO_IDX,
    AG_PREDICATE_TO_IDX,
)


# COCO to AG mapping
COCO_TO_AG = {
    'person': 'person', 'chair': 'chair', 'couch': 'sofa', 'bed': 'bed',
    'dining table': 'table', 'tv': 'television', 'laptop': 'laptop',
    'cell phone': 'phone/camera', 'book': 'book', 'cup': 'cup/glass/bottle',
    'bottle': 'cup/glass/bottle', 'bowl': 'dish', 'knife': 'knife',
    'refrigerator': 'refrigerator', 'oven': 'stove', 'microwave': 'stove',
    'sink': 'sink', 'toilet': 'toilet', 'backpack': 'bag', 'handbag': 'bag',
}


def get_gt_future_relations(annotations: Dict, video_id: str, future_frame_indices: List[int]):
    """Get ground truth relations for future frames."""
    relations = []
    
    for f_idx, frame_idx in enumerate(future_frame_indices):
        frame_key = f"{video_id}/{frame_idx:06d}.png"
        if frame_key not in annotations:
            continue
        
        for obj in annotations[frame_key]:
            if not obj.get('visible', False):
                continue
            
            obj_class = obj.get('class', '')
            if obj_class not in AG_OBJECT_TO_IDX:
                continue
            
            obj_idx = AG_OBJECT_TO_IDX[obj_class]
            
            # Get relations (person is always subject in AG)
            for rel_type in ['spatial_relationship', 'contacting_relationship']:
                rels = obj.get(rel_type) or []
                for rel in rels:
                    if rel in AG_PREDICATE_TO_IDX:
                        pred_idx = AG_PREDICATE_TO_IDX[rel]
                        relations.append((f_idx, 0, pred_idx, obj_idx))  # (future_idx, subj=person, pred, obj)
    
    return relations


def run_yolo_on_frames(video_path: str, detector, frame_indices: List[int]):
    """Run YOLO on specific frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    detections = []
    for target_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            detections.append({'frame_idx': target_idx, 'detections': []})
            continue
        
        results = detector.model(frame, verbose=False)
        
        frame_dets = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                cls_name = detector.class_names[int(boxes.cls[i])]
                ag_class = COCO_TO_AG.get(cls_name)
                if ag_class and ag_class in AG_OBJECT_TO_IDX:
                    frame_dets.append({
                        'class': ag_class,
                        'class_idx': AG_OBJECT_TO_IDX[ag_class],
                        'bbox': boxes.xyxy[i].cpu().numpy(),
                        'confidence': float(boxes.conf[i]),
                    })
        
        detections.append({'frame_idx': target_idx, 'detections': frame_dets})
    
    cap.release()
    return detections


def prepare_batch(detections: List[Dict], max_objects: int = 10):
    """Prepare model input from detections."""
    all_objects = set()
    for frame in detections:
        for det in frame['detections']:
            all_objects.add(det['class'])
    
    object_list = ['person'] if 'person' in all_objects else []
    for obj in sorted(all_objects):
        if obj != 'person' and len(object_list) < max_objects:
            object_list.append(obj)
    
    if len(object_list) == 0:
        return None
    
    object_to_idx = {obj: i for i, obj in enumerate(object_list)}
    num_frames = len(detections)
    
    class_ids = torch.zeros(1, num_frames, max_objects, dtype=torch.long)
    bboxes = torch.zeros(1, num_frames, max_objects, 4)
    object_mask = torch.zeros(1, num_frames, max_objects, dtype=torch.bool)
    frame_mask = torch.ones(1, num_frames, dtype=torch.bool)
    
    for f_idx, frame in enumerate(detections):
        for det in frame['detections']:
            if det['class'] in object_to_idx:
                o_idx = object_to_idx[det['class']]
                if o_idx < max_objects:
                    class_ids[0, f_idx, o_idx] = det['class_idx']
                    bbox = det['bbox']
                    bboxes[0, f_idx, o_idx] = torch.tensor([
                        float(bbox[0]) / 1920, float(bbox[1]) / 1080,
                        float(bbox[2]) / 1920, float(bbox[3]) / 1080
                    ])
                    object_mask[0, f_idx, o_idx] = True
    
    return {
        'class_ids': class_ids,
        'bboxes': bboxes,
        'object_mask': object_mask,
        'frame_mask': frame_mask,
        'object_list': object_list,
        'object_to_idx': object_to_idx,
    }


def compute_recall(pred_logits, exist_logits, gt_relations, object_to_idx, k_values=[10, 20, 50]):
    """Compute R@K against GT relations."""
    if len(gt_relations) == 0:
        return {}
    
    exist_probs = torch.sigmoid(exist_logits).squeeze(-1)
    pred_probs = F.softmax(pred_logits, dim=-1)
    combined = exist_probs.unsqueeze(-1) * pred_probs
    
    num_future, num_pairs, num_preds = combined.shape
    num_objs = int(np.sqrt(num_pairs)) + 1
    
    # Build GT set in prediction format
    gt_set = set()
    for (f_idx, s_idx, pred_idx, obj_class_idx) in gt_relations:
        # Find object index in our object list
        for obj_name, o_idx in object_to_idx.items():
            if AG_OBJECT_TO_IDX.get(obj_name) == obj_class_idx:
                if num_objs > 1:
                    adj_o = o_idx if o_idx < s_idx else o_idx - 1
                    pair_idx = s_idx * (num_objs - 1) + adj_o
                else:
                    pair_idx = 0
                if f_idx < num_future and pair_idx < num_pairs:
                    gt_set.add((f_idx, pair_idx, pred_idx))
                break
    
    if len(gt_set) == 0:
        return {}
    
    flat = combined.flatten()
    recalls = {}
    
    for k in k_values:
        _, topk_indices = torch.topk(flat, min(k, len(flat)))
        hits = 0
        for idx in topk_indices:
            idx = idx.item()
            f_idx = idx // (num_pairs * num_preds)
            remainder = idx % (num_pairs * num_preds)
            p_idx = remainder // num_preds
            pred_idx = remainder % num_preds
            if (f_idx, p_idx, pred_idx) in gt_set:
                hits += 1
        recalls[f'R@{k}'] = hits / len(gt_set)
    
    return recalls


def main():
    print("=" * 70)
    print("SGA EVALUATION WITH REAL YOLO DETECTIONS")
    print("=" * 70)
    
    # Load annotations
    ann_path = 'datasets/ActionGenome/annotations/action_genome_v1.0/object_bbox_and_relationship.pkl'
    with open(ann_path, 'rb') as f:
        annotations = pickle.load(f)
    print(f"Loaded {len(annotations)} frame annotations")
    
    # Get video list
    video_dir = Path('datasets/ActionGenome/videos/Charades_v1_480')
    videos = sorted(video_dir.glob('*.mp4'))[:10]  # Test on 10 videos
    print(f"Testing on {len(videos)} videos")
    
    # Load detector
    print("\nLoading YOLO detector...")
    detector = YOLODetector(model_name="yolo11m", confidence_threshold=0.3, device="mps")
    
    # Load SGA model
    print("Loading SGA model...")
    checkpoint = torch.load('models/temporal_sga_best.pt', map_location='cpu', weights_only=False)
    model = TemporalSGAModel(checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(f"  Model loaded (GT R@20={checkpoint['best_r20']:.2f}%)")
    
    # Evaluate
    print("\n" + "-" * 70)
    all_recalls = defaultdict(list)
    
    for i, video_path in enumerate(videos):
        video_id = video_path.stem
        
        # Get frame list from annotations
        video_frames = sorted([k for k in annotations.keys() if k.startswith(f"{video_id}.mp4/")])
        if len(video_frames) < 15:
            print(f"[{i+1}] {video_id}: Skipping (only {len(video_frames)} frames)")
            continue
        
        # Split: 50% observed, 50% future
        split = len(video_frames) // 2
        observed_keys = video_frames[:split][-10:]  # Last 10 observed
        future_keys = video_frames[split:][:5]  # First 5 future
        
        # Get frame indices
        observed_indices = [int(k.split('/')[-1].replace('.png', '')) for k in observed_keys]
        future_indices = [int(k.split('/')[-1].replace('.png', '')) for k in future_keys]
        
        # Run YOLO on observed frames
        detections = run_yolo_on_frames(str(video_path), detector, observed_indices)
        
        # Prepare batch
        batch = prepare_batch(detections)
        if batch is None:
            print(f"[{i+1}] {video_id}: No valid detections")
            continue
        
        # Get GT future relations
        gt_relations = get_gt_future_relations(annotations, f"{video_id}.mp4", future_indices)
        
        if len(gt_relations) == 0:
            print(f"[{i+1}] {video_id}: No GT relations")
            continue
        
        # Run model
        with torch.no_grad():
            outputs = model(
                class_ids=batch['class_ids'],
                bboxes=batch['bboxes'],
                object_mask=batch['object_mask'],
                frame_mask=batch['frame_mask'],
                num_future_frames=5,
            )
        
        # Compute recall
        recalls = compute_recall(
            outputs['predicate_logits'][0],
            outputs['existence_logits'][0],
            gt_relations,
            batch['object_to_idx'],
        )
        
        if recalls:
            r20 = recalls.get('R@20', 0) * 100
            r50 = recalls.get('R@50', 0) * 100
            print(f"[{i+1}] {video_id}: objs={len(batch['object_list'])}, GT={len(gt_relations)} | R@20={r20:.1f}% R@50={r50:.1f}%")
            
            for k, v in recalls.items():
                all_recalls[k].append(v)
        else:
            print(f"[{i+1}] {video_id}: Could not compute recall")
    
    # Summary
    print("-" * 70)
    print("\nRESULTS WITH REAL YOLO DETECTIONS:")
    for k in ['R@10', 'R@20', 'R@50']:
        if all_recalls[k]:
            avg = np.mean(all_recalls[k]) * 100
            print(f"  {k}: {avg:.2f}%")
    
    print("\nComparison:")
    print(f"  With GT detections:   R@20 = 38.01%")
    print(f"  With YOLO detections: R@20 = {np.mean(all_recalls['R@20'])*100:.2f}%")


if __name__ == "__main__":
    main()
