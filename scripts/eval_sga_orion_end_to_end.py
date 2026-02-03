#!/usr/bin/env python3
"""
End-to-End Scene Graph Anticipation Evaluation

This script uses the EXISTING Orion detection pipeline outputs (tracks.jsonl)
and evaluates Scene Graph Anticipation against Action Genome ground truth.

Pipeline:
1. Load Orion detections from tracks.jsonl (already generated)
2. Map Orion class names to Action Genome indices
3. Run trained TemporalSGAModel to predict future scene graphs  
4. Evaluate R@K metrics against Action Genome ground truth

This is TRUE SGA evaluation with REAL detections, not GT!
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.sga.temporal_model import TemporalSGAConfig, TemporalSGAModel
from orion.sga.ag_dataset_v2 import AG_OBJECT_CLASSES as OBJECT_CLASSES, AG_ALL_PREDICATES as PREDICATE_CLASSES


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_OBJECTS = len(OBJECT_CLASSES)
NUM_PREDICATES = len(PREDICATE_CLASSES)

OBJECT_TO_IDX = {c: i for i, c in enumerate(OBJECT_CLASSES)}
PREDICATE_TO_IDX = {c: i for i, c in enumerate(PREDICATE_CLASSES)}
IDX_TO_OBJECT = {i: c for i, c in enumerate(OBJECT_CLASSES)}
IDX_TO_PREDICATE = {i: c for i, c in enumerate(PREDICATE_CLASSES)}


# =============================================================================
# ORION → ACTION GENOME CLASS MAPPING  
# =============================================================================

ORION_TO_AG = {
    # Person variants
    'person': 'person', 'adult': 'person', 'child': 'person',
    'baby': 'person', 'man': 'person', 'woman': 'person',
    
    # Furniture  
    'chair': 'chair', 'sofa': 'sofa/couch', 'couch': 'sofa/couch',
    'bed': 'bed', 'table': 'table', 'desk': 'table', 'dining table': 'table',
    
    # Containers
    'bag': 'bag', 'handbag': 'bag', 'backpack': 'bag', 'suitcase': 'bag',
    'box': 'box', 'cabinet': 'closet/cabinet', 'closet': 'closet/cabinet',
    'cupboard': 'closet/cabinet', 'refrigerator': 'refrigerator',
    'fridge': 'refrigerator', 'shelf': 'shelf',
    
    # Household
    'door': 'door', 'window': 'window', 'mirror': 'mirror',
    'light': 'light', 'lamp': 'light',
    
    # Textiles
    'blanket': 'blanket', 'pillow': 'pillow', 'towel': 'towel',
    'clothes': 'clothes', 'shirt': 'clothes', 'pants': 'clothes',
    
    # Electronics
    'laptop': 'laptop', 'computer': 'laptop', 'tv': 'television',
    'television': 'television', 'phone': 'phone/camera',
    'cellphone': 'phone/camera', 'camera': 'phone/camera',
    
    # Kitchen
    'cup': 'cup/glass/bottle', 'glass': 'cup/glass/bottle',
    'bottle': 'cup/glass/bottle', 'beverage': 'cup/glass/bottle',
    'dish': 'dish', 'plate': 'dish', 'bowl': 'dish',
    'food': 'food', 'sandwich': 'sandwich', 'cake': 'food',
    
    # Other
    'book': 'book', 'paper': 'paper/notebook', 'notebook': 'paper/notebook',
    'shoe': 'shoe', 'broom': 'broom', 'vacuum': 'vacuum',
    'picture': 'picture', 'floor': 'floor', 'doorway': 'doorway',
    'countertop': 'table', 'hat': 'clothes',
}


def map_to_ag_class(label: str) -> Optional[int]:
    """Map Orion label to Action Genome class index."""
    label_lower = label.lower().strip()
    
    if label_lower in ORION_TO_AG:
        ag_class = ORION_TO_AG[label_lower]
        return OBJECT_TO_IDX.get(ag_class, 0)
    
    if label_lower in OBJECT_TO_IDX:
        return OBJECT_TO_IDX[label_lower]
    
    return None


# =============================================================================
# LOAD ORION DETECTIONS
# =============================================================================

def load_orion_tracks(tracks_path: Path) -> Dict[int, List[Dict]]:
    """Load detections from Orion tracks.jsonl."""
    dets = defaultdict(list)
    
    with open(tracks_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            dets[d['frame_id']].append({
                'track_id': d['track_id'],
                'label': d['label'],
                'bbox': d['bbox'],
                'confidence': d.get('confidence', 1.0)
            })
    
    return dict(dets)


def tracks_to_tensors(
    dets: Dict[int, List[Dict]],
    frame_ids: List[int],
    max_obj: int = 20
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert Orion detections to model tensors."""
    T = len(frame_ids)
    
    boxes = torch.zeros(T, max_obj, 4, dtype=torch.float32)
    labels = torch.zeros(T, max_obj, dtype=torch.long)
    masks = torch.zeros(T, max_obj, dtype=torch.float32)
    
    for t, fid in enumerate(frame_ids):
        frame_dets = dets.get(fid, [])
        n = 0
        for det in frame_dets:
            if n >= max_obj:
                break
            
            ag_idx = map_to_ag_class(det['label'])
            if ag_idx is None or ag_idx == 0:
                continue
            
            boxes[t, n] = torch.tensor(det['bbox'][:4], dtype=torch.float32)
            labels[t, n] = ag_idx
            masks[t, n] = 1.0
            n += 1
    
    return boxes, labels, masks


# =============================================================================
# LOAD ACTION GENOME GROUND TRUTH
# =============================================================================

def load_ag_ground_truth(
    annotation_path: str,
    video_id: str
) -> Dict[int, List[Tuple[int, int, int]]]:
    """
    Load AG ground truth triplets for a specific video.
    
    Returns:
        Dict[frame_idx, List[(subj_class_idx, pred_idx, obj_class_idx)]]
    """
    with open(annotation_path, 'rb') as f:
        annotations = pickle.load(f)
    
    gt_by_frame = defaultdict(list)
    
    for frame_key, frame_data in annotations.items():
        # frame_key format: "video_id/frame_number.png" or similar
        if isinstance(frame_key, str):
            parts = frame_key.split('/')
            if len(parts) >= 1:
                vid = parts[0]
                if vid != video_id:
                    continue
                
                # Extract frame number
                if len(parts) > 1:
                    frame_str = parts[1].replace('.png', '').replace('.jpg', '')
                    try:
                        frame_idx = int(frame_str)
                    except ValueError:
                        continue
                else:
                    continue
        else:
            continue
        
        # Extract triplets from frame data
        if isinstance(frame_data, dict):
            relations = frame_data.get('relationships', [])
            for rel in relations:
                subj_class = rel.get('subject_class', '')
                obj_class = rel.get('object_class', '')
                predicate = rel.get('predicate', '')
                
                subj_idx = OBJECT_TO_IDX.get(subj_class, 0)
                obj_idx = OBJECT_TO_IDX.get(obj_class, 0)
                pred_idx = PREDICATE_TO_IDX.get(predicate, 0)
                
                if subj_idx > 0 and obj_idx > 0 and pred_idx > 0:
                    gt_by_frame[frame_idx].append((subj_idx, pred_idx, obj_idx))
    
    return dict(gt_by_frame)


def load_ag_gt_simple(annotation_path: str, video_id: str) -> Dict[int, List[Tuple[int, int, int]]]:
    """
    Load AG GT - format is dict with keys like "001YG.mp4/000264.png"
    Value is list of objects, each with spatial_relationship, contacting_relationship
    """
    with open(annotation_path, 'rb') as f:
        data = pickle.load(f)
    
    gt_by_frame = defaultdict(list)
    
    # Keys are "{video_id}.mp4/{frame_num}.png"
    video_prefix = f"{video_id}.mp4/"
    
    for key, objects in data.items():
        if not key.startswith(video_prefix):
            continue
        
        # Extract frame number: "001YG.mp4/000264.png" -> 264
        try:
            frame_str = key.split('/')[-1].replace('.png', '')
            frame_idx = int(frame_str)
        except:
            continue
        
        # Person is always subject (index 0 in AG)
        person_idx = OBJECT_TO_IDX.get('person', 0)
        
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            
            obj_class = obj.get('class', '')
            obj_idx = OBJECT_TO_IDX.get(obj_class, 0)
            
            if obj_idx == 0:
                continue
            
            # Spatial relationships
            spatial_rels = obj.get('spatial_relationship') or []
            for pred in spatial_rels:
                pred_idx = PREDICATE_TO_IDX.get(pred, 0)
                if pred_idx > 0:  # Skip background predicate
                    # Triplet: (person, predicate, object)
                    gt_by_frame[frame_idx].append((person_idx, pred_idx, obj_idx))
            
            # Contact relationships
            contact_rels = obj.get('contacting_relationship') or []
            for pred in contact_rels:
                pred_idx = PREDICATE_TO_IDX.get(pred, 0)
                if pred_idx > 0:
                    gt_by_frame[frame_idx].append((person_idx, pred_idx, obj_idx))
    
    return dict(gt_by_frame)


# =============================================================================
# SGA EVALUATION
# =============================================================================

def recall_at_k(
    preds: List[Tuple[int, int, int, float]],  # (subj, pred, obj, score)
    gt: List[Tuple[int, int, int]],  # (subj, pred, obj)
    k: int
) -> float:
    """Compute R@K."""
    if not gt:
        return 0.0
    
    top_k = sorted(preds, key=lambda x: x[3], reverse=True)[:k]
    pred_set = set((p[0], p[1], p[2]) for p in top_k)
    gt_set = set(gt)
    
    return len(pred_set & gt_set) / len(gt_set)


class OrionSGAEvaluator:
    """Evaluate SGA using Orion detections."""
    
    def __init__(self, model_path: str, device: str = "mps"):
        self.device = device
        
        # Load model
        print(f"Loading model from {model_path}...")
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        
        cfg = ckpt.get('config', TemporalSGAConfig())
        if isinstance(cfg, dict):
            cfg = TemporalSGAConfig(**cfg)
        
        self.model = TemporalSGAModel(cfg)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.to(device)
        self.model.eval()
        
        self.cfg = cfg
        print(f"✓ Loaded model ({sum(p.numel() for p in self.model.parameters()):,} params)")
    
    @torch.no_grad()
    def predict(
        self,
        boxes: torch.Tensor,  # [T, N, 4]
        labels: torch.Tensor,  # [T, N]
        masks: torch.Tensor,  # [T, N]
    ) -> List[Tuple[int, int, int, float]]:
        """Predict future scene graph triplets."""
        # Add batch dim - model expects: class_ids, bboxes, appearance_features, object_mask
        # class_ids: (batch, num_frames, num_objects)
        # bboxes: (batch, num_frames, num_objects, 4)
        # object_mask: (batch, num_frames, num_objects)
        
        class_ids = labels.unsqueeze(0).to(self.device)  # [1, T, N]
        bboxes = boxes.unsqueeze(0).to(self.device)      # [1, T, N, 4]
        object_mask = masks.unsqueeze(0).to(self.device).bool()  # [1, T, N]
        
        # No appearance features available from Orion tracks
        out = self.model(
            class_ids=class_ids,
            bboxes=bboxes,
            appearance_features=None,
            object_mask=object_mask,
            frame_mask=None,
            num_future_frames=1
        )
        
        # Extract triplet predictions from model output
        # Model outputs: predicate_logits (batch, num_future, num_object_positions, num_predicates)
        preds = []
        
        pred_logits = out.get('predicate_logits')  # [B, F, P, num_preds]
        
        if pred_logits is None:
            return preds
        
        # Get number of objects from last frame
        T, N = labels.shape
        valid_objects = masks[-1].nonzero().squeeze(-1).tolist()
        if not isinstance(valid_objects, list):
            valid_objects = [valid_objects]
        
        n_valid = len(valid_objects)
        if n_valid < 1:
            return preds
        
        # In Action Genome, person is ALWAYS the subject (idx 0)
        # Model output dimension corresponds to objects, not pairs
        # Use sigmoid (multi-label) not softmax (mutually exclusive)
        pred_probs = torch.sigmoid(pred_logits[0, 0])  # [num_object_positions, num_predicates]
        
        person_idx = 0  # Person is always subject in Action Genome
        
        # For each valid object position, predict predicates
        for oi in range(min(n_valid, pred_probs.shape[0])):
            obj_class_idx = int(labels[-1, valid_objects[oi]].item())
            
            # Get all predicates above threshold
            for pred_idx in range(pred_probs.shape[1]):
                score = pred_probs[oi, pred_idx].item()
                if score > 0.1:  # threshold
                    preds.append((person_idx, pred_idx, obj_class_idx, score))
        
        # Sort by score
        preds.sort(key=lambda x: -x[3])
        
        return preds
    
    def evaluate_video(
        self,
        tracks_path: Path,
        gt_by_frame: Dict[int, List[Tuple[int, int, int]]],
        observe: int = 8,
        anticipate: int = 3
    ) -> Dict[str, float]:
        """Evaluate a single video."""
        # Load Orion detections
        dets = load_orion_tracks(tracks_path)
        
        if not dets:
            return {'R@10': 0, 'R@20': 0, 'R@50': 0, 'n': 0}
        
        orion_frames = sorted(dets.keys())
        gt_frames = sorted(gt_by_frame.keys())
        
        if len(orion_frames) < observe + anticipate:
            return {'R@10': 0, 'R@20': 0, 'R@50': 0, 'n': 0}
        
        # Map GT frames to nearest Orion frames
        def find_nearest_orion(gt_frame):
            best = min(orion_frames, key=lambda x: abs(x - gt_frame))
            return best if abs(best - gt_frame) < 10 else None  # 10 frame tolerance
        
        # Build map from orion frame → GT triplets
        gt_mapped = defaultdict(list)
        for gt_frame, triplets in gt_by_frame.items():
            nearest = find_nearest_orion(gt_frame)
            if nearest is not None:
                gt_mapped[nearest].extend(triplets)
        
        if not gt_mapped:
            return {'R@10': 0, 'R@20': 0, 'R@50': 0, 'n': 0}
        
        r10, r20, r50 = [], [], []
        
        # Sample evaluation windows where we have GT
        mapped_frames = sorted(gt_mapped.keys())
        
        print(f"  Mapped frames to evaluate: {len(mapped_frames)}")
        
        for fut_frame in mapped_frames:
            # Find observation window before this frame
            try:
                fut_idx = orion_frames.index(fut_frame)
            except ValueError:
                print(f"    Frame {fut_frame} not in orion_frames")
                continue
            
            if fut_idx < observe:
                print(f"    Frame {fut_frame} idx={fut_idx} < {observe}")
                continue
            
            obs_frames = orion_frames[fut_idx - observe:fut_idx]
            
            # Get tensors
            boxes, labels, masks = tracks_to_tensors(dets, obs_frames)
            
            mask_sum = masks.sum().item()
            if mask_sum < observe:
                print(f"    Frame {fut_frame}: mask_sum={mask_sum} < {observe}")
                continue
            
            # Predict
            try:
                preds = self.predict(boxes, labels, masks)
            except Exception as e:
                print(f"    Frame {fut_frame}: predict error: {e}")
                continue
            
            # Evaluate
            gt = gt_mapped[fut_frame]
            if not gt:
                print(f"    Frame {fut_frame}: no GT")
                continue
            
            r10_val = recall_at_k(preds, gt, 10)
            r20_val = recall_at_k(preds, gt, 20)
            r50_val = recall_at_k(preds, gt, 50)
            
            r10.append(r10_val)
            r20.append(r20_val)
            r50.append(r50_val)
            
            print(f"    Frame {fut_frame}: preds={len(preds)}, gt={len(gt)}, r20={r20_val:.3f}")
        
        if not r20:
            return {'R@10': 0, 'R@20': 0, 'R@50': 0, 'n': 0}
        
        return {
            'R@10': np.mean(r10) * 100,
            'R@20': np.mean(r20) * 100,
            'R@50': np.mean(r50) * 100,
            'n': len(r20)
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/temporal_sga_best.pt')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--ag-annotations', 
                       default='datasets/ActionGenome/annotations/action_genome_v1.0/object_bbox_and_relationship.pkl')
    parser.add_argument('--ag-videos', default='datasets/ActionGenome/videos/Charades_v1_480')
    parser.add_argument('--max-videos', type=int, default=10)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    
    # Find videos that have BOTH Orion results AND AG ground truth
    results_dir = Path(args.results_dir)
    ag_videos_dir = Path(args.ag_videos)
    
    # Get available Charades videos
    charades_videos = set()
    if ag_videos_dir.exists():
        charades_videos = {p.stem for p in ag_videos_dir.glob('*.mp4')}
    
    print(f"Found {len(charades_videos)} Charades videos in AG dataset")
    
    # Find matching Orion results
    matching = []
    for d in results_dir.iterdir():
        if not d.is_dir():
            continue
        tracks_path = d / 'tracks.jsonl'
        if not tracks_path.exists() or tracks_path.stat().st_size == 0:
            continue
        
        video_id = d.name
        if video_id in charades_videos:
            matching.append((video_id, tracks_path))
    
    print(f"Found {len(matching)} videos with both Orion results and AG GT")
    
    if not matching:
        print("\nNo matching videos found!")
        print("Run Orion detection on Charades videos first:")
        print("  python run_and_eval.py --video datasets/ActionGenome/videos/Charades_v1_480/001YG.mp4")
        return
    
    # Initialize evaluator
    evaluator = OrionSGAEvaluator(args.model, args.device)
    
    # Load AG annotations
    print(f"\nLoading AG annotations from {args.ag_annotations}...")
    with open(args.ag_annotations, 'rb') as f:
        ag_data = pickle.load(f)
    print(f"Loaded {len(ag_data)} annotation entries")
    
    # Evaluate
    print(f"\n{'='*60}")
    print("SCENE GRAPH ANTICIPATION WITH ORION DETECTIONS")
    print(f"{'='*60}")
    
    all_metrics = []
    
    for video_id, tracks_path in matching[:args.max_videos]:
        print(f"\n{video_id}:")
        
        # Load GT for this video
        gt_by_frame = load_ag_gt_simple(args.ag_annotations, video_id)
        
        if not gt_by_frame:
            print(f"  No GT found for {video_id}, skipping")
            continue
        
        print(f"  GT frames: {len(gt_by_frame)}")
        
        metrics = evaluator.evaluate_video(tracks_path, gt_by_frame)
        
        if metrics['n'] > 0:
            print(f"  R@10: {metrics['R@10']:.2f}%")
            print(f"  R@20: {metrics['R@20']:.2f}%")
            print(f"  R@50: {metrics['R@50']:.2f}%")
            print(f"  Samples: {metrics['n']}")
            all_metrics.append(metrics)
        else:
            print(f"  No valid samples")
    
    # Summary
    if all_metrics:
        print(f"\n{'='*60}")
        print(f"SUMMARY ({len(all_metrics)} videos)")
        print(f"{'='*60}")
        
        r10 = np.mean([m['R@10'] for m in all_metrics])
        r20 = np.mean([m['R@20'] for m in all_metrics])
        r50 = np.mean([m['R@50'] for m in all_metrics])
        n = sum(m['n'] for m in all_metrics)
        
        print(f"R@10: {r10:.2f}%")
        print(f"R@20: {r20:.2f}%")
        print(f"R@50: {r50:.2f}%")
        print(f"Total samples: {n}")
    else:
        print("\nNo valid evaluations completed.")


if __name__ == '__main__':
    main()
