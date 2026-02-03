#!/usr/bin/env python3
"""Debug SGA predictions vs GT - understand why R@20 = 0%"""

import sys
import json
import pickle
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from orion.sga.temporal_model import TemporalSGAConfig, TemporalSGAModel
from orion.sga.ag_dataset_v2 import AG_OBJECT_CLASSES, AG_ALL_PREDICATES

# Mappings
OBJECT_TO_IDX = {c: i for i, c in enumerate(AG_OBJECT_CLASSES)}
PREDICATE_TO_IDX = {p: i for i, p in enumerate(AG_ALL_PREDICATES)}
IDX_TO_OBJECT = {i: c for i, c in enumerate(AG_OBJECT_CLASSES)}
IDX_TO_PREDICATE = {i: p for i, p in enumerate(AG_ALL_PREDICATES)}

# Orion label mapping
ORION_TO_AG = {
    'adult': 'person', 'child': 'person', 'person': 'person',
    'computer': 'laptop', 'desk': 'table', 'cellphone': 'phone/camera',
    'bookcase': 'shelf', 'couch': 'sofa', 'tv': 'television',
    'refrigerator': 'refrigerator', 'counter': 'table',
    'glass': 'cup/glass/bottle', 'bottle': 'cup/glass/bottle',
    'cup': 'cup/glass/bottle', 'beverage': 'cup/glass/bottle',
}

def load_orion_tracks(results_dir):
    """Load tracks from Orion results."""
    tracks_file = Path(results_dir) / 'tracks.jsonl'
    if not tracks_file.exists():
        return []
    
    tracks = []
    with open(tracks_file) as f:
        for line in f:
            tracks.append(json.loads(line))
    return tracks

def load_gt(video_id, frame_num):
    """Load GT for a specific frame."""
    pkl_path = Path('datasets/ActionGenome/annotations/action_genome_v1.0/object_bbox_and_relationship.pkl')
    with open(pkl_path, 'rb') as f:
        ag_data = pickle.load(f)
    
    # Find matching key
    frame_str = f'{frame_num:06d}'
    search_key = f'{video_id}.mp4/{frame_str}.png'
    
    triplets = []
    person_idx = OBJECT_TO_IDX['person']
    
    if search_key in ag_data:
        for obj in ag_data[search_key]:
            obj_class = obj.get('class', '')
            obj_idx = OBJECT_TO_IDX.get(obj_class, -1)
            if obj_idx < 0:
                continue
            
            for pred in (obj.get('spatial_relationship') or []):
                pred_idx = PREDICATE_TO_IDX.get(pred, -1)
                if pred_idx > 0:
                    triplets.append((person_idx, pred_idx, obj_idx))
            
            for pred in (obj.get('contacting_relationship') or []):
                pred_idx = PREDICATE_TO_IDX.get(pred, -1)
                if pred_idx > 0:
                    triplets.append((person_idx, pred_idx, obj_idx))
    
    return triplets, ag_data.get(search_key, [])

def tracks_to_model_input(tracks, target_frame):
    """Convert Orion tracks to model input tensors."""
    # Get tracks for frames BEFORE target frame (for prediction)
    frame_data = defaultdict(list)
    for t in tracks:
        fid = t['frame_id']
        if fid < target_frame:
            frame_data[fid].append(t)
    
    if not frame_data:
        return None
    
    # Use last 8 frames
    sorted_frames = sorted(frame_data.keys())[-8:]
    
    # Collect unique objects
    obj_labels = []
    for fid in sorted_frames:
        for det in frame_data[fid]:
            label = det['label'].lower()
            ag_label = ORION_TO_AG.get(label, label)
            if ag_label not in obj_labels and ag_label in OBJECT_TO_IDX:
                obj_labels.append(ag_label)
    
    if not obj_labels:
        return None
    
    num_obj = len(obj_labels)
    num_frames = len(sorted_frames)
    
    # Build tensors
    class_ids = torch.zeros(1, num_frames, num_obj, dtype=torch.long)
    bboxes = torch.zeros(1, num_frames, num_obj, 4)
    appearance = torch.randn(1, num_frames, num_obj, 2048)  # Model expects 2048-dim features
    obj_mask = torch.ones(1, num_frames, num_obj, dtype=torch.bool)
    frame_mask = torch.ones(1, num_frames, dtype=torch.bool)
    
    for fi, fid in enumerate(sorted_frames):
        for det in frame_data[fid]:
            label = det['label'].lower()
            ag_label = ORION_TO_AG.get(label, label)
            if ag_label in obj_labels:
                oi = obj_labels.index(ag_label)
                class_ids[0, fi, oi] = OBJECT_TO_IDX.get(ag_label, 0)
                bbox = det['bbox']
                bboxes[0, fi, oi] = torch.tensor([
                    bbox[0] / 1920, bbox[1] / 1080,
                    (bbox[2] - bbox[0]) / 1920, (bbox[3] - bbox[1]) / 1080
                ])
    
    return {
        'class_ids': class_ids,
        'bboxes': bboxes,
        'appearance': appearance,
        'obj_mask': obj_mask,
        'frame_mask': frame_mask,
        'obj_labels': obj_labels
    }

def extract_predictions(output, obj_labels, topk=20):
    """Extract top-k triplet predictions from model output."""
    pred_logits = output['predicate_logits']  # [B, F, P, num_predicates]
    B, F, P, num_pred = pred_logits.shape
    
    # Get probabilities
    pred_probs = torch.sigmoid(pred_logits[0, 0])  # [P, num_predicates]
    
    triplets = []
    person_idx = OBJECT_TO_IDX['person']
    
    for oi, obj_label in enumerate(obj_labels):
        obj_idx = OBJECT_TO_IDX.get(obj_label, 0)
        for pi in range(num_pred):
            score = pred_probs[oi, pi].item()
            if score > 0.1:  # threshold
                triplets.append((person_idx, pi, obj_idx, score))
    
    # Sort by score and take top-k
    triplets.sort(key=lambda x: -x[3])
    return triplets[:topk]

def main():
    print("="*60)
    print("DEBUG: SGA Predictions vs GT")
    print("="*60)
    
    print("\nObject classes:", AG_OBJECT_CLASSES[:10], "...")
    print("Predicates:", AG_ALL_PREDICATES[:10], "...")
    print(f"person idx = {OBJECT_TO_IDX['person']}")
    
    # Load model
    print("\nLoading model...")
    ckpt = torch.load('models/temporal_sga_best.pt', map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', TemporalSGAConfig())
    if isinstance(cfg, dict):
        cfg = TemporalSGAConfig(**cfg)
    model = TemporalSGAModel(cfg)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"Model loaded, num_predicate_classes = {cfg.num_predicate_classes}")
    
    # Load Orion tracks
    print("\nLoading Orion tracks for 001YG...")
    tracks = load_orion_tracks('results/001YG')
    print(f"Loaded {len(tracks)} track entries")
    
    # Check what frames we have
    frame_ids = sorted(set(t['frame_id'] for t in tracks))
    print(f"Frame range: {frame_ids[0]} - {frame_ids[-1]}")
    
    # Load GT for frame 89 (which maps to ~90 in 1-indexed)
    print("\nLoading GT for frame 89 (001YG)...")
    gt_triplets, gt_objects = load_gt('001YG', 89)
    print(f"GT has {len(gt_objects)} objects, {len(gt_triplets)} triplets")
    
    print("\nGT Objects:")
    for obj in gt_objects:
        print(f"  {obj.get('class')}: spatial={obj.get('spatial_relationship')}, contact={obj.get('contacting_relationship')}")
    
    print("\nGT Triplets:")
    for t in gt_triplets:
        print(f"  ({IDX_TO_OBJECT[t[0]]}, {IDX_TO_PREDICATE[t[1]]}, {IDX_TO_OBJECT[t[2]]})")
    
    # Prepare input for target frame 90
    print("\n" + "="*60)
    print("Preparing model input...")
    model_input = tracks_to_model_input(tracks, target_frame=90)
    
    if model_input is None:
        print("ERROR: Could not prepare model input")
        return
    
    print(f"Objects detected: {model_input['obj_labels']}")
    print(f"Class IDs shape: {model_input['class_ids'].shape}")
    
    # Run model
    print("\nRunning model inference...")
    with torch.no_grad():
        output = model(
            model_input['class_ids'],
            model_input['bboxes'],
            model_input['appearance'],
            model_input['obj_mask'],
            model_input['frame_mask'],
            num_future_frames=1
        )
    
    print(f"Output predicate_logits shape: {output['predicate_logits'].shape}")
    
    # Extract predictions
    print("\nExtracting predictions...")
    predictions = extract_predictions(output, model_input['obj_labels'], topk=20)
    
    print(f"\nTop-20 Predictions:")
    for p in predictions[:20]:
        print(f"  ({IDX_TO_OBJECT[p[0]]}, {IDX_TO_PREDICATE[p[1]]}, {IDX_TO_OBJECT[p[2]]}) score={p[3]:.3f}")
    
    # Check overlap
    print("\n" + "="*60)
    print("CHECKING OVERLAP")
    print("="*60)
    
    pred_set = set((p[0], p[1], p[2]) for p in predictions)
    gt_set = set(gt_triplets)
    
    print(f"\nPrediction triplets (no score): {pred_set}")
    print(f"GT triplets: {gt_set}")
    print(f"Intersection: {pred_set & gt_set}")
    
    # Calculate recall
    if gt_set:
        recall = len(pred_set & gt_set) / len(gt_set)
        print(f"\nR@20 = {recall:.2%}")
    
    # Debug: what predicates does the model predict?
    print("\n" + "="*60)
    print("PREDICATE DISTRIBUTION")
    print("="*60)
    
    pred_logits = output['predicate_logits'][0, 0]  # [num_obj, num_predicates]
    pred_probs = torch.sigmoid(pred_logits)
    
    print("\nPer-object predicate probabilities:")
    for oi, label in enumerate(model_input['obj_labels']):
        probs = pred_probs[oi]
        top_preds = torch.topk(probs, 5)
        print(f"  {label}:")
        for idx, prob in zip(top_preds.indices.tolist(), top_preds.values.tolist()):
            print(f"    {IDX_TO_PREDICATE[idx]}: {prob:.3f}")

if __name__ == '__main__':
    main()
