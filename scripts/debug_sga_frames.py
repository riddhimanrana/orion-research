#!/usr/bin/env python3
"""Debug script for SGA evaluation."""

import sys
import json
import pickle
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np

sys.path.insert(0, '/Users/yogeshatluru/orion-research')
from orion.sga.temporal_model import TemporalSGAConfig, TemporalSGAModel
from orion.sga.ag_dataset_v2 import AG_OBJECT_CLASSES, AG_ALL_PREDICATES

OBJECT_TO_IDX = {c: i for i, c in enumerate(AG_OBJECT_CLASSES)}
PREDICATE_TO_IDX = {p: i for i, p in enumerate(AG_ALL_PREDICATES)}

ORION_TO_AG = {
    'person': 'person', 'adult': 'person', 'child': 'person',
    'chair': 'chair', 'table': 'table', 'bed': 'bed',
    'cup': 'cup/glass/bottle', 'glass': 'cup/glass/bottle',
    'bottle': 'cup/glass/bottle', 'beverage': 'cup/glass/bottle',
    'countertop': 'table', 'hat': 'clothes', 'computer': 'laptop',
}

def map_to_ag(label):
    label_lower = label.lower().strip()
    if label_lower in ORION_TO_AG:
        ag_class = ORION_TO_AG[label_lower]
        return OBJECT_TO_IDX.get(ag_class, 0)
    if label_lower in OBJECT_TO_IDX:
        return OBJECT_TO_IDX[label_lower]
    return None

# Load Orion detections
with open('/Users/yogeshatluru/orion-research/results/001YG/tracks.jsonl') as f:
    dets = defaultdict(list)
    for line in f:
        d = json.loads(line)
        dets[d['frame_id']].append(d)

orion_frames = sorted(dets.keys())
print(f'Orion frames: {len(orion_frames)}')

# Load AG GT
with open('/Users/yogeshatluru/orion-research/datasets/ActionGenome/annotations/action_genome_v1.0/object_bbox_and_relationship.pkl', 'rb') as f:
    ag_data = pickle.load(f)

gt_by_frame = defaultdict(list)
person_idx = OBJECT_TO_IDX.get('person', 0)
for key, objects in ag_data.items():
    if not key.startswith('001YG.mp4/'):
        continue
    frame_idx = int(key.split('/')[-1].replace('.png', ''))
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        obj_class = obj.get('class', '')
        obj_idx = OBJECT_TO_IDX.get(obj_class, 0)
        if obj_idx == 0:
            continue
        for pred in obj.get('spatial_relationship') or []:
            pred_idx = PREDICATE_TO_IDX.get(pred, 0)
            if pred_idx > 0:
                gt_by_frame[frame_idx].append((person_idx, pred_idx, obj_idx))
        for pred in obj.get('contacting_relationship') or []:
            pred_idx = PREDICATE_TO_IDX.get(pred, 0)
            if pred_idx > 0:
                gt_by_frame[frame_idx].append((person_idx, pred_idx, obj_idx))

print(f'GT frames: {len(gt_by_frame)}')
print(f'GT frame IDs: {sorted(gt_by_frame.keys())}')

# Find nearest orion frame for each GT frame
def find_nearest(gt_frame):
    best = min(orion_frames, key=lambda x: abs(x - gt_frame))
    return best if abs(best - gt_frame) < 10 else None

gt_mapped = defaultdict(list)
for gt_frame, triplets in gt_by_frame.items():
    nearest = find_nearest(gt_frame)
    if nearest is not None:
        gt_mapped[nearest].extend(triplets)
        
print(f'Mapped frames: {len(gt_mapped)}')
print(f'Mapped frame IDs: {sorted(gt_mapped.keys())[:10]}...')

# Check first observation window
mapped_frames = sorted(gt_mapped.keys())
first_fut = mapped_frames[0]
fut_idx = orion_frames.index(first_fut)
print(f'\nFirst future frame: {first_fut} at index {fut_idx}')
print(f'Need 8 observation frames before, have: {fut_idx}')

if fut_idx >= 8:
    obs_frames = orion_frames[fut_idx - 8:fut_idx]
    print(f'Obs frames: {obs_frames}')
    
    # Check detections in obs frames
    for fid in obs_frames:
        frame_dets = dets.get(fid, [])
        mapped_count = sum(1 for d in frame_dets if map_to_ag(d['label']) is not None)
        print(f'  Frame {fid}: {len(frame_dets)} dets, {mapped_count} mapped to AG')
