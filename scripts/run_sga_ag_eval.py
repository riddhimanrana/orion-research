#!/usr/bin/env python3
"""
Run Orion detection pipeline on Action Genome videos and evaluate SGA.

This script:
1. Runs Orion detection on AG videos (using run_showcase or similar)
2. Evaluates the trained SGA model using Orion detections
"""

import os
import sys
import json
import pickle
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from orion.sga.temporal_model import TemporalSGAConfig, TemporalSGAModel
from orion.sga.ag_dataset_v2 import AG_OBJECT_CLASSES, AG_ALL_PREDICATES

# ============================================================================
# CONFIG
# ============================================================================

AG_VIDEOS_DIR = Path("datasets/ActionGenome/videos/Charades_v1_480")
AG_ANNOTATIONS_PATH = Path("datasets/ActionGenome/annotations/action_genome_v1.0/object_bbox_and_relationship.pkl")
RESULTS_DIR = Path("results")
MODEL_PATH = Path("models/temporal_sga_best.pt")

# Mapping from Orion labels to AG classes
ORION_TO_AG = {
    'adult': 'person', 'child': 'person', 'person': 'person',
    'computer': 'laptop', 'desk': 'table', 'cellphone': 'phone/camera',
    'bookcase': 'shelf', 'couch': 'sofa', 'tv': 'television',
    'refrigerator': 'refrigerator', 'counter': 'table',
    'glass': 'cup/glass/bottle', 'bottle': 'cup/glass/bottle',
    'cup': 'cup/glass/bottle', 'beverage': 'cup/glass/bottle',
}

# AG vocabulary
OBJECT_TO_IDX = {c: i for i, c in enumerate(AG_OBJECT_CLASSES)}
PREDICATE_TO_IDX = {p: i for i, p in enumerate(AG_ALL_PREDICATES)}


# ============================================================================
# DETECTION (using Orion pipeline)
# ============================================================================

def run_orion_detection(video_path: Path, output_dir: Path, force: bool = False) -> bool:
    """Run Orion detection pipeline on a video."""
    tracks_file = output_dir / "tracks.jsonl"
    
    if tracks_file.exists() and not force:
        print(f"  Tracks exist: {tracks_file}")
        return True
    
    # Use the Orion CLI for detection
    cmd = [
        "python", "-m", "orion.cli.run_showcase",
        "--episode", output_dir.name,
        "--video", str(video_path),
        "--skip-scene-graph",  # We only need tracks
        "--no-overlay",
    ]
    
    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  Error: {result.stderr[:500]}")
            return False
        return tracks_file.exists()
    except subprocess.TimeoutExpired:
        print(f"  Timeout running detection")
        return False
    except Exception as e:
        print(f"  Exception: {e}")
        return False


def run_simple_yolo_detection(video_path: Path, output_dir: Path, force: bool = False) -> bool:
    """Run detection using run_and_eval.py (GroundingDINO + DINOv3)."""
    tracks_file = output_dir / "tracks.jsonl"
    
    if tracks_file.exists() and not force:
        print(f"  Using existing tracks: {tracks_file}")
        return True
    
    print(f"  Running GroundingDINO detection via run_and_eval.py...")
    
    # Use the existing run_and_eval.py which uses GroundingDINO
    cmd = [
        "python3", "run_and_eval.py",
        "--video", str(video_path),
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(Path(__file__).parent.parent))
        if result.returncode != 0:
            print(f"  Error: {result.stderr[:500] if result.stderr else 'Unknown error'}")
            return False
        
        # Check if tracks file was created
        # run_and_eval.py puts results in results/<video_name>/
        video_name = video_path.stem
        expected_tracks = RESULTS_DIR / video_name / "tracks.jsonl"
        if expected_tracks.exists():
            print(f"  ✓ Detection complete: {expected_tracks}")
            return True
        else:
            print(f"  Tracks not found at {expected_tracks}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  Timeout running detection")
        return False
    except Exception as e:
        print(f"  Exception: {e}")
        return False


# ============================================================================
# LOAD FUNCTIONS
# ============================================================================

def load_tracks(tracks_path: Path) -> Dict[int, List[Dict]]:
    """Load tracks from JSONL file."""
    dets = defaultdict(list)
    
    with open(tracks_path) as f:
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


def load_ag_ground_truth(video_id: str) -> Dict[int, List]:
    """Load AG ground truth for a video."""
    with open(AG_ANNOTATIONS_PATH, 'rb') as f:
        annotations = pickle.load(f)
    
    gt = defaultdict(list)
    person_idx = OBJECT_TO_IDX['person']
    
    for key, objects in annotations.items():
        if not key.startswith(f"{video_id}.mp4/"):
            continue
        
        # Extract frame number
        frame_str = key.split('/')[-1].replace('.png', '')
        try:
            frame_idx = int(frame_str)
        except ValueError:
            continue
        
        for obj in objects:
            obj_class = obj.get('class', '')
            obj_idx = OBJECT_TO_IDX.get(obj_class, -1)
            if obj_idx < 0:
                continue
            
            for pred in (obj.get('spatial_relationship') or []):
                pred_idx = PREDICATE_TO_IDX.get(pred, -1)
                if pred_idx >= 0:
                    gt[frame_idx].append((person_idx, pred_idx, obj_idx))
            
            for pred in (obj.get('contacting_relationship') or []):
                pred_idx = PREDICATE_TO_IDX.get(pred, -1)
                if pred_idx >= 0:
                    gt[frame_idx].append((person_idx, pred_idx, obj_idx))
    
    return dict(gt)


def map_to_ag_class(label: str) -> int:
    """Map detection label to AG class index."""
    label = label.lower()
    if label in ORION_TO_AG:
        return OBJECT_TO_IDX.get(ORION_TO_AG[label], -1)
    return OBJECT_TO_IDX.get(label, -1)


# ============================================================================
# SGA EVALUATION
# ============================================================================

class SGAEvaluator:
    """Evaluate SGA model with detections."""
    
    def __init__(self, model_path: Path, device: str = "cpu"):
        self.device = device
        
        # Load model
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        cfg = ckpt.get('config', TemporalSGAConfig())
        if isinstance(cfg, dict):
            cfg = TemporalSGAConfig(**cfg)
        
        self.model = TemporalSGAModel(cfg)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.to(device)
        self.model.eval()
        print(f"✓ Loaded SGA model ({sum(p.numel() for p in self.model.parameters()):,} params)")
    
    def prepare_input(self, dets: Dict[int, List], frame_ids: List[int], max_obj: int = 20):
        """Prepare model input from detections."""
        T = len(frame_ids)
        
        labels = torch.zeros(T, max_obj, dtype=torch.long)
        boxes = torch.zeros(T, max_obj, 4, dtype=torch.float32)
        masks = torch.zeros(T, max_obj, dtype=torch.bool)
        
        for t, fid in enumerate(frame_ids):
            n = 0
            for det in dets.get(fid, []):
                if n >= max_obj:
                    break
                ag_idx = map_to_ag_class(det['label'])
                if ag_idx < 0:
                    continue
                
                labels[t, n] = ag_idx
                bbox = det['bbox']
                # Normalize bbox (assume 1920x1080)
                boxes[t, n] = torch.tensor([
                    bbox[0] / 1920, bbox[1] / 1080,
                    (bbox[2] - bbox[0]) / 1920, (bbox[3] - bbox[1]) / 1080
                ])
                masks[t, n] = True
                n += 1
        
        return labels, boxes, masks
    
    @torch.no_grad()
    def predict(self, labels, boxes, masks) -> List:
        """Run SGA prediction."""
        # Check if we have any valid objects
        if not masks.any():
            return []
        
        # Add batch dim
        class_ids = labels.unsqueeze(0).to(self.device)
        bboxes = boxes.unsqueeze(0).to(self.device)
        obj_mask = masks.unsqueeze(0).to(self.device)
        
        try:
            # Run model
            out = self.model(
                class_ids=class_ids,
                bboxes=bboxes,
                appearance_features=None,
                object_mask=obj_mask,
                frame_mask=None,
                num_future_frames=1
            )
        except RuntimeError as e:
            # Handle empty tensor errors
            if "non-zero numel" in str(e):
                return []
            raise
        
        # Extract predictions
        pred_logits = out.get('predicate_logits')
        if pred_logits is None:
            return []
        
        pred_probs = torch.sigmoid(pred_logits[0, 0])  # [num_positions, num_predicates]
        
        T, N = labels.shape
        valid_obj = masks[-1].nonzero().squeeze(-1).tolist()
        if not isinstance(valid_obj, list):
            valid_obj = [valid_obj]
        
        person_idx = 0  # Person is always subject in AG
        preds = []
        
        for oi in range(min(len(valid_obj), pred_probs.shape[0])):
            obj_class = int(labels[-1, valid_obj[oi]].item())
            for pi in range(pred_probs.shape[1]):
                score = pred_probs[oi, pi].item()
                if score > 0.1:
                    preds.append((person_idx, pi, obj_class, score))
        
        preds.sort(key=lambda x: -x[3])
        return preds
    
    def evaluate_video(self, tracks_path: Path, gt: Dict[int, List], observe: int = 8) -> Dict:
        """Evaluate a single video."""
        dets = load_tracks(tracks_path)
        
        if not dets:
            return {'r10': 0, 'r20': 0, 'r50': 0, 'n': 0}
        
        det_frames = sorted(dets.keys())
        gt_frames = sorted(gt.keys())
        
        if len(det_frames) < observe:
            return {'r10': 0, 'r20': 0, 'r50': 0, 'n': 0}
        
        # Map GT frames to nearest detection frames
        def find_nearest(gf):
            best = min(det_frames, key=lambda x: abs(x - gf))
            return best if abs(best - gf) < 10 else None
        
        r10_sum, r20_sum, r50_sum = 0.0, 0.0, 0.0
        n_samples = 0
        
        for gt_frame in gt_frames:
            gt_triplets = set(gt[gt_frame])
            if not gt_triplets:
                continue
            
            det_frame = find_nearest(gt_frame)
            if det_frame is None:
                continue
            
            # Get observed frames (8 frames before target)
            idx = det_frames.index(det_frame)
            if idx < observe:
                continue
            
            obs_frames = det_frames[idx - observe:idx]
            
            # Prepare input
            labels, boxes, masks = self.prepare_input(dets, obs_frames)
            
            # Predict
            preds = self.predict(labels, boxes, masks)
            
            # Calculate recall@k
            for k in [10, 20, 50]:
                top_k = set((p[0], p[1], p[2]) for p in preds[:k])
                matches = len(top_k & gt_triplets)
                recall = matches / len(gt_triplets)
                
                if k == 10:
                    r10_sum += recall
                elif k == 20:
                    r20_sum += recall
                else:
                    r50_sum += recall
            
            n_samples += 1
        
        if n_samples == 0:
            return {'r10': 0, 'r20': 0, 'r50': 0, 'n': 0}
        
        return {
            'r10': r10_sum / n_samples,
            'r20': r20_sum / n_samples,
            'r50': r50_sum / n_samples,
            'n': n_samples
        }


# ============================================================================
# MAIN
# ============================================================================

def get_ag_video_ids(max_videos: int = None) -> List[str]:
    """Get list of AG video IDs that have ground truth."""
    with open(AG_ANNOTATIONS_PATH, 'rb') as f:
        ag = pickle.load(f)
    
    video_ids = set()
    for key in ag.keys():
        vid = key.split('.mp4/')[0] if '.mp4/' in key else key.split('/')[0]
        video_ids.add(vid)
    
    # Filter to videos that exist
    valid = []
    for vid in sorted(video_ids):
        vpath = AG_VIDEOS_DIR / f"{vid}.mp4"
        if vpath.exists():
            valid.append(vid)
            if max_videos and len(valid) >= max_videos:
                break
    
    return valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-videos", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--force-detect", action="store_true")
    parser.add_argument("--skip-detect", action="store_true")
    args = parser.parse_args()
    
    print("=" * 60)
    print("SGA EVALUATION WITH ORION DETECTIONS")
    print("=" * 60)
    
    # Get video IDs
    video_ids = get_ag_video_ids(args.max_videos)
    print(f"\nFound {len(video_ids)} AG videos with GT")
    
    # Load evaluator
    evaluator = SGAEvaluator(MODEL_PATH, device=args.device)
    
    # Process each video
    results = []
    
    for vid in tqdm(video_ids, desc="Processing videos"):
        video_path = AG_VIDEOS_DIR / f"{vid}.mp4"
        output_dir = RESULTS_DIR / vid
        tracks_path = output_dir / "tracks.jsonl"
        
        print(f"\n{vid}:")
        
        # Run detection if needed
        if not args.skip_detect:
            if not tracks_path.exists() or args.force_detect:
                success = run_simple_yolo_detection(video_path, output_dir, force=args.force_detect)
                if not success:
                    print(f"  Skipping - detection failed")
                    continue
        
        if not tracks_path.exists():
            print(f"  Skipping - no tracks")
            continue
        
        # Load GT
        gt = load_ag_ground_truth(vid)
        if not gt:
            print(f"  Skipping - no GT")
            continue
        
        # Evaluate
        metrics = evaluator.evaluate_video(tracks_path, gt)
        
        if metrics['n'] > 0:
            print(f"  R@10={metrics['r10']:.1%}, R@20={metrics['r20']:.1%}, R@50={metrics['r50']:.1%} ({metrics['n']} samples)")
            results.append(metrics)
        else:
            print(f"  No valid samples")
    
    # Summary
    if results:
        avg_r10 = sum(r['r10'] for r in results) / len(results)
        avg_r20 = sum(r['r20'] for r in results) / len(results)
        avg_r50 = sum(r['r50'] for r in results) / len(results)
        total_n = sum(r['n'] for r in results)
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Videos evaluated: {len(results)}")
        print(f"Total samples: {total_n}")
        print(f"Average R@10: {avg_r10:.2%}")
        print(f"Average R@20: {avg_r20:.2%}")
        print(f"Average R@50: {avg_r50:.2%}")


if __name__ == "__main__":
    main()
