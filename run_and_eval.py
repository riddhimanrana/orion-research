#!/usr/bin/env python3
"""
Run SGG Generation and Evaluation on a batch of videos.
"""

import os
import json
import argparse
import subprocess
import logging
from pathlib import Path
from typing import List, Dict

# Import evaluation logic (assuming it's in the python path or same dir)
import cv2
import torch
import numpy as np
from collections import defaultdict
import logging

try:
    from ultralytics import YOLOWorld
except ImportError:
    YOLOWorld = None

# Import rich vocabulary for better recall
try:
    from orion.perception.config import YOLOWORLD_PROMPT_INDOOR_FULL
except ImportError:
    # Fallback if config not found
    YOLOWORLD_PROMPT_INDOOR_FULL = "person . chair . table . ball . dog . cat ."

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleTracker:
    """Simple IoU-based tracker for object tracking."""
    def __init__(self, max_age=30, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 1
        self.track_ages = {}
    
    def update(self, detections):
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
                self.tracks[best_track_id]['bbox'] = det_box
                self.tracks[best_track_id]['confidence'] = det['confidence']
                self.track_ages[best_track_id] = 0
                matched.add(best_track_id)
                results.append({**det, 'track_id': best_track_id})
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
                results.append({**det, 'track_id': track_id})
        
        to_remove = []
        for track_id in self.tracks:
            if track_id not in matched:
                self.track_ages[track_id] += 1
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

def run_detection(video_path: Path, output_dir: Path, conf_threshold=0.05) -> Path:
    """Run Phase 1: Detection + Tracking -> tracks.jsonl"""
    if YOLOWorld is None:
        logger.error("ultralytics not installed. Cannot run detection.")
        return None
        
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = output_dir / "tracks.jsonl"
    
    logger.info(f"Running Detection on {video_path}...")
    
    # Setup Model with Rich Vocab
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLOWorld("yolov8x-worldv2.pt")
    model.to(device)
    
    # Load Dynamic Vocabulary from PVSG GT
    # This ensures "ALL videos" work by using the exact vocabulary the dataset expects.
    try:
        with open("datasets/PVSG/pvsg.json", 'r') as f:
            pvsg_data = json.load(f)
            # Combine 'thing' and 'stuff' (if exists)
            things = pvsg_data.get('objects', {}).get('thing', [])
            stuff = pvsg_data.get('objects', {}).get('stuff', [])
            vocab = list(set(things + stuff))
            
            # Sanitization
            vocab = [c.strip() for c in vocab if c.strip()]
            vocab.sort()
            
            if not vocab:
                raise ValueError("Empty vocabulary found in PVSG JSON")
                
            model.set_classes(vocab)
            logger.info(f"Using PVSG DYNAMIC VOCABULARY ({len(vocab)} classes)")
            logger.info(f"Sample: {vocab[:10]}...")
            
    except Exception as e:
        logger.error(f"Failed to load PVSG vocabulary: {e}")
        logger.info("Falling back to rich indoor vocabulary...")
        classes = [c.strip() for c in YOLOWORLD_PROMPT_INDOOR_FULL.split('.') if c.strip()]
        model.set_classes(classes)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video {video_path}")
        return None
        
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 5.0
    frame_interval = max(1, int(video_fps / target_fps))
    
    tracker = SimpleTracker(max_age=30, iou_threshold=0.3)
    all_tracks = []
    frame_idx = 0
    sampled_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            # Predict
            results = model.predict(source=frame, conf=conf_threshold, device=device, verbose=False)
            
            detections = []
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
            
            # Track
            tracked = tracker.update(detections)
            for t in tracked:
                all_tracks.append({
                    'frame_id': frame_idx,
                    'track_id': t['track_id'],
                    'label': t['label'],
                    'confidence': t['confidence'],
                    'bbox': t['bbox'],
                })
                
            sampled_count += 1
            if sampled_count % 50 == 0:
                logger.info(f"Processed {sampled_count} frames...")
                
        frame_idx += 1
        
    cap.release()
    
    # Save tracks
    with open(tracks_path, 'w') as f:
        for t in all_tracks:
            f.write(json.dumps(t) + '\n')
            
    logger.info(f"Detection complete. Saved {len(all_tracks)} tracks to {tracks_path}")
    return tracks_path

def run_sgg(tracks_path: Path, output_path: Path) -> bool:
    """Run test_pvsg_sgg.py for a single video."""
    cmd = [
        "python", "test_pvsg_sgg.py",
        "--tracks", str(tracks_path),
        "--out", str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run SGG for {tracks_path}: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_dir", default="results/pvsg_batch_1_100", help="Directory containing video subdirectories with tracks.jsonl")
    parser.add_argument("--gt_path", default="datasets/PVSG/pvsg.json", help="Path to PVSG GT json")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process")
    parser.add_argument("--video_id", default=None, help="Specific video ID to process (within batch_dir)")
    parser.add_argument("--video", default=None, help="Path to raw video file for full run (Detection -> Tracking -> SGG -> Eval)")
    args = parser.parse_args()
    
    processed_videos = []
    
    # Case A: Single Video Full Run
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return
            
        # Infer Video ID
        # PVSG filenames are like "0003_3396832512.mp4"
        video_id = video_path.stem
        
        logger.info(f"Starting Full Pipeline for VIDEO: {video_id}")
        
        # Output dir
        output_dir = Path("results") / video_id
        
        # 1. Run Detection (Phase 1)
        tracks_path = run_detection(video_path, output_dir)
        if not tracks_path:
            logger.error("Detection failed.")
            return

        # 2. Run SGG (Phase 2)
        sgg_out = output_dir / "scene_graph.jsonl"
        if run_sgg(tracks_path, sgg_out):
            processed_videos.append(video_id)
        else:
            logger.error("SGG failed.")
            return
            
        batch_dir = Path("results") # For eval loader

    # Case B: Batch Run (Existing Tracks)
    else: 
        batch_dir = Path(args.batch_dir)
        if not batch_dir.exists():
            logger.error(f"Batch directory not found: {batch_dir}")
            return

        # Find videos
        if args.video_id:
            target_dir = batch_dir / args.video_id
            if not target_dir.exists():
                 found = list(batch_dir.glob(f"*{args.video_id}*"))
                 if found: target_dir = found[0]
            
            if target_dir.exists() and (target_dir / "tracks.jsonl").exists():
                video_dirs = [target_dir]
            else:
                logger.error(f"Video {args.video_id} not found in {batch_dir} (or no tracks.jsonl)")
                return
        else:
            video_dirs = [d for d in batch_dir.iterdir() if d.is_dir() and (d / "tracks.jsonl").exists()]
            video_dirs.sort(key=lambda x: x.name)
            if args.limit:
                video_dirs = video_dirs[:args.limit]
            
        logger.info(f"Found {len(video_dirs)} videos to process")
        
        # Run SGG Generation
        for v_dir in video_dirs:
            video_id = v_dir.name
            tracks_path = v_dir / "tracks.jsonl"
            output_path = v_dir / "scene_graph.jsonl"
            
            logger.info(f"[{video_id}] Generating Scene Graph...")
            if run_sgg(tracks_path, output_path):
                processed_videos.append(video_id)
            else:
                logger.warning(f"[{video_id}] SGG Generation failed.")

    # 3. Run Evaluation (Common)
    logger.info("Starting Evaluation...")
    
    # Load GT
    with open(args.gt_path, 'r') as f:
        pvsg_data = json.load(f)
    gt_map = {v['video_id']: v for v in pvsg_data['data']}
    
    total_recall = {'R@20': [], 'R@50': [], 'R@100': []}
    
    # Import evaluation logic
    try:
        from scripts.eval_sgg_recall import evaluate_video, load_pvsg_ground_truth
    except ImportError:
        import sys
        sys.path.append(os.getcwd())
        from scripts.eval_sgg_recall import evaluate_video, load_pvsg_ground_truth

    # We'll also store per-video metrics to print specific one if requested
    video_metrics = {}

    for video_id in processed_videos:
        # Load GT dictionary for evaluate_video compatibility
        gt_videos = {v['video_id']: v for v in pvsg_data['data']}
        res = evaluate_video(video_id, str(batch_dir), gt_videos)
        
        if 'error' in res:
             logger.warning(f"[{video_id}] Eval Error: {res['error']}")
             continue
             
        video_metrics[video_id] = res
        logger.info(f"[{video_id}] Pred={res['pred_count']} GT={res['gt_count']} R@20: {res['R@20']:.1f} | mR@20: {res['mR@20']:.1f}")
        logger.info(f"[{video_id}] R@20: {res['R@20']:.1f} | R@50: {res['R@50']:.1f} | R@100: {res['R@100']:.1f}")
        
    # Summary
    if video_metrics:
        
        print("\n\n")
        print("############################################################")
        print("#                   SGG EVALUATION RESULTS                 #")
        print("############################################################")
        
        if (args.video or args.video_id) and len(video_metrics) == 1:
             vid = list(video_metrics.keys())[0]
             m = video_metrics[vid]
             print(f"\nVIDEO: {vid}")
             print("-" * 60)
             print(f"R@20:   {m['R@20']:.2f}%  (mR@20:  {m['mR@20']:.2f})")
             print(f"R@50:   {m['R@50']:.2f}%  (mR@50:  {m['mR@50']:.2f})")
             print(f"R@100:  {m['R@100']:.2f}%  (mR@100: {m['mR@100']:.2f})")
             print("-" * 60)
        else:
             # Aggregate pooled mR@K
             def get_pooled_mr(k):
                 cat_stats = defaultdict(lambda: [0, 0])
                 for res in video_metrics.values():
                     for p, (m, t) in res['pred_stats'][k].items():
                         cat_stats[p][0] += m
                         cat_stats[p][1] += t
                 recalls = [m/t for m, t in cat_stats.values() if t > 0]
                 return np.mean(recalls) * 100.0 if recalls else 0.0

             mR20_avg = get_pooled_mr(20)
             mR50_avg = get_pooled_mr(50)
             mR100_avg = get_pooled_mr(100)
             
             R20_avg = np.mean([m['R@20'] for m in video_metrics.values()])
             R50_avg = np.mean([m['R@50'] for m in video_metrics.values()])
             R100_avg = np.mean([m['R@100'] for m in video_metrics.values()])
             
             print(f"\nAGGREGATE OVER {len(video_metrics)} VIDEOS")
             print("-" * 60)
             print(f"MEAN R@20:    {R20_avg:.2f}%  |  MEAN mR@20:   {mR20_avg:.2f}%")
             print(f"MEAN R@50:    {R50_avg:.2f}%  |  MEAN mR@50:   {mR50_avg:.2f}%")
             print(f"MEAN R@100:   {R100_avg:.2f}%  |  MEAN mR@100:  {mR100_avg:.2f}%")
             print("-" * 60)
             
        print("############################################################\n")
    else:
        logger.warning("No valid evaluation results found.")

if __name__ == "__main__":
    main()
