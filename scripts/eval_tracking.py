#!/usr/bin/env python3
"""
Re-ID / Tracking Evaluation
Evaluates tracking consistency using V-JEPA2 embeddings and IoU-based matching
"""

import argparse
import cv2
import json
import logging
import numpy as np
import time
from collections import defaultdict
from pathlib import Path
from PIL import Image
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_device():
    """Auto-detect best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def evaluate_iou_tracking(video_path: str, base_detections: list, iou_threshold: float = 0.3):
    """Evaluate simple IoU-based tracking."""
    logger.info("\n" + "="*60)
    logger.info(f"IoU-BASED TRACKING (threshold={iou_threshold})")
    logger.info("="*60)
    
    tracks = {}  # track_id -> list of (frame_idx, detection)
    next_track_id = 0
    active_tracks = {}  # track_id -> last detection
    
    track_switches = 0
    fragmentations = 0
    
    # Sort detections by frame
    sorted_frames = sorted(base_detections, key=lambda x: x["frame_idx"])
    
    for frame_data in sorted_frames:
        frame_idx = frame_data["frame_idx"]
        detections = frame_data.get("detections", [])
        
        if not detections:
            continue
        
        if not active_tracks:
            # Initialize tracks
            for det in detections:
                tracks[next_track_id] = [(frame_idx, det)]
                active_tracks[next_track_id] = det
                next_track_id += 1
            continue
        
        # Build cost matrix based on IoU
        det_indices = list(range(len(detections)))
        track_ids = list(active_tracks.keys())
        
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                iou = compute_iou(active_tracks[tid]["bbox"], det["bbox"])
                cost_matrix[i, j] = 1 - iou  # Convert to cost
        
        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_tracks = set()
        matched_dets = set()
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1 - iou_threshold:  # IoU > threshold
                tid = track_ids[r]
                det = detections[c]
                
                # Check for class consistency
                if active_tracks[tid]["class"] != det["class"]:
                    track_switches += 1
                
                tracks[tid].append((frame_idx, det))
                active_tracks[tid] = det
                matched_tracks.add(tid)
                matched_dets.add(c)
        
        # Handle unmatched tracks (potential fragmentations)
        for tid in track_ids:
            if tid not in matched_tracks:
                # Track lost
                fragmentations += 1
                del active_tracks[tid]
        
        # Handle unmatched detections (new tracks)
        for j, det in enumerate(detections):
            if j not in matched_dets:
                tracks[next_track_id] = [(frame_idx, det)]
                active_tracks[next_track_id] = det
                next_track_id += 1
    
    # Compute metrics
    track_lengths = [len(t) for t in tracks.values()]
    
    results = {
        "method": f"iou_{iou_threshold}",
        "stats": {
            "total_tracks": len(tracks),
            "avg_track_length": float(np.mean(track_lengths)) if track_lengths else 0,
            "max_track_length": max(track_lengths) if track_lengths else 0,
            "min_track_length": min(track_lengths) if track_lengths else 0,
            "track_switches": track_switches,
            "fragmentations": fragmentations,
            "tracks_gt_3_frames": sum(1 for l in track_lengths if l > 3),
            "tracks_gt_5_frames": sum(1 for l in track_lengths if l > 5),
        },
        "track_lengths": track_lengths,
        "tracks_by_class": defaultdict(list)
    }
    
    # Analyze tracks by class
    for tid, track_history in tracks.items():
        if track_history:
            primary_class = max(set(d["class"] for _, d in track_history), 
                              key=lambda c: sum(1 for _, d in track_history if d["class"] == c))
            results["tracks_by_class"][primary_class].append(len(track_history))
    
    results["tracks_by_class"] = dict(results["tracks_by_class"])
    
    return results


def evaluate_embedding_tracking(video_path: str, base_detections: list, 
                                similarity_threshold: float = 0.7):
    """Evaluate embedding-based tracking using DINOv2."""
    import torch
    from transformers import AutoModel, AutoProcessor
    
    logger.info("\n" + "="*60)
    logger.info(f"EMBEDDING-BASED TRACKING (DINOv2, threshold={similarity_threshold})")
    logger.info("="*60)
    
    device = get_device()
    
    # Load DINOv2
    logger.info("Loading DINOv2...")
    dino_processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-base")
    dino_model = dino_model.to(device)
    dino_model.eval()
    logger.info(f"DINOv2 loaded on {device}")
    
    cap = cv2.VideoCapture(video_path)
    
    tracks = {}  # track_id -> list of (frame_idx, detection, embedding)
    next_track_id = 0
    active_tracks = {}  # track_id -> (last_detection, gallery_embeddings)
    
    track_switches = 0
    fragmentations = 0
    processing_times = []
    
    sorted_frames = sorted(base_detections, key=lambda x: x["frame_idx"])
    
    for frame_data in sorted_frames:
        frame_idx = frame_data["frame_idx"]
        detections = frame_data.get("detections", [])
        
        if not detections:
            continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        start = time.time()
        
        # Extract embeddings for all detections
        det_embeddings = []
        valid_detections = []
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            crop = frame[max(0,y1):y2, max(0,x1):x2]
            if crop.size == 0:
                continue
            
            try:
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                inputs = dino_processor(images=crop_pil, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = dino_model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
                
                det_embeddings.append(embedding.cpu())
                valid_detections.append(det)
            except Exception as e:
                logger.debug(f"Embedding extraction error: {e}")
                continue
        
        if not valid_detections:
            continue
        
        det_embeddings = torch.cat(det_embeddings, dim=0)
        
        elapsed = time.time() - start
        processing_times.append(elapsed)
        
        if not active_tracks:
            # Initialize tracks
            for det, emb in zip(valid_detections, det_embeddings):
                tracks[next_track_id] = [(frame_idx, det)]
                active_tracks[next_track_id] = {
                    "detection": det,
                    "gallery": [emb]
                }
                next_track_id += 1
            continue
        
        # Build cost matrix based on embedding similarity + IoU
        track_ids = list(active_tracks.keys())
        
        cost_matrix = np.zeros((len(track_ids), len(valid_detections)))
        
        for i, tid in enumerate(track_ids):
            track_data = active_tracks[tid]
            gallery = torch.stack(track_data["gallery"][-10:])  # Keep last 10
            gallery_mean = gallery.mean(dim=0, keepdim=True)
            
            for j, (det, emb) in enumerate(zip(valid_detections, det_embeddings)):
                # Embedding similarity
                emb_sim = float(torch.nn.functional.cosine_similarity(
                    emb.unsqueeze(0), gallery_mean, dim=-1
                ))
                
                # IoU
                iou = compute_iou(track_data["detection"]["bbox"], det["bbox"])
                
                # Combined score (weighted)
                combined = 0.6 * emb_sim + 0.4 * iou
                cost_matrix[i, j] = 1 - combined
        
        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_tracks = set()
        matched_dets = set()
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1 - similarity_threshold:
                tid = track_ids[r]
                det = valid_detections[c]
                emb = det_embeddings[c]
                
                if active_tracks[tid]["detection"]["class"] != det["class"]:
                    track_switches += 1
                
                tracks[tid].append((frame_idx, det))
                active_tracks[tid]["detection"] = det
                active_tracks[tid]["gallery"].append(emb)
                
                # Limit gallery size
                if len(active_tracks[tid]["gallery"]) > 30:
                    active_tracks[tid]["gallery"] = active_tracks[tid]["gallery"][-30:]
                
                matched_tracks.add(tid)
                matched_dets.add(c)
        
        # Handle unmatched tracks
        for tid in track_ids:
            if tid not in matched_tracks:
                fragmentations += 1
                del active_tracks[tid]
        
        # Handle unmatched detections
        for j, (det, emb) in enumerate(zip(valid_detections, det_embeddings)):
            if j not in matched_dets:
                tracks[next_track_id] = [(frame_idx, det)]
                active_tracks[next_track_id] = {
                    "detection": det,
                    "gallery": [emb]
                }
                next_track_id += 1
    
    cap.release()
    
    # Compute metrics
    track_lengths = [len(t) for t in tracks.values()]
    
    results = {
        "method": f"embedding_{similarity_threshold}",
        "stats": {
            "total_tracks": len(tracks),
            "avg_track_length": float(np.mean(track_lengths)) if track_lengths else 0,
            "max_track_length": max(track_lengths) if track_lengths else 0,
            "min_track_length": min(track_lengths) if track_lengths else 0,
            "track_switches": track_switches,
            "fragmentations": fragmentations,
            "tracks_gt_3_frames": sum(1 for l in track_lengths if l > 3),
            "tracks_gt_5_frames": sum(1 for l in track_lengths if l > 5),
            "avg_processing_time": float(np.mean(processing_times)) if processing_times else 0,
        },
        "track_lengths": track_lengths,
        "tracks_by_class": defaultdict(list)
    }
    
    for tid, track_history in tracks.items():
        if track_history:
            primary_class = max(set(d["class"] for _, d in track_history),
                              key=lambda c: sum(1 for _, d in track_history if d["class"] == c))
            results["tracks_by_class"][primary_class].append(len(track_history))
    
    results["tracks_by_class"] = dict(results["tracks_by_class"])
    
    return results


def compare_tracking_methods(results_list: list):
    """Compare tracking results across methods."""
    logger.info("\n" + "="*70)
    logger.info("TRACKING METHOD COMPARISON")
    logger.info("="*70)
    
    print("\n{:<20} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Method", "Tracks", "Avg Len", "Max Len", "Switches", "Frags", ">5 frames"))
    print("-" * 85)
    
    for r in results_list:
        stats = r["stats"]
        print("{:<20} {:>10} {:>10.1f} {:>10} {:>10} {:>10} {:>10}".format(
            r["method"][:20],
            stats["total_tracks"],
            stats["avg_track_length"],
            stats["max_track_length"],
            stats["track_switches"],
            stats["fragmentations"],
            stats["tracks_gt_5_frames"]
        ))
    
    # Track quality comparison
    print("\n\nTRACK QUALITY ANALYSIS:")
    print("-" * 70)
    
    for r in results_list:
        stats = r["stats"]
        total = stats["total_tracks"]
        long_tracks = stats["tracks_gt_5_frames"]
        quality = long_tracks / total if total > 0 else 0
        
        print(f"{r['method']}:")
        print(f"  Track quality (>5 frames): {quality:.1%} ({long_tracks}/{total})")
        print(f"  Fragmentation rate: {stats['fragmentations'] / max(1, total):.2f} per track")
        print(f"  Switch rate: {stats['track_switches'] / max(1, total):.2f} per track")
    
    # Class-level analysis
    print("\n\nCLASS-LEVEL TRACKING:")
    print("-" * 70)
    
    all_classes = set()
    for r in results_list:
        all_classes.update(r["tracks_by_class"].keys())
    
    for cls_name in sorted(all_classes):
        print(f"\n{cls_name}:")
        for r in results_list:
            lengths = r["tracks_by_class"].get(cls_name, [])
            if lengths:
                print(f"  {r['method']}: {len(lengths)} tracks, avg len={np.mean(lengths):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Re-ID / Tracking Evaluation")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--detections", help="Path to base detections JSON")
    parser.add_argument("--output-dir", default="results/tracking_eval", help="Output directory")
    parser.add_argument("--skip-embedding", action="store_true", help="Skip embedding-based tracking")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base detections
    if args.detections and Path(args.detections).exists():
        logger.info(f"Loading base detections from {args.detections}")
        with open(args.detections) as f:
            base_data = json.load(f)
            base_detections = base_data.get("frames", [])
    else:
        logger.error("Base detections required. Run eval_detections_simple.py first.")
        return
    
    logger.info(f"Loaded {len(base_detections)} frames")
    
    all_results = []
    
    # 1. IoU-based tracking (baseline)
    iou_results = evaluate_iou_tracking(args.video, base_detections, iou_threshold=0.3)
    all_results.append(iou_results)
    with open(output_dir / "iou_tracking.json", "w") as f:
        json.dump(iou_results, f, indent=2, default=str)
    
    # 2. IoU with higher threshold
    iou_strict_results = evaluate_iou_tracking(args.video, base_detections, iou_threshold=0.5)
    all_results.append(iou_strict_results)
    with open(output_dir / "iou_strict_tracking.json", "w") as f:
        json.dump(iou_strict_results, f, indent=2, default=str)
    
    # 3. Embedding-based tracking
    if not args.skip_embedding:
        try:
            emb_results = evaluate_embedding_tracking(args.video, base_detections)
            all_results.append(emb_results)
            with open(output_dir / "embedding_tracking.json", "w") as f:
                json.dump(emb_results, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Embedding tracking failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare
    if len(all_results) > 1:
        compare_tracking_methods(all_results)
    
    logger.info("\n" + "="*70)
    logger.info("TRACKING EVALUATION COMPLETE")
    logger.info(f"Results saved to {output_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
