#!/usr/bin/env python3
"""
Re-ID Diagnostic Script
========================

Diagnoses why Re-ID is failing by analyzing embedding similarity
across frames for specific objects.

Usage:
    python scripts/diagnose_reid.py --results results/test_demo --video data/examples/test.mp4 --track-id 5
    python scripts/diagnose_reid.py --results results/test_demo --video data/examples/test.mp4 --class-name bed
    python scripts/diagnose_reid.py --results results/test_demo --video data/examples/test.mp4 --compare-backends

This script will:
1. Load tracks.jsonl
2. Extract crops for the specified track/class from the video
3. Compute embeddings using CLIP, DINO, and V-JEPA2
4. Print pairwise similarities to identify why Re-ID is failing

Author: Orion Research Team
Date: January 2026
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_tracks(results_dir: Path) -> List[dict]:
    """Load all detections from tracks.jsonl."""
    tracks_path = results_dir / "tracks.jsonl"
    if not tracks_path.exists():
        raise FileNotFoundError(f"tracks.jsonl not found: {tracks_path}")
    
    detections = []
    with open(tracks_path) as f:
        for line in f:
            line = line.strip()
            if line:
                detections.append(json.loads(line))
    
    return detections


def group_by_track(detections: List[dict]) -> Dict[int, List[dict]]:
    """Group detections by track_id."""
    by_track = defaultdict(list)
    for det in detections:
        track_id = det.get("track_id")
        if track_id is not None:
            by_track[track_id].append(det)
    return dict(by_track)


def group_by_class(detections: List[dict]) -> Dict[str, List[dict]]:
    """Group detections by class_name."""
    by_class = defaultdict(list)
    for det in detections:
        class_name = det.get("class_name", "unknown")
        by_class[class_name].append(det)
    return dict(by_class)


def extract_crop(
    video_path: Path,
    frame_id: int,
    bbox: Tuple[float, float, float, float],
    padding: float = 0.1
) -> Optional[np.ndarray]:
    """Extract a crop from the video at the given frame and bbox."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        logger.warning(f"Failed to read frame {frame_id}")
        return None
    
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Handle portrait videos that may have been rotated during detection
    # If bbox coords exceed frame dims, try unrotating
    if max(x1, x2) > w + 2 or max(y1, y2) > h + 2:
        # Coords are in rotated space; unrotate
        orig_w = w
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        unrot = []
        for xr, yr in corners:
            xo = (orig_w - 1) - yr
            yo = xr
            unrot.append((xo, yo))
        xs = [p[0] for p in unrot]
        ys = [p[1] for p in unrot]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
    
    # Add padding
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)
    
    x1 = max(0, int(x1) - pad_x)
    y1 = max(0, int(y1) - pad_y)
    x2 = min(w, int(x2) + pad_x)
    y2 = min(h, int(y2) + pad_y)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    return frame[y1:y2, x1:x2]


def compute_clip_embedding(crop: np.ndarray, clip_model) -> np.ndarray:
    """Compute CLIP embedding for a crop."""
    from PIL import Image
    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_crop)
    embedding = clip_model.encode_image(pil_image, normalize=True)
    return embedding


def compute_dino_embedding(crop: np.ndarray, dino_model) -> np.ndarray:
    """Compute DINO embedding for a crop."""
    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    embedding = dino_model.encode_image(rgb_crop, normalize=True)
    return embedding


def compute_vjepa2_embedding(crop: np.ndarray, vjepa2_model) -> np.ndarray:
    """Compute V-JEPA2 embedding for a crop."""
    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    embedding_tensor = vjepa2_model.embed_single_image(rgb_crop)
    return embedding_tensor.numpy().flatten()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = a.flatten()
    b = b.flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def analyze_track(
    track_dets: List[dict],
    video_path: Path,
    backends: Dict[str, object],
    max_samples: int = 10
) -> Dict[str, Dict]:
    """
    Analyze Re-ID similarity across frames for a track.
    
    Returns:
        Dictionary mapping backend name -> {
            "embeddings": list of embeddings,
            "similarities": matrix of pairwise similarities,
            "mean_similarity": average similarity,
            "min_similarity": minimum similarity
        }
    """
    # Sort by frame_id and sample evenly
    sorted_dets = sorted(track_dets, key=lambda d: d.get("frame_id", 0))
    if len(sorted_dets) > max_samples:
        indices = np.linspace(0, len(sorted_dets) - 1, max_samples, dtype=int)
        sorted_dets = [sorted_dets[i] for i in indices]
    
    # Extract crops
    crops = []
    frame_ids = []
    for det in sorted_dets:
        bbox = det.get("bbox_2d") or det.get("bbox")
        if not bbox:
            continue
        if isinstance(bbox, dict):
            bbox = (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"])
        
        frame_id = det.get("frame_id", 0)
        crop = extract_crop(video_path, frame_id, bbox)
        if crop is not None and crop.size > 0:
            crops.append(crop)
            frame_ids.append(frame_id)
    
    if len(crops) < 2:
        logger.warning("Not enough valid crops to analyze")
        return {}
    
    results = {}
    
    for backend_name, model in backends.items():
        logger.info(f"  Computing {backend_name} embeddings for {len(crops)} crops...")
        
        embeddings = []
        for crop in crops:
            if backend_name == "clip":
                emb = compute_clip_embedding(crop, model)
            elif backend_name == "dino":
                emb = compute_dino_embedding(crop, model)
            elif backend_name == "vjepa2":
                emb = compute_vjepa2_embedding(crop, model)
            else:
                continue
            embeddings.append(emb)
        
        # Compute pairwise similarities
        n = len(embeddings)
        similarities = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                similarities[i, j] = cosine_similarity(embeddings[i], embeddings[j])
        
        # Get off-diagonal stats (exclude self-comparisons)
        off_diag = similarities[~np.eye(n, dtype=bool)]
        
        results[backend_name] = {
            "embeddings": embeddings,
            "similarities": similarities,
            "frame_ids": frame_ids,
            "mean_similarity": float(np.mean(off_diag)),
            "min_similarity": float(np.min(off_diag)),
            "max_similarity": float(np.max(off_diag)),
            "std_similarity": float(np.std(off_diag))
        }
    
    return results


def print_similarity_matrix(
    results: Dict[str, Dict],
    backend_name: str,
    track_info: str
):
    """Pretty-print the similarity matrix for a backend."""
    if backend_name not in results:
        return
    
    data = results[backend_name]
    sims = data["similarities"]
    frame_ids = data["frame_ids"]
    n = len(frame_ids)
    
    print(f"\n{'='*60}")
    print(f"{backend_name.upper()} Similarity Matrix for {track_info}")
    print(f"{'='*60}")
    
    # Header row
    header = "Frame |" + " | ".join([f"F{fid:>4}" for fid in frame_ids])
    print(header)
    print("-" * len(header))
    
    # Matrix rows
    for i in range(n):
        row = f"F{frame_ids[i]:>4} |"
        for j in range(n):
            sim = sims[i, j]
            if i == j:
                row += "  ---- |"
            elif sim < 0.5:
                row += f" {sim:.2f}* |"  # Mark low similarity
            else:
                row += f" {sim:.2f}  |"
        print(row)
    
    print(f"\nStats: mean={data['mean_similarity']:.3f}, min={data['min_similarity']:.3f}, "
          f"max={data['max_similarity']:.3f}, std={data['std_similarity']:.3f}")
    
    # Recommendation
    if data["min_similarity"] < 0.5:
        print(f"⚠️  WARNING: Min similarity {data['min_similarity']:.3f} is below typical Re-ID threshold (0.5-0.6)")
        print("   This explains why Re-ID may be failing for this object.")


def main():
    parser = argparse.ArgumentParser(description="Diagnose Re-ID failures")
    parser.add_argument("--results", type=str, required=True, help="Path to results directory")
    parser.add_argument("--video", type=str, required=True, help="Path to source video")
    parser.add_argument("--track-id", type=int, help="Specific track ID to analyze")
    parser.add_argument("--class-name", type=str, help="Analyze all tracks of this class")
    parser.add_argument("--compare-backends", action="store_true", help="Compare CLIP vs V-JEPA2")
    parser.add_argument("--backend", type=str, default="vjepa2", choices=["clip", "vjepa2", "all"],
                        help="Which backend(s) to use")
    parser.add_argument("--max-samples", type=int, default=10, help="Max samples per track")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/mps/cpu)")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    video_path = Path(args.video)
    
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)
    
    # Load tracks
    logger.info(f"Loading tracks from {results_dir}")
    detections = load_tracks(results_dir)
    logger.info(f"Loaded {len(detections)} detections")
    
    # Group tracks
    by_track = group_by_track(detections)
    by_class = group_by_class(detections)
    
    logger.info(f"Found {len(by_track)} unique tracks")
    logger.info(f"Classes: {list(by_class.keys())}")
    
    # Initialize backends
    backends = {}
    device = args.device
    
    if args.backend in ["clip", "all"]:
        try:
            from orion.backends.clip_backend import CLIPEmbedder
            backends["clip"] = CLIPEmbedder(device=device)
            logger.info("✓ CLIP backend loaded")
        except Exception as e:
            logger.warning(f"Failed to load CLIP: {e}")
    
    if args.backend in ["vjepa2", "all"]:
        try:
            from orion.backends.vjepa2_backend import VJepa2Embedder
            backends["vjepa2"] = VJepa2Embedder(device=device)
            logger.info("✓ V-JEPA2 backend loaded")
        except Exception as e:
            logger.warning(f"Failed to load V-JEPA2: {e}")
    
    if not backends:
        logger.error("No backends available!")
        sys.exit(1)
    
    # Determine which tracks to analyze
    tracks_to_analyze = []
    
    if args.track_id is not None:
        if args.track_id in by_track:
            tracks_to_analyze.append((args.track_id, by_track[args.track_id]))
        else:
            logger.error(f"Track ID {args.track_id} not found. Available: {list(by_track.keys())[:20]}")
            sys.exit(1)
    
    elif args.class_name:
        class_name = args.class_name.lower()
        for tid, dets in by_track.items():
            if dets and dets[0].get("class_name", "").lower() == class_name:
                tracks_to_analyze.append((tid, dets))
        if not tracks_to_analyze:
            logger.error(f"No tracks found for class '{args.class_name}'")
            sys.exit(1)
    
    else:
        # Default: analyze top 5 longest tracks
        sorted_tracks = sorted(by_track.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        tracks_to_analyze = sorted_tracks
    
    # Analyze each track
    print("\n" + "="*80)
    print("RE-ID DIAGNOSTIC REPORT")
    print("="*80)
    
    all_results = {}
    
    for track_id, track_dets in tracks_to_analyze:
        class_name = track_dets[0].get("class_name", "unknown") if track_dets else "unknown"
        track_info = f"Track {track_id} ({class_name}, {len(track_dets)} detections)"
        
        logger.info(f"\nAnalyzing {track_info}...")
        
        results = analyze_track(
            track_dets,
            video_path,
            backends,
            max_samples=args.max_samples
        )
        
        all_results[track_id] = results
        
        # Print results for each backend
        for backend_name in backends:
            print_similarity_matrix(results, backend_name, track_info)
    
    # Summary comparison
    if len(backends) > 1 and all_results:
        print("\n" + "="*80)
        print("BACKEND COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Track':<15} | {'Backend':<10} | {'Mean Sim':<10} | {'Min Sim':<10} | {'Verdict'}")
        print("-" * 70)
        
        for track_id, results in all_results.items():
            for backend_name, data in results.items():
                mean_sim = data["mean_similarity"]
                min_sim = data["min_similarity"]
                
                if min_sim >= 0.6:
                    verdict = "✅ Good"
                elif min_sim >= 0.5:
                    verdict = "⚠️ Marginal"
                else:
                    verdict = "❌ Failing"
                
                print(f"T{track_id:<13} | {backend_name:<10} | {mean_sim:<10.3f} | {min_sim:<10.3f} | {verdict}")
        
        # Recommendations
        print("\n" + "-"*70)
        print("RECOMMENDATIONS:")
        
        # Find best backend
        backend_scores = defaultdict(list)
        for results in all_results.values():
            for backend_name, data in results.items():
                backend_scores[backend_name].append(data["min_similarity"])
        
        best_backend = max(backend_scores.items(), key=lambda x: np.mean(x[1]))
        print(f"  • Best backend: {best_backend[0]} (avg min similarity: {np.mean(best_backend[1]):.3f})")
        
        if "vjepa2" in backend_scores:
            vjepa2_mean = np.mean(backend_scores["vjepa2"])
            clip_mean = np.mean(backend_scores.get("clip", [0]))
            if vjepa2_mean > clip_mean:
                print(f"  • V-JEPA2 outperforms CLIP by {(vjepa2_mean - clip_mean)*100:.1f}% on min similarity")
            else:
                print(f"  • CLIP outperforms V-JEPA2 by {(clip_mean - vjepa2_mean)*100:.1f}% on min similarity")


if __name__ == "__main__":
    main()
