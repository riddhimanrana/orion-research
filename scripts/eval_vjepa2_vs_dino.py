#!/usr/bin/env python3
"""
V-JEPA2 vs DINOv2 Re-ID Evaluation
==================================

Compare V-JEPA2 and DINOv2 embedders for Re-ID quality on existing tracks.

This script:
1. Loads tracks from a completed Phase 1 run
2. Extracts crops for each track
3. Embeds with both V-JEPA2 and DINOv2
4. Computes intra-track (same object) and inter-track (different objects) similarities
5. Evaluates which embedder has better separation

Usage:
    python scripts/eval_vjepa2_vs_dino.py --episode phase1_test_v2
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_tracks(tracks_path: Path) -> List[Dict]:
    """Load tracks from JSONL file."""
    tracks = []
    with open(tracks_path, 'r') as f:
        for line in f:
            if line.strip():
                tracks.append(json.loads(line))
    return tracks


def group_by_track_id(tracks: List[Dict]) -> Dict[int, List[Dict]]:
    """Group track records by track_id."""
    by_id = defaultdict(list)
    for t in tracks:
        track_id = t.get('track_id', t.get('id'))
        if track_id is not None:
            by_id[track_id].append(t)
    return dict(by_id)


def extract_crop(frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
    """Extract crop from frame using bbox."""
    if bbox is None or len(bbox) != 4:
        return None
    
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def load_dino_embedder(device: str = "mps"):
    """Load DINOv2 embedder."""
    from orion.backends.dino_backend import DINOEmbedder
    return DINOEmbedder(model_name="facebook/dinov2-base", device=device)


def load_vjepa2_embedder(device: str = "cuda"):
    """Load V-JEPA2 embedder (if available)."""
    try:
        from orion.backends.vjepa2_backend import VJepa2Embedder
        embedder = VJepa2Embedder(device=device)
        # Test load
        embedder._ensure_loaded()
        return embedder
    except Exception as e:
        logger.warning(f"V-JEPA2 not available: {e}")
        return None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < 1e-8 or b_norm < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def compute_embeddings(
    video_path: Path,
    tracks_by_id: Dict[int, List[Dict]],
    embedder,
    embedder_name: str,
    max_tracks: int = 50,
    samples_per_track: int = 5
) -> Dict[int, List[np.ndarray]]:
    """
    Compute embeddings for crops from each track.
    
    Returns:
        Dict mapping track_id -> list of embeddings
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    
    # Select subset of tracks
    track_ids = list(tracks_by_id.keys())[:max_tracks]
    
    # Collect frame_ids needed
    frame_to_tracks = defaultdict(list)
    for tid in track_ids:
        observations = tracks_by_id[tid]
        # Sample evenly across track
        n = len(observations)
        indices = np.linspace(0, n - 1, min(n, samples_per_track), dtype=int)
        for idx in indices:
            obs = observations[idx]
            frame_id = obs.get('frame_id', obs.get('frame_idx'))
            if frame_id is not None:
                frame_to_tracks[frame_id].append((tid, obs))
    
    # Read frames and compute embeddings
    embeddings_by_track: Dict[int, List[np.ndarray]] = defaultdict(list)
    
    frame_ids_sorted = sorted(frame_to_tracks.keys())
    logger.info(f"[{embedder_name}] Processing {len(frame_ids_sorted)} frames for {len(track_ids)} tracks")
    
    for frame_id in frame_ids_sorted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue
        
        for (tid, obs) in frame_to_tracks[frame_id]:
            bbox = obs.get('bbox', obs.get('bbox_2d'))
            crop = extract_crop(frame, bbox)
            if crop is None or crop.size == 0:
                continue
            
            # Resize to consistent size for embedding
            crop_resized = cv2.resize(crop, (224, 224))
            
            try:
                if hasattr(embedder, 'embed_single_image'):
                    # V-JEPA2 interface
                    emb = embedder.embed_single_image(crop_resized)
                    emb = emb.numpy().flatten()
                elif hasattr(embedder, 'encode_image'):
                    # DINO interface
                    emb = embedder.encode_image(crop_resized)
                    if hasattr(emb, 'numpy'):
                        emb = emb.numpy()
                    emb = emb.flatten()
                else:
                    continue
                
                embeddings_by_track[tid].append(emb)
            except Exception as e:
                logger.debug(f"Embedding failed for track {tid}: {e}")
                continue
    
    cap.release()
    return dict(embeddings_by_track)


def evaluate_embeddings(embeddings_by_track: Dict[int, List[np.ndarray]]) -> Dict:
    """
    Evaluate Re-ID quality of embeddings.
    
    Returns:
        Dict with metrics:
        - intra_similarities: similarities within same track
        - inter_similarities: similarities between different tracks
        - separation: mean(intra) - mean(inter) (higher is better)
    """
    track_ids = list(embeddings_by_track.keys())
    if len(track_ids) < 2:
        return {"error": "Need at least 2 tracks for evaluation"}
    
    intra_sims = []
    inter_sims = []
    
    # Compute mean embedding per track for efficiency
    track_means = {}
    for tid, embs in embeddings_by_track.items():
        if len(embs) > 0:
            track_means[tid] = np.mean(embs, axis=0)
    
    # Intra-track similarity (same object, different frames)
    for tid, embs in embeddings_by_track.items():
        if len(embs) < 2:
            continue
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                sim = cosine_sim(embs[i], embs[j])
                intra_sims.append(sim)
    
    # Inter-track similarity (different objects)
    for i, tid1 in enumerate(track_ids):
        for tid2 in track_ids[i + 1:]:
            if tid1 not in track_means or tid2 not in track_means:
                continue
            sim = cosine_sim(track_means[tid1], track_means[tid2])
            inter_sims.append(sim)
    
    if len(intra_sims) == 0 or len(inter_sims) == 0:
        return {"error": "Insufficient data for evaluation"}
    
    intra_mean = np.mean(intra_sims)
    intra_std = np.std(intra_sims)
    inter_mean = np.mean(inter_sims)
    inter_std = np.std(inter_sims)
    separation = intra_mean - inter_mean
    
    # Compute optimal threshold (midpoint)
    optimal_threshold = (intra_mean + inter_mean) / 2
    
    # Compute accuracy at common thresholds
    def accuracy_at_threshold(thresh):
        tp = sum(1 for s in intra_sims if s >= thresh)  # Same track, above threshold
        tn = sum(1 for s in inter_sims if s < thresh)   # Different track, below threshold
        fp = sum(1 for s in inter_sims if s >= thresh)  # Different track, above threshold
        fn = sum(1 for s in intra_sims if s < thresh)   # Same track, below threshold
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return {"accuracy": accuracy, "precision": precision, "recall": recall}
    
    return {
        "intra_similarity": {"mean": intra_mean, "std": intra_std, "count": len(intra_sims)},
        "inter_similarity": {"mean": inter_mean, "std": inter_std, "count": len(inter_sims)},
        "separation": separation,
        "optimal_threshold": optimal_threshold,
        "metrics_at_0.70": accuracy_at_threshold(0.70),
        "metrics_at_0.75": accuracy_at_threshold(0.75),
        "metrics_at_optimal": accuracy_at_threshold(optimal_threshold),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate V-JEPA2 vs DINOv2 for Re-ID")
    parser.add_argument("--episode", "-e", required=True, help="Episode ID")
    parser.add_argument("--video", help="Video path (auto-detected if not provided)")
    parser.add_argument("--max-tracks", type=int, default=50, help="Max tracks to evaluate")
    parser.add_argument("--samples-per-track", type=int, default=5, help="Samples per track")
    parser.add_argument("--device", default="mps", help="Device (cuda/mps/cpu)")
    args = parser.parse_args()
    
    # Find results
    results_dir = Path(f"results/{args.episode}")
    tracks_path = results_dir / "tracks.jsonl"
    
    if not tracks_path.exists():
        logger.error(f"Tracks not found at {tracks_path}")
        sys.exit(1)
    
    # Find video
    if args.video:
        video_path = Path(args.video)
    else:
        from orion.config import get_episode_video_path
        video_path = get_episode_video_path(args.episode)
        if video_path is None:
            logger.error("Could not find video for episode. Use --video to specify.")
            sys.exit(1)
    
    logger.info(f"Loading tracks from {tracks_path}")
    tracks = load_tracks(tracks_path)
    tracks_by_id = group_by_track_id(tracks)
    logger.info(f"Found {len(tracks_by_id)} unique tracks")
    
    results = {}
    
    # Evaluate DINOv2
    logger.info("\n=== Evaluating DINOv2 ===")
    try:
        dino = load_dino_embedder(device=args.device)
        t0 = time.time()
        dino_embeddings = compute_embeddings(
            video_path, tracks_by_id, dino, "DINOv2",
            max_tracks=args.max_tracks, samples_per_track=args.samples_per_track
        )
        dino_time = time.time() - t0
        dino_metrics = evaluate_embeddings(dino_embeddings)
        dino_metrics["embedding_time_sec"] = dino_time
        results["dinov2"] = dino_metrics
        
        logger.info(f"DINOv2 Results:")
        logger.info(f"  Intra-track similarity: {dino_metrics['intra_similarity']['mean']:.3f} ± {dino_metrics['intra_similarity']['std']:.3f}")
        logger.info(f"  Inter-track similarity: {dino_metrics['inter_similarity']['mean']:.3f} ± {dino_metrics['inter_similarity']['std']:.3f}")
        logger.info(f"  Separation (intra - inter): {dino_metrics['separation']:.3f}")
        logger.info(f"  Optimal threshold: {dino_metrics['optimal_threshold']:.3f}")
        logger.info(f"  Accuracy @ 0.70: {dino_metrics['metrics_at_0.70']['accuracy']:.1%}")
        logger.info(f"  Time: {dino_time:.1f}s")
    except Exception as e:
        logger.error(f"DINOv2 evaluation failed: {e}")
        results["dinov2"] = {"error": str(e)}
    
    # Evaluate V-JEPA2
    logger.info("\n=== Evaluating V-JEPA2 ===")
    try:
        vjepa2 = load_vjepa2_embedder(device=args.device)
        if vjepa2 is not None:
            t0 = time.time()
            vjepa2_embeddings = compute_embeddings(
                video_path, tracks_by_id, vjepa2, "V-JEPA2",
                max_tracks=args.max_tracks, samples_per_track=args.samples_per_track
            )
            vjepa2_time = time.time() - t0
            vjepa2_metrics = evaluate_embeddings(vjepa2_embeddings)
            vjepa2_metrics["embedding_time_sec"] = vjepa2_time
            results["vjepa2"] = vjepa2_metrics
            
            logger.info(f"V-JEPA2 Results:")
            logger.info(f"  Intra-track similarity: {vjepa2_metrics['intra_similarity']['mean']:.3f} ± {vjepa2_metrics['intra_similarity']['std']:.3f}")
            logger.info(f"  Inter-track similarity: {vjepa2_metrics['inter_similarity']['mean']:.3f} ± {vjepa2_metrics['inter_similarity']['std']:.3f}")
            logger.info(f"  Separation (intra - inter): {vjepa2_metrics['separation']:.3f}")
            logger.info(f"  Optimal threshold: {vjepa2_metrics['optimal_threshold']:.3f}")
            logger.info(f"  Accuracy @ 0.70: {vjepa2_metrics['metrics_at_0.70']['accuracy']:.1%}")
            logger.info(f"  Time: {vjepa2_time:.1f}s")
        else:
            logger.warning("V-JEPA2 not available, skipping")
            results["vjepa2"] = {"error": "Not available"}
    except Exception as e:
        logger.error(f"V-JEPA2 evaluation failed: {e}")
        results["vjepa2"] = {"error": str(e)}
    
    # Summary comparison
    logger.info("\n=== Summary ===")
    if "error" not in results.get("dinov2", {}) and "error" not in results.get("vjepa2", {}):
        dino_sep = results["dinov2"]["separation"]
        vjepa2_sep = results["vjepa2"]["separation"]
        winner = "V-JEPA2" if vjepa2_sep > dino_sep else "DINOv2"
        logger.info(f"Better separation: {winner}")
        logger.info(f"  DINOv2:  {dino_sep:.3f}")
        logger.info(f"  V-JEPA2: {vjepa2_sep:.3f}")
        
        # Speed comparison
        dino_time = results["dinov2"].get("embedding_time_sec", 0)
        vjepa2_time = results["vjepa2"].get("embedding_time_sec", 0)
        faster = "DINOv2" if dino_time < vjepa2_time else "V-JEPA2"
        logger.info(f"Faster: {faster} ({min(dino_time, vjepa2_time):.1f}s vs {max(dino_time, vjepa2_time):.1f}s)")
    
    # Save results
    output_path = results_dir / "embedder_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
