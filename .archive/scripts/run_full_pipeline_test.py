#!/usr/bin/env python3
"""
Full Pipeline Test with Diagnostics
====================================

Tests YOLO-World (yolov8x open-vocab) + V-JEPA2 Re-ID + FastVLM Semantic Filtering
with detailed timing, accuracy metrics, and Gemini validation.

Target: Real-time on A10 GPU (~24fps for perception)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Timing and quality metrics for pipeline stages."""
    # Timing (seconds)
    detection_time: float = 0.0
    embedding_time: float = 0.0
    tracking_time: float = 0.0
    filtering_time: float = 0.0
    total_time: float = 0.0
    
    # Counts
    frames_processed: int = 0
    raw_detections: int = 0
    tracked_objects: int = 0
    filtered_objects: int = 0
    final_tracks: int = 0
    
    # Performance
    avg_fps: float = 0.0
    detection_fps: float = 0.0
    
    # Quality
    reid_avg_similarity: float = 0.0
    track_id_switches: int = 0
    avg_track_length: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


def get_device() -> str:
    """Auto-detect best device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_detection_stage(video_path: str, target_fps: float = 8.0, 
                        yoloworld_model: str = "yolov8x-worldv2.pt",
                        open_vocab: bool = True,
                        vocab_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Stage 1: YOLO-World detection with timing.
    
    Returns dict with detections per frame and timing info.
    """
    from ultralytics import YOLO
    
    device = get_device()
    logger.info(f"🎯 Loading YOLO-World: {yoloworld_model} on {device}")
    
    # Check if model exists, download if needed
    model_path = Path("models") / yoloworld_model
    if not model_path.exists():
        logger.info(f"Downloading {yoloworld_model}...")
        model = YOLO(yoloworld_model)  # Auto-downloads
    else:
        model = YOLO(str(model_path))
    
    model.to(device)
    
    # Set vocabulary for open-vocab mode
    if open_vocab and vocab_prompt:
        # Parse dot-separated prompt
        classes = [c.strip() for c in vocab_prompt.split('.') if c.strip()]
        logger.info(f"Setting open vocab classes: {len(classes)} categories")
        model.set_classes(classes)
    elif open_vocab:
        # Default indoor vocabulary
        default_vocab = [
            "person", "chair", "couch", "table", "desk", "laptop", "monitor", "keyboard",
            "phone", "bottle", "cup", "book", "remote", "lamp", "tv", "refrigerator",
            "microwave", "oven", "sink", "cabinet", "door", "window", "plant", "clock",
            "bag", "backpack", "umbrella", "shoe", "hat", "glasses", "watch", "jewelry"
        ]
        logger.info(f"Using default open vocab: {len(default_vocab)} categories")
        model.set_classes(default_vocab)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(1, int(video_fps / target_fps))
    
    logger.info(f"📹 Video: {video_fps:.1f}fps, {total_frames} frames, skip={frame_skip}")
    
    detections_by_frame = {}
    frame_times = []
    frame_idx = 0
    processed = 0
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip == 0:
            t0 = time.time()
            
            # Run inference
            results = model(frame, conf=0.15, iou=0.45, verbose=False)
            
            t1 = time.time()
            frame_times.append(t1 - t0)
            
            # Extract detections
            frame_dets = []
            for r in results:
                boxes = r.boxes
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    conf = float(boxes.conf[i])
                    cls_id = int(boxes.cls[i])
                    cls_name = r.names.get(cls_id, f"class_{cls_id}")
                    
                    frame_dets.append({
                        "bbox": bbox,
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "frame_idx": frame_idx
                    })
            
            detections_by_frame[frame_idx] = frame_dets
            processed += 1
            
            if processed % 50 == 0:
                avg_ms = np.mean(frame_times[-50:]) * 1000
                logger.info(f"  Frame {frame_idx}/{total_frames}: {len(frame_dets)} dets, {avg_ms:.1f}ms/frame")
        
        frame_idx += 1
    
    cap.release()
    total_time = time.time() - start_time
    
    # Compute stats
    total_dets = sum(len(d) for d in detections_by_frame.values())
    avg_per_frame = total_dets / max(processed, 1)
    detection_fps = processed / total_time
    
    logger.info(f"✅ Detection complete: {total_dets} detections in {processed} frames")
    logger.info(f"   Avg: {avg_per_frame:.1f} det/frame, {detection_fps:.1f} fps")
    
    return {
        "detections_by_frame": detections_by_frame,
        "total_detections": total_dets,
        "frames_processed": processed,
        "detection_time": total_time,
        "detection_fps": detection_fps,
        "avg_frame_time_ms": np.mean(frame_times) * 1000 if frame_times else 0,
        "video_fps": video_fps,
        "total_video_frames": total_frames
    }


def run_embedding_stage(video_path: str, detections_by_frame: Dict[int, List[Dict]],
                        batch_size: int = 16) -> Dict[str, Any]:
    """
    Stage 2: V-JEPA2 embeddings for Re-ID.
    """
    logger.info("🧠 Loading V-JEPA2 for embeddings...")
    
    from orion.backends.vjepa2_backend import VJepa2Embedder
    
    embedder = VJepa2Embedder(
        model_name="facebook/vjepa2-vitl-fpc64-256",
        device=get_device()
    )
    
    cap = cv2.VideoCapture(video_path)
    
    start_time = time.time()
    embedding_count = 0
    embeddings_by_frame = {}
    
    for frame_idx, dets in detections_by_frame.items():
        if not dets:
            continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Crop and embed each detection
        frame_embeddings = []
        for det in dets:
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                # Embed single crop using single image mode
                try:
                    emb = embedder.embed_single_image(crop)
                    frame_embeddings.append(emb.squeeze().numpy())
                    embedding_count += 1
                except Exception as e:
                    logger.warning(f"Embedding failed for crop: {e}")
                    frame_embeddings.append(None)
        
        embeddings_by_frame[frame_idx] = frame_embeddings
    
    cap.release()
    total_time = time.time() - start_time
    
    logger.info(f"✅ Embeddings: {embedding_count} in {total_time:.2f}s ({embedding_count/total_time:.1f}/s)")
    
    return {
        "embeddings_by_frame": embeddings_by_frame,
        "embedding_count": embedding_count,
        "embedding_time": total_time,
        "embedding_dim": embedder.embedding_dim
    }


def run_tracking_stage(detections_by_frame: Dict, embeddings_by_frame: Dict,
                       reid_threshold: float = 0.65) -> Dict[str, Any]:
    """
    Stage 3: Hungarian algorithm tracking with Re-ID.
    """
    logger.info(f"🔗 Running tracker (reid_threshold={reid_threshold})...")
    
    from orion.perception.trackers.enhanced import EnhancedTracker
    
    tracker = EnhancedTracker(
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        reid_threshold=reid_threshold,
        use_reid=True
    )
    
    start_time = time.time()
    
    tracks_by_frame = {}
    all_track_ids = set()
    
    # Sort frames
    sorted_frames = sorted(detections_by_frame.keys())
    
    for frame_idx in sorted_frames:
        dets = detections_by_frame[frame_idx]
        embeddings = embeddings_by_frame.get(frame_idx, [])
        
        if not dets:
            tracks_by_frame[frame_idx] = []
            continue
        
        # Build detection array for tracker
        det_array = []
        for i, det in enumerate(dets):
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cls_id = det["class_id"]
            det_array.append([x1, y1, x2, y2, conf, cls_id])
        
        det_array = np.array(det_array) if det_array else np.empty((0, 6))
        
        # Update tracker
        tracks = tracker.update(det_array, embeddings if embeddings else None)
        
        frame_tracks = []
        for track in tracks:
            x1, y1, x2, y2, track_id = track[:5]
            all_track_ids.add(int(track_id))
            frame_tracks.append({
                "track_id": int(track_id),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "frame_idx": frame_idx
            })
        
        tracks_by_frame[frame_idx] = frame_tracks
    
    total_time = time.time() - start_time
    
    # Compute track statistics
    track_lengths = {}
    for frame_idx, tracks in tracks_by_frame.items():
        for t in tracks:
            tid = t["track_id"]
            track_lengths[tid] = track_lengths.get(tid, 0) + 1
    
    avg_track_length = np.mean(list(track_lengths.values())) if track_lengths else 0
    
    logger.info(f"✅ Tracking: {len(all_track_ids)} unique tracks, avg length: {avg_track_length:.1f}")
    
    return {
        "tracks_by_frame": tracks_by_frame,
        "unique_tracks": len(all_track_ids),
        "tracking_time": total_time,
        "avg_track_length": avg_track_length,
        "track_lengths": track_lengths
    }


def run_semantic_filtering(video_path: str, tracks_by_frame: Dict, 
                           similarity_threshold: float = 0.4) -> Dict[str, Any]:
    """
    Stage 4: FastVLM descriptions + SentenceTransformer filtering.
    """
    logger.info("🔍 Running semantic filtering (FastVLM + SentenceTransformer)...")
    
    try:
        from orion.perception.filters import SemanticFilter
        
        filter = SemanticFilter(
            similarity_threshold=similarity_threshold,
            device=get_device()
        )
    except ImportError:
        logger.warning("SemanticFilter not available, skipping filtering stage")
        return {
            "filtered_tracks": tracks_by_frame,
            "filtering_time": 0,
            "rejected_count": 0
        }
    
    cap = cv2.VideoCapture(video_path)
    start_time = time.time()
    
    # Sample some tracks for filtering
    filtered_tracks = {}
    rejected_count = 0
    valid_count = 0
    
    for frame_idx, tracks in tracks_by_frame.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_filtered = []
        for track in tracks:
            x1, y1, x2, y2 = [int(c) for c in track["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                
                # Run filter (only on first frame of each track to save time)
                # In production this would be cached per track_id
                is_valid, score = filter.validate(crop)
                
                if is_valid:
                    track["filter_score"] = score
                    frame_filtered.append(track)
                    valid_count += 1
                else:
                    rejected_count += 1
        
        filtered_tracks[frame_idx] = frame_filtered
    
    cap.release()
    total_time = time.time() - start_time
    
    logger.info(f"✅ Filtering: {valid_count} valid, {rejected_count} rejected in {total_time:.2f}s")
    
    return {
        "filtered_tracks": filtered_tracks,
        "filtering_time": total_time,
        "valid_count": valid_count,
        "rejected_count": rejected_count
    }


def validate_with_gemini(video_path: str, tracks_by_frame: Dict, 
                         sample_frames: int = 5) -> Dict[str, Any]:
    """
    Use Gemini to validate tracking quality.
    """
    logger.info("🤖 Running Gemini validation...")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set, skipping validation")
        return {"validation_skipped": True}
    
    try:
        import PIL.Image
        from orion.utils.gemini_client import GeminiClientError, get_gemini_model

        model = get_gemini_model("gemini-2.0-flash-exp", api_key=api_key)
        
        cap = cv2.VideoCapture(video_path)
        
        # Sample frames with tracks
        frames_with_tracks = [f for f, t in tracks_by_frame.items() if t]
        sample_indices = np.linspace(0, len(frames_with_tracks)-1, sample_frames, dtype=int)
        sample_frame_ids = [frames_with_tracks[i] for i in sample_indices]
        
        validation_results = []
        
        for frame_idx in sample_frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            tracks = tracks_by_frame[frame_idx]
            
            # Draw tracks on frame
            annotated = frame.copy()
            for track in tracks:
                x1, y1, x2, y2 = [int(c) for c in track["bbox"]]
                tid = track["track_id"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"ID:{tid}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert to PIL
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_image = PIL.Image.fromarray(annotated_rgb)
            
            # Query Gemini
            prompt = f"""Analyze this video frame with {len(tracks)} tracked objects (green boxes with IDs).
            
Answer these questions:
1. Are the bounding boxes correctly placed around distinct objects?
2. Are there any obvious missed detections (objects without boxes)?
3. Are there any false positives (boxes around non-objects)?
4. Rate the overall detection quality from 1-10.

Be concise and specific."""

            try:
                response = model.generate_content([prompt, pil_image])
                validation_results.append({
                    "frame_idx": frame_idx,
                    "num_tracks": len(tracks),
                    "analysis": response.text
                })
            except Exception as e:
                logger.warning(f"Gemini error on frame {frame_idx}: {e}")
        
        cap.release()
        
        return {
            "validation_results": validation_results,
            "frames_validated": len(validation_results)
        }
        
    except Exception as e:
        logger.error(f"Gemini validation failed: {e}")
        return {"validation_error": str(e)}


def run_full_pipeline(video_path: str, target_fps: float = 8.0,
                      yoloworld_model: str = "yolov8x-worldv2.pt",
                      open_vocab: bool = True,
                      vocab_prompt: Optional[str] = None,
                      reid_threshold: float = 0.65,
                      filter_threshold: float = 0.4,
                      skip_filtering: bool = False,
                      skip_gemini: bool = False,
                      output_dir: str = "results/pipeline_test") -> PipelineMetrics:
    """
    Run the complete pipeline and return metrics.
    """
    logger.info("="*80)
    logger.info("ORION FULL PIPELINE TEST")
    logger.info("="*80)
    logger.info(f"Video: {video_path}")
    logger.info(f"Model: {yoloworld_model}, Open-vocab: {open_vocab}")
    logger.info(f"Target FPS: {target_fps}, Re-ID threshold: {reid_threshold}")
    logger.info("="*80)
    
    metrics = PipelineMetrics()
    start_time = time.time()
    
    # Stage 1: Detection
    logger.info("\n" + "="*40)
    logger.info("STAGE 1: YOLO-World Detection")
    logger.info("="*40)
    det_results = run_detection_stage(
        video_path, target_fps, yoloworld_model, open_vocab, vocab_prompt
    )
    metrics.detection_time = det_results["detection_time"]
    metrics.raw_detections = det_results["total_detections"]
    metrics.frames_processed = det_results["frames_processed"]
    metrics.detection_fps = det_results["detection_fps"]
    
    # Stage 2: Embeddings
    logger.info("\n" + "="*40)
    logger.info("STAGE 2: V-JEPA2 Embeddings")
    logger.info("="*40)
    emb_results = run_embedding_stage(
        video_path, det_results["detections_by_frame"]
    )
    metrics.embedding_time = emb_results["embedding_time"]
    
    # Stage 3: Tracking
    logger.info("\n" + "="*40)
    logger.info("STAGE 3: Hungarian Tracking + Re-ID")
    logger.info("="*40)
    track_results = run_tracking_stage(
        det_results["detections_by_frame"],
        emb_results["embeddings_by_frame"],
        reid_threshold
    )
    metrics.tracking_time = track_results["tracking_time"]
    metrics.tracked_objects = track_results["unique_tracks"]
    metrics.avg_track_length = track_results["avg_track_length"]
    
    # Stage 4: Semantic Filtering (optional)
    if not skip_filtering:
        logger.info("\n" + "="*40)
        logger.info("STAGE 4: Semantic Filtering")
        logger.info("="*40)
        filter_results = run_semantic_filtering(
            video_path, track_results["tracks_by_frame"], filter_threshold
        )
        metrics.filtering_time = filter_results["filtering_time"]
        metrics.filtered_objects = filter_results.get("rejected_count", 0)
        final_tracks = filter_results["filtered_tracks"]
    else:
        final_tracks = track_results["tracks_by_frame"]
    
    metrics.final_tracks = len(set(
        t["track_id"] for tracks in final_tracks.values() for t in tracks
    ))
    
    # Gemini validation (optional)
    if not skip_gemini:
        logger.info("\n" + "="*40)
        logger.info("STAGE 5: Gemini Validation")
        logger.info("="*40)
        gemini_results = validate_with_gemini(video_path, final_tracks)
    else:
        gemini_results = {"skipped": True}
    
    metrics.total_time = time.time() - start_time
    metrics.avg_fps = metrics.frames_processed / metrics.total_time
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    
    with open(f"{output_dir}/tracks.jsonl", "w") as f:
        for frame_idx, tracks in sorted(final_tracks.items()):
            for track in tracks:
                track["frame_idx"] = frame_idx
                f.write(json.dumps(track) + "\n")
    
    if not skip_gemini and "validation_results" in gemini_results:
        with open(f"{output_dir}/gemini_validation.json", "w") as f:
            json.dump(gemini_results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*80)
    logger.info(f"Total time: {metrics.total_time:.2f}s")
    logger.info(f"Frames processed: {metrics.frames_processed}")
    logger.info(f"Raw detections: {metrics.raw_detections}")
    logger.info(f"Final tracks: {metrics.final_tracks}")
    logger.info(f"")
    logger.info("Stage Timing:")
    logger.info(f"  Detection:  {metrics.detection_time:.2f}s ({metrics.detection_fps:.1f} fps)")
    logger.info(f"  Embedding:  {metrics.embedding_time:.2f}s")
    logger.info(f"  Tracking:   {metrics.tracking_time:.2f}s")
    logger.info(f"  Filtering:  {metrics.filtering_time:.2f}s")
    logger.info(f"")
    logger.info(f"Overall: {metrics.avg_fps:.1f} fps")
    logger.info(f"Target for real-time: 24+ fps")
    logger.info("="*80)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Full Pipeline Test")
    parser.add_argument("--video", default="data/examples/test.mp4", help="Video path")
    parser.add_argument("--fps", type=float, default=8.0, help="Target FPS")
    parser.add_argument("--model", default="yolov8x-worldv2.pt", help="YOLO-World model")
    parser.add_argument("--open-vocab", action="store_true", default=True, help="Use open vocabulary")
    parser.add_argument("--vocab-prompt", type=str, default=None, help="Custom vocabulary (dot-separated)")
    parser.add_argument("--reid-threshold", type=float, default=0.65, help="Re-ID similarity threshold")
    parser.add_argument("--filter-threshold", type=float, default=0.4, help="Semantic filter threshold")
    parser.add_argument("--skip-filtering", action="store_true", help="Skip semantic filtering")
    parser.add_argument("--skip-gemini", action="store_true", help="Skip Gemini validation")
    parser.add_argument("--output", default="results/pipeline_test", help="Output directory")
    
    args = parser.parse_args()
    
    run_full_pipeline(
        video_path=args.video,
        target_fps=args.fps,
        yoloworld_model=args.model,
        open_vocab=args.open_vocab,
        vocab_prompt=args.vocab_prompt,
        reid_threshold=args.reid_threshold,
        filter_threshold=args.filter_threshold,
        skip_filtering=args.skip_filtering,
        skip_gemini=args.skip_gemini,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
