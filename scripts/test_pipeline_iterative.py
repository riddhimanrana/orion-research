#!/usr/bin/env python3
"""
Orion Pipeline Test - Iterative Development
============================================

Test the full Orion pipeline on Lambda A10 GPU with timing diagnostics.
Uses the existing PerceptionEngine instead of raw components.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResults:
    """Pipeline test results."""
    video_path: str = ""
    video_duration_s: float = 0.0
    total_frames: int = 0
    frames_processed: int = 0
    
    # Counts
    raw_detections: int = 0
    valid_tracks: int = 0
    filtered_tracks: int = 0
    
    # Timing
    total_time_s: float = 0.0
    detection_time_s: float = 0.0
    embedding_time_s: float = 0.0
    tracking_time_s: float = 0.0
    filtering_time_s: float = 0.0
    
    # Performance
    detection_fps: float = 0.0
    overall_fps: float = 0.0
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)


def test_yoloworld_detection(video_path: str, target_fps: float = 8.0,
                              model: str = "yolov8x-worldv2.pt",
                              vocab: Optional[List[str]] = None) -> TestResults:
    """
    Test Stage 1: YOLO-World detection only.
    
    This tests the core detection speed and accuracy.
    """
    import cv2
    import numpy as np
    from ultralytics import YOLO
    import torch
    
    results = TestResults(video_path=video_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🎯 Testing YOLO-World detection on {device}")
    logger.info(f"   Model: {model}")
    
    # Load model
    t0 = time.time()
    yolo = YOLO(model)
    yolo.to(device)
    model_load_time = time.time() - t0
    logger.info(f"   Model loaded in {model_load_time:.2f}s")
    
    # Set vocabulary
    if vocab is None:
        vocab = [
            # People
            "person", "man", "woman", "child",
            # Furniture
            "chair", "couch", "sofa", "table", "desk", "bed",
            # Electronics
            "laptop", "computer", "monitor", "screen", "keyboard", "mouse", "phone", "tv", "television",
            # Containers
            "bottle", "cup", "mug", "glass", "bowl", "plate",
            # Personal items
            "bag", "backpack", "purse", "wallet", "keys", "glasses", "watch",
            # Office
            "book", "paper", "pen", "lamp", "clock",
            # Kitchen
            "refrigerator", "microwave", "oven", "sink", "cabinet",
            # Misc
            "plant", "pillow", "blanket", "remote", "door", "window"
        ]
    
    logger.info(f"   Vocabulary: {len(vocab)} classes")
    yolo.set_classes(vocab)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(1, int(video_fps / target_fps))
    
    results.total_frames = total_frames
    results.video_duration_s = total_frames / video_fps
    
    logger.info(f"📹 Video: {video_fps:.1f}fps, {total_frames} frames, {results.video_duration_s:.1f}s")
    logger.info(f"   Processing at {target_fps}fps (skip={frame_skip})")
    
    # Process frames
    frame_times = []
    class_counts = {}
    frame_idx = 0
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip == 0:
            t0 = time.time()
            
            # Run detection
            det_results = yolo(frame, conf=0.15, iou=0.45, verbose=False)
            
            frame_time = time.time() - t0
            frame_times.append(frame_time)
            
            # Count detections
            for r in det_results:
                for i in range(len(r.boxes)):
                    cls_id = int(r.boxes.cls[i])
                    cls_name = r.names.get(cls_id, "unknown")
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    results.raw_detections += 1
            
            results.frames_processed += 1
            
            if results.frames_processed % 100 == 0:
                avg_ms = np.mean(frame_times[-100:]) * 1000
                logger.info(f"   Frame {frame_idx}/{total_frames}: {avg_ms:.1f}ms/frame, {len(det_results[0].boxes)} dets")
        
        frame_idx += 1
    
    cap.release()
    
    results.detection_time_s = time.time() - start_time
    results.total_time_s = results.detection_time_s
    results.detection_fps = results.frames_processed / results.detection_time_s
    results.overall_fps = results.frames_processed / results.total_time_s
    
    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info("DETECTION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Frames processed: {results.frames_processed}")
    logger.info(f"Total detections: {results.raw_detections}")
    logger.info(f"Avg detections/frame: {results.raw_detections / max(1, results.frames_processed):.1f}")
    logger.info(f"Detection time: {results.detection_time_s:.2f}s")
    logger.info(f"Detection FPS: {results.detection_fps:.1f}")
    logger.info(f"\nTop classes detected:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1])[:10]:
        logger.info(f"  {cls}: {count}")
    
    return results


def test_vjepa2_embedding(video_path: str, max_crops: int = 100) -> Dict[str, Any]:
    """
    Test Stage 2: V-JEPA2 embedding speed.
    
    Extracts random crops and measures embedding throughput.
    """
    import cv2
    import numpy as np
    import torch
    
    logger.info(f"🧠 Testing V-JEPA2 embedding speed...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load embedder
    t0 = time.time()
    from orion.backends.vjepa2_backend import VJepa2Embedder
    embedder = VJepa2Embedder(device=device)
    load_time = time.time() - t0
    logger.info(f"   Model loaded in {load_time:.2f}s on {device}")
    
    # Extract random crops from video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    crops = []
    frame_indices = np.random.choice(total_frames, min(max_crops, total_frames), replace=False)
    
    for idx in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Random crop (simulating detection)
            h, w = frame.shape[:2]
            x1 = np.random.randint(0, w // 2)
            y1 = np.random.randint(0, h // 2)
            x2 = np.random.randint(w // 2, w)
            y2 = np.random.randint(h // 2, h)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
    
    cap.release()
    logger.info(f"   Extracted {len(crops)} random crops")
    
    # Embed crops
    embed_times = []
    embeddings = []
    
    start_time = time.time()
    for crop in crops:
        t0 = time.time()
        emb = embedder.embed_single_image(crop)
        embed_times.append(time.time() - t0)
        embeddings.append(emb)
    
    total_time = time.time() - start_time
    
    results = {
        "crops_embedded": len(crops),
        "total_time_s": total_time,
        "avg_time_ms": np.mean(embed_times) * 1000,
        "throughput_per_s": len(crops) / total_time,
        "embedding_dim": embeddings[0].shape[-1] if embeddings else 0
    }
    
    logger.info(f"\n{'='*60}")
    logger.info("EMBEDDING RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Crops embedded: {results['crops_embedded']}")
    logger.info(f"Total time: {results['total_time_s']:.2f}s")
    logger.info(f"Avg per crop: {results['avg_time_ms']:.1f}ms")
    logger.info(f"Throughput: {results['throughput_per_s']:.1f} crops/s")
    logger.info(f"Embedding dim: {results['embedding_dim']}")
    
    return results


def test_full_perception_engine(video_path: str, episode_id: str = "test_run",
                                 target_fps: float = 8.0,
                                 skip_filtering: bool = False) -> Dict[str, Any]:
    """
    Test the full PerceptionEngine pipeline.
    """
    logger.info(f"🚀 Testing full PerceptionEngine pipeline...")
    
    from orion.perception.config import PerceptionConfig, DetectionConfig, EmbeddingConfig
    from orion.perception.engine import PerceptionEngine
    
    # Configure for YOLO-World + V-JEPA2
    config = PerceptionConfig(
        detection=DetectionConfig(
            backend="yoloworld",
            yoloworld_model="yolov8x-worldv2.pt",
            confidence_threshold=0.15,
            iou_threshold=0.45,
        ),
        embedding=EmbeddingConfig(
            backend="vjepa2",
            embedding_dim=1024,
        ),
        target_fps=target_fps,
        enable_tracking=True,
        enable_3d=False,  # Skip depth for speed test
    )
    
    logger.info(f"   Config: {config.detection.backend} + {config.embedding.backend}")
    
    # Run engine
    engine = PerceptionEngine(config=config)
    
    start_time = time.time()
    result = engine.process_video(video_path, episode_id=episode_id)
    total_time = time.time() - start_time
    
    # Extract metrics
    metrics = result.metrics or {}
    
    results = {
        "video_path": video_path,
        "total_time_s": total_time,
        "total_frames": result.total_frames,
        "processing_time_s": result.processing_time_seconds,
        "entities": len(result.entities),
        "observations": len(result.raw_observations) if result.raw_observations else 0,
        "fps": result.total_frames / total_time,
        "metrics": metrics
    }
    
    logger.info(f"\n{'='*60}")
    logger.info("PERCEPTION ENGINE RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {results['total_time_s']:.2f}s")
    logger.info(f"Frames: {results['total_frames']}")
    logger.info(f"Entities: {results['entities']}")
    logger.info(f"FPS: {results['fps']:.1f}")
    
    return results


def run_gemini_analysis(video_path: str, tracks_path: str, 
                        sample_frames: int = 5) -> Dict[str, Any]:
    """
    Use Gemini to analyze detection quality.
    """
    import cv2
    import numpy as np
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set, skipping Gemini analysis")
        return {"skipped": True, "reason": "no_api_key"}
    
    try:
        import google.generativeai as genai
        import PIL.Image
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        # Load tracks
        tracks_by_frame = {}
        if os.path.exists(tracks_path):
            with open(tracks_path) as f:
                for line in f:
                    track = json.loads(line)
                    frame_idx = track.get("frame_idx", 0)
                    if frame_idx not in tracks_by_frame:
                        tracks_by_frame[frame_idx] = []
                    tracks_by_frame[frame_idx].append(track)
        
        cap = cv2.VideoCapture(video_path)
        
        # Sample frames
        frames_with_tracks = sorted(tracks_by_frame.keys())
        if not frames_with_tracks:
            return {"skipped": True, "reason": "no_tracks"}
        
        sample_indices = np.linspace(0, len(frames_with_tracks)-1, 
                                     min(sample_frames, len(frames_with_tracks)), 
                                     dtype=int)
        
        analyses = []
        
        for i in sample_indices:
            frame_idx = frames_with_tracks[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            tracks = tracks_by_frame[frame_idx]
            
            # Draw tracks
            for track in tracks:
                bbox = track.get("bbox") or track.get("bbox_2d", [])
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = [int(c) for c in bbox[:4]]
                    tid = track.get("track_id") or track.get("id", 0)
                    label = track.get("class_name") or track.get("category", "")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label}:{tid}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = PIL.Image.fromarray(frame_rgb)
            
            prompt = f"""Analyze this video frame with {len(tracks)} detected objects (green boxes).

1. Are the bounding boxes correctly placed around real objects?
2. Any obvious missed detections?
3. Any false positives (boxes around non-objects)?
4. Overall quality rating 1-10?

Be concise."""

            try:
                response = model.generate_content([prompt, pil_img])
                analyses.append({
                    "frame_idx": frame_idx,
                    "num_tracks": len(tracks),
                    "analysis": response.text
                })
            except Exception as e:
                analyses.append({
                    "frame_idx": frame_idx,
                    "error": str(e)
                })
        
        cap.release()
        
        return {
            "analyses": analyses,
            "frames_analyzed": len(analyses)
        }
        
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Orion Pipeline Tests")
    parser.add_argument("--video", default="data/examples/test.mp4")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--test", choices=["detection", "embedding", "full", "gemini", "all"],
                        default="detection")
    parser.add_argument("--output", default="results/pipeline_test")
    parser.add_argument("--episode", default="test_run")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.test in ["detection", "all"]:
        det_results = test_yoloworld_detection(args.video, args.fps)
        with open(f"{args.output}/detection_results.json", "w") as f:
            json.dump(det_results.to_dict(), f, indent=2)
    
    if args.test in ["embedding", "all"]:
        emb_results = test_vjepa2_embedding(args.video)
        with open(f"{args.output}/embedding_results.json", "w") as f:
            json.dump(emb_results, f, indent=2)
    
    if args.test in ["full", "all"]:
        full_results = test_full_perception_engine(args.video, args.episode, args.fps)
        with open(f"{args.output}/full_results.json", "w") as f:
            json.dump(full_results, f, indent=2)
    
    if args.test in ["gemini", "all"]:
        tracks_path = f"results/{args.episode}/tracks.jsonl"
        gemini_results = run_gemini_analysis(args.video, tracks_path)
        with open(f"{args.output}/gemini_results.json", "w") as f:
            json.dump(gemini_results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to {args.output}/")


if __name__ == "__main__":
    main()
