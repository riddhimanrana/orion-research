#!/usr/bin/env python3
"""
Full Phase 1 Validation Script

Runs detection on ALL frames at target FPS and validates with Gemini on sampled frames.
Shows real-time progress and comprehensive statistics.

Usage:
    python scripts/full_validation.py --video data/examples/video.mp4 --episode phase1_full_v2
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_dotenv():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


def setup_gemini():
    """Initialize Gemini API."""
    load_dotenv()
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("WARNING: GOOGLE_API_KEY not found - skipping Gemini validation")
        return None
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai
    except ImportError:
        print("WARNING: google-generativeai not installed")
        return None


def run_detection_realtime(video_path: str, episode: str, device: str, fps: float, confidence: float):
    """Run detection with real-time progress output."""
    import cv2
    import torch
    
    print("=" * 80)
    print(f"PHASE 1 DETECTION: {video_path}")
    print("=" * 80)
    
    # Check device
    if device == "cuda":
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠ CUDA not available, falling back to CPU")
            device = "cpu"
    
    # Load video info
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / video_fps
    cap.release()
    
    # Calculate sampling
    sample_interval = max(1, int(video_fps / fps))
    expected_samples = total_frames // sample_interval
    
    print(f"\nVideo: {video_fps:.1f} FPS, {total_frames} frames, {duration:.1f}s")
    print(f"Resolution: {width}x{height}")
    print(f"Sampling: every {sample_interval} frames → ~{expected_samples} samples at {fps} FPS")
    
    # Setup output directory
    results_dir = Path(f"results/{episode}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save episode metadata
    meta = {
        "episode_id": episode,
        "video_path": str(Path(video_path).resolve()),
        "video": {
            "fps": video_fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_seconds": duration
        },
        "config": {
            "detector": "yolov8x-worldv2",
            "device": device,
            "target_fps": fps,
            "sample_interval": sample_interval,
            "confidence_threshold": confidence
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(results_dir / "episode_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    # Load detector
    print("\nLoading YOLO-World detector...")
    from orion.backends.yoloworld_backend import YOLOWorldDetector, YOLOWorldConfig
    from orion.perception.config import DetectionConfig
    
    det_config = DetectionConfig()
    yolo_config = YOLOWorldConfig(
        model=det_config.yoloworld_model,
        confidence=confidence,
        device=device
    )
    detector = YOLOWorldDetector(yolo_config)
    
    # Set custom classes from config
    classes = det_config.yoloworld_categories()
    detector.set_classes(classes)
    print(f"✓ Detector loaded with {len(classes)} classes")
    
    # Process video
    print("\nProcessing video...")
    cap = cv2.VideoCapture(video_path)
    
    tracks_file = open(results_dir / "tracks.jsonl", "w")
    
    frame_idx = 0
    processed = 0
    total_detections = 0
    track_counter = 0
    active_tracks = {}  # track_id -> last_bbox for simple IoU tracking
    
    start_time = time.time()
    last_print = start_time
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_interval == 0:
            # Run detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.detect(rgb_frame)
            
            timestamp = frame_idx / video_fps
            
            # Simple IoU-based tracking
            frame_tracks = []
            for det in detections:
                bbox = det["bbox"]
                label = det["class_name"]
                conf = det["confidence"]
                
                # Try to match with existing track
                best_track_id = None
                best_iou = 0.3  # threshold
                
                for tid, (last_bbox, last_label) in list(active_tracks.items()):
                    if last_label != label:
                        continue
                    iou = compute_iou(bbox, last_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_track_id = tid
                
                if best_track_id is None:
                    track_counter += 1
                    best_track_id = track_counter
                
                active_tracks[best_track_id] = (bbox, label)
                
                track_data = {
                    "frame_id": frame_idx,
                    "timestamp": round(timestamp, 3),
                    "track_id": best_track_id,
                    "bbox": [round(b, 1) for b in bbox],
                    "confidence": round(conf, 4),
                    "label": label,
                    "class_id": det.get("class_id", 0)
                }
                tracks_file.write(json.dumps(track_data) + "\n")
                frame_tracks.append(track_data)
                total_detections += 1
            
            processed += 1
            
            # Progress update every 2 seconds
            now = time.time()
            if now - last_print >= 2.0:
                elapsed = now - start_time
                fps_actual = processed / elapsed if elapsed > 0 else 0
                pct = (frame_idx / total_frames) * 100
                eta = (total_frames - frame_idx) / (frame_idx / elapsed) if frame_idx > 0 else 0
                
                print(f"  [{pct:5.1f}%] Frame {frame_idx:5d}/{total_frames} | "
                      f"Detections: {total_detections:5d} | Tracks: {track_counter:4d} | "
                      f"FPS: {fps_actual:.1f} | ETA: {eta:.0f}s")
                last_print = now
        
        frame_idx += 1
    
    cap.release()
    tracks_file.close()
    
    elapsed = time.time() - start_time
    
    # Save stats
    stats = {
        "total_frames": total_frames,
        "processed_frames": processed,
        "total_detections": total_detections,
        "unique_tracks": track_counter,
        "elapsed_seconds": round(elapsed, 2),
        "fps": round(processed / elapsed, 2) if elapsed > 0 else 0
    }
    
    with open(results_dir / "detection_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*80}")
    print("DETECTION COMPLETE")
    print(f"{'='*80}")
    print(f"  Frames processed: {processed}")
    print(f"  Total detections: {total_detections}")
    print(f"  Unique tracks:    {track_counter}")
    print(f"  Processing time:  {elapsed:.1f}s")
    print(f"  Processing FPS:   {processed/elapsed:.1f}")
    print(f"  Results saved to: {results_dir}")
    
    return results_dir, stats


def compute_iou(bbox1, bbox2):
    """Compute IoU between two bboxes [x1, y1, x2, y2]."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def load_tracks(results_dir: Path):
    """Load tracks from tracks.jsonl."""
    tracks = []
    with open(results_dir / "tracks.jsonl") as f:
        for line in f:
            if line.strip():
                tracks.append(json.loads(line))
    return tracks


def analyze_tracking(tracks: list):
    """Analyze tracking quality."""
    print(f"\n{'='*80}")
    print("TRACKING ANALYSIS")
    print(f"{'='*80}")
    
    by_track = defaultdict(list)
    for t in tracks:
        by_track[t['track_id']].append(t)
    
    by_label = defaultdict(list)
    for tid, dets in by_track.items():
        label = dets[0]['label']
        by_label[label].append({
            'track_id': tid,
            'length': len(dets),
            'first_frame': dets[0]['frame_id'],
            'last_frame': dets[-1]['frame_id'],
            'avg_conf': sum(d['confidence'] for d in dets) / len(dets)
        })
    
    print(f"\n  {'Class':<20} {'Tracks':>8} {'Avg Len':>10} {'Avg Conf':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10}")
    
    for label in sorted(by_label.keys(), key=lambda x: -len(by_label[x])):
        tracks_for_label = by_label[label]
        avg_len = sum(t['length'] for t in tracks_for_label) / len(tracks_for_label)
        avg_conf = sum(t['avg_conf'] for t in tracks_for_label) / len(tracks_for_label)
        print(f"  {label:<20} {len(tracks_for_label):>8} {avg_len:>10.1f} {avg_conf:>10.2f}")
    
    # Find fragmented tracks (same object, multiple track IDs)
    print(f"\n  Potential ID Switches (tracks that appear/disappear frequently):")
    short_tracks = [t for t in by_track.values() if len(t) <= 3]
    print(f"  Very short tracks (<=3 frames): {len(short_tracks)}")
    
    return by_label


def validate_with_gemini(genai, video_path: str, tracks: list, results_dir: Path, 
                         sample_every_n: int = 30):
    """Validate frames with Gemini."""
    import cv2
    import PIL.Image
    
    print(f"\n{'='*80}")
    print("GEMINI VALIDATION")
    print(f"{'='*80}")
    
    model = genai.GenerativeModel("gemini-2.0-flash")  # Using stable model for reliability
    
    # Get unique frames from tracks
    frame_ids = sorted(set(t['frame_id'] for t in tracks))
    
    # Sample frames for validation
    sample_frames = frame_ids[::sample_every_n]
    print(f"\n  Validating {len(sample_frames)} frames (every {sample_every_n}th)")
    
    # Group tracks by frame
    by_frame = defaultdict(list)
    for t in tracks:
        by_frame[t['frame_id']].append(t)
    
    # Extract and validate frames
    cap = cv2.VideoCapture(video_path)
    frames_dir = results_dir / "validation_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    validations = []
    
    for i, frame_id in enumerate(sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_path = frames_dir / f"frame_{frame_id:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        
        frame_dets = by_frame.get(frame_id, [])
        labels = [t['label'] for t in frame_dets]
        label_counts = dict(Counter(labels))
        
        prompt = f"""Analyze this video frame. A YOLO-World detector found these objects:
{json.dumps(label_counts, indent=2)}

Respond with JSON:
{{
    "objects_visible": ["all", "visible", "objects"],
    "false_positives": ["YOLO detections that are WRONG"],
    "false_negatives": ["objects YOLO MISSED"],
    "accuracy_score": 0.0-1.0,
    "comments": "brief assessment"
}}"""

        try:
            image = PIL.Image.open(frame_path)
            response = model.generate_content([prompt, image])
            text = response.text.strip()
            
            # Clean markdown
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
            
            result = json.loads(text)
            result["frame_id"] = frame_id
            result["yolo_detections"] = label_counts
            validations.append(result)
            
            acc = result.get("accuracy_score", "?")
            print(f"  [{i+1:3d}/{len(sample_frames)}] Frame {frame_id:5d}: Accuracy {acc}")
            
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower():
                print(f"  [{i+1:3d}/{len(sample_frames)}] Frame {frame_id:5d}: Rate limited, waiting 30s...")
                time.sleep(30)
            else:
                print(f"  [{i+1:3d}/{len(sample_frames)}] Frame {frame_id:5d}: Error - {err_str[:50]}")
        
        # Small delay to avoid rate limits
        time.sleep(0.5)
    
    cap.release()
    
    # Save validations
    with open(results_dir / "gemini_validation.json", "w") as f:
        json.dump({"validations": validations}, f, indent=2)
    
    # Summary
    if validations:
        accuracies = [v.get("accuracy_score", 0) for v in validations 
                     if isinstance(v.get("accuracy_score"), (int, float))]
        avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0
        
        all_fn = []
        all_fp = []
        for v in validations:
            all_fn.extend(v.get("false_negatives", []))
            all_fp.extend(v.get("false_positives", []))
        
        print(f"\n  {'='*60}")
        print(f"  VALIDATION SUMMARY")
        print(f"  {'='*60}")
        print(f"  Average Accuracy: {avg_acc:.1%}")
        print(f"  Total False Positives: {len(all_fp)}")
        print(f"  Total False Negatives: {len(all_fn)}")
        
        if all_fn:
            print(f"\n  Top Missed Objects:")
            for item, cnt in Counter(all_fn).most_common(10):
                print(f"    {item}: {cnt}")
    
    return validations


def main():
    parser = argparse.ArgumentParser(description="Full Phase 1 Validation")
    parser.add_argument("--video", required=True, help="Video path")
    parser.add_argument("--episode", required=True, help="Episode ID")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--fps", type=float, default=5.0, help="Target FPS")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--skip-detect", action="store_true", help="Skip detection")
    parser.add_argument("--validate-every", type=int, default=30, 
                        help="Validate every Nth frame (default: 30)")
    args = parser.parse_args()
    
    results_dir = Path(f"results/{args.episode}")
    
    # Step 1: Run detection
    if args.skip_detect and (results_dir / "tracks.jsonl").exists():
        print("Skipping detection, using existing tracks...")
        with open(results_dir / "detection_stats.json") as f:
            stats = json.load(f)
    else:
        results_dir, stats = run_detection_realtime(
            args.video, args.episode, args.device, args.fps, args.confidence
        )
    
    # Step 2: Load and analyze tracks
    tracks = load_tracks(results_dir)
    analyze_tracking(tracks)
    
    # Step 3: Gemini validation
    genai = setup_gemini()
    if genai:
        with open(results_dir / "episode_meta.json") as f:
            meta = json.load(f)
        video_path = meta["video_path"]
        validate_with_gemini(genai, video_path, tracks, results_dir, 
                            sample_every_n=args.validate_every)
    
    print(f"\n{'='*80}")
    print("VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results: {results_dir}")


if __name__ == "__main__":
    main()
