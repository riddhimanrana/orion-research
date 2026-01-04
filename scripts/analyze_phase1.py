#!/usr/bin/env python3
"""
Comprehensive Phase 1 Analysis & Testing

Full detection + tracking analysis on videos with:
- Frame-by-frame detection quality metrics
- Tracking continuity analysis (ID switches, fragmentation)
- Hungarian algorithm performance
- Gemini ground truth validation on key frames
- Missing detection analysis by class and scene context

Usage:
    python scripts/analyze_phase1.py --video data/examples/test.mp4 --episode phase1_full
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

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TrackMetrics:
    """Metrics for a single track."""
    track_id: int
    label: str
    first_frame: int
    last_frame: int
    total_detections: int
    expected_frames: int  # Frames in range where we expected to see it
    gaps: list = field(default_factory=list)  # List of (start, end) gaps
    avg_confidence: float = 0.0
    bbox_stability: float = 0.0  # Lower = more stable
    
    @property
    def fragmentation_ratio(self) -> float:
        """How fragmented is this track? 0 = continuous, 1 = very fragmented"""
        if self.expected_frames <= 1:
            return 0.0
        return 1.0 - (self.total_detections / self.expected_frames)
    
    @property
    def gap_count(self) -> int:
        return len(self.gaps)


def load_tracks(results_dir: Path):
    """Load tracks from tracks.jsonl."""
    tracks_path = results_dir / "tracks.jsonl"
    if not tracks_path.exists():
        raise FileNotFoundError(f"tracks.jsonl not found at {tracks_path}")
    
    tracks = []
    with open(tracks_path) as f:
        for line in f:
            if line.strip():
                tracks.append(json.loads(line))
    return tracks


def analyze_tracking_quality(tracks: list) -> dict:
    """Analyze tracking quality metrics."""
    # Group by track_id
    by_track = defaultdict(list)
    for t in tracks:
        by_track[t['track_id']].append(t)
    
    # Get frame range
    all_frames = sorted(set(t['frame_id'] for t in tracks))
    frame_interval = all_frames[1] - all_frames[0] if len(all_frames) > 1 else 1
    
    track_metrics = []
    
    for track_id, detections in by_track.items():
        detections = sorted(detections, key=lambda x: x['frame_id'])
        
        first_frame = detections[0]['frame_id']
        last_frame = detections[-1]['frame_id']
        expected_frames = ((last_frame - first_frame) // frame_interval) + 1
        
        # Find gaps
        gaps = []
        for i in range(len(detections) - 1):
            gap = detections[i+1]['frame_id'] - detections[i]['frame_id']
            if gap > frame_interval:
                gaps.append((detections[i]['frame_id'], detections[i+1]['frame_id']))
        
        # Calculate bbox stability (average IoU between consecutive frames)
        ious = []
        for i in range(len(detections) - 1):
            b1 = detections[i]['bbox']
            b2 = detections[i+1]['bbox']
            iou = compute_iou(b1, b2)
            ious.append(iou)
        
        avg_iou = sum(ious) / len(ious) if ious else 1.0
        bbox_stability = 1.0 - avg_iou  # Lower = more stable
        
        metrics = TrackMetrics(
            track_id=track_id,
            label=detections[0]['label'],
            first_frame=first_frame,
            last_frame=last_frame,
            total_detections=len(detections),
            expected_frames=expected_frames,
            gaps=gaps,
            avg_confidence=sum(d['confidence'] for d in detections) / len(detections),
            bbox_stability=bbox_stability
        )
        track_metrics.append(metrics)
    
    # Aggregate stats
    total_tracks = len(track_metrics)
    fragmented_tracks = [t for t in track_metrics if t.fragmentation_ratio > 0.2]
    short_tracks = [t for t in track_metrics if t.total_detections < 3]
    
    # Class-wise stats
    class_stats = defaultdict(lambda: {'count': 0, 'avg_length': 0, 'fragmented': 0})
    for t in track_metrics:
        class_stats[t.label]['count'] += 1
        class_stats[t.label]['avg_length'] += t.total_detections
        if t.fragmentation_ratio > 0.2:
            class_stats[t.label]['fragmented'] += 1
    
    for label, stats in class_stats.items():
        stats['avg_length'] /= stats['count']
        stats['fragmentation_rate'] = stats['fragmented'] / stats['count']
    
    return {
        'total_tracks': total_tracks,
        'fragmented_tracks': len(fragmented_tracks),
        'short_tracks': len(short_tracks),
        'class_stats': dict(class_stats),
        'track_metrics': track_metrics,
        'frame_interval': frame_interval
    }


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def analyze_detections_per_frame(tracks: list) -> dict:
    """Analyze detection patterns per frame."""
    by_frame = defaultdict(list)
    for t in tracks:
        by_frame[t['frame_id']].append(t)
    
    frames = sorted(by_frame.keys())
    detection_counts = [len(by_frame[f]) for f in frames]
    
    # Find frames with unusual detection counts
    mean_detections = np.mean(detection_counts)
    std_detections = np.std(detection_counts)
    
    low_detection_frames = []
    high_detection_frames = []
    
    for frame in frames:
        count = len(by_frame[frame])
        if count < mean_detections - 2 * std_detections:
            low_detection_frames.append((frame, count))
        elif count > mean_detections + 2 * std_detections:
            high_detection_frames.append((frame, count))
    
    return {
        'total_frames': len(frames),
        'mean_detections_per_frame': mean_detections,
        'std_detections': std_detections,
        'min_detections': min(detection_counts),
        'max_detections': max(detection_counts),
        'low_detection_frames': low_detection_frames[:10],  # First 10
        'high_detection_frames': high_detection_frames[:10],
        'detections_by_frame': {f: len(by_frame[f]) for f in frames}
    }


def validate_with_gemini(video_path: str, tracks: list, sample_frames: list, results_dir: Path):
    """Validate specific frames with Gemini Vision."""
    from scripts.validate_phase1 import (
        setup_gemini, get_frame_detections, validate_frame_with_gemini
    )
    
    genai = setup_gemini()
    
    # Extract the specific frames
    cap = cv2.VideoCapture(video_path)
    frames_dir = results_dir / "validation_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    validations = []
    
    for frame_id in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_path = frames_dir / f"frame_{frame_id:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        
        frame_detections = get_frame_detections(tracks, frame_id, window=2)
        validation = validate_frame_with_gemini(genai, frame_path, frame_id, frame_detections)
        
        if validation:
            validations.append(validation)
            print(f"  Frame {frame_id}: Accuracy {validation.get('accuracy_score', 'N/A')}")
    
    cap.release()
    return validations


def run_detection(video_path: str, episode: str, device: str = "cuda", fps: float = 5.0, 
                  confidence: float = 0.25):
    """Run Phase 1 detection pipeline."""
    results_dir = Path(f"results/{episode}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"PHASE 1 DETECTION: {video_path}")
    print(f"{'='*80}")
    
    # Import detection components
    from orion.perception.config import DetectionConfig
    from orion.backends.yoloworld_backend import YOLOWorldBackend
    
    # Initialize detector
    config = DetectionConfig(
        backend="yoloworld",
        yoloworld_model="yolov8x-worldv2.pt",
        confidence_threshold=confidence,
        device=device,
    )
    
    detector = YOLOWorldBackend(config)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    sample_interval = max(1, int(video_fps / fps))
    
    print(f"  Video: {total_frames} frames @ {video_fps:.2f} fps")
    print(f"  Sampling every {sample_interval} frames (target {fps} fps)")
    print(f"  Confidence threshold: {confidence}")
    
    # Process frames with tracking
    from orion.perception.trackers.enhanced import EnhancedTracker
    
    tracker = EnhancedTracker(
        max_age=30,
        min_hits=3,
        iou_threshold=0.3
    )
    
    tracks = []
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        
        if frame_id % sample_interval != 0:
            continue
        
        # Detect
        detections = detector.detect(frame)
        
        # Track
        if detections:
            bboxes = np.array([d['bbox'] for d in detections])
            scores = np.array([d['confidence'] for d in detections])
            class_ids = np.array([d.get('class_id', 0) for d in detections])
            
            tracked = tracker.update(bboxes, scores, class_ids)
            
            timestamp = frame_id / video_fps
            
            for track in tracked:
                track_entry = {
                    'frame_id': frame_id,
                    'timestamp': round(timestamp, 3),
                    'track_id': int(track[4]),
                    'bbox': [round(x, 1) for x in track[:4]],
                    'confidence': round(float(track[5]) if len(track) > 5 else scores[0], 4),
                    'label': detections[0]['label'] if detections else 'unknown',
                    'class_id': int(track[6]) if len(track) > 6 else 0
                }
                
                # Find matching detection for label
                for det in detections:
                    if compute_iou(track[:4], det['bbox']) > 0.5:
                        track_entry['label'] = det['label']
                        track_entry['class_id'] = det.get('class_id', 0)
                        break
                
                tracks.append(track_entry)
        
        frame_count += 1
        if frame_count % 50 == 0:
            elapsed = time.time() - start_time
            print(f"    Processed {frame_count} frames ({elapsed:.1f}s, {frame_count/elapsed:.1f} fps)")
    
    cap.release()
    elapsed = time.time() - start_time
    
    # Save tracks
    tracks_path = results_dir / "tracks.jsonl"
    with open(tracks_path, 'w') as f:
        for t in tracks:
            f.write(json.dumps(t) + "\n")
    
    # Save metadata
    meta = {
        'episode_id': episode,
        'video_path': str(Path(video_path).resolve()),
        'video': {
            'fps': video_fps,
            'total_frames': total_frames,
            'width': width,
            'height': height
        },
        'config': {
            'detector': 'yolov8x-worldv2',
            'device': device,
            'target_fps': fps,
            'sample_interval': sample_interval,
            'confidence_threshold': confidence
        },
        'stats': {
            'processed_frames': frame_count,
            'total_detections': len(tracks),
            'unique_tracks': len(set(t['track_id'] for t in tracks)),
            'elapsed_seconds': round(elapsed, 1)
        }
    }
    
    with open(results_dir / "episode_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n  Completed: {frame_count} frames, {len(tracks)} detections, {meta['stats']['unique_tracks']} tracks")
    print(f"  Time: {elapsed:.1f}s ({frame_count/elapsed:.1f} fps)")
    
    return tracks, results_dir


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Phase 1 Analysis")
    parser.add_argument("--video", required=True, help="Video path")
    parser.add_argument("--episode", required=True, help="Episode ID for results")
    parser.add_argument("--device", default="cuda", help="Device (cuda/mps/cpu)")
    parser.add_argument("--fps", type=float, default=5.0, help="Target FPS")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--skip-detect", action="store_true", help="Skip detection, analyze existing results")
    parser.add_argument("--validate-frames", type=int, default=10, help="Number of frames to validate with Gemini")
    args = parser.parse_args()
    
    results_dir = Path(f"results/{args.episode}")
    
    # Step 1: Run detection (or load existing)
    if args.skip_detect and (results_dir / "tracks.jsonl").exists():
        print("Loading existing tracks...")
        tracks = load_tracks(results_dir)
    else:
        tracks, results_dir = run_detection(
            args.video, args.episode, args.device, args.fps, args.confidence
        )
    
    # Step 2: Analyze tracking quality
    print(f"\n{'='*80}")
    print("TRACKING QUALITY ANALYSIS")
    print(f"{'='*80}")
    
    tracking_stats = analyze_tracking_quality(tracks)
    
    print(f"\n  Total tracks: {tracking_stats['total_tracks']}")
    print(f"  Fragmented tracks (>20% gaps): {tracking_stats['fragmented_tracks']}")
    print(f"  Short tracks (<3 detections): {tracking_stats['short_tracks']}")
    
    print(f"\n  Class-wise tracking quality:")
    sorted_classes = sorted(tracking_stats['class_stats'].items(), 
                           key=lambda x: x[1]['count'], reverse=True)
    for label, stats in sorted_classes[:15]:
        print(f"    {label:20s}: {stats['count']:3d} tracks, "
              f"avg length {stats['avg_length']:.1f}, "
              f"frag rate {stats['fragmentation_rate']:.1%}")
    
    # Step 3: Frame-by-frame analysis
    print(f"\n{'='*80}")
    print("DETECTION PATTERNS")
    print(f"{'='*80}")
    
    frame_stats = analyze_detections_per_frame(tracks)
    
    print(f"\n  Frames processed: {frame_stats['total_frames']}")
    print(f"  Detections per frame: {frame_stats['mean_detections_per_frame']:.1f} ± {frame_stats['std_detections']:.1f}")
    print(f"  Range: {frame_stats['min_detections']} - {frame_stats['max_detections']}")
    
    if frame_stats['low_detection_frames']:
        print(f"\n  Frames with unusually LOW detections:")
        for frame, count in frame_stats['low_detection_frames'][:5]:
            print(f"    Frame {frame}: {count} detections")
    
    if frame_stats['high_detection_frames']:
        print(f"\n  Frames with unusually HIGH detections:")
        for frame, count in frame_stats['high_detection_frames'][:5]:
            print(f"    Frame {frame}: {count} detections")
    
    # Step 4: Gemini validation
    if args.validate_frames > 0:
        print(f"\n{'='*80}")
        print(f"GEMINI VALIDATION ({args.validate_frames} frames)")
        print(f"{'='*80}")
        
        # Select frames to validate: low detection, high detection, and random
        frames_to_validate = []
        
        # Add low detection frames
        for frame, _ in frame_stats['low_detection_frames'][:3]:
            frames_to_validate.append(frame)
        
        # Add high detection frames  
        for frame, _ in frame_stats['high_detection_frames'][:3]:
            frames_to_validate.append(frame)
        
        # Add evenly distributed frames
        all_frames = sorted(frame_stats['detections_by_frame'].keys())
        step = len(all_frames) // (args.validate_frames - len(frames_to_validate))
        for i in range(0, len(all_frames), max(1, step)):
            if len(frames_to_validate) >= args.validate_frames:
                break
            if all_frames[i] not in frames_to_validate:
                frames_to_validate.append(all_frames[i])
        
        frames_to_validate = sorted(set(frames_to_validate))[:args.validate_frames]
        print(f"\n  Validating frames: {frames_to_validate}")
        
        try:
            meta = json.load(open(results_dir / "episode_meta.json"))
            video_path = meta['video_path']
            
            validations = validate_with_gemini(video_path, tracks, frames_to_validate, results_dir)
            
            if validations:
                print(f"\n  Validation Summary:")
                avg_accuracy = sum(v.get('accuracy_score', 0) for v in validations) / len(validations)
                print(f"    Average accuracy: {avg_accuracy:.2%}")
                
                all_fp = []
                all_fn = []
                for v in validations:
                    all_fp.extend(v.get('false_positives', []))
                    all_fn.extend(v.get('false_negatives', []))
                
                if all_fp:
                    print(f"\n    Common False Positives:")
                    for item, count in Counter(all_fp).most_common(5):
                        print(f"      • {item}: {count}")
                
                if all_fn:
                    print(f"\n    Common False Negatives (MISSED):")
                    for item, count in Counter(all_fn).most_common(10):
                        print(f"      • {item}: {count}")
                
                # Save validation results
                validation_output = {
                    'validations': validations,
                    'summary': {
                        'average_accuracy': avg_accuracy,
                        'false_positives': dict(Counter(all_fp)),
                        'false_negatives': dict(Counter(all_fn))
                    }
                }
                with open(results_dir / "gemini_validation.json", 'w') as f:
                    json.dump(validation_output, f, indent=2)
        
        except Exception as e:
            print(f"  ERROR in Gemini validation: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 5: Save full analysis
    analysis = {
        'tracking': {
            'total_tracks': tracking_stats['total_tracks'],
            'fragmented_tracks': tracking_stats['fragmented_tracks'],
            'short_tracks': tracking_stats['short_tracks'],
            'class_stats': tracking_stats['class_stats']
        },
        'detection_patterns': {
            'total_frames': frame_stats['total_frames'],
            'mean_detections': frame_stats['mean_detections_per_frame'],
            'std_detections': frame_stats['std_detections'],
            'low_detection_frames': frame_stats['low_detection_frames'],
            'high_detection_frames': frame_stats['high_detection_frames']
        }
    }
    
    with open(results_dir / "analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\n  Results saved to: {results_dir}")
    print(f"  Files: tracks.jsonl, episode_meta.json, analysis.json, gemini_validation.json")


if __name__ == "__main__":
    main()
