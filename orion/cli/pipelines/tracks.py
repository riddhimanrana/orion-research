#!/usr/bin/env python3
"""
Run Tracks Pipeline
===================

Process video through detection + tracking and output tracks.jsonl.

Usage:
    python -m orion.cli.pipelines.tracks --video path/to/video.mp4 --episode demo_room
    python -m orion.cli.pipelines.tracks --episode demo_room --fps 5
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from orion.perception.detectors.yolo import YOLODetector
from orion.perception.hand_classifier import HandClassifier
from orion.perception.trackers.enhanced import EnhancedTracker as ObjectTracker
from orion.config import (
    get_episode_dir,
    get_episode_video_path,
    ensure_results_dir,
    save_results_jsonl,
    load_episode_meta,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_video_to_tracks(
    video_path: str,
    episode_id: str,
    target_fps: float = 5.0,
    yolo_model: str = "yolo11m",
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.3,
    max_age: int = 30,
    device: str = "mps",
    save_viz: bool = False,
    enable_hand_detector: bool = False,
    hand_max_hands: int = 2,
    hand_detection_confidence: float = 0.5,
    hand_tracking_confidence: float = 0.3,
) -> dict:
    """
    Process video through detection + tracking pipeline.
    
    Args:
        video_path: Path to video file
        episode_id: Episode identifier for results
        target_fps: Target FPS for processing
        yolo_model: YOLO variant to use
        confidence_threshold: Min detection confidence
        iou_threshold: Min IoU for track association
        max_age: Frames to keep unmatched tracks
        device: Device to run on
        save_viz: Save visualization outputs
        
    Returns:
        Dictionary with statistics
    """
    logger.info("="*80)
    logger.info("PHASE 1: DETECTION + TRACKING BASELINE")
    logger.info("="*80)
    
    start_time = time.time()
    total_steps = 5 + (1 if enable_hand_detector else 0)
    current_step = 1
    
    # Initialize detector
    logger.info(f"\n[{current_step}/{total_steps}] Initializing detector...")
    detector = YOLODetector(
        model_name=yolo_model,
        confidence_threshold=confidence_threshold,
        device=device,
    )
    current_step += 1
    
    # Initialize tracker
    logger.info(f"\n[{current_step}/{total_steps}] Initializing tracker...")
    tracker = ObjectTracker(
        iou_threshold=iou_threshold,
        max_age=max_age,
    )
    current_step += 1

    # Detect objects
    logger.info(f"\n[{current_step}/{total_steps}] Detecting objects (target FPS: {target_fps})...")
    detections = detector.detect_video(
        video_path=video_path,
        target_fps=target_fps,
        show_progress=True,
    )
    current_step += 1

    hand_detector_info = None
    if enable_hand_detector:
        logger.info(f"\n[{current_step}/{total_steps}] Running hand classifier...")
        hand_detector = HandClassifier(
            max_hands=hand_max_hands,
            min_detection_confidence=hand_detection_confidence,
            min_tracking_confidence=hand_tracking_confidence,
        )
        hand_detections = hand_detector.detect_video(
            video_path=video_path,
            target_fps=target_fps,
        )
        detections.extend(hand_detections)
        hand_detector_info = hand_detector.get_model_info()
        logger.info(f"  ✓ Added {len(hand_detections)} hand detections → total {len(detections)}")
        current_step += 1
    
    # Group detections by frame
    logger.info(f"\n[{current_step}/{total_steps}] Tracking objects...")
    detections_by_frame = {}
    for det in detections:
        frame_id = det["frame_id"]
        if frame_id not in detections_by_frame:
            detections_by_frame[frame_id] = []
        detections_by_frame[frame_id].append(det)
    
    # Track frame by frame
    all_tracks = []
    for frame_id in sorted(detections_by_frame.keys()):
        frame_dets = detections_by_frame[frame_id]
        tracked_dets = tracker.update(frame_dets)
        all_tracks.extend(tracked_dets)
    current_step += 1
    
    # Save tracks
    logger.info(f"\n[{current_step}/{total_steps}] Saving results to episode: {episode_id}")
    results_dir = ensure_results_dir(episode_id)
    
    # Save tracks.jsonl
    tracks_path = save_results_jsonl(episode_id, "tracks.jsonl", all_tracks)
    logger.info(f"  ✓ Saved {len(all_tracks)} track observations → {tracks_path}")
    
    # Save run metadata
    tracker_stats = tracker.get_statistics()
    detector_info = detector.get_model_info()
    
    run_meta = {
        "episode_id": episode_id,
        "video_path": str(video_path),
        "processing_time_seconds": time.time() - start_time,
        "target_fps": target_fps,
        "detector": detector_info,
        "hand_detector": hand_detector_info,
        "tracker": {
            "iou_threshold": iou_threshold,
            "max_age": max_age,
            **tracker_stats,
        },
        "statistics": {
            "total_detections": len(detections),
            "total_track_observations": len(all_tracks),
            "unique_tracks": tracker_stats["total_tracks"],
            "frames_processed": tracker_stats["frame_count"],
        }
    }
    
    meta_path = results_dir / "run_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(run_meta, f, indent=2)
    
    logger.info(f"  ✓ Saved run metadata → {meta_path}")
    
    # Print summary
    elapsed = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info("TRACKING COMPLETE")
    logger.info("="*80)
    logger.info(f"  Frames processed: {tracker_stats['frame_count']}")
    logger.info(f"  Total detections: {len(detections)}")
    logger.info(f"  Track observations: {len(all_tracks)}")
    logger.info(f"  Unique tracks: {tracker_stats['total_tracks']}")
    logger.info(f"  Active tracks: {tracker_stats['active_tracks']}")
    logger.info(f"  Processing time: {elapsed:.2f}s")
    logger.info(f"  Results: results/{episode_id}/")
    logger.info("="*80 + "\n")
    
    return run_meta


def main():
    parser = argparse.ArgumentParser(
        description="Run detection + tracking pipeline on video"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file (overrides episode video)"
    )
    parser.add_argument(
        "--episode",
        type=str,
        required=True,
        help="Episode ID for results output"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Target FPS for processing (default: 5.0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11m",
        choices=["yolo11n", "yolo11s", "yolo11m", "yolo11x"],
        help="YOLO model variant (default: yolo11m)"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="Tracking IoU threshold (default: 0.3)"
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Max frames to keep unmatched tracks (default: 30)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["cuda", "mps", "cpu"],
        help="Device to run on (default: mps)"
    )
    parser.add_argument(
        "--detect-hands",
        action="store_true",
        help="Enable MediaPipe-based hand classification and keypoint extraction",
    )
    parser.add_argument(
        "--hand-max",
        type=int,
        default=2,
        help="Maximum number of hands per frame (default: 2)",
    )
    parser.add_argument(
        "--hand-det-conf",
        type=float,
        default=0.5,
        help="Minimum detection confidence for hand classifier (default: 0.5)",
    )
    parser.add_argument(
        "--hand-track-conf",
        type=float,
        default=0.3,
        help="Minimum tracking confidence for hand classifier (default: 0.3)",
    )
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Save visualization outputs"
    )
    
    args = parser.parse_args()
    
    # Determine video path
    if args.video:
        video_path = Path(args.video)
    else:
        # Try to get from episode
        video_path = get_episode_video_path(args.episode)
        if video_path is None:
            logger.error(f"No video found for episode '{args.episode}'")
            logger.error("Please specify --video or add video.mp4 to episode directory")
            sys.exit(1)
    
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)
    
    # Process
    try:
        process_video_to_tracks(
            video_path=str(video_path),
            episode_id=args.episode,
            target_fps=args.fps,
            yolo_model=args.model,
            enable_hand_detector=args.detect_hands,
            hand_max_hands=args.hand_max,
            hand_detection_confidence=args.hand_det_conf,
            hand_tracking_confidence=args.hand_track_conf,
            confidence_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            max_age=args.max_age,
            device=args.device,
            save_viz=args.save_viz,
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
