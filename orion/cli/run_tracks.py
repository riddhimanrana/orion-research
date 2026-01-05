#!/usr/bin/env python3
"""
Run Tracks Pipeline
===================

Process video through detection + tracking and output tracks.jsonl.

Usage:
    python -m orion.cli.run_tracks --video path/to/video.mp4 --episode demo_room
    python -m orion.cli.run_tracks --episode demo_room --fps 5
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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


def get_default_device() -> str:
    """Auto-detect the best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def process_video_to_tracks(
    video_path: str,
    episode_id: str,
    target_fps: float = 5.0,
    yolo_model: str = "yolo11m",
    detector_backend: str = "yolo",
    yoloworld_open_vocab: bool = False,
    yoloworld_prompt: str | None = None,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.3,
    max_age: int = 30,
    device: str = "auto",
    save_viz: bool = False,
    enable_hand_detector: bool = False,
    hand_max_hands: int = 2,
    hand_detection_confidence: float = 0.5,
    hand_tracking_confidence: float = 0.3,
    enable_3d: bool = False,
    depth_model_size: str = "small",
    disable_slam: bool = False,
) -> dict:
    """
    Process video through detection + tracking pipeline.
    
    Args:
        video_path: Path to video file
        episode_id: Episode identifier for results
        target_fps: Target FPS for processing
        yolo_model: YOLO variant to use
        detector_backend: Detection backend ('yolo', 'yoloworld')
        confidence_threshold: Min detection confidence
        iou_threshold: Min IoU for track association
        max_age: Frames to keep unmatched tracks
        device: Device to run on ('auto', 'cuda', 'mps', 'cpu')
        save_viz: Save visualization outputs
        enable_3d: Enable 3D perception (Depth/SLAM)

    Returns:
        Dictionary with statistics
    """
    # Auto-detect device if needed
    if device == "auto":
        device = get_default_device()
        logger.info(f"Auto-detected device: {device}")
    
    logger.info("="*80)
    logger.info("PHASE 1: DETECTION + TRACKING BASELINE")
    logger.info("="*80)
    
    start_time = time.time()
    total_steps = 5 + (1 if enable_hand_detector else 0)
    current_step = 1
    
    # Initialize detector
    logger.info(f"\n[{current_step}/{total_steps}] Initializing detector...")
    
    # Use FrameObserver for advanced detection (supports 3D/Depth)
    from orion.perception.observer import FrameObserver
    from orion.perception.config import DetectionConfig
    from ultralytics import YOLO
    
    # Initialize detector based on backend
    yolo = None
    yoloworld = None

    if detector_backend == "yolo":
        logger.info(f"Loading YOLO model: {yolo_model}")
        yolo = YOLO(f"{yolo_model}.pt")
    elif detector_backend == "yoloworld":
        # Get model from config (default: yolov8l-worldv2.pt)
        det_config_temp = DetectionConfig(backend="yoloworld")
        if yoloworld_prompt:
            det_config_temp.yoloworld_prompt_preset = "custom"
            det_config_temp.yoloworld_prompt = yoloworld_prompt
        yoloworld_model = det_config_temp.yoloworld_model
        logger.info(f"Loading YOLO-World model: {yoloworld_model}")
        yoloworld = YOLO(yoloworld_model)
        
        if yoloworld_open_vocab:
            logger.info("  YOLO-World mode: default vocabulary (no set_classes)")
        elif "custom" in yoloworld_model or "general" in yoloworld_model:
            logger.info("  YOLO-World mode: using pre-baked custom/general vocabulary (skipping set_classes)")
            # Verify classes match config if possible, but for now trust the file
            logger.info(f"  Model classes: {yoloworld.names}")
        else:
            # Constrain YOLO-World to our configured prompt/categories
            custom_classes = det_config_temp.yoloworld_categories()
            logger.info(f"  Setting {len(custom_classes)} custom classes for open-vocab detection")
            yoloworld.set_classes(custom_classes)

    det_config = DetectionConfig(
        backend=detector_backend,
        model=yolo_model,
        confidence_threshold=confidence_threshold,
        yoloworld_use_custom_classes=(False if yoloworld_open_vocab else True),
    )
    if yoloworld_prompt:
        det_config.yoloworld_prompt_preset = "custom"
        det_config.yoloworld_prompt = yoloworld_prompt

    observer = FrameObserver(
        config=det_config,
        detector_backend=detector_backend,
        yolo_model=yolo,
        yoloworld_model=yoloworld,
        target_fps=target_fps,
        enable_3d=enable_3d,
        depth_model_size=depth_model_size,
        enable_slam=(not disable_slam),
        show_progress=True,
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
    detections = observer.process_video(video_path=video_path)
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
        # Get timestamp from detections (all dets in frame share same timestamp)
        timestamp = frame_dets[0].get("timestamp", 0.0) if frame_dets else 0.0
        # Annotate each track with frame_id and timestamp before saving
        for track in tracked_dets:
            track_dict = track.to_dict() if hasattr(track, 'to_dict') else track
            track_dict['frame_id'] = frame_id
            track_dict['timestamp'] = timestamp
            all_tracks.append(track_dict)
    current_step += 1
    
    # Save tracks
    logger.info(f"\n[{current_step}/{total_steps}] Saving results to episode: {episode_id}")
    results_dir = ensure_results_dir(episode_id)
    
    # Save tracks.jsonl
    tracks_path = save_results_jsonl(episode_id, "tracks.jsonl", all_tracks)
    logger.info(f"  ✓ Saved {len(all_tracks)} track observations → {tracks_path}")
    
    # Save run metadata
    tracker_stats = tracker.get_statistics()
    detector_info = {
        "backend": detector_backend,
        "model": yolo_model if detector_backend == "yolo" else ("yolov8m-worldv2" if detector_backend == "yoloworld" else "grounding-dino-base"),
        "confidence_threshold": confidence_threshold
    }
    
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
            "unique_tracks": tracker_stats.get("total_tracks", 0),
            "frames_processed": tracker_stats.get("frame_count", len(detections_by_frame)),
        }
    }
    
    meta_path = results_dir / "run_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(run_meta, f, indent=2)
    
    logger.info(f"  ✓ Saved run metadata → {meta_path}")
    
    # Print summary
    elapsed = time.time() - start_time
    frames_processed = tracker_stats.get('frame_count', len(detections_by_frame))
    logger.info("\n" + "="*80)
    logger.info("TRACKING COMPLETE")
    logger.info("="*80)
    logger.info(f"  Frames processed: {frames_processed}")
    logger.info(f"  Total detections: {len(detections)}")
    logger.info(f"  Track observations: {len(all_tracks)}")
    logger.info(f"  Unique tracks: {tracker_stats.get('total_tracks', 0)}")
    logger.info(f"  Active tracks: {tracker_stats.get('active_tracks', 0)}")
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
        "--detector-backend",
        type=str,
        default="yoloworld",
        choices=["yolo", "yoloworld"],
        help="Detection backend (default: yoloworld)"
    )
    parser.add_argument(
        "--yoloworld-open-vocab",
        action="store_true",
        help="When using --detector-backend yoloworld, do NOT call set_classes(); run YOLO-World in open-vocab mode",
    )
    parser.add_argument(
        "--yoloworld-prompt",
        type=str,
        default=None,
        help=(
            "Dot-separated prompt/classes for YOLO-World set_classes() (e.g. 'chair . table . lamp'). "
            "Only used when --detector-backend yoloworld and --yoloworld-open-vocab is NOT set."
        ),
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
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to run on (default: auto-detect)"
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

    parser.add_argument(
        "--enable-3d",
        action="store_true",
        help="Enable DepthAnythingV3 depth stats (and optional SLAM) during detection",
    )
    parser.add_argument(
        "--depth-model-size",
        type=str,
        default="small",
        choices=["small", "base", "large", "giant"],
        help="DepthAnythingV3 model size/preset (default: small)",
    )
    parser.add_argument(
        "--disable-slam",
        action="store_true",
        help="If --enable-3d is set, disable SLAM (keeps depth only; faster)",
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
            detector_backend=args.detector_backend,
            yoloworld_open_vocab=args.yoloworld_open_vocab,
            yoloworld_prompt=args.yoloworld_prompt,
            enable_hand_detector=args.detect_hands,
            hand_max_hands=args.hand_max,
            hand_detection_confidence=args.hand_det_conf,
            hand_tracking_confidence=args.hand_track_conf,
            confidence_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            max_age=args.max_age,
            device=args.device,
            save_viz=args.save_viz,
            enable_3d=args.enable_3d,
            depth_model_size=args.depth_model_size,
            disable_slam=args.disable_slam,
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
