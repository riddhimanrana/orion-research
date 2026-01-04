"""
orion init - Initialize a new episode from video

Creates the episode directory structure and metadata.
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import cv2

logger = logging.getLogger(__name__)


def run_init(args) -> int:
    """Initialize a new episode."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return 1
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results") / args.episode
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video metadata
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return 1
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    # Detect orientation
    is_portrait = height > width
    
    cap.release()
    
    # Create episode metadata
    metadata = {
        "episode_id": args.episode,
        "video_path": str(video_path.absolute()),
        "output_dir": str(output_dir.absolute()),
        "created_at": datetime.now().isoformat(),
        "video": {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": duration,
            "is_portrait": is_portrait
        },
        "config": {
            "target_fps": args.fps
        },
        "status": {
            "initialized": True,
            "detected": False,
            "embedded": False,
            "filtered": False,
            "graphed": False,
            "exported": False
        },
        "version": "2.0.0"
    }
    
    # Write metadata
    meta_path = output_dir / "episode_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Create symlink to video (optional, for convenience)
    video_link = output_dir / "video_source.mp4"
    if not video_link.exists():
        try:
            video_link.symlink_to(video_path.absolute())
        except OSError:
            pass  # Symlinks may not work on all systems
    
    # Print summary
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    ORION V2 - EPISODE INITIALIZED                ║
╠══════════════════════════════════════════════════════════════════╣
║  Episode:    {args.episode:<50} ║
║  Video:      {video_path.name:<50} ║
║  Duration:   {duration:.1f}s ({frame_count} frames @ {fps:.1f} FPS){' ' * max(0, 28 - len(f'{duration:.1f}s ({frame_count} frames @ {fps:.1f} FPS)'))}║
║  Resolution: {width}x{height} {'(portrait)' if is_portrait else '(landscape)':<30} ║
║  Output:     {str(output_dir):<50} ║
╠══════════════════════════════════════════════════════════════════╣
║  Next step:  orion analyze --episode {args.episode:<24} ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    return 0
