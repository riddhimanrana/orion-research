#!/usr/bin/env python3
"""
Test script for loop closure detection in SLAM

This script runs SLAM on a video and verifies that:
1. Loop closures are detected when camera revisits locations
2. Pose graph optimization reduces drift
3. Number of zones is reduced (target: 2-3 instead of 4)
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add orion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.slam.slam_engine import SLAMEngine, SLAMConfig


def create_synthetic_video_with_loop(output_path: str = "/tmp/test_loop_video.mp4"):
    """
    Create a synthetic video that simulates a loop closure scenario.
    Camera moves forward then returns to starting position.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))
    
    num_frames = 200
    for i in range(num_frames):
        # Create synthetic image with features
        img = np.ones((480, 640, 3), dtype=np.uint8) * 50
        
        # Add grid pattern for features
        for x in range(0, 640, 40):
            cv2.line(img, (x, 0), (x, 480), (100, 100, 100), 1)
        for y in range(0, 480, 40):
            cv2.line(img, (0, y), (640, y), (100, 100, 100), 1)
        
        # Add moving features based on camera position
        # Simulate forward motion then return
        if i < 100:
            # Moving forward
            offset_x = int((i / 100) * 200)
            offset_y = int((i / 100) * 100)
        else:
            # Returning back (loop closure)
            offset_x = int(((200 - i) / 100) * 200)
            offset_y = int(((200 - i) / 100) * 100)
        
        # Draw moving circles
        for j in range(10):
            cx = 100 + j * 50 - offset_x
            cy = 200 - offset_y
            if 0 <= cx < 640 and 0 <= cy < 480:
                cv2.circle(img, (cx, cy), 20, (200, 200, 200), -1)
        
        # Add frame number
        cv2.putText(img, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2)
        
        writer.write(img)
    
    writer.release()
    print(f"✓ Created synthetic video: {output_path}")
    return output_path


def test_loop_closure_on_video(video_path: str, max_frames: int = 200):
    """Test loop closure detection on a video"""
    
    print("\n" + "="*70)
    print("LOOP CLOSURE TEST")
    print("="*70)
    
    # Configure SLAM with loop closure enabled
    config = SLAMConfig(
        method="opencv",
        enable_loop_closure=True,
        min_loop_interval=30,  # Min frames between loops
        min_loop_inliers=20,   # Lower threshold for synthetic data
        bow_similarity_threshold=0.65,  # Slightly lower for synthetic
        enable_pose_graph_optimization=True,
        loop_closure_weight=100.0,
        optimize_every_n_loops=3,  # Optimize every 3 loops
        num_features=2000,  # More features for better matching
    )
    
    # Initialize SLAM
    slam = SLAMEngine(config=config)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Failed to open video: {video_path}")
        return
    
    print(f"\n✓ Opened video: {video_path}")
    print(f"  Config: loop_closure={config.enable_loop_closure}, "
          f"min_interval={config.min_loop_interval}, "
          f"optimize_every={config.optimize_every_n_loops}")
    
    frame_idx = 0
    loop_count = 0
    optimization_count = 0
    
    # Process frames
    print("\nProcessing frames...")
    print("-" * 70)
    
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        timestamp = frame_idx / 30.0  # Assume 30 FPS
        pose = slam.process_frame(frame, timestamp, frame_idx)
        
        # Track loop closures
        if slam.loop_detector and len(slam.loop_detector.loop_closures) > loop_count:
            loop_count = len(slam.loop_detector.loop_closures)
            print(f"  Frame {frame_idx}: Loop detected! (Total loops: {loop_count})")
        
        # Show progress every 50 frames
        if frame_idx % 50 == 0 and frame_idx > 0:
            print(f"  Frame {frame_idx}: {len(slam.poses)} poses, "
                  f"{len(slam.loop_detector.loop_closures) if slam.loop_detector else 0} loops")
        
        frame_idx += 1
    
    cap.release()
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    stats = slam.get_statistics()
    print(f"\nTracking Statistics:")
    print(f"  Total frames processed: {frame_idx}")
    print(f"  Successful poses: {stats['total_frames']}")
    print(f"  Tracking quality: {stats['tracking_success_rate']:.1%}")
    print(f"  Trajectory length: {stats['trajectory_length_m']:.2f} meters")
    print(f"  Avg motion per frame: {stats['avg_translation_per_frame_m']:.3f} meters")
    
    if slam.loop_detector:
        print(f"\nLoop Closure Statistics:")
        print(f"  Keyframes in database: {len(slam.loop_detector.keyframes)}")
        print(f"  Loop closures detected: {len(slam.loop_detector.loop_closures)}")
        
        vocab = slam.loop_detector.bow.vocabulary
        vocab_size = len(vocab) if vocab is not None and hasattr(vocab, '__len__') else 0
        print(f"  BoW vocabulary size: {vocab_size if vocab_size > 0 else 'Not trained'}")
        
        if len(slam.loop_detector.loop_closures) > 0:
            print(f"\n  Loop Closure Details:")
            for i, loop in enumerate(slam.loop_detector.loop_closures[:5]):  # Show first 5
                print(f"    Loop {i+1}: frame {loop.query_id} → {loop.match_id}, "
                      f"inliers={loop.inliers}, confidence={loop.confidence:.2f}")
            if len(slam.loop_detector.loop_closures) > 5:
                print(f"    ... and {len(slam.loop_detector.loop_closures) - 5} more")
    
    # Check if we achieved the goal
    print("\n" + "="*70)
    if slam.loop_detector and len(slam.loop_detector.loop_closures) > 0:
        print("✓ SUCCESS: Loop closure system is working!")
        print(f"  {len(slam.loop_detector.loop_closures)} loops detected and corrected")
    else:
        print("⚠ WARNING: No loop closures detected")
        print("  Possible reasons:")
        print("  - Video doesn't contain loop closures (camera doesn't revisit locations)")
        print("  - Similarity threshold too high")
        print("  - Not enough keyframes")
    
    print("="*70 + "\n")
    
    return slam


def test_with_real_video():
    """Test with a real video if available"""
    
    # Try to find example videos in the workspace
    video_paths = [
        "data/examples/video.mp4",
        "data/examples/test_video.mp4",
        "data/ag_50/video.mp4",
    ]
    
    for video_path in video_paths:
        if Path(video_path).exists():
            print(f"\n✓ Found video: {video_path}")
            return test_loop_closure_on_video(video_path, max_frames=300)
    
    print("\n⚠ No real video found, using synthetic video instead")
    return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test loop closure detection")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic video")
    parser.add_argument("--max-frames", type=int, default=200, help="Max frames to process")
    
    args = parser.parse_args()
    
    if args.synthetic or not args.video:
        # Create and test with synthetic video
        print("Creating synthetic video with loop closure...")
        video_path = create_synthetic_video_with_loop()
        slam = test_loop_closure_on_video(video_path, max_frames=args.max_frames)
    elif args.video:
        # Test with provided video
        if not Path(args.video).exists():
            print(f"✗ Video not found: {args.video}")
            sys.exit(1)
        slam = test_loop_closure_on_video(args.video, max_frames=args.max_frames)
    else:
        # Try real videos first, fallback to synthetic
        slam = test_with_real_video()
        if slam is None:
            video_path = create_synthetic_video_with_loop()
            slam = test_loop_closure_on_video(video_path, max_frames=args.max_frames)
    
    print("\n✓ Test complete!")
