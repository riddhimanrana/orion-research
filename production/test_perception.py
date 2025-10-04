"""
Test Script for Part 1: Asynchronous Perception Engine
=======================================================

This script demonstrates how to use the perception engine with a sample video.

Usage:
    python production/test_perception.py --video path/to/video.mp4
    python production/test_perception.py --generate-sample  # Creates a test video
"""

import os
import sys
import argparse
import json
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from perception_engine import run_perception_engine, logger


def generate_sample_video(output_path: str, duration: int = 10, fps: int = 30):
    """
    Generate a simple test video with moving objects
    
    Args:
        output_path: Where to save the video
        duration: Video duration in seconds
        fps: Frames per second
    """
    logger.info(f"Generating sample video: {output_path}")
    
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
    
    total_frames = duration * fps
    
    for i in range(total_frames):
        # Create frame with gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [50, 100, 150]
        
        # Add moving rectangle (simulates a "car")
        x = int(50 + 400 * (i / total_frames))
        cv2.rectangle(frame, (x, 200), (x + 80, 280), (0, 0, 255), -1)
        cv2.putText(frame, "CAR", (x + 15, 245), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 2)
        
        # Add stationary circle (simulates a "person")
        cv2.circle(frame, (320, 350), 40, (0, 255, 0), -1)
        cv2.putText(frame, "PERSON", (280, 355), cv2.FONT_HERSHEY_SIMPLEX,
                   0.4, (255, 255, 255), 1)
        
        # Add oscillating triangle (simulates a "bird")
        y_offset = int(30 * np.sin(i * 0.2))
        pts = np.array([[100, 100 + y_offset], 
                       [120, 80 + y_offset], 
                       [140, 100 + y_offset]], np.int32)
        cv2.fillPoly(frame, [pts], (255, 255, 0))
        
        # Add frame number
        cv2.putText(frame, f"Frame: {i}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    logger.info(f"Sample video created: {output_path}")
    logger.info(f"  Duration: {duration}s, FPS: {fps}, Frames: {total_frames}")


def analyze_perception_log(perception_log: list):
    """
    Analyze and print statistics about the perception log
    
    Args:
        perception_log: List of RichPerceptionObject dictionaries
    """
    if not perception_log:
        logger.warning("Perception log is empty!")
        return
    
    print("\n" + "="*80)
    print("PERCEPTION LOG ANALYSIS")
    print("="*80)
    
    # Basic statistics
    total_objects = len(perception_log)
    complete_objects = sum(1 for obj in perception_log if obj['rich_description'] is not None)
    
    print(f"\nTotal objects detected: {total_objects}")
    print(f"Complete descriptions: {complete_objects} ({complete_objects/total_objects*100:.1f}%)")
    
    # Time range
    timestamps = [obj['timestamp'] for obj in perception_log]
    print(f"\nTime range: {min(timestamps):.2f}s - {max(timestamps):.2f}s")
    print(f"Duration covered: {max(timestamps) - min(timestamps):.2f}s")
    
    # Object class distribution
    class_counts = {}
    for obj in perception_log:
        cls = obj['object_class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    print("\nObject class distribution:")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {count} ({count/total_objects*100:.1f}%)")
    
    # Confidence statistics
    confidences = [obj['detection_confidence'] for obj in perception_log]
    print(f"\nDetection confidence:")
    print(f"  Mean: {np.mean(confidences):.3f}")
    print(f"  Min: {np.min(confidences):.3f}")
    print(f"  Max: {np.max(confidences):.3f}")
    
    # Sample objects
    print("\n" + "-"*80)
    print("SAMPLE PERCEPTION OBJECTS (first 3)")
    print("-"*80)
    for i, obj in enumerate(perception_log[:3]):
        print(f"\n[Object {i+1}]")
        print(f"  Timestamp: {obj['timestamp']:.2f}s")
        print(f"  Class: {obj['object_class']}")
        print(f"  Confidence: {obj['detection_confidence']:.3f}")
        print(f"  Bounding box: {obj['bounding_box']}")
        print(f"  Description: {obj['rich_description']}")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Test the Asynchronous Perception Engine'
    )
    parser.add_argument(
        '--video',
        type=str,
        help='Path to input video file'
    )
    parser.add_argument(
        '--generate-sample',
        action='store_true',
        help='Generate a sample test video'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/testing',
        help='Directory for output files (default: data/testing)'
    )
    parser.add_argument(
        '--sample-duration',
        type=int,
        default=10,
        help='Duration of generated sample video in seconds (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine video path
    if args.generate_sample:
        video_path = str(output_dir / 'sample_video.mp4')
        generate_sample_video(video_path, duration=args.sample_duration)
    elif args.video:
        video_path = args.video
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            sys.exit(1)
    else:
        logger.error("Please provide --video or --generate-sample")
        parser.print_help()
        sys.exit(1)
    
    # Run perception engine
    logger.info(f"\nProcessing video: {video_path}")
    
    try:
        perception_log = run_perception_engine(video_path)
        
        # Save results
        output_json = output_dir / 'perception_log.json'
        with open(output_json, 'w') as f:
            json.dump(perception_log, f, indent=2)
        
        logger.info(f"\nPerception log saved to: {output_json}")
        
        # Analyze results
        analyze_perception_log(perception_log)
        
        # Success message
        print("✅ Test completed successfully!")
        print(f"📁 Output files saved in: {output_dir}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
