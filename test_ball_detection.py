#!/usr/bin/env python3
"""
Test ball detection with explicit ball vocabulary
"""

import cv2
import json
from pathlib import Path
from orion.managers.model_manager import ModelManager
from orion.backends.gdino_backend import GroundingDINOBackend

def test_ball_detection():
    """Test if we can detect balls with explicit vocabulary."""
    
    video_path = "datasets/PVSG/VidOR/mnt/lustre/jkyang/CVPR23/openpvsg/data/vidor/videos/0003_6141007489.mp4"
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    
    # Sample frames where ball should be visible (based on GT)
    # GT shows ball interactions in frames 0-124
    test_frames = [0, 15, 30, 60, 90, 115]
    
    # Initialize detector
    print("Initializing GroundingDINO...")
    model_manager = ModelManager(device="mps")
    gdino = model_manager.get_grounding_dino("IDEA-Research/grounding-dino-tiny")
    
    # Test with explicit ball prompts
    prompts = [
        "ball",
        "sports ball",
        "toy ball",
        "ball . person . chair .",
    ]
    
    results = {}
    
    for prompt in prompts:
        print(f"\n{'='*80}")
        print(f"Testing prompt: '{prompt}'")
        print(f"{'='*80}")
        
        detections_found = 0
        
        for frame_id in test_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Run detection
            dets = gdino.detect(frame, prompt, confidence_threshold=0.10)
            
            # Check for ball detections
            ball_dets = [d for d in dets if 'ball' in d['label'].lower()]
            
            if ball_dets:
                print(f"  Frame {frame_id}: Found {len(ball_dets)} ball(s)")
                for det in ball_dets:
                    print(f"    - {det['label']}: conf={det['confidence']:.3f}, bbox={det['bbox']}")
                detections_found += len(ball_dets)
            else:
                print(f"  Frame {frame_id}: No balls detected (total dets: {len(dets)})")
        
        results[prompt] = detections_found
        print(f"\nTotal ball detections with '{prompt}': {detections_found}")
    
    cap.release()
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for prompt, count in results.items():
        print(f"  '{prompt}': {count} ball detections")
    
    # Find best prompt
    best_prompt = max(results.items(), key=lambda x: x[1])
    print(f"\nBest prompt: '{best_prompt[0]}' with {best_prompt[1]} detections")
    
    return results


if __name__ == '__main__':
    test_ball_detection()
