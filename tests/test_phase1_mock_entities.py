#!/usr/bin/env python3
"""
Test Phase 1 with mock entities to verify 3D backprojection and occlusion detection.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from orion.perception import (
    PerceptionEngine,
    PerceptionConfig,
    CameraIntrinsics,
    EntityState,
)
from orion.perception.config import (
    DepthConfig,
    HandTrackingConfig,
    OcclusionConfig,
    CameraConfig,
)


def create_mock_entities():
    """Create mock YOLO detections for testing."""
    detections = [
        {
            'entity_id': 'obj_1',
            'class': 'cup',
            'confidence': 0.95,
            'bbox': (300, 400, 500, 700),  # (x1, y1, x2, y2)
        },
        {
            'entity_id': 'obj_2',
            'class': 'bottle',
            'confidence': 0.87,
            'bbox': (600, 300, 750, 650),
        },
        {
            'entity_id': 'obj_3',
            'class': 'phone',
            'confidence': 0.92,
            'bbox': (200, 800, 400, 1100),
        },
    ]
    return detections


def test_mock_entities():
    """Test Phase 1 perception with mock entities."""
    
    print("="*80)
    print("Phase 1 Mock Entities Test")
    print("="*80)
    
    # Load test frame
    video_path = "data/examples/video_short.mp4"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Failed to read video frame")
        return
    
    height, width = frame.shape[:2]
    print(f"\n📐 Frame size: {width}x{height}")
    
    # Initialize perception engine
    config = PerceptionConfig(
        depth=DepthConfig(model_name="zoe", device=None),
        hand_tracking=HandTrackingConfig(max_num_hands=2),
        occlusion=OcclusionConfig(depth_margin_mm=100.0),
        camera=CameraConfig(width=width, height=height, auto_estimate=True),
        enable_hands=True,
        enable_depth=True,
        enable_occlusion=True,
    )
    
    print("\n🔧 Initializing perception engine...")
    engine = PerceptionEngine(config)
    print("✅ Engine initialized")
    
    # Create mock entities
    mock_detections = create_mock_entities()
    print(f"\n📦 Created {len(mock_detections)} mock detections:")
    for det in mock_detections:
        print(f"  - {det['class']}: bbox={det['bbox']}, conf={det['confidence']:.2f}")
    
    # Process frame
    print("\n⚙️  Processing frame with mock detections...")
    result = engine.process_frame(frame, mock_detections, frame_number=0, timestamp=0.0)
    
    # Print results
    print(f"\n{'='*80}")
    print("Results")
    print(f"{'='*80}")
    
    # Depth statistics
    if result.depth_map is not None:
        print(f"\n📏 Depth Map:")
        print(f"  Shape: {result.depth_map.shape}")
        print(f"  Range: {np.min(result.depth_map):.0f}mm - {np.max(result.depth_map):.0f}mm")
        print(f"  Mean: {np.mean(result.depth_map):.0f}mm")
    
    # Hands
    print(f"\n👋 Hands detected: {len(result.hands)}")
    for i, hand in enumerate(result.hands):
        palm = hand.palm_center_3d
        print(f"  Hand {i}: {hand.handedness}, {hand.pose.value}")
        print(f"    Palm 3D: ({palm[0]:.0f}, {palm[1]:.0f}, {palm[2]:.0f})mm")
    
    # Entities with 3D info
    print(f"\n🎯 Entities with 3D coordinates: {len(result.entities)}")
    for entity in result.entities:
        print(f"\n  {entity.class_label} (id={entity.entity_id}):")
        print(f"    2D bbox: {entity.bbox_2d_px}")
        if entity.centroid_3d_mm:
            x, y, z = entity.centroid_3d_mm
            print(f"    3D centroid: ({x:.0f}, {y:.0f}, {z:.0f})mm")
        if entity.depth_mean_mm:
            print(f"    Mean depth: {entity.depth_mean_mm:.0f}mm")
        if entity.bbox_3d_mm:
            print(f"    3D bbox: {entity.bbox_3d_mm}")
        print(f"    Visibility: {entity.visibility_state.value}")
        print(f"    Occlusion ratio: {entity.occlusion_ratio:.2f}")
        if entity.occluded_by:
            print(f"    Occluded by: {entity.occluded_by}")
    
    print(f"\n⏱️  Processing time: {result.processing_time_ms:.1f}ms")
    
    print(f"\n{'='*80}")
    print("✅ Test completed successfully!")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_mock_entities()
