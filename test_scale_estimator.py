#!/usr/bin/env python3
"""
Quick test for ScaleEstimator functionality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from orion.perception.scale_estimator import ScaleEstimator, OBJECT_SIZE_PRIORS

def test_scale_estimator():
    """Test scale estimation with synthetic data"""
    
    print("="*80)
    print("SCALE ESTIMATOR TEST")
    print("="*80)
    
    # Initialize
    estimator = ScaleEstimator(
        min_estimates=5,  # Lower threshold for testing
        confidence_threshold=0.7,
        outlier_threshold=2.0
    )
    
    print(f"\n✓ Initialized estimator (min_estimates=5)")
    print(f"✓ Loaded {len(OBJECT_SIZE_PRIORS)} object size priors")
    
    # Test 1: Door detection (most reliable)
    print("\n" + "="*80)
    print("TEST 1: Door Detection (2.1m tall)")
    print("="*80)
    
    # Simulate: Door appears 200px tall at 5m distance
    # Expected scale: 2.1m / (200px * 5m) = 0.0021 m/px/m = 2.1/1000 = 0.0021
    # But our formula is: scale = real_height / (bbox_height * depth / 1000)
    # So: scale = 2.1 / (200 * 5000/1000) = 2.1 / 1000 = 0.0021
    
    bbox_door = (100, 100, 200, 300)  # 100px wide, 200px tall
    depth_roi_door = np.full((200, 100), 5000.0)  # 5000mm (5m)
    
    estimate1 = estimator.estimate_from_object(bbox_door, depth_roi_door, 'door', frame_idx=10)
    
    if estimate1:
        print(f"✓ Estimate 1: scale={estimate1.scale:.6f} m/unit, confidence={estimate1.confidence:.2f}")
        estimator.add_estimate(estimate1)
    else:
        print("✗ Failed to generate estimate 1")
    
    # Test 2: Person detection
    print("\n" + "="*80)
    print("TEST 2: Person Detection (1.7m tall)")
    print("="*80)
    
    bbox_person = (300, 150, 400, 350)  # 100px wide, 200px tall
    depth_roi_person = np.full((200, 100), 4500.0)  # 4500mm (4.5m)
    
    estimate2 = estimator.estimate_from_object(bbox_person, depth_roi_person, 'person', frame_idx=20)
    
    if estimate2:
        print(f"✓ Estimate 2: scale={estimate2.scale:.6f} m/unit, confidence={estimate2.confidence:.2f}")
        estimator.add_estimate(estimate2)
    else:
        print("✗ Failed to generate estimate 2")
    
    # Test 3: Another door (consistency check)
    print("\n" + "="*80)
    print("TEST 3: Another Door (consistency)")
    print("="*80)
    
    bbox_door2 = (500, 50, 600, 250)  # 100px wide, 200px tall
    depth_roi_door2 = np.full((200, 100), 5200.0)  # 5200mm (5.2m)
    
    estimate3 = estimator.estimate_from_object(bbox_door2, depth_roi_door2, 'door', frame_idx=30)
    
    if estimate3:
        print(f"✓ Estimate 3: scale={estimate3.scale:.6f} m/unit, confidence={estimate3.confidence:.2f}")
        estimator.add_estimate(estimate3)
    else:
        print("✗ Failed to generate estimate 3")
    
    # Test 4: Laptop (smaller object)
    print("\n" + "="*80)
    print("TEST 4: Laptop Detection (0.35m wide)")
    print("="*80)
    
    bbox_laptop = (200, 400, 350, 450)  # 150px wide, 50px tall
    depth_roi_laptop = np.full((50, 150), 2000.0)  # 2000mm (2m)
    
    estimate4 = estimator.estimate_from_object(bbox_laptop, depth_roi_laptop, 'laptop', frame_idx=40)
    
    if estimate4:
        print(f"✓ Estimate 4: scale={estimate4.scale:.6f} m/unit, confidence={estimate4.confidence:.2f}")
        estimator.add_estimate(estimate4)
    else:
        print("✗ Failed to generate estimate 4")
    
    # Test 5: Chair
    print("\n" + "="*80)
    print("TEST 5: Chair Detection (0.9m tall)")
    print("="*80)
    
    bbox_chair = (400, 300, 500, 480)  # 100px wide, 180px tall
    depth_roi_chair = np.full((180, 100), 3500.0)  # 3500mm (3.5m)
    
    estimate5 = estimator.estimate_from_object(bbox_chair, depth_roi_chair, 'chair', frame_idx=50)
    
    if estimate5:
        print(f"✓ Estimate 5: scale={estimate5.scale:.6f} m/unit, confidence={estimate5.confidence:.2f}")
        estimator.add_estimate(estimate5)
    else:
        print("✗ Failed to generate estimate 5")
    
    # Get final statistics
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    stats = estimator.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"  Total estimates: {stats['total_estimates']}")
    print(f"  Scale locked: {'✓' if stats['scale_locked'] else '✗'}")
    
    if stats['committed_scale']:
        print(f"  ✓ Committed scale: {stats['committed_scale']:.6f} meters/unit")
        print(f"  ✓ Source objects: {', '.join(stats['source_classes'])}")
    elif stats['provisional_scale']:
        print(f"  Provisional scale: {stats['provisional_scale']:.6f} meters/unit")
        print(f"  ⚠️  Not locked yet (need {5 - stats['total_estimates']} more estimates)")
    else:
        print(f"  ✗ No scale estimates available")
    
    # Interpretation
    print(f"\n💡 Interpretation:")
    if stats['committed_scale']:
        scale = stats['committed_scale']
        print(f"  - A SLAM coordinate of 1000 units = {scale * 1000:.2f} meters")
        print(f"  - A SLAM coordinate of 1 unit = {scale * 1000:.2f} millimeters")
        print(f"  - Example: Object at (1500, 2000, 3000) SLAM units")
        print(f"    → Real position: ({1500*scale:.2f}m, {2000*scale:.2f}m, {3000*scale:.2f}m)")
    
    print("\n" + "="*80)
    print("✓ Test completed successfully!")
    print("="*80)

if __name__ == "__main__":
    test_scale_estimator()
