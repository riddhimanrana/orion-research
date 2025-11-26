"""
Standalone Phase 1 unit tests.

Tests individual components without full integration.
"""

import sys
from pathlib import Path
import numpy as np

# Direct imports to avoid existing orion module conflicts
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_camera_intrinsics():
    """Test camera intrinsics calculation."""
    print("\n" + "="*60)
    print("Test: CameraIntrinsics")
    print("="*60)

    from orion.perception.types import CameraIntrinsics

    # Test auto-estimation
    intrinsics = CameraIntrinsics.auto_estimate(1920, 1080)

    assert intrinsics.width == 1920
    assert intrinsics.height == 1080
    assert intrinsics.fx > 0
    assert intrinsics.fy > 0
    assert intrinsics.cx == 960.0  # Center
    assert intrinsics.cy == 540.0  # Center

    print("✓ Auto-estimated intrinsics:")
    print(f"  fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")
    print(f"  cx={intrinsics.cx:.1f}, cy={intrinsics.cy:.1f}")


def test_backprojection():
    """Test 3D backprojection."""
    print("\n" + "="*60)
    print("Test: Backprojection")
    print("="*60)

    from orion.perception.types import CameraIntrinsics
    from orion.perception.camera_intrinsics import backproject_point, backproject_bbox

    intrinsics = CameraIntrinsics.auto_estimate(640, 480)

    # Test single point backprojection
    u, v = 320, 240  # Image center
    depth = 1000.0  # 1 meter

    X, Y, Z = backproject_point(u, v, depth, intrinsics)

    # At image center with correct intrinsics, X and Y should be ~0
    assert abs(X) < 10  # Within 1cm
    assert abs(Y) < 10
    assert abs(Z - depth) < 0.1

    print(f"✓ Backprojected point at center: ({X:.1f}, {Y:.1f}, {Z:.1f}) mm")

    # Test bbox backprojection
    bbox = (100, 100, 200, 200)
    depth_map = np.ones((480, 640), dtype=np.float32) * 1500.0  # Uniform 1.5m

    result = backproject_bbox(bbox, depth_map, intrinsics)

    assert 'centroid_3d' in result
    assert 'depth_mean' in result
    assert result['depth_mean'] > 0

    print("✓ Backprojected bbox:")
    print(f"  Centroid 3D: {result['centroid_3d']}")
    print(f"  Depth mean: {result['depth_mean']:.1f} mm")


def test_entity_state():
    """Test EntityState dataclass."""
    print("\n" + "="*60)
    print("Test: EntityState")
    print("="*60)

    from orion.perception.types import EntityState3D, VisibilityState

    entity = EntityState3D(
        entity_id='test_obj',
        frame_number=0,
        timestamp=0.0,
        class_label='cup',
        class_confidence=0.95,
        bbox_2d_px=(100, 100, 200, 200),
        centroid_2d_px=(150, 150),
        centroid_3d_mm=(0.0, 0.0, 1500.0),
        depth_mean_mm=1500.0,
        visibility_state=VisibilityState.FULLY_VISIBLE,
    )

    # Test serialization
    entity_dict = entity.to_dict()

    assert entity_dict['entity_id'] == 'test_obj'
    assert entity_dict['class_label'] == 'cup'
    assert entity_dict['visibility_state'] == 'fully_visible'

    print("✓ EntityState created and serialized:")
    print(f"  ID: {entity.entity_id}")
    print(f"  Class: {entity.class_label}")
    print(f"  3D position: {entity.centroid_3d_mm}")
    print(f"  Visibility: {entity.visibility_state.value}")


def test_hand_pose():
    """Test Hand dataclass."""
    print("\n" + "="*60)
    print("Test: Hand")
    print("="*60)

    from orion.perception.types import Hand, HandPose

    # Create dummy landmarks
    landmarks_2d = [(0.5, 0.5) for _ in range(21)]
    landmarks_3d = [(0.0, 0.0, 1000.0) for _ in range(21)]

    hand = Hand(
        id='hand_0',
        landmarks_2d=landmarks_2d,
        landmarks_3d=landmarks_3d,
        palm_center_3d=(0.0, 0.0, 1000.0),
        pose=HandPose.OPEN,
        confidence=0.95,
        handedness='Right',
    )

    # Test serialization
    hand_dict = hand.to_dict()

    assert hand_dict['id'] == 'hand_0'
    assert hand_dict['pose'] == 'open'
    assert hand_dict['handedness'] == 'Right'
    assert len(hand_dict['landmarks_2d']) == 21

    print("✓ Hand created and serialized:")
    print(f"  ID: {hand.id}")
    print(f"  Pose: {hand.pose.value}")
    print(f"  Handedness: {hand.handedness}")
    print(f"  Landmarks: {len(hand.landmarks_2d)}")


def test_perception_config():
    """Test PerceptionConfig."""
    print("\n" + "="*60)
    print("Test: PerceptionConfig")
    print("="*60)

    from orion.perception.config import PerceptionConfig

    config = PerceptionConfig()

    assert config.enable_depth is True
    assert config.enable_hands is True
    assert config.enable_occlusion is True
    assert config.depth.model_name in ["zoe", "midas"]

    print("✓ PerceptionConfig created:")
    print(f"  Depth enabled: {config.enable_depth}")
    print(f"  Hands enabled: {config.enable_hands}")
    print(f"  Occlusion enabled: {config.enable_occlusion}")
    print(f"  Depth model: {config.depth.model_name}")


def main():
    """Run all unit tests."""
    print("\n" + "="*60)
    print("Phase 1 Unit Tests")
    print("="*60)
    
    tests = [
        ('CameraIntrinsics', test_camera_intrinsics),
        ('Backprojection', test_backprojection),
        ('EntityState', test_entity_state),
        ('Hand', test_hand_pose),
        ('PerceptionConfig', test_perception_config),
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All Phase 1 unit tests passed!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install torch torchvision mediapipe")
        print("  2. Test with real models: python test_phase1_perception.py")
        print("  3. Integrate with YOLO pipeline")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
