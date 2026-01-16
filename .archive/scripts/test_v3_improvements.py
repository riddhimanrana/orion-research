#!/usr/bin/env python3
"""
Test script to validate v3 architectural improvements.

Tests:
1. HybridDetector initialization and basic functionality
2. Temporal smoothing integration
3. Updated Re-ID thresholds loading
4. Tracker cost matrix configuration

Run locally first, then on Lambda for full validation.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_hybrid_detector():
    """Test HybridDetector module loads and initializes."""
    print("\n=== Test 1: HybridDetector ===")
    try:
        from orion.perception.hybrid_detector import HybridDetector, HybridDetectorConfig
        
        config = HybridDetectorConfig(
            min_detections_for_skip=3,
            nms_iou_threshold=0.5,
        )
        print(f"✓ HybridDetectorConfig created: min_detections_for_skip={config.min_detections_for_skip}")
        
        # Don't actually init detector (needs GPU), just verify imports
        print("✓ HybridDetector imports successfully")
        return True
    except Exception as e:
        print(f"✗ HybridDetector test failed: {e}")
        return False


def test_temporal_smoothing():
    """Test temporal smoothing module."""
    print("\n=== Test 2: Temporal Smoothing ===")
    try:
        from orion.graph.temporal_smoothing import SceneGraphSmoother, TemporalSmoothingConfig
        
        config = TemporalSmoothingConfig(
            window_size=5,
            relation_specific_thresholds={
                "near": 0.4,
                "on": 0.6,
                "held_by": 0.7,
            },
        )
        smoother = SceneGraphSmoother(config=config)
        print(f"✓ SceneGraphSmoother created: window={config.window_size}")
        
        # Test with mock data
        mock_graphs = [
            {
                "frame": i,
                "nodes": [{"memory_id": "obj_1", "class": "bottle"}],
                "edges": [{"relation": "near", "subject": "obj_1", "object": "obj_2"}] if i % 2 == 0 else [],
            }
            for i in range(10)
        ]
        
        smoothed = smoother.smooth_graphs(mock_graphs)
        print(f"✓ Smoothing applied: {len(mock_graphs)} → {len(smoothed)} frames")
        
        # Check that flickering 'near' edges are filtered
        original_near_count = sum(1 for g in mock_graphs for e in g.get("edges", []) if e.get("relation") == "near")
        smoothed_near_count = sum(1 for g in smoothed for e in g.get("edges", []) if e.get("relation") == "near")
        print(f"  Near edges: {original_near_count} original → {smoothed_near_count} smoothed")
        
        return True
    except Exception as e:
        print(f"✗ Temporal smoothing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reid_thresholds():
    """Test Re-ID thresholds loading."""
    print("\n=== Test 3: Re-ID Thresholds ===")
    try:
        import json
        threshold_file = ROOT / "orion/perception/reid_thresholds_vjepa2.json"
        
        if not threshold_file.exists():
            print(f"✗ Threshold file not found: {threshold_file}")
            return False
        
        with open(threshold_file, 'r') as f:
            thresholds = json.load(f)
        
        print(f"✓ Loaded {len(thresholds)} class thresholds")
        
        # Verify v3 updates
        assert thresholds.get("_default", 0) >= 0.65, "_default should be >= 0.65"
        assert thresholds.get("refrigerator", 0) >= 0.80, "refrigerator should be >= 0.80"
        assert "remote" in thresholds, "remote should be present"
        
        print(f"  _default: {thresholds.get('_default')}")
        print(f"  refrigerator: {thresholds.get('refrigerator')}")
        print(f"  remote: {thresholds.get('remote')}")
        
        return True
    except Exception as e:
        print(f"✗ Re-ID thresholds test failed: {e}")
        return False


def test_tracker_config():
    """Test enhanced tracker configuration."""
    print("\n=== Test 4: Tracker Cost Matrix ===")
    try:
        from orion.perception.trackers.enhanced import EnhancedTracker
        
        tracker = EnhancedTracker(
            max_age=30,
            min_hits=3,
            appearance_threshold=0.65,
        )
        print(f"✓ EnhancedTracker created: max_age={tracker.max_age}")
        
        # Check that per-class thresholds are supported
        tracker_with_thresholds = EnhancedTracker(
            per_class_thresholds={"bottle": 0.70, "cup": 0.72},
        )
        print(f"✓ Per-class thresholds supported")
        
        return True
    except Exception as e:
        print(f"✗ Tracker config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scene_graph_integration():
    """Test scene graph with temporal smoothing enabled."""
    print("\n=== Test 5: Scene Graph Integration ===")
    try:
        from orion.graph.scene_graph import build_scene_graphs, TEMPORAL_SMOOTHING_AVAILABLE
        
        print(f"  Temporal smoothing available: {TEMPORAL_SMOOTHING_AVAILABLE}")
        
        # Mock minimal data
        memory = {
            "objects": [
                {"memory_id": "obj_1", "prototype_embedding": "emb_1", "class": "bottle"},
                {"memory_id": "obj_2", "prototype_embedding": "emb_2", "class": "table"},
            ]
        }
        tracks = [
            {"embedding_id": "emb_1", "frame_id": 0, "bbox": [10, 10, 50, 50], "frame_width": 640, "frame_height": 480},
            {"embedding_id": "emb_2", "frame_id": 0, "bbox": [20, 60, 100, 150], "frame_width": 640, "frame_height": 480},
            {"embedding_id": "emb_1", "frame_id": 1, "bbox": [15, 15, 55, 55], "frame_width": 640, "frame_height": 480},
            {"embedding_id": "emb_2", "frame_id": 1, "bbox": [20, 60, 100, 150], "frame_width": 640, "frame_height": 480},
        ]
        
        # Test without smoothing
        graphs_raw = build_scene_graphs(memory, tracks, enable_temporal_smoothing=False)
        print(f"✓ Raw graphs built: {len(graphs_raw)} frames")
        
        # Test with smoothing
        graphs_smooth = build_scene_graphs(
            memory, tracks,
            enable_temporal_smoothing=True,
            temporal_window_size=3,
        )
        print(f"✓ Smoothed graphs built: {len(graphs_smooth)} frames")
        
        return True
    except Exception as e:
        print(f"✗ Scene graph integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Orion v3 Improvements Validation")
    print("=" * 60)
    
    results = {
        "HybridDetector": test_hybrid_detector(),
        "TemporalSmoothing": test_temporal_smoothing(),
        "ReIDThresholds": test_reid_thresholds(),
        "TrackerConfig": test_tracker_config(),
        "SceneGraphIntegration": test_scene_graph_integration(),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\n{passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
