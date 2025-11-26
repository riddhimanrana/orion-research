"""
Test Tracking Baseline
=======================

Validate detection + tracking pipeline with metrics.
"""

import pytest
import json
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.perception.yolo_detector import YOLODetector
from orion.perception.byte_tracker import ObjectTracker
from orion.config import list_episodes, get_results_dir


class TestYOLODetector:
    """Test YOLO detection wrapper."""
    
    def test_detector_initialization(self):
        """Detector should initialize without errors."""
        detector = YOLODetector(
            model_name="yolo11n",  # Use nano for faster testing
            confidence_threshold=0.25,
            device="cpu",  # Use CPU for CI compatibility
        )
        
        assert detector.model is not None
        assert detector.confidence_threshold == 0.25
        assert len(detector.class_names) == 80  # COCO classes
    
    def test_detect_frame(self):
        """Should detect objects in a frame."""
        detector = YOLODetector(model_name="yolo11n", device="cpu")
        
        # Create dummy frame (640x480 RGB)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add a white square (simulates object)
        frame[200:300, 250:350, :] = 255
        
        detections = detector.detect_frame(frame, frame_idx=0, timestamp=0.0)
        
        # May or may not detect anything in random noise, but should not crash
        assert isinstance(detections, list)
        
        for det in detections:
            assert "bbox" in det
            assert "centroid" in det
            assert "category" in det
            assert "confidence" in det
            assert "frame_id" in det
            assert len(det["bbox"]) == 4
            assert len(det["centroid"]) == 2
    
    def test_get_model_info(self):
        """Should return model information."""
        detector = YOLODetector(model_name="yolo11n", device="cpu")
        info = detector.get_model_info()
        
        assert "model_name" in info
        assert "num_classes" in info
        assert "confidence_threshold" in info
        assert info["model_name"] == "yolo11n"


class TestObjectTracker:
    """Test object tracking."""
    
    def test_tracker_initialization(self):
        """Tracker should initialize with correct params."""
        tracker = ObjectTracker(
            iou_threshold=0.3,
            max_age=30,
        )
        
        assert tracker.iou_threshold == 0.3
        assert tracker.max_age == 30
        assert tracker.next_id == 1
        assert len(tracker.tracks) == 0
    
    def test_single_object_tracking(self):
        """Should track a single object across frames."""
        tracker = ObjectTracker()
        
        # Frame 1: First detection
        det1 = {
            "bbox": [100, 100, 200, 200],
            "centroid": [150, 150],
            "category": "person",
            "confidence": 0.9,
            "frame_id": 1,
            "timestamp": 0.0,
        }
        
        tracks1 = tracker.update([det1])
        
        # Should not confirm on first frame (tentative)
        assert len(tracks1) == 0
        
        # Frame 2: Same object moved slightly
        det2 = {
            "bbox": [105, 105, 205, 205],
            "centroid": [155, 155],
            "category": "person",
            "confidence": 0.9,
            "frame_id": 2,
            "timestamp": 0.2,
        }
        
        tracks2 = tracker.update([det2])
        assert len(tracks2) == 0  # Still tentative
        
        # Frame 3: Confirm
        det3 = {
            "bbox": [110, 110, 210, 210],
            "centroid": [160, 160],
            "category": "person",
            "confidence": 0.9,
            "frame_id": 3,
            "timestamp": 0.4,
        }
        
        tracks3 = tracker.update([det3])
        assert len(tracks3) == 1  # Now confirmed
        assert tracks3[0]["track_id"] == 1
        assert tracks3[0]["category"] == "person"
    
    def test_multiple_object_tracking(self):
        """Should track multiple objects."""
        tracker = ObjectTracker()
        
        # Frame 1: Two objects
        dets1 = [
            {
                "bbox": [100, 100, 200, 200],
                "centroid": [150, 150],
                "category": "person",
                "confidence": 0.9,
                "frame_id": 1,
                "timestamp": 0.0,
            },
            {
                "bbox": [300, 300, 400, 400],
                "centroid": [350, 350],
                "category": "car",
                "confidence": 0.85,
                "frame_id": 1,
                "timestamp": 0.0,
            }
        ]
        
        # Process 3 frames to confirm tracks
        for i in range(3):
            dets = [
                {
                    **dets1[0],
                    "bbox": [100 + i*5, 100 + i*5, 200 + i*5, 200 + i*5],
                    "frame_id": i + 1,
                },
                {
                    **dets1[1],
                    "bbox": [300 + i*3, 300 + i*3, 400 + i*3, 400 + i*3],
                    "frame_id": i + 1,
                }
            ]
            tracks = tracker.update(dets)
        
        # Both should be confirmed
        assert len(tracks) == 2
        track_ids = {t["track_id"] for t in tracks}
        assert len(track_ids) == 2  # Unique IDs
    
    def test_track_deletion(self):
        """Should delete tracks after max_age frames."""
        tracker = ObjectTracker(max_age=2)
        
        # Create and confirm a track
        det = {
            "bbox": [100, 100, 200, 200],
            "centroid": [150, 150],
            "category": "person",
            "confidence": 0.9,
            "frame_id": 1,
            "timestamp": 0.0,
        }
        
        # Confirm track over 3 frames
        for i in range(3):
            tracker.update([{**det, "frame_id": i+1}])
        
        stats1 = tracker.get_statistics()
        assert stats1["total_tracks"] == 1
        
        # Miss next 3 frames (exceeds max_age=2)
        for i in range(3):
            tracker.update([])
        
        stats2 = tracker.get_statistics()
        assert stats2["active_tracks"] == 0
    
    def test_get_statistics(self):
        """Should return tracking statistics."""
        tracker = ObjectTracker()
        stats = tracker.get_statistics()
        
        assert "total_tracks" in stats
        assert "active_tracks" in stats
        assert "frame_count" in stats
        assert stats["total_tracks"] == 0
        assert stats["frame_count"] == 0


class TestIntegration:
    """Test full detection + tracking pipeline."""
    
    @pytest.mark.skipif(
        not any("demo" in ep for ep in list_episodes()),
        reason="No demo episodes available"
    )
    def test_demo_episode_results(self):
        """Demo episode should have valid results if processed."""
        # Check if demo_room has been processed
        results_dir = get_results_dir("demo_room")
        tracks_file = results_dir / "tracks.jsonl"
        
        if not tracks_file.exists():
            pytest.skip("demo_room not yet processed")
        
        # Load and validate tracks
        tracks = []
        with open(tracks_file) as f:
            for line in f:
                track = json.loads(line)
                tracks.append(track)
        
        assert len(tracks) > 0, "Should have at least one track"
        
        # Validate track format
        for track in tracks:
            assert "track_id" in track
            assert "bbox" in track
            assert "category" in track
            assert "confidence" in track
            assert "frame_id" in track
            
            # Check types
            assert isinstance(track["track_id"], int)
            assert isinstance(track["bbox"], list)
            assert len(track["bbox"]) == 4
            assert isinstance(track["category"], str)
            assert 0 <= track["confidence"] <= 1
        
        # Check track ID consistency
        track_ids = {t["track_id"] for t in tracks}
        assert len(track_ids) > 0, "Should have unique track IDs"
        
        print(f"\n✓ Validated {len(tracks)} track observations")
        print(f"  Unique tracks: {len(track_ids)}")
        print(f"  Frames: {max(t['frame_id'] for t in tracks)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
