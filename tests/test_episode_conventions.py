"""
Test Episode Conventions
========================

Validate episode and results directory structure and schema compliance.
"""

import json
import pytest
from pathlib import Path

from orion.config.data_paths import (
    get_episode_dir,
    get_results_dir,
    load_episode_meta,
    load_episode_gt,
    validate_episode_structure,
    list_episodes,
)


class TestEpisodeStructure:
    """Test episode directory structure and metadata."""
    
    def test_demo_room_exists(self):
        """Demo room episode should exist."""
        episodes = list_episodes()
        assert "demo_room" in episodes, "demo_room episode not found"
    
    def test_demo_room_meta_schema(self):
        """Demo room meta.json should have required fields."""
        meta = load_episode_meta("demo_room")
        
        # Required fields
        assert "episode_id" in meta
        assert "source" in meta
        assert "video" in meta
        assert "created_at" in meta
        
        # Video fields
        video = meta["video"]
        assert "fps" in video
        assert "resolution" in video
        assert "duration_seconds" in video
        
        # Validate types
        assert isinstance(meta["episode_id"], str)
        assert isinstance(video["fps"], (int, float))
        assert isinstance(video["resolution"], list)
        assert len(video["resolution"]) == 2
        assert isinstance(video["duration_seconds"], (int, float))
    
    def test_demo_room_gt_schema(self):
        """Demo room gt.json should have valid structure if present."""
        gt = load_episode_gt("demo_room")
        
        if gt is None:
            pytest.skip("No ground truth for demo_room")
        
        # Required fields
        assert "objects" in gt
        assert isinstance(gt["objects"], list)
        
        # Validate object schema
        for obj in gt["objects"]:
            assert "id" in obj
            assert "class" in obj
            assert "spans" in obj
            assert isinstance(obj["spans"], list)
            
            # Validate spans
            for span in obj["spans"]:
                assert "start_frame" in span
                assert "end_frame" in span
                assert isinstance(span["start_frame"], int)
                assert isinstance(span["end_frame"], int)
                assert span["start_frame"] <= span["end_frame"]
    
    def test_validate_demo_room_structure(self):
        """Validate demo_room structure using validator."""
        checks = validate_episode_structure("demo_room")
        
        assert checks["episode_dir_exists"], "Episode directory must exist"
        assert checks["meta_json_exists"], "meta.json must exist"
        
        # Note: video and frames are optional for Phase 0


class TestResultsStructure:
    """Test results directory conventions."""
    
    def test_results_dir_creation(self):
        """Results directory should be creatable."""
        from orion.config.data_paths import ensure_results_dir
        
        test_episode = "test_episode_12345"
        results_dir = ensure_results_dir(test_episode)
        
        assert results_dir.exists()
        assert results_dir.is_dir()
        
        # Cleanup
        results_dir.rmdir()
    
    def test_save_results_json(self):
        """Should save results as JSON."""
        from orion.config.data_paths import save_results_json
        
        test_episode = "test_episode_12345"
        test_data = {
            "total_entities": 3,
            "entities": [
                {"id": 0, "class": "mug", "confidence": 0.95}
            ]
        }
        
        output_path = save_results_json(test_episode, "entities.json", test_data)
        
        assert output_path.exists()
        
        # Verify contents
        with open(output_path) as f:
            loaded = json.load(f)
        
        assert loaded["total_entities"] == 3
        assert len(loaded["entities"]) == 1
        
        # Cleanup
        output_path.unlink()
        output_path.parent.rmdir()
    
    def test_save_results_jsonl(self):
        """Should save results as JSONL."""
        from orion.config.data_paths import save_results_jsonl
        
        test_episode = "test_episode_12345"
        test_data = [
            {"frame_id": 1, "track_id": 1, "bbox": [100, 200, 300, 400]},
            {"frame_id": 2, "track_id": 1, "bbox": [105, 205, 305, 405]},
        ]
        
        output_path = save_results_jsonl(test_episode, "tracks.jsonl", test_data)
        
        assert output_path.exists()
        
        # Verify contents (line-delimited JSON)
        with open(output_path) as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        
        frame1 = json.loads(lines[0])
        assert frame1["frame_id"] == 1
        assert frame1["track_id"] == 1
        
        # Cleanup
        output_path.unlink()
        output_path.parent.rmdir()


class TestEpisodeMetadataValidation:
    """Validate metadata field constraints."""
    
    def test_meta_fps_positive(self):
        """FPS should be positive."""
        meta = load_episode_meta("demo_room")
        assert meta["video"]["fps"] > 0
    
    def test_meta_resolution_valid(self):
        """Resolution should be [width, height] with positive values."""
        meta = load_episode_meta("demo_room")
        width, height = meta["video"]["resolution"]
        assert width > 0
        assert height > 0
    
    def test_meta_duration_positive(self):
        """Duration should be positive."""
        meta = load_episode_meta("demo_room")
        assert meta["video"]["duration_seconds"] > 0
    
    def test_meta_episode_id_matches_directory(self):
        """Episode ID in meta.json should match directory name."""
        meta = load_episode_meta("demo_room")
        assert meta["episode_id"] == "demo_room"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
