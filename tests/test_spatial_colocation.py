"""
Tests for spatial co-location zone system
"""

import pytest
import numpy as np

from orion.spatial_colocation import (
    BoundingBox,
    EntityAppearance,
    SpatialCoLocationZone,
    SpatialCoLocationAnalyzer,
    extract_appearances_from_observations
)


class TestBoundingBox:
    def test_center(self):
        bbox = BoundingBox(x1=10, y1=20, x2=30, y2=40)
        assert bbox.center == (20, 30)
    
    def test_area(self):
        bbox = BoundingBox(x1=10, y1=20, x2=30, y2=40)
        assert bbox.area == 400  # (30-10) * (40-20)
    
    def test_iou_no_overlap(self):
        bbox1 = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        bbox2 = BoundingBox(x1=20, y1=20, x2=30, y2=30)
        assert bbox1.iou(bbox2) == 0.0
    
    def test_iou_full_overlap(self):
        bbox1 = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        bbox2 = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        assert bbox1.iou(bbox2) == 1.0
    
    def test_iou_partial_overlap(self):
        bbox1 = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        bbox2 = BoundingBox(x1=5, y1=5, x2=15, y2=15)
        # Intersection = 5*5 = 25
        # Union = 100 + 100 - 25 = 175
        # IoU = 25/175 ≈ 0.1429
        iou = bbox1.iou(bbox2)
        assert 0.14 < iou < 0.15


class TestEntityAppearance:
    def test_creation(self):
        bbox = BoundingBox(x1=10, y1=20, x2=30, y2=40)
        appearance = EntityAppearance(
            entity_id="entity_001",
            frame_idx=10,
            timestamp_ms=333.33,
            bbox=bbox,
            class_name="person",
            confidence=0.95
        )
        
        assert appearance.entity_id == "entity_001"
        assert appearance.frame_idx == 10
        assert appearance.class_name == "person"


class TestSpatialCoLocationZone:
    def test_zone_creation(self):
        bbox = BoundingBox(x1=100, y1=200, x2=300, y2=400)
        zone = SpatialCoLocationZone(
            zone_id="zone_001",
            spatial_bounds=bbox,
            frame_width=1920,
            frame_height=1080,
            frame_range=(10, 50),
            duration_ms=1333.33,
            entity_ids={"entity_001", "entity_002"},
            entity_appearances={
                "entity_001": [10, 11, 12, 20, 30],
                "entity_002": [10, 15, 20, 30, 40]
            },
            location_descriptor="left side"
        )
        
        assert zone.zone_id == "zone_001"
        assert len(zone.entity_ids) == 2
        assert zone.total_frames == 41  # 50 - 10 + 1
        assert zone.max_concurrent_entities == 2  # Both entities appear in frames 10, 20, 30
    
    def test_location_descriptor_center(self):
        # Center of frame
        bbox = BoundingBox(x1=900, y1=500, x2=1000, y2=600)
        zone = SpatialCoLocationZone(
            zone_id="zone_001",
            spatial_bounds=bbox,
            frame_width=1920,
            frame_height=1080,
            frame_range=(0, 10),
            duration_ms=333.33,
            entity_ids={"e1"},
            entity_appearances={"e1": [0]},
            location_descriptor=""
        )
        
        descriptor = zone.get_location_descriptor()
        assert "center" in descriptor.lower()
    
    def test_location_descriptor_left(self):
        # Left side
        bbox = BoundingBox(x1=100, y1=500, x2=200, y2=600)
        zone = SpatialCoLocationZone(
            zone_id="zone_001",
            spatial_bounds=bbox,
            frame_width=1920,
            frame_height=1080,
            frame_range=(0, 10),
            duration_ms=333.33,
            entity_ids={"e1"},
            entity_appearances={"e1": [0]},
            location_descriptor=""
        )
        
        descriptor = zone.get_location_descriptor()
        assert "left" in descriptor.lower()


class TestSpatialCoLocationAnalyzer:
    def test_single_entity_no_zones(self):
        """Single entity should not create a co-location zone"""
        analyzer = SpatialCoLocationAnalyzer()
        
        appearances = [
            EntityAppearance(
                entity_id="e1",
                frame_idx=0,
                timestamp_ms=0,
                bbox=BoundingBox(100, 100, 200, 200),
                class_name="person",
                confidence=0.9
            )
        ]
        
        zones = analyzer.analyze_colocation_zones(appearances, 1920, 1080)
        assert len(zones) == 0  # Need at least 2 entities
    
    def test_two_entities_close_together(self):
        """Two entities close together should create a zone"""
        analyzer = SpatialCoLocationAnalyzer(
            proximity_threshold_pixels=500,  # Increased threshold to ensure they're considered "close"
            min_frames_for_zone=2
        )
        
        appearances = []
        # Entity 1 and 2 are close together for 10 frames
        for frame_idx in range(10):
            appearances.append(EntityAppearance(
                entity_id="e1",
                frame_idx=frame_idx,
                timestamp_ms=frame_idx * 33.33,
                bbox=BoundingBox(100, 100, 200, 200),
                class_name="person",
                confidence=0.9
            ))
            appearances.append(EntityAppearance(
                entity_id="e2",
                frame_idx=frame_idx,
                timestamp_ms=frame_idx * 33.33,
                bbox=BoundingBox(150, 150, 250, 250),  # Close to e1
                class_name="person",
                confidence=0.9
            ))
        
        zones = analyzer.analyze_colocation_zones(appearances, 1920, 1080)
        
        # Should create at least one zone with both entities
        assert len(zones) > 0
        zone = zones[0]
        assert len(zone.entity_ids) == 2
        assert "e1" in zone.entity_ids
        assert "e2" in zone.entity_ids
    
    def test_two_entities_far_apart_no_zones(self):
        """Two entities far apart should not create a zone"""
        analyzer = SpatialCoLocationAnalyzer(
            proximity_threshold_pixels=100,
            min_frames_for_zone=2
        )
        
        appearances = []
        # Entity 1 and 2 are far apart
        for frame_idx in range(10):
            appearances.append(EntityAppearance(
                entity_id="e1",
                frame_idx=frame_idx,
                timestamp_ms=frame_idx * 33.33,
                bbox=BoundingBox(100, 100, 200, 200),
                class_name="person",
                confidence=0.9
            ))
            appearances.append(EntityAppearance(
                entity_id="e2",
                frame_idx=frame_idx,
                timestamp_ms=frame_idx * 33.33,
                bbox=BoundingBox(1000, 1000, 1100, 1100),  # Far from e1
                class_name="person",
                confidence=0.9
            ))
        
        zones = analyzer.analyze_colocation_zones(appearances, 1920, 1080)
        
        # Should not create a zone (too far apart)
        assert len(zones) == 0


class TestExtractAppearances:
    def test_extract_from_observations(self):
        observations = [
            {
                "entity_id": "e1",
                "frame_idx": 0,
                "bbox": [100, 100, 200, 200],
                "class": "person",
                "confidence": 0.9
            },
            {
                "entity_id": "e2",
                "frame_idx": 0,
                "bbox": [300, 300, 400, 400],
                "class": "car",
                "confidence": 0.85
            }
        ]
        
        appearances = extract_appearances_from_observations(observations, fps=30.0)
        
        assert len(appearances) == 2
        assert appearances[0].entity_id == "e1"
        assert appearances[0].class_name == "person"
        assert appearances[0].bbox.center == (150, 150)
        assert appearances[1].entity_id == "e2"
        assert appearances[1].class_name == "car"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
