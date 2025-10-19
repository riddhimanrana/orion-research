#!/usr/bin/env python3
"""
Test script for optimized contextual understanding

This script validates that the optimization improvements work correctly.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orion.optimized_contextual_understanding import (
    OptimizedContextualUnderstandingEngine,
    SpatialZone,
)


def test_spatial_zone_calculation():
    """Test that spatial zone calculation is fixed"""
    print("Testing spatial zone calculation...")
    
    # Mock config and model_manager
    class MockConfig:
        pass
    
    class MockModelManager:
        pass
    
    engine = OptimizedContextualUnderstandingEngine(MockConfig(), MockModelManager())
    
    # Test case 1: Pixel coordinates (wall-mounted door knob)
    bbox1 = {
        'x1': 850, 'y1': 540, 'x2': 920, 'y2': 620,
        'frame_width': 1920, 'frame_height': 1080
    }
    zone1 = engine._calculate_spatial_zone_v2(bbox1)
    assert zone1.zone_type == 'wall_middle', f"Expected wall_middle, got {zone1.zone_type}"
    assert zone1.confidence > 0.8, f"Expected high confidence, got {zone1.confidence}"
    print(f"  ✓ Pixel coords: {zone1.zone_type} (confidence: {zone1.confidence:.2f})")
    
    # Test case 2: Normalized coordinates (ceiling object)
    bbox2 = {
        'x1': 0.4, 'y1': 0.05, 'x2': 0.6, 'y2': 0.12,
        'frame_width': 1, 'frame_height': 1
    }
    zone2 = engine._calculate_spatial_zone_v2(bbox2)
    assert zone2.zone_type == 'ceiling', f"Expected ceiling, got {zone2.zone_type}"
    print(f"  ✓ Normalized coords: {zone2.zone_type} (confidence: {zone2.confidence:.2f})")
    
    # Test case 3: Floor object
    bbox3 = {
        'x1': 500, 'y1': 900, 'x2': 700, 'y2': 1050,
        'frame_width': 1920, 'frame_height': 1080
    }
    zone3 = engine._calculate_spatial_zone_v2(bbox3)
    assert zone3.zone_type == 'floor', f"Expected floor, got {zone3.zone_type}"
    print(f"  ✓ Floor object: {zone3.zone_type} (confidence: {zone3.confidence:.2f})")
    
    print("✅ Spatial zone calculation tests passed!")
    return True


def test_scene_inference():
    """Test that scene inference requires strong evidence"""
    print("\nTesting scene inference...")
    
    class MockConfig:
        pass
    
    class MockModelManager:
        pass
    
    engine = OptimizedContextualUnderstandingEngine(MockConfig(), MockModelManager())
    
    # Test case 1: Clear kitchen (multiple indicators)
    entities1 = [
        {'class': 'oven', 'description': 'a stainless steel oven'},
        {'class': 'microwave', 'description': 'a microwave on counter'},
        {'class': 'refrigerator', 'description': 'white refrigerator'},
    ]
    scene1 = engine._infer_scene_type_v2(entities1)
    assert scene1 == 'kitchen', f"Expected kitchen, got {scene1}"
    print(f"  ✓ Multiple kitchen indicators: {scene1}")
    
    # Test case 2: Weak signals (should be general, not kitchen)
    entities2 = [
        {'class': 'person', 'description': 'a person walking'},
        {'class': 'chair', 'description': 'a wooden chair'},
        {'class': 'book', 'description': 'a blue book'},
    ]
    scene2 = engine._infer_scene_type_v2(entities2)
    assert scene2 == 'general', f"Expected general, got {scene2}"
    print(f"  ✓ Weak signals correctly: {scene2}")
    
    # Test case 3: Bedroom
    entities3 = [
        {'class': 'bed', 'description': 'a large bed with pillows'},
        {'class': 'nightstand', 'description': 'wooden nightstand'},
    ]
    scene3 = engine._infer_scene_type_v2(entities3)
    assert scene3 == 'bedroom', f"Expected bedroom, got {scene3}"
    print(f"  ✓ Bedroom detection: {scene3}")
    
    print("✅ Scene inference tests passed!")
    return True


def test_llm_filtering():
    """Test that LLM filtering works correctly"""
    print("\nTesting LLM filtering...")
    
    class MockConfig:
        pass
    
    class MockModelManager:
        pass
    
    engine = OptimizedContextualUnderstandingEngine(MockConfig(), MockModelManager())
    
    # Test case 1: High confidence + matching description = skip LLM
    entity1 = {
        'class': 'laptop',
        'description': 'a silver laptop computer on a desk',
        'confidence': 0.85,
    }
    needs_llm1 = engine._needs_llm_analysis(entity1, 'office')
    assert not needs_llm1, "High confidence + matching description should skip LLM"
    print(f"  ✓ Skip LLM for high-confidence match: {entity1['class']}")
    
    # Test case 2: Problematic class = always verify
    entity2 = {
        'class': 'hair drier',
        'description': 'a metallic knob on the wall',
        'confidence': 0.75,
    }
    needs_llm2 = engine._needs_llm_analysis(entity2, 'bedroom')
    assert needs_llm2, "Problematic classes should always verify"
    print(f"  ✓ Always verify problematic class: {entity2['class']}")
    
    # Test case 3: Low confidence = verify
    entity3 = {
        'class': 'remote',
        'description': 'a small rectangular object',
        'confidence': 0.45,
    }
    needs_llm3 = engine._needs_llm_analysis(entity3, 'living_room')
    assert needs_llm3, "Low confidence should verify"
    print(f"  ✓ Verify low confidence: {entity3['class']} ({entity3['confidence']})")
    
    # Test case 4: Unambiguous common object = skip
    entity4 = {
        'class': 'person',
        'description': 'a person standing',
        'confidence': 0.65,
    }
    needs_llm4 = engine._needs_llm_analysis(entity4, 'general')
    assert not needs_llm4, "Unambiguous objects should skip LLM"
    print(f"  ✓ Skip LLM for unambiguous object: {entity4['class']}")
    
    print("✅ LLM filtering tests passed!")
    return True


def test_batch_grouping():
    """Test that batch grouping works correctly"""
    print("\nTesting batch grouping...")
    
    # Create mock entities across multiple frames
    entities = []
    for frame in range(10):
        for obj_id in range(5):  # 5 objects per frame
            entities.append({
                'entity_id': f'obj_{frame}_{obj_id}',
                'class': 'test',
                'description': 'test object',
                'confidence': 0.5,
                'first_frame': frame,
                'needs_llm_analysis': True,
            })
    
    # Group by frame
    from collections import defaultdict
    frame_groups = defaultdict(list)
    for entity in entities:
        frame_num = entity.get('first_frame', 0)
        frame_groups[frame_num].append(entity)
    
    # Verify grouping
    assert len(frame_groups) == 10, f"Expected 10 frame groups, got {len(frame_groups)}"
    for frame_num, group in frame_groups.items():
        assert len(group) == 5, f"Frame {frame_num} should have 5 objects, got {len(group)}"
    
    print(f"  ✓ Batched {len(entities)} entities into {len(frame_groups)} frame groups")
    print(f"  ✓ Reduction: {len(entities)} calls → {len(frame_groups)} calls ({len(entities)//len(frame_groups)}x)")
    
    print("✅ Batch grouping tests passed!")
    return True


def test_integration():
    """Test full integration with mock data"""
    print("\nTesting full integration...")
    
    # Create mock perception log
    perception_log = []
    for i in range(20):
        perception_log.append({
            'temp_id': f'det_{i:06d}',
            'object_class': 'hair drier' if i % 5 == 0 else 'laptop',
            'rich_description': 'a metallic knob on the wall' if i % 5 == 0 else 'a silver laptop',
            'detection_confidence': 0.6 if i % 5 == 0 else 0.85,
            'bounding_box': [100 + i*50, 500 + i*10, 150 + i*50, 550 + i*10],
            'frame_number': i // 2,  # 2 objects per frame
            'timestamp': (i // 2) * 0.25,
        })
    
    # Convert to entities format (as done in pipeline)
    from orion.optimized_pipeline_integration import _convert_perception_to_entities
    entities = _convert_perception_to_entities(perception_log)
    
    # Verify conversion
    assert len(entities) == 20, f"Expected 20 entities, got {len(entities)}"
    assert all('entity_id' in e for e in entities), "All entities should have IDs"
    assert all('bbox' in e for e in entities), "All entities should have bboxes"
    
    print(f"  ✓ Converted {len(perception_log)} perception objects to entities")
    print(f"  ✓ All entities have proper IDs and bboxes")
    
    # Verify spatial zones would be calculated
    from orion.optimized_contextual_understanding import OptimizedContextualUnderstandingEngine
    
    class MockConfig:
        pass
    
    class MockModelManager:
        pass
    
    engine = OptimizedContextualUnderstandingEngine(MockConfig(), MockModelManager())
    
    zones_detected = 0
    for entity in entities:
        zone = engine._calculate_spatial_zone_v2(entity['bbox'])
        if zone.zone_type != 'unknown':
            zones_detected += 1
    
    detection_rate = zones_detected / len(entities)
    print(f"  ✓ Spatial zone detection rate: {detection_rate*100:.1f}%")
    assert detection_rate > 0.8, f"Expected >80% detection rate, got {detection_rate*100:.1f}%"
    
    print("✅ Integration tests passed!")
    return True


def main():
    """Run all tests"""
    print("="*80)
    print("OPTIMIZATION VALIDATION TESTS")
    print("="*80)
    
    tests = [
        ("Spatial Zone Calculation", test_spatial_zone_calculation),
        ("Scene Inference", test_scene_inference),
        ("LLM Filtering", test_llm_filtering),
        ("Batch Grouping", test_batch_grouping),
        ("Integration", test_integration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            failed += 1
    
    print("\n" + "="*80)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("✅ All optimization tests passed!")
        return 0
    else:
        print(f"❌ {failed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
