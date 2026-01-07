#!/usr/bin/env python3
"""
Test script for CIS (Causal Influence Scoring) and Spatial Relations.

This validates:
1. 3D velocity alignment computation
2. Hand interaction detection
3. Spatial relationship computation
4. Edge generation and stability tracking
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class MockEntity:
    """Mock entity for testing CIS computation."""
    id: int
    object_class: str
    class_name: str  # Alias
    state: Optional[np.ndarray] = None  # [x, y, z, vx, vy, vz]
    bbox_3d: Optional[np.ndarray] = None  # [x, y, z, w, h, d]
    observations: List = None
    
    def __post_init__(self):
        if self.observations is None:
            self.observations = []
        if self.class_name is None:
            self.class_name = self.object_class


def test_velocity_alignment():
    """Test 3D velocity alignment computation."""
    print("\n=== Test: 3D Velocity Alignment ===")
    
    from orion.analysis.cis_scorer import CausalInfluenceScorer
    scorer = CausalInfluenceScorer()
    
    # Test 1: Same direction movement
    person = MockEntity(
        id=1,
        object_class="person",
        class_name="person",
        state=np.array([0, 0, 1000, 10, 0, 5]),  # Moving right and forward
    )
    cup = MockEntity(
        id=2,
        object_class="cup",
        class_name="cup",
        state=np.array([50, 0, 1000, 10, 0, 5]),  # Same velocity
    )
    
    score, cosine = scorer._motion_score_3d(person, cup)
    print(f"Same direction: score={score:.3f}, cosine={cosine:.3f}")
    assert score > 0.9, f"Expected high score for same direction, got {score}"
    
    # Test 2: Opposite direction
    cup_opposite = MockEntity(
        id=3,
        object_class="cup",
        class_name="cup",
        state=np.array([50, 0, 1000, -10, 0, -5]),  # Opposite velocity
    )
    score, cosine = scorer._motion_score_3d(person, cup_opposite)
    print(f"Opposite direction: score={score:.3f}, cosine={cosine:.3f}")
    assert score < 0.2, f"Expected low score for opposite direction, got {score}"
    
    # Test 3: Both stationary
    person_still = MockEntity(
        id=4,
        object_class="person",
        class_name="person",
        state=np.array([0, 0, 1000, 0, 0, 0]),
    )
    cup_still = MockEntity(
        id=5,
        object_class="cup",
        class_name="cup",
        state=np.array([50, 0, 1000, 0, 0, 0]),
    )
    score, _ = scorer._motion_score_3d(person_still, cup_still)
    print(f"Both stationary: score={score:.3f}")
    assert score > 0.7, f"Expected high score for both stationary, got {score}"
    
    # Test 4: One moving, one stationary
    score, _ = scorer._motion_score_3d(person, cup_still)
    print(f"One moving: score={score:.3f}")
    assert score < 0.3, f"Expected low score for one moving, got {score}"
    
    print("✓ Velocity alignment tests passed!")


def test_spatial_score():
    """Test 3D spatial score computation."""
    print("\n=== Test: 3D Spatial Score ===")
    
    from orion.analysis.cis_scorer import CausalInfluenceScorer
    scorer = CausalInfluenceScorer(max_spatial_distance_mm=600.0)
    
    # Test 1: Very close (< 5cm)
    person = MockEntity(
        id=1,
        object_class="person",
        class_name="person",
        state=np.array([0, 0, 1000, 0, 0, 0]),
    )
    cup_close = MockEntity(
        id=2,
        object_class="cup",
        class_name="cup",
        state=np.array([30, 0, 1000, 0, 0, 0]),  # 30mm away
    )
    score, dist = scorer._spatial_score_3d(person, cup_close)
    print(f"Close (30mm): score={score:.3f}, dist={dist:.1f}mm")
    assert score > 0.95, f"Expected very high score for close, got {score}"
    
    # Test 2: At threshold (600mm)
    cup_far = MockEntity(
        id=3,
        object_class="cup",
        class_name="cup",
        state=np.array([600, 0, 1000, 0, 0, 0]),
    )
    score, dist = scorer._spatial_score_3d(person, cup_far)
    print(f"At threshold (600mm): score={score:.3f}, dist={dist:.1f}mm")
    assert score < 0.1, f"Expected low score at threshold, got {score}"
    
    # Test 3: Beyond threshold
    cup_very_far = MockEntity(
        id=4,
        object_class="cup",
        class_name="cup",
        state=np.array([1000, 0, 1000, 0, 0, 0]),
    )
    score, dist = scorer._spatial_score_3d(person, cup_very_far)
    print(f"Beyond threshold (1000mm): score={score:.3f}, dist={dist:.1f}mm")
    assert score == 0.0, f"Expected zero score beyond threshold, got {score}"
    
    print("✓ Spatial score tests passed!")


def test_hand_interaction():
    """Test hand interaction bonus computation."""
    print("\n=== Test: Hand Interaction Detection ===")
    
    from orion.analysis.cis_scorer import CausalInfluenceScorer
    scorer = CausalInfluenceScorer()
    
    # Test 1: Hand keypoint inside object
    person = MockEntity(
        id=1,
        object_class="person",
        class_name="person",
        state=np.array([0, 0, 1000, 0, 0, 0]),
        bbox_3d=np.array([0, 0, 950, 500, 1800, 100]),  # Person bbox
    )
    cup = MockEntity(
        id=2,
        object_class="cup",
        class_name="cup",
        state=np.array([100, 900, 1000, 0, 0, 0]),
        bbox_3d=np.array([100, 900, 1000, 80, 100, 80]),  # Cup bbox
    )
    
    # Hand keypoint inside cup bbox
    hand_kps = [(100, 920, 1010)]  # Inside cup
    bonus, dist, interaction = scorer._hand_bonus_3d(person, cup, hand_kps)
    print(f"Hand inside object: bonus={bonus:.3f}, type={interaction}")
    assert interaction == "grasping", f"Expected grasping, got {interaction}"
    assert bonus == scorer.hand_grasping_bonus
    
    # Test 2: Hand keypoint near object (touching)
    hand_kps_near = [(130, 900, 1000)]  # ~30mm away from cup center
    bonus, dist, interaction = scorer._hand_bonus_3d(person, cup, hand_kps_near)
    print(f"Hand near object ({dist:.1f}mm): bonus={bonus:.3f}, type={interaction}")
    assert interaction == "touching", f"Expected touching, got {interaction}"
    
    # Test 3: No person - no bonus
    table = MockEntity(
        id=3,
        object_class="table",
        class_name="table",
        state=np.array([0, 0, 1000, 0, 0, 0]),
    )
    bonus, _, interaction = scorer._hand_bonus_3d(table, cup, None)
    print(f"Non-person agent: bonus={bonus:.3f}, type={interaction}")
    assert bonus == 0.0, f"Expected no bonus for non-person, got {bonus}"
    
    print("✓ Hand interaction tests passed!")


def test_full_cis_computation():
    """Test full CIS score computation."""
    print("\n=== Test: Full CIS Computation ===")
    
    from orion.analysis.cis_scorer import CausalInfluenceScorer
    scorer = CausalInfluenceScorer(cis_threshold=0.5)
    
    # Person picking up cup scenario
    person = MockEntity(
        id=1,
        object_class="person",
        class_name="person",
        state=np.array([0, 0, 1000, 10, 0, 5]),  # Moving
        bbox_3d=np.array([0, 0, 950, 500, 1800, 100]),
    )
    cup = MockEntity(
        id=2,
        object_class="cup",
        class_name="cup",
        state=np.array([50, 900, 1010, 10, 0, 5]),  # Moving with person
        bbox_3d=np.array([50, 900, 1010, 80, 100, 80]),
    )
    
    # Hand near cup
    hand_kps = [(60, 910, 1015)]
    
    cis_score, components = scorer.calculate_cis(
        person, cup, time_delta=0.0, hand_keypoints=hand_kps
    )
    
    print(f"\nCIS Score: {cis_score:.3f}")
    print(f"  Temporal: {components.temporal:.3f}")
    print(f"  Spatial:  {components.spatial:.3f} (dist={components.distance_3d_mm:.1f}mm)")
    print(f"  Motion:   {components.motion:.3f} (align={components.velocity_alignment:.3f})")
    print(f"  Semantic: {components.semantic:.3f}")
    print(f"  Hand:     {components.hand_bonus:.3f} ({components.interaction_type})")
    
    assert cis_score > 0.5, f"Expected high CIS for person-cup interaction, got {cis_score}"
    
    print("✓ Full CIS computation test passed!")


def test_edge_generation():
    """Test CIS edge generation for a frame."""
    print("\n=== Test: CIS Edge Generation ===")
    
    from orion.analysis.cis_scorer import CausalInfluenceScorer
    scorer = CausalInfluenceScorer(cis_threshold=0.4)
    
    # Create a scene with person and objects
    person = MockEntity(
        id=1,
        object_class="person",
        class_name="person",
        state=np.array([0, 0, 1000, 5, 0, 0]),
        bbox_3d=np.array([0, 0, 950, 500, 1800, 100]),
    )
    cup = MockEntity(
        id=2,
        object_class="cup",
        class_name="cup",
        state=np.array([100, 900, 1050, 5, 0, 0]),  # Near person, moving with
        bbox_3d=np.array([100, 900, 1050, 80, 100, 80]),
    )
    laptop = MockEntity(
        id=3,
        object_class="laptop",
        class_name="laptop",
        state=np.array([500, 800, 1200, 0, 0, 0]),  # Farther, stationary
        bbox_3d=np.array([500, 800, 1200, 300, 20, 200]),
    )
    
    entities = [person, cup, laptop]
    
    edges = scorer.compute_frame_edges(
        entities=entities,
        frame_id=0,
        timestamp=0.0,
    )
    
    print(f"\nGenerated {len(edges)} edges:")
    for edge in edges:
        print(f"  {edge.agent_class} --[{edge.relation_type}]--> {edge.patient_class}")
        print(f"    CIS: {edge.cis_score:.3f}, frame={edge.frame_id}")
    
    # Should have at least person→cup interaction
    person_cup_edges = [e for e in edges if e.agent_id == 1 and e.patient_id == 2]
    assert len(person_cup_edges) > 0, "Expected person→cup edge"
    
    print("✓ Edge generation test passed!")


def test_spatial_relations_engine():
    """Test the SpatialRelationEngine."""
    print("\n=== Test: Spatial Relations Engine ===")
    
    from orion.graph.spatial_relations import SpatialRelationEngine, Entity3D
    
    engine = SpatialRelationEngine(
        stability_frames=3,
        near_threshold_mm=300.0,
    )
    
    # Create entities
    table = Entity3D(
        id=1,
        class_name="table",
        bbox_3d=np.array([0, 700, 1000, 800, 800, 1400]),
        bbox_2d=np.array([100, 300, 700, 500]),
        centroid_3d=np.array([400, 750, 1200]),
    )
    cup = Entity3D(
        id=2,
        class_name="cup",
        bbox_3d=np.array([350, 650, 1150, 430, 700, 1250]),
        bbox_2d=np.array([350, 280, 430, 350]),
        centroid_3d=np.array([390, 675, 1200]),
    )
    person = Entity3D(
        id=3,
        class_name="person",
        bbox_3d=np.array([500, 0, 900, 1000, 1800, 1300]),
        bbox_2d=np.array([200, 50, 600, 550]),
        centroid_3d=np.array([750, 900, 1100]),
        velocity_3d=np.array([0, 0, 0]),
    )
    
    entities = [table, cup, person]
    
    # Compute relations for multiple frames
    all_edges = []
    for frame in range(5):
        edges = engine.compute_frame_relations(entities, frame_id=frame)
        all_edges.extend(edges)
        print(f"Frame {frame}: {len(edges)} edges")
    
    # Check stable edges
    stable = engine.get_stable_edges()
    print(f"\nStable edges: {len(stable)}")
    for edge in stable:
        print(f"  {edge.subject_id} --[{edge.predicate.value}]--> {edge.object_id}")
        print(f"    confidence={edge.confidence:.3f}, consecutive={edge.consecutive_frames}")
    
    print("✓ Spatial relations engine test passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("CIS & Spatial Relations Test Suite")
    print("=" * 60)
    
    try:
        test_velocity_alignment()
        test_spatial_score()
        test_hand_interaction()
        test_full_cis_computation()
        test_edge_generation()
        test_spatial_relations_engine()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
