#!/usr/bin/env python3
"""
Quick Reference: What to Test for CIS & Temporal
=================================================

Simplified, actionable testing checklist for you.
"""

# =============================================================================
# CIS COMPONENT VALUES (from hpo_results/cis_weights.json)
# =============================================================================

CIS_CONFIG = {
    "weight_temporal": 0.296,        # 30%  - When did it happen?
    "weight_spatial": 0.436,         # 44%  - How close? (DOMINANT)
    "weight_motion": 0.208,          # 21%  - Moving toward?
    "weight_semantic": 0.060,        # 6%   - Related objects?
    "cis_threshold": 0.543,          # 54%  - Decision boundary
    "max_spatial_distance": 600.0,   # px
    "temporal_decay_seconds": 4.0,   # s
}

# =============================================================================
# TEST THESE 10 THINGS (PRIORITY ORDER)
# =============================================================================

MUST_TEST = """
1. TEMPORAL PROXIMITY
   ✓ Formula: decay = exp(-time_delta / 4.0)
   ✓ Test: t=0 → score=1.0
   ✓ Test: t=4 → score≈0.37 (1/e)
   ✓ Test: t>20 → score→0

2. SPATIAL PROXIMITY
   ✓ Formula: (1 - min(distance/600, 1))^2
   ✓ Test: distance=0 → score=1.0
   ✓ Test: distance=150 → score≈0.56
   ✓ Test: distance=600 → score=0.0

3. MOTION ALIGNMENT
   ✓ Formula: 0.6*alignment + 0.4*speed_norm
   ✓ Test: Moving toward target → high score
   ✓ Test: Moving away from target → low score
   ✓ Test: Stationary → ~0.25

4. SEMANTIC PROXIMITY
   ✓ Formula: (cosine_similarity + 1) / 2
   ✓ Test: Same embedding → score=1.0
   ✓ Test: Opposite embedding → score=0.0
   ✓ Test: Orthogonal → score≈0.5

5. PERFECT CAUSAL SCENARIO
   ✓ agent at (100, 100), patient at (105, 100) [touching]
   ✓ agent moving toward patient at 10 px/s
   ✓ time_delta = 0.5s [recent]
   ✓ semantic_similarity = 0.8 [related]
   ✓ Expected CIS: 0.85-1.0

6. NON-CAUSAL SCENARIO
   ✓ agent at (100, 100), patient at (900, 100) [far]
   ✓ agent moving away at -10 px/s
   ✓ time_delta = 20s [old]
   ✓ semantic_similarity = -0.8 [unrelated]
   ✓ Expected CIS: 0.0-0.05

7. BOUNDARY SCENARIO
   ✓ Right at threshold (0.543)
   ✓ Test rounding, stability
   ✓ Should consistently classify as CAUSAL or NON-CAUSAL

8. TEMPORAL WINDOW CREATION
   ✓ Single change → 1 window
   ✓ Changes within 1.5s gap → same window
   ✓ Changes >1.5s apart → separate windows
   ✓ Respect max_duration_seconds = 5.0

9. TEMPORAL DECAY IN WINDOWS
   ✓ Changes at t=1.0, t=1.5, t=2.0 → single window
   ✓ Changes at t=1.0, t=10.0 → separate windows
   ✓ Verify window boundaries

10. WEIGHT SENSITIVITY
    ✓ Spatial weight (0.436) is most important
    ✓ Semantic weight (0.060) is least important
    ✓ Removing spatial weight → F1 drops significantly
    ✓ Removing semantic weight → F1 barely changes
"""

# =============================================================================
# SIMPLE TEST TEMPLATE
# =============================================================================

SIMPLE_TEST_TEMPLATE = """
import math
import numpy as np
from orion.semantic.causal_scorer import CausalInfluenceScorer
from orion.semantic.config import CausalConfig
from orion.semantic.types import StateChange

def test_temporal_proximity_immediate():
    '''Test: t=0 should score 1.0'''
    config = CausalConfig()
    scorer = CausalInfluenceScorer(config)
    
    # Create two state changes at same time
    change_a = StateChange(
        entity_id="hand",
        timestamp_after=1.0,
        change_magnitude=0.5,
    )
    change_b = StateChange(
        entity_id="door",
        timestamp_after=1.0,  # Same time
        change_magnitude=0.7,
    )
    
    time_delta = change_b.timestamp_after - change_a.timestamp_after
    score = scorer._temporal_proximity(change_a, change_b, time_delta)
    
    # Should be close to 1.0 (accounting for magnitude weighting)
    assert 0.8 < score <= 1.0, f"Expected ~1.0, got {score}"
    print(f"✓ Temporal proximity at t=0: {score:.3f}")

def test_spatial_proximity_touching():
    '''Test: distance=0 should score 1.0'''
    config = CausalConfig()
    scorer = CausalInfluenceScorer(config)
    
    change_a = StateChange(
        entity_id="hand",
        centroid_after=(100.0, 100.0),
        bounding_box_after=(95, 95, 105, 105),
    )
    change_b = StateChange(
        entity_id="door",
        centroid_after=(100.0, 100.0),  # Same position
        bounding_box_after=(95, 95, 105, 105),
    )
    
    score, distance = scorer._spatial_proximity(change_a, change_b)
    
    # Should be ~1.0
    assert 0.9 < score <= 1.0, f"Expected ~1.0, got {score}"
    print(f"✓ Spatial proximity at distance=0: {score:.3f}")

def test_perfect_causal_case():
    '''Full CIS: should be high'''
    config = CausalConfig()
    scorer = CausalInfluenceScorer(config)
    
    change_a = StateChange(
        entity_id="hand",
        timestamp_after=1.0,
        centroid_after=(100.0, 100.0),
        bounding_box_after=(95, 95, 105, 105),
        change_magnitude=0.8,
    )
    change_b = StateChange(
        entity_id="door",
        timestamp_after=1.1,  # 0.1s later
        centroid_after=(105.0, 100.0),  # 5 pixels away
        bounding_box_after=(100, 95, 110, 105),
        change_magnitude=0.7,
    )
    
    embeddings = {
        "hand": np.ones(512) / np.sqrt(512),
        "door": np.ones(512) / np.sqrt(512),  # Same embedding
    }
    
    cis, details = scorer._compute_cis(change_a, change_b, embeddings)
    
    # Should be high
    assert cis > 0.7, f"Expected >0.7, got {cis:.3f}"
    print(f"✓ Perfect causal CIS: {cis:.3f}")
    print(f"  Components: T={details['temporal']:.2f}, S={details['spatial']:.2f}, "
          f"M={details['motion']:.2f}, Se={details['semantic']:.2f}")
"""

# =============================================================================
# EXPECTED VALUES FOR YOUR DATA
# =============================================================================

EXPECTED_RANGES = {
    "temporal_at_1ms": (0.998, 1.0),
    "temporal_at_100ms": (0.976, 0.980),
    "temporal_at_1s": (0.778, 0.783),
    "temporal_at_4s": (0.366, 0.370),
    "temporal_at_10s": (0.080, 0.085),
    
    "spatial_at_0px": (0.95, 1.0),
    "spatial_at_50px": (0.926, 0.934),
    "spatial_at_150px": (0.562, 0.564),
    "spatial_at_300px": (0.249, 0.251),
    "spatial_at_600px": (0.0, 0.01),
    
    "motion_aligned": (0.8, 1.0),
    "motion_perpendicular": (0.0, 0.2),
    "motion_opposite": (0.0, 0.2),
    "motion_stationary": (0.0, 0.25),
    
    "semantic_identical": (0.95, 1.0),
    "semantic_related": (0.75, 0.85),
    "semantic_neutral": (0.45, 0.55),
    "semantic_opposite": (0.0, 0.1),
    
    "perfect_causal_cis": (0.85, 1.0),
    "non_causal_cis": (0.0, 0.05),
    "boundary_cis": (0.51, 0.57),
}

# =============================================================================
# PERFORMANCE TARGETS
# =============================================================================

PERFORMANCE_TARGETS = {
    "cis_computation_per_link": "< 1 ms",
    "temporal_window_creation": "< 10 ms (100 changes)",
    "embedding_lookup": "< 0.1 ms per entity",
    "f1_improvement_vs_defaults": "> 5%",
}

# =============================================================================
# QUICK DIAGNOSTIC SCRIPT
# =============================================================================

def quick_diagnostic():
    """Run quick sanity checks on CIS implementation"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    config = CausalConfig()
    scorer = CausalInfluenceScorer(config)
    
    print("\n" + "="*60)
    print("CIS QUICK DIAGNOSTIC")
    print("="*60)
    
    print("\n✓ Config loaded:")
    print(f"  Weights: T={config.weight_temporal:.3f}, "
          f"S={config.weight_spatial:.3f}, "
          f"M={config.weight_motion:.3f}, "
          f"Se={config.weight_semantic:.3f}")
    print(f"  Threshold: {config.cis_threshold:.3f}")
    print(f"  Decay: {config.temporal_decay_seconds}s")
    
    # Test formulas
    print("\n✓ Formula checks:")
    print(f"  exp(-0/4) = {math.exp(0):.4f} (expect 1.0000)")
    print(f"  exp(-4/4) = {math.exp(-1):.4f} (expect 0.3679)")
    print(f"  exp(-10/4) = {math.exp(-2.5):.4f} (expect 0.0821)")
    
    print("\n✓ Distance formulas:")
    for distance in [0, 150, 300, 600]:
        score = max(0, (1 - distance/600)**2)
        print(f"  distance={distance}: score={score:.4f}")
    
    print("\n✓ Weights sum to: {:.4f} (expect ~1.0)".format(
        config.weight_temporal + config.weight_spatial +
        config.weight_motion + config.weight_semantic
    ))
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE - Ready to test!")
    print("="*60 + "\n")

if __name__ == "__main__":
    quick_diagnostic()
