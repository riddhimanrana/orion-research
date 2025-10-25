"""
CIS & Temporal Testing Guide
============================

Comprehensive testing strategy for Causal Influence Score (CIS) and temporal components.
Use the HPO-learned weights from hpo_results/cis_weights.json for reference.

Author: Orion Research Team
Date: October 25, 2025
"""

# ============================================================================
# CIS COMPONENT TESTS
# ============================================================================

CIS_COMPONENT_TESTS = {
    "temporal_proximity": {
        "description": "Test temporal decay function exp(-t / decay_constant)",
        "config": {
            "decay_constant": 4.0,  # seconds
            "formula": "decay = exp(-time_delta / 4.0)"
        },
        "tests": [
            {
                "name": "test_temporal_proximity_immediate",
                "description": "t=0 should score 1.0 (immediate causality)",
                "time_delta": 0.0,
                "expected_score_range": [0.95, 1.0],
                "formula_check": "exp(-0/4) = 1.0"
            },
            {
                "name": "test_temporal_proximity_1second",
                "description": "t=1s should score high",
                "time_delta": 1.0,
                "expected_score_range": [0.75, 0.85],
                "formula_check": "exp(-1/4) ≈ 0.78"
            },
            {
                "name": "test_temporal_proximity_decay_constant",
                "description": "t=decay_constant should score ~0.37 (1/e)",
                "time_delta": 4.0,
                "expected_score_range": [0.35, 0.40],
                "formula_check": "exp(-4/4) = exp(-1) ≈ 0.368"
            },
            {
                "name": "test_temporal_proximity_10seconds",
                "description": "t=10s should score low",
                "time_delta": 10.0,
                "expected_score_range": [0.05, 0.10],
                "formula_check": "exp(-10/4) ≈ 0.082"
            },
            {
                "name": "test_temporal_proximity_beyond_horizon",
                "description": "t > 30s should score near 0",
                "time_delta": 30.0,
                "expected_score_range": [0.0, 0.01],
                "formula_check": "exp(-30/4) ≈ 0.00007"
            },
            {
                "name": "test_temporal_proximity_with_magnitude",
                "description": "Should weight by magnitude (0.5 + 0.5*magnitude)",
                "time_delta": 4.0,
                "change_magnitude_a": 0.8,
                "change_magnitude_b": 0.6,
                "expected_score_range": [0.26, 0.28],
                "formula_check": "0.368 * (0.5 + 0.5*0.7) ≈ 0.276"
            }
        ]
    },
    
    "spatial_proximity": {
        "description": "Test spatial distance falloff (quadratic)",
        "config": {
            "max_distance": 600.0,  # pixels
            "formula": "(1 - min(distance/600, 1))^2"
        },
        "tests": [
            {
                "name": "test_spatial_proximity_zero_distance",
                "description": "distance=0 should score 1.0",
                "distance": 0.0,
                "expected_score_range": [0.95, 1.0],
                "formula_check": "(1 - 0/600)^2 = 1.0"
            },
            {
                "name": "test_spatial_proximity_150pixels",
                "description": "Quarter distance should score well",
                "distance": 150.0,
                "expected_score_range": [0.55, 0.65],
                "formula_check": "(1 - 150/600)^2 = 0.5625"
            },
            {
                "name": "test_spatial_proximity_300pixels",
                "description": "Half distance should score 0.25",
                "distance": 300.0,
                "expected_score_range": [0.24, 0.26],
                "formula_check": "(1 - 300/600)^2 = 0.25"
            },
            {
                "name": "test_spatial_proximity_600pixels",
                "description": "Max distance should score ~0",
                "distance": 600.0,
                "expected_score_range": [0.0, 0.01],
                "formula_check": "(1 - 600/600)^2 = 0.0"
            },
            {
                "name": "test_spatial_proximity_beyond_max",
                "description": "Beyond max should score 0",
                "distance": 1000.0,
                "expected_score_range": [0.0, 0.0],
                "formula_check": "clamped to 0"
            },
            {
                "name": "test_spatial_proximity_same_location",
                "description": "Same location bonus +0.2",
                "distance": 0.0,
                "location_match": True,
                "expected_score_range": [1.0, 1.0],
                "formula_check": "1.0 + bonus (clamped)"
            },
            {
                "name": "test_spatial_proximity_same_zone",
                "description": "Same zone bonus +0.1",
                "distance": 50.0,
                "zone_match": True,
                "expected_score_range": [0.89, 0.91],
                "formula_check": "(1 - 50/600)^2 + 0.1"
            }
        ]
    },
    
    "motion_alignment": {
        "description": "Test motion vector alignment to target",
        "config": {
            "angle_threshold": 0.785,  # π/4 radians (45 degrees)
            "min_speed": 5.0,  # pixels/sec
            "max_speed": 200.0,  # pixels/sec
            "formula": "0.6*alignment + 0.4*speed_norm"
        },
        "tests": [
            {
                "name": "test_motion_alignment_perfect",
                "description": "Moving directly at target",
                "agent_motion": (10.0, 0.0),  # velocity
                "agent_position": (100.0, 100.0),
                "patient_position": (110.0, 100.0),
                "expected_alignment": 1.0,
                "expected_score_range": [0.80, 1.0],
                "notes": "Perfect alignment (0° angle)"
            },
            {
                "name": "test_motion_alignment_perpendicular",
                "description": "Moving perpendicular to target",
                "agent_motion": (0.0, 10.0),  # moving up
                "agent_position": (100.0, 100.0),
                "patient_position": (110.0, 100.0),  # target is right
                "expected_alignment": 0.0,
                "expected_score_range": [0.0, 0.2],
                "notes": "90° angle, poor alignment"
            },
            {
                "name": "test_motion_alignment_opposite",
                "description": "Moving away from target",
                "agent_motion": (-10.0, 0.0),  # moving left
                "agent_position": (100.0, 100.0),
                "patient_position": (110.0, 100.0),  # target is right
                "expected_alignment": -1.0,
                "expected_score_range": [0.0, 0.2],
                "notes": "180° angle (opposite), worst case"
            },
            {
                "name": "test_motion_alignment_angle_threshold",
                "description": "At angle threshold (45°)",
                "agent_motion": (7.07, 7.07),  # 45° angle
                "agent_position": (100.0, 100.0),
                "patient_position": (110.0, 100.0),
                "expected_alignment": 0.707,  # cos(45°)
                "expected_score_range": [0.35, 0.55],
                "notes": "At threshold, begins penalty"
            },
            {
                "name": "test_motion_alignment_stationary",
                "description": "Agent not moving",
                "agent_motion": (0.0, 0.0),
                "agent_position": (100.0, 100.0),
                "patient_position": (105.0, 100.0),
                "expected_score_range": [0.0, 0.25],
                "notes": "No motion = low score"
            },
            {
                "name": "test_motion_alignment_low_speed",
                "description": "Speed below minimum",
                "agent_motion": (2.0, 0.0),  # speed=2, below min=5
                "agent_position": (100.0, 100.0),
                "patient_position": (110.0, 100.0),
                "expected_speed": 2.0,
                "expected_score_range": [0.0, 0.15],
                "notes": "Speed penalty: score *= 0.3"
            },
            {
                "name": "test_motion_alignment_high_speed",
                "description": "Speed above saturation",
                "agent_motion": (250.0, 0.0),  # speed=250, max=200
                "agent_position": (100.0, 100.0),
                "patient_position": (110.0, 100.0),
                "expected_speed_norm": 1.0,
                "expected_score_range": [0.80, 1.0],
                "notes": "Speed saturates at max_speed"
            }
        ]
    },
    
    "semantic_proximity": {
        "description": "Test embedding-based semantic similarity",
        "config": {
            "formula": "(cosine_similarity + 1) / 2",
            "min_similarity_threshold": 0.3,
            "penalty_threshold": 0.3
        },
        "tests": [
            {
                "name": "test_semantic_proximity_identical",
                "description": "Same embedding (e.g., two 'hands')",
                "embedding_a": [1.0, 0.0, 0.0],
                "embedding_b": [1.0, 0.0, 0.0],
                "cosine_similarity": 1.0,
                "expected_score_range": [0.95, 1.0],
                "formula_check": "(1.0 + 1) / 2 = 1.0"
            },
            {
                "name": "test_semantic_proximity_identical_512dim",
                "description": "Identical 512-dim embeddings",
                "embedding_dims": 512,
                "similarity_type": "identical",
                "cosine_similarity": 1.0,
                "expected_score_range": [0.95, 1.0],
                "notes": "Should work with real CLIP embeddings"
            },
            {
                "name": "test_semantic_proximity_related",
                "description": "Related objects (hand → door)",
                "cosine_similarity": 0.7,
                "expected_score_range": [0.80, 0.90],
                "formula_check": "(0.7 + 1) / 2 = 0.85"
            },
            {
                "name": "test_semantic_proximity_neutral",
                "description": "Orthogonal embeddings",
                "embedding_a": [1.0, 0.0, 0.0],
                "embedding_b": [0.0, 1.0, 0.0],
                "cosine_similarity": 0.0,
                "expected_score_range": [0.45, 0.55],
                "formula_check": "(0.0 + 1) / 2 = 0.5"
            },
            {
                "name": "test_semantic_proximity_opposite",
                "description": "Opposite embeddings",
                "embedding_a": [1.0, 0.0, 0.0],
                "embedding_b": [-1.0, 0.0, 0.0],
                "cosine_similarity": -1.0,
                "expected_score_range": [0.0, 0.1],
                "formula_check": "(-1.0 + 1) / 2 = 0.0"
            },
            {
                "name": "test_semantic_proximity_below_threshold",
                "description": "Below min_similarity → penalty",
                "cosine_similarity": 0.2,
                "expected_score_range": [0.25, 0.35],
                "notes": "Score *= 0.5 (penalty for low similarity)"
            },
            {
                "name": "test_semantic_proximity_missing_embedding",
                "description": "No embedding available",
                "embedding_a": None,
                "embedding_b": [1.0, 0.0, 0.0],
                "expected_score": 0.5,
                "notes": "Default neutral score"
            }
        ]
    }
}


# ============================================================================
# CIS FORMULA INTEGRATION TESTS
# ============================================================================

CIS_INTEGRATION_TESTS = [
    {
        "name": "perfect_causal_scenario",
        "description": "Clear causality: close + aligned + recent + related",
        "scenario": {
            "agent_position": (100.0, 100.0),
            "patient_position": (105.0, 100.0),
            "distance": 5.0,
            "agent_motion": (10.0, 0.0),
            "time_delta": 0.5,  # 500ms
            "semantic_similarity": 0.8,  # related objects
        },
        "component_expectations": {
            "temporal": {"range": [0.85, 1.0], "reason": "very recent"},
            "spatial": {"range": [0.99, 1.0], "reason": "touching"},
            "motion": {"range": [0.80, 1.0], "reason": "aligned to target"},
            "semantic": {"range": [0.85, 0.95], "reason": "related"},
        },
        "weights": {
            "temporal": 0.296,
            "spatial": 0.436,
            "motion": 0.208,
            "semantic": 0.060,
        },
        "expected_cis_range": [0.85, 1.0],
        "expected_decision": "CAUSAL (above 0.543 threshold)",
    },
    {
        "name": "non_causal_scenario",
        "description": "Clear non-causality: far + opposite motion + old + unrelated",
        "scenario": {
            "agent_position": (100.0, 100.0),
            "patient_position": (900.0, 100.0),
            "distance": 800.0,  # beyond max
            "agent_motion": (-10.0, 0.0),  # moving away
            "time_delta": 20.0,  # 20 seconds
            "semantic_similarity": -0.8,  # very unrelated
        },
        "component_expectations": {
            "temporal": {"range": [0.01, 0.05], "reason": "distant in time"},
            "spatial": {"range": [0.0, 0.01], "reason": "far apart"},
            "motion": {"range": [0.0, 0.1], "reason": "opposite direction"},
            "semantic": {"range": [0.0, 0.1], "reason": "unrelated"},
        },
        "weights": {
            "temporal": 0.296,
            "spatial": 0.436,
            "motion": 0.208,
            "semantic": 0.060,
        },
        "expected_cis_range": [0.0, 0.05],
        "expected_decision": "NON-CAUSAL (below 0.543 threshold)",
    },
    {
        "name": "spatial_dominant_scenario",
        "description": "Spatial proximity dominates (weight=0.436)",
        "scenario": {
            "agent_position": (100.0, 100.0),
            "patient_position": (110.0, 100.0),
            "distance": 10.0,
            "agent_motion": (0.0, 0.0),  # not moving
            "time_delta": 5.0,  # recent but not immediate
            "semantic_similarity": 0.0,  # unrelated
        },
        "component_expectations": {
            "temporal": {"range": [0.28, 0.32], "reason": "exp(-5/4) ≈ 0.29"},
            "spatial": {"range": [0.96, 0.98], "reason": "very close"},
            "motion": {"range": [0.0, 0.25], "reason": "not moving"},
            "semantic": {"range": [0.45, 0.55], "reason": "unrelated"},
        },
        "weights": {
            "temporal": 0.296,
            "spatial": 0.436,  # DOMINANT
            "motion": 0.208,
            "semantic": 0.060,
        },
        "expected_cis_range": [0.55, 0.70],
        "expected_decision": "MARGINAL CAUSAL (spatial saves it)",
    },
    {
        "name": "threshold_boundary",
        "description": "Right at threshold (0.543)",
        "scenario": {
            "agent_position": (100.0, 100.0),
            "patient_position": (150.0, 100.0),
            "distance": 50.0,
            "agent_motion": (5.0, 0.0),  # moving toward
            "time_delta": 2.0,
            "semantic_similarity": 0.5,  # somewhat related
        },
        "component_expectations": {
            "temporal": {"range": [0.60, 0.65], "reason": "2s decay"},
            "spatial": {"range": [0.92, 0.94], "reason": "50px distance"},
            "motion": {"range": [0.40, 0.60], "reason": "aligned + low speed"},
            "semantic": {"range": [0.70, 0.80], "reason": "somewhat related"},
        },
        "weights": {
            "temporal": 0.296,
            "spatial": 0.436,
            "motion": 0.208,
            "semantic": 0.060,
        },
        "expected_cis_range": [0.53, 0.57],
        "expected_decision": "BOUNDARY (test rounding, stability)",
    }
]


# ============================================================================
# TEMPORAL WINDOW TESTS
# ============================================================================

TEMPORAL_WINDOW_TESTS = [
    {
        "name": "single_state_change",
        "description": "Only one state change → single window",
        "state_changes": [
            {"entity_id": "door", "timestamp": 5.0, "frame": 150}
        ],
        "config": {
            "max_duration_seconds": 5.0,
            "max_gap_between_changes": 1.5,
            "max_changes_per_window": 20,
        },
        "expected_windows": 1,
        "expected_window_0": {
            "start_time": 5.0,
            "end_time": 5.0,
            "duration": 0.0,
            "change_count": 1,
        }
    },
    {
        "name": "clustered_changes",
        "description": "Multiple changes within max_gap → same window",
        "state_changes": [
            {"entity_id": "door", "timestamp": 5.0, "frame": 150},
            {"entity_id": "light", "timestamp": 5.5, "frame": 165},
            {"entity_id": "window", "timestamp": 6.0, "frame": 180},
        ],
        "config": {
            "max_duration_seconds": 5.0,
            "max_gap_between_changes": 1.5,
            "max_changes_per_window": 20,
        },
        "expected_windows": 1,
        "expected_window_0": {
            "start_time": 5.0,
            "end_time": 6.0,
            "duration": 1.0,
            "change_count": 3,
        }
    },
    {
        "name": "sparse_changes",
        "description": "Changes beyond max_gap → separate windows",
        "state_changes": [
            {"entity_id": "door", "timestamp": 5.0, "frame": 150},
            {"entity_id": "light", "timestamp": 10.0, "frame": 300},  # 5s gap > 1.5s threshold
            {"entity_id": "window", "timestamp": 15.0, "frame": 450},
        ],
        "config": {
            "max_duration_seconds": 5.0,
            "max_gap_between_changes": 1.5,
            "max_changes_per_window": 20,
        },
        "expected_windows": 3,
        "expected_window_0": {"start_time": 5.0, "change_count": 1},
        "expected_window_1": {"start_time": 10.0, "change_count": 1},
        "expected_window_2": {"start_time": 15.0, "change_count": 1},
    },
    {
        "name": "max_duration_split",
        "description": "Changes within time but exceeding max_duration → split",
        "state_changes": [
            {"entity_id": "door", "timestamp": 5.0, "frame": 150},
            {"entity_id": "light", "timestamp": 6.0, "frame": 180},
            {"entity_id": "window", "timestamp": 11.0, "frame": 330},  # 6s span > 5s max
        ],
        "config": {
            "max_duration_seconds": 5.0,
            "max_gap_between_changes": 1.5,
            "max_changes_per_window": 20,
        },
        "expected_windows": 2,
        "expected_window_0": {
            "start_time": 5.0,
            "end_time": 6.0,
            "change_count": 2,
        },
        "expected_window_1": {
            "start_time": 11.0,
            "end_time": 11.0,
            "change_count": 1,
        }
    },
    {
        "name": "max_changes_per_window",
        "description": "More changes than max_changes_per_window → split",
        "state_changes": [
            {"entity_id": f"entity_{i}", "timestamp": 5.0 + i*0.1, "frame": 150 + i*3}
            for i in range(25)  # 25 changes, max is 20
        ],
        "config": {
            "max_duration_seconds": 5.0,
            "max_gap_between_changes": 1.5,
            "max_changes_per_window": 20,
        },
        "expected_windows": 2,
        "expected_window_0": {"change_count": 20},
        "expected_window_1": {"change_count": 5},
    }
]


# ============================================================================
# TEMPORAL DECAY TESTS
# ============================================================================

TEMPORAL_DECAY_TESTS = [
    {
        "name": "decay_formula_verification",
        "description": "Verify exp(-t/4) with various time values",
        "formula": "decay = exp(-time_delta / 4.0)",
        "test_points": [
            {"time": 0.0, "expected": 1.0, "description": "t=0 (immediate)"},
            {"time": 1.0, "expected": 0.7788, "description": "t=1s"},
            {"time": 2.0, "expected": 0.6065, "description": "t=2s"},
            {"time": 4.0, "expected": 0.3679, "description": "t=decay_constant (1/e)"},
            {"time": 8.0, "expected": 0.1353, "description": "t=2*decay_constant"},
            {"time": 12.0, "expected": 0.0498, "description": "t=3*decay_constant"},
            {"time": 20.0, "expected": 0.0067, "description": "t=5*decay_constant"},
        ]
    },
    {
        "name": "causal_link_temporal_decay",
        "description": "Test how temporal decay affects causal link scoring",
        "agent_change": {
            "entity_id": "hand",
            "timestamp": 1.0,
            "frame": 30,
        },
        "patient_changes": [
            {
                "entity_id": "door",
                "timestamp": 1.1,  # 0.1s later
                "expected_temporal_score": 0.975,
                "expected_link_likely": True,
            },
            {
                "entity_id": "light",
                "timestamp": 1.5,  # 0.5s later
                "expected_temporal_score": 0.881,
                "expected_link_likely": True,
            },
            {
                "entity_id": "window",
                "timestamp": 3.0,  # 2.0s later
                "expected_temporal_score": 0.607,
                "expected_link_likely": True,  # might need spatial/motion boost
            },
            {
                "entity_id": "fan",
                "timestamp": 6.0,  # 5.0s later
                "expected_temporal_score": 0.290,
                "expected_link_likely": False,  # temporal alone too weak
            },
            {
                "entity_id": "speaker",
                "timestamp": 11.0,  # 10.0s later
                "expected_temporal_score": 0.082,
                "expected_link_likely": False,  # beyond horizon
            },
        ]
    }
]


# ============================================================================
# WEIGHT SENSITIVITY TESTS (from HPO analysis)
# ============================================================================

WEIGHT_SENSITIVITY_TESTS = {
    "description": "Test sensitivity to each weight component",
    "current_weights": {
        "temporal": 0.296,
        "spatial": 0.436,
        "motion": 0.208,
        "semantic": 0.060,
        "threshold": 0.543,
    },
    "test_ablations": [
        {
            "name": "remove_spatial_weight",
            "modify_weight": {"spatial": 0.0},
            "expected_impact": "HIGH",
            "reason": "Spatial is dominant (43.6%)",
        },
        {
            "name": "remove_temporal_weight",
            "modify_weight": {"temporal": 0.0},
            "expected_impact": "MEDIUM",
            "reason": "Temporal is significant (29.6%)",
        },
        {
            "name": "remove_motion_weight",
            "modify_weight": {"motion": 0.0},
            "expected_impact": "MEDIUM",
            "reason": "Motion is meaningful (20.8%)",
        },
        {
            "name": "remove_semantic_weight",
            "modify_weight": {"semantic": 0.0},
            "expected_impact": "LOW",
            "reason": "Semantic is minor (6.0%)",
        },
        {
            "name": "equal_weights",
            "modify_weight": {
                "temporal": 0.25,
                "spatial": 0.25,
                "motion": 0.25,
                "semantic": 0.25
            },
            "expected_impact": "SIGNIFICANT",
            "reason": "Semantic becomes 4x more important",
        },
    ]
}


# ============================================================================
# RECOMMENDATION SUMMARY
# ============================================================================

TESTING_PRIORITY = """
1. CRITICAL (Do First):
   - test_cis_weights_match_hpo()
     → Verify CIS formula uses learned weights from JSON
   - test_perfect_causal_scenario()
     → Hand touches door should score high
   - test_non_causal_scenario()
     → Random distant objects should score low
   - test_threshold_boundary()
     → Values near 0.543 threshold should be stable

2. HIGH (Do Early):
   - All temporal_proximity tests
     → Core component, cheap to verify
   - All spatial_proximity tests
     → Dominant weight (43.6%), must be correct
   - test_temporal_window_creation_sparse()
   - test_temporal_window_creation_clustered()

3. MEDIUM (Do Mid):
   - All motion_alignment tests
     → More complex, but manageable
   - All semantic_proximity tests
     → Includes 512-dim real embeddings
   - Temporal decay formula verification
   - Weight sensitivity ablations

4. LOW (Nice to Have):
   - Edge case tests (missing data, invalid values)
   - Performance benchmarks
   - Large-scale integration tests

Performance Metrics to Measure:
- CIS computation time per link (target: <1ms)
- Temporal window creation time (target: <10ms for 100 changes)
- Memory usage for embeddings cache (target: <100MB)
- F1 score improvement from HPO weights vs defaults
"""

print(TESTING_PRIORITY)
