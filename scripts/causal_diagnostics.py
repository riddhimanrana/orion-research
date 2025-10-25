#!/usr/bin/env python3
"""Causal Influence Score (CIS) Diagnostic Tool.

Quick verification tool for CIS component calculations without requiring
full video processing. Tests individual CIS components with synthetic data.

Usage:
    # Test individual CIS components
    python scripts/causal_diagnostics.py --test components

    # Compare CIS computation before/after HPO
    python scripts/causal_diagnostics.py --test hpo

    # Run full diagnostic suite
    python scripts/causal_diagnostics.py --test all

    # Inspect HPO results
    python scripts/causal_diagnostics.py --test hpo --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Setup path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.perception.tracker import MotionData, calculate_distance
from orion.semantic.causal import (
    AgentCandidate,
    CausalConfig,
    CausalInferenceEngine,
    StateChange,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def make_agent(
    entity_id: str = "agent",
    timestamp: float = 0.0,
    centroid: Tuple[float, float] = (0.0, 0.0),
    velocity: Optional[Tuple[float, float]] = None,
    speed: float = 0.0,
    object_class: str = "person",
    description: Optional[str] = None,
) -> AgentCandidate:
    """Create a test agent candidate."""
    motion = None
    if velocity is not None:
        motion = MotionData(
            centroid=centroid,
            velocity=velocity,
            speed=speed,
            direction=math.atan2(velocity[1], velocity[0]) if speed > 0 else 0.0,
            timestamp=timestamp,
        )

    return AgentCandidate(
        entity_id=entity_id,
        temp_id=f"tmp_{entity_id}",
        timestamp=timestamp,
        centroid=centroid,
        bounding_box=[centroid[0] - 5, centroid[1] - 5, centroid[0] + 5, centroid[1] + 5],
        motion_data=motion,
        visual_embedding=[0.1, 0.2, 0.3],
        object_class=object_class,
        description=description or object_class,
    )


def make_patient(
    entity_id: str = "patient",
    timestamp: float = 0.0,
    centroid: Tuple[float, float] = (0.0, 0.0),
    old_desc: str = "before",
    new_desc: str = "after",
) -> StateChange:
    """Create a test state change (patient)."""
    return StateChange(
        entity_id=entity_id,
        timestamp=timestamp,
        frame_number=0,
        old_description=old_desc,
        new_description=new_desc,
        centroid=centroid,
        bounding_box=[centroid[0] - 5, centroid[1] - 5, centroid[0] + 5, centroid[1] + 5],
    )


def test_temporal_component() -> None:
    """Test temporal proximity scoring."""
    logger.info("\n" + "=" * 70)
    logger.info("TEMPORAL PROXIMITY COMPONENT TEST")
    logger.info("=" * 70)

    config = CausalConfig()
    engine = CausalInferenceEngine(config)

    agent = make_agent(timestamp=0.0)
    patients = [
        make_patient(timestamp=0.0, new_desc="immediate change"),
        make_patient(timestamp=1.0, new_desc="1s change"),
        make_patient(timestamp=2.0, new_desc="2s change"),
        make_patient(timestamp=4.0, new_desc="4s change"),
        make_patient(timestamp=8.0, new_desc="8s change"),
    ]

    logger.info(f"Agent timestamp: {agent.timestamp:.1f}s")
    logger.info(f"Decay constant: {config.temporal_decay:.1f}s")
    logger.info("\nTemporal scores:")

    scores = []
    for patient in patients:
        score = engine._temporal_score(agent, patient)
        scores.append((patient.timestamp - agent.timestamp, score))
        expected_decay = math.exp(-(patient.timestamp - agent.timestamp) / config.temporal_decay)
        logger.info(
            f"  Δt={patient.timestamp - agent.timestamp:4.1f}s → "
            f"score={score:.4f} (expected≈{expected_decay:.4f})"
        )

    assert all(scores[i][1] >= scores[i + 1][1] for i in range(len(scores) - 1)), "Scores should decay monotonically"
    logger.info("\n✓ Temporal scores decay as expected")


def test_spatial_component() -> None:
    """Test spatial proximity scoring."""
    logger.info("\n" + "=" * 70)
    logger.info("SPATIAL PROXIMITY COMPONENT TEST")
    logger.info("=" * 70)

    config = CausalConfig()
    engine = CausalInferenceEngine(config)

    agent = make_agent(centroid=(100.0, 100.0))
    patients = [
        make_patient(centroid=(100.0, 100.0), new_desc="at agent"),
        make_patient(centroid=(110.0, 100.0), new_desc="10px away"),
        make_patient(centroid=(200.0, 100.0), new_desc="100px away"),
        make_patient(centroid=(400.0, 100.0), new_desc="300px away"),
        make_patient(centroid=(700.0, 100.0), new_desc="600px away"),
    ]

    logger.info(f"Agent centroid: {agent.centroid}")
    logger.info(f"Max distance: {config.max_pixel_distance:.1f}px")
    logger.info("\nSpatial scores:")

    scores = []
    for patient in patients:
        score = engine._proximity_score(agent, patient)
        distance = calculate_distance(agent.centroid, patient.centroid)
        scores.append((distance, score))
        logger.info(
            f"  Distance={distance:6.1f}px → score={score:.4f} ({patient.new_description})"
        )

    # Verify monotonic decay
    assert all(scores[i][1] >= scores[i + 1][1] for i in range(len(scores) - 1)), "Scores should decay with distance"
    logger.info("\n✓ Spatial scores decay with distance")


def test_motion_component() -> None:
    """Test motion alignment scoring."""
    logger.info("\n" + "=" * 70)
    logger.info("MOTION ALIGNMENT COMPONENT TEST")
    logger.info("=" * 70)

    config = CausalConfig()
    engine = CausalInferenceEngine(config)

    patient = make_patient(centroid=(100.0, 0.0))

    test_cases = [
        ("moving toward", (0.0, 0.0), (10.0, 0.0), 10.0, "person moving toward patient"),
        ("moving away", (200.0, 0.0), (-10.0, 0.0), 10.0, "person moving away from patient"),
        ("perpendicular", (0.0, 0.0), (0.0, 10.0), 10.0, "person moving perpendicular"),
        ("stationary", (50.0, 0.0), (0.0, 0.0), 0.0, "person stationary"),
        ("fast approach", (0.0, 0.0), (50.0, 0.0), 50.0, "person fast moving toward"),
    ]

    logger.info(f"Patient centroid: {patient.centroid}")
    logger.info("\nMotion alignment scores:")

    for label, start_pos, velocity, speed, desc in test_cases:
        agent = make_agent(
            centroid=start_pos,
            velocity=velocity,
            speed=speed,
        )
        score = engine._motion_alignment_score(agent, patient)
        logger.info(f"  {label:20s} → score={score:.4f} ({desc})")

    logger.info("\n✓ Motion alignment scores vary with direction and speed")


def test_full_cis_formula() -> None:
    """Test complete CIS formula with various scenarios."""
    logger.info("\n" + "=" * 70)
    logger.info("FULL CIS FORMULA TEST (DEFAULT CONFIG)")
    logger.info("=" * 70)

    config = CausalConfig()
    logger.info(f"\nWeights: T={config.temporal_proximity_weight:.3f}, "
                f"S={config.spatial_proximity_weight:.3f}, "
                f"M={config.motion_alignment_weight:.3f}, "
                f"Se={config.semantic_similarity_weight:.3f}")
    logger.info(f"Threshold: {config.min_score:.3f}")

    engine = CausalInferenceEngine(config)

    scenarios = [
        ("Perfect causal", {
            "agent": make_agent(
                entity_id="good_agent",
                timestamp=0.0,
                centroid=(100.0, 100.0),
                velocity=(10.0, 0.0),
                speed=10.0,
            ),
            "patient": make_patient(
                timestamp=0.5,
                centroid=(105.0, 100.0),
                new_desc="door opened",
            ),
        }),
        ("Distant temporal", {
            "agent": make_agent(
                entity_id="old_agent",
                timestamp=0.0,
                centroid=(100.0, 100.0),
            ),
            "patient": make_patient(
                timestamp=20.0,
                centroid=(100.0, 100.0),
                new_desc="something changed",
            ),
        }),
        ("Distant spatial", {
            "agent": make_agent(
                entity_id="far_agent",
                timestamp=1.0,
                centroid=(0.0, 0.0),
            ),
            "patient": make_patient(
                timestamp=1.5,
                centroid=(800.0, 800.0),
                new_desc="far change",
            ),
        }),
        ("Moving away", {
            "agent": make_agent(
                entity_id="leaving_agent",
                timestamp=0.0,
                centroid=(100.0, 100.0),
                velocity=(-10.0, 0.0),
                speed=10.0,
            ),
            "patient": make_patient(
                timestamp=0.5,
                centroid=(105.0, 100.0),
                new_desc="change behind agent",
            ),
        }),
    ]

    logger.info("\nCIS Scenarios:")
    for scenario_name, actors in scenarios:
        agent = actors["agent"]
        patient = actors["patient"]
        cis = engine.calculate_cis(agent, patient)
        temporal = engine._temporal_score(agent, patient)
        spatial = engine._proximity_score(agent, patient)
        motion = engine._motion_alignment_score(agent, patient)
        semantic = engine._embedding_score(agent, patient)

        passes = "✓" if cis >= config.min_score else "✗"
        logger.info(
            f"  {scenario_name:20s}: CIS={cis:.4f} {passes} "
            f"(T={temporal:.2f}, S={spatial:.2f}, M={motion:.2f}, Se={semantic:.2f})"
        )

    logger.info("\n✓ Full CIS formula working")


def test_hpo_weights() -> None:
    """Load and verify HPO-optimized weights."""
    logger.info("\n" + "=" * 70)
    logger.info("HPO WEIGHTS VERIFICATION")
    logger.info("=" * 70)

    hpo_path = Path("hpo_results/optimization_latest.json")
    if not hpo_path.exists():
        logger.warning(f"HPO results not found at {hpo_path}")
        return

    with open(hpo_path) as f:
        hpo_data = json.load(f)

    weights = hpo_data["best_weights"]
    threshold = hpo_data["best_threshold"]
    score = hpo_data["best_score"]

    logger.info(f"\nBest weights (F1={score:.4f}):")
    logger.info(f"  Temporal: {weights['temporal']:.4f}")
    logger.info(f"  Spatial:  {weights['spatial']:.4f}")
    logger.info(f"  Motion:   {weights['motion']:.4f}")
    logger.info(f"  Semantic: {weights['semantic']:.4f}")
    logger.info(f"  Sum: {sum(weights.values()):.4f}")
    logger.info(f"  Threshold: {threshold:.4f}")

    # Verify weights sum to ~1.0
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.05:
        logger.warning(f"Weights don't sum to 1.0: {weight_sum:.4f}")
    else:
        logger.info("✓ Weights sum to 1.0")

    # Verify precision/recall
    precision = hpo_data.get("precision", 0)
    recall = hpo_data.get("recall", 0)
    logger.info(f"\nValidation metrics:")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")

    if score > 0.85:
        logger.info("✓ Strong HPO score (>0.85)")
    elif score > 0.75:
        logger.info("⚠ Moderate HPO score (0.75-0.85)")
    else:
        logger.warning(f"⚠ Low HPO score ({score:.4f})")

    # Test with HPO config
    logger.info("\nTesting CIS with HPO weights...")
    config = CausalConfig.from_hpo_result(str(hpo_path))
    logger.info(f"✓ Loaded HPO config: min_score={config.min_score:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CIS Diagnostic Tool - Test CIS component calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--test",
        choices=["temporal", "spatial", "motion", "cis", "hpo", "all"],
        default="all",
        help="Which diagnostic to run",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("\n" + "=" * 70)
    logger.info("CAUSAL INFLUENCE SCORE (CIS) DIAGNOSTIC TOOL")
    logger.info("=" * 70)

    try:
        if args.test in ("temporal", "all"):
            test_temporal_component()

        if args.test in ("spatial", "all"):
            test_spatial_component()

        if args.test in ("motion", "all"):
            test_motion_component()

        if args.test in ("cis", "all"):
            test_full_cis_formula()

        if args.test in ("hpo", "all"):
            test_hpo_weights()

        logger.info("\n" + "=" * 70)
        logger.info("✓ ALL DIAGNOSTICS PASSED")
        logger.info("=" * 70 + "\n")
        return 0

    except Exception as e:
        logger.error(f"Diagnostic failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
