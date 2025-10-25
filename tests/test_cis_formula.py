"""Causal Influence Score (CIS) formula tests aligned with semantic module."""

from __future__ import annotations

import numpy as np
import pytest

from orion.perception.tracker import MotionData
from orion.semantic.causal import (
    AgentCandidate,
    CausalConfig,
    CausalInferenceEngine,
    StateChange,
)


def make_agent(
    *,
    entity_id: str = "agent",
    timestamp: float = 0.0,
    centroid: tuple[float, float] = (0.0, 0.0),
    velocity: tuple[float, float] | None = None,
    speed: float = 0.0,
    object_class: str = "person",
    description: str | None = None,
    visual_embedding: list[float] | None = None,
) -> AgentCandidate:
    motion = None
    if velocity is not None:
        motion = MotionData(
            centroid=centroid,
            velocity=velocity,
            speed=speed,
            direction=0.0,
            timestamp=timestamp,
        )

    return AgentCandidate(
        entity_id=entity_id,
        temp_id=f"temp_{entity_id}",
        timestamp=timestamp,
        centroid=centroid,
        bounding_box=[centroid[0] - 5, centroid[1] - 5, centroid[0] + 5, centroid[1] + 5],
        motion_data=motion,
        visual_embedding=visual_embedding if visual_embedding is not None else [0.0, 1.0, 0.0],
        object_class=object_class,
        description=description,
    )


def make_state_change(
    *,
    entity_id: str = "patient",
    timestamp: float = 0.0,
    centroid: tuple[float, float] = (0.0, 0.0),
    old_description: str = "state before",
    new_description: str = "state after",
    frame_number: int = 0,
) -> StateChange:
    return StateChange(
        entity_id=entity_id,
        timestamp=timestamp,
        frame_number=frame_number,
        old_description=old_description,
        new_description=new_description,
        centroid=centroid,
        bounding_box=[centroid[0] - 5, centroid[1] - 5, centroid[0] + 5, centroid[1] + 5],
    )


class TestCISComponents:
    """Validate individual CIS component functions."""

    def setup_method(self) -> None:
        self.config = CausalConfig(
            temporal_proximity_weight=0.3,
            spatial_proximity_weight=0.3,
            motion_alignment_weight=0.2,
            semantic_similarity_weight=0.2,
            min_score=0.5,
        )
        self.engine = CausalInferenceEngine(self.config)

    def test_temporal_score_immediate(self) -> None:
        agent = make_agent(timestamp=10.0)
        patient = make_state_change(timestamp=10.0)
        assert self.engine._temporal_score(agent, patient) == pytest.approx(1.0)

    def test_temporal_score_decay(self) -> None:
        agent = make_agent(timestamp=0.0)
        patient_near = make_state_change(timestamp=1.0)
        patient_mid = make_state_change(timestamp=3.0)
        patient_far = make_state_change(timestamp=6.0)

        score_near = self.engine._temporal_score(agent, patient_near)
        score_mid = self.engine._temporal_score(agent, patient_mid)
        score_far = self.engine._temporal_score(agent, patient_far)

        assert 1.0 > score_near > score_mid > score_far >= 0.0

    def test_temporal_score_far_future(self) -> None:
        agent = make_agent(timestamp=0.0)
        patient = make_state_change(timestamp=100.0)
        # With pure exponential decay (no threshold), score approaches 0 but never reaches it
        # exp(-100/4) = exp(-25) ≈ 1.39e-11, which is effectively 0
        assert self.engine._temporal_score(agent, patient) == pytest.approx(0.0, abs=1e-10)

    def test_proximity_score_zero_distance(self) -> None:
        agent = make_agent(centroid=(10.0, 10.0))
        patient = make_state_change(centroid=(10.0, 10.0))
        assert self.engine._proximity_score(agent, patient) == pytest.approx(1.0)

    def test_proximity_score_max_distance(self) -> None:
        max_d = self.config.max_pixel_distance
        agent = make_agent(centroid=(0.0, 0.0))
        patient = make_state_change(centroid=(max_d, 0.0))
        assert self.engine._proximity_score(agent, patient) == pytest.approx(0.0)

    def test_proximity_score_decay(self) -> None:
        agent = make_agent(centroid=(0.0, 0.0))
        patient_near = make_state_change(centroid=(50.0, 0.0))
        patient_far = make_state_change(centroid=(300.0, 0.0))
        patient_very_far = make_state_change(centroid=(550.0, 0.0))

        scores = [
            self.engine._proximity_score(agent, patient_near),
            self.engine._proximity_score(agent, patient_far),
            self.engine._proximity_score(agent, patient_very_far),
        ]
        assert scores == sorted(scores, reverse=True)

    def test_motion_alignment_same_direction(self) -> None:
        agent = make_agent(
            centroid=(0.0, 0.0),
            timestamp=0.0,
            velocity=(10.0, 0.0),
            speed=10.0,
        )
        patient = make_state_change(centroid=(10.0, 0.0))
        score = self.engine._motion_alignment_score(agent, patient)
        assert score > 0.0

    def test_motion_alignment_opposite_direction(self) -> None:
        agent = make_agent(
            centroid=(0.0, 0.0),
            timestamp=0.0,
            velocity=(10.0, 0.0),
            speed=10.0,
        )
        patient = make_state_change(centroid=(-10.0, 0.0))
        assert self.engine._motion_alignment_score(agent, patient) == 0.0

    def test_motion_alignment_stationary(self) -> None:
        agent = make_agent(
            centroid=(0.0, 0.0),
            timestamp=0.0,
            velocity=(0.0, 0.0),
            speed=0.1,
        )
        patient = make_state_change(centroid=(10.0, 0.0))
        assert self.engine._motion_alignment_score(agent, patient) == 0.0

    def test_embedding_score_neutral_without_clip(self) -> None:
        agent = make_agent(visual_embedding=[])
        patient = make_state_change()
        self.engine._clip_model = False  # Force fallback path
        assert self.engine._embedding_score(agent, patient) == pytest.approx(0.5)

    def test_embedding_score_uses_clip_value(self) -> None:
        agent = make_agent()
        patient = make_state_change()

        def fake_clip(agent_candidate, patient_state):
            assert agent_candidate is agent
            assert patient_state is patient
            return 0.87

        self.engine._clip_semantic_causality = fake_clip  # type: ignore[method-assign]
        assert self.engine._embedding_score(agent, patient) == pytest.approx(0.87)


class TestCISFormula:
    """Exercise the combined CIS calculation."""

    def setup_method(self) -> None:
        self.config = CausalConfig(
            temporal_proximity_weight=0.25,
            spatial_proximity_weight=0.25,
            motion_alignment_weight=0.25,
            semantic_similarity_weight=0.25,
            min_score=0.5,
        )
        self.engine = CausalInferenceEngine(self.config)

    def test_weights_sum_to_one(self) -> None:
        total = (
            self.config.temporal_proximity_weight
            + self.config.spatial_proximity_weight
            + self.config.motion_alignment_weight
            + self.config.semantic_similarity_weight
        )
        assert total == pytest.approx(1.0)

    def test_score_range(self) -> None:
        agent = make_agent(
            timestamp=10.0,
            centroid=(0.0, 0.0),
            velocity=(20.0, 0.0),
            speed=20.0,
            visual_embedding=list(np.random.randn(4)),
        )
        patient = make_state_change(timestamp=10.5, centroid=(5.0, 0.0))
        score = self.engine.calculate_cis(agent, patient)
        assert 0.0 <= score <= 1.0

    def test_perfect_causal_case(self) -> None:
        agent = make_agent(
            entity_id="hand",
            timestamp=5.0,
            centroid=(0.0, 0.0),
            velocity=(25.0, 0.0),
            speed=25.0,
            visual_embedding=list(np.ones(8)),
        )
        patient = make_state_change(
            entity_id="object",
            timestamp=5.2,
            centroid=(5.0, 0.0),
            old_description="closed",
            new_description="open",
        )
        score = self.engine.calculate_cis(agent, patient)
        assert score > 0.6

    def test_non_causal_case(self) -> None:
        agent = make_agent(
            timestamp=0.0,
            centroid=(0.0, 0.0),
            velocity=(0.0, 0.0),
            speed=0.0,
            visual_embedding=list(np.zeros(4)),
        )
        patient = make_state_change(
            timestamp=20.0,
            centroid=(1000.0, 1000.0),
            old_description="idle",
            new_description="moved",
        )
        score = self.engine.calculate_cis(agent, patient)
        assert score < 0.3

    def test_threshold_application(self) -> None:
        self.config.min_score = 0.6
        engine = CausalInferenceEngine(self.config)
        agent = make_agent(
            timestamp=0.0,
            centroid=(0.0, 0.0),
            velocity=(5.0, 0.0),
            speed=5.0,
        )
        patient = make_state_change(timestamp=3.5, centroid=(400.0, 0.0))
        score = engine.calculate_cis(agent, patient)
        assert (score < self.config.min_score) or score == pytest.approx(self.config.min_score)


class TestCISIntegration:
    """Integration-style tests that mirror usage in semantic engine."""

    def test_score_multiple_agents(self) -> None:
        config = CausalConfig(top_k_per_event=3)
        engine = CausalInferenceEngine(config)

        agents = [
            make_agent(
                entity_id=f"agent_{i}",
                timestamp=1.0 + 0.1 * i,
                centroid=(float(i) * 20.0, 0.0),
                velocity=(15.0 - i, 0.0),
                speed=15.0 - i,
            )
            for i in range(5)
        ]
        patient = make_state_change(timestamp=1.6, centroid=(10.0, 0.0))

        links = engine.score_all_agents(agents, patient)

        assert len(links) <= config.top_k_per_event
        assert all(link.cis_score >= config.min_score for link in links)
        scores = [link.cis_score for link in links]
        assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
