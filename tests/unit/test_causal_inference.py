"""
"""Unit tests for `orion.semantic.causal`."""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

from orion.perception.tracker import MotionData
from orion.semantic.causal import (
    AgentCandidate,
    CausalConfig,
    CausalInferenceEngine,
    CausalLink,
    StateChange,
    cosine_similarity,
)


def make_agent(
    *,
    entity_id: str = "agent",
    timestamp: float = 0.0,
    centroid: tuple[float, float] = (0.0, 0.0),
    velocity: tuple[float, float] | None = None,
    speed: float = 0.0,
    embedding: List[float] | None = None,
) -> AgentCandidate:
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
        visual_embedding=embedding if embedding is not None else [0.1, 0.0, 0.2],
        object_class="person",
        description="a person",
    )


def make_state_change(
    *,
    entity_id: str = "patient",
    timestamp: float = 0.0,
    centroid: tuple[float, float] = (0.0, 0.0),
    frame_number: int = 0,
    old_description: str = "before",
    new_description: str = "after",
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


class TestCausalConfig:
    def test_default_weights_sum_to_one(self) -> None:
        config = CausalConfig()
        total = (
            config.temporal_proximity_weight
            + config.spatial_proximity_weight
            + config.motion_alignment_weight
            + config.semantic_similarity_weight
        )
        assert total == pytest.approx(1.0, abs=0.05)

    def test_custom_config_overrides(self) -> None:
        config = CausalConfig(
            temporal_proximity_weight=0.4,
            spatial_proximity_weight=0.1,
            motion_alignment_weight=0.3,
            semantic_similarity_weight=0.2,
            min_score=0.7,
        )
        assert config.temporal_proximity_weight == 0.4
        assert config.min_score == 0.7


class TestCausalInferenceEngine:
    def test_initialization(self) -> None:
        engine = CausalInferenceEngine()
        assert isinstance(engine.config, CausalConfig)

    def test_proximity_score_close_vs_far(self) -> None:
        engine = CausalInferenceEngine()
        agent = make_agent(centroid=(100.0, 100.0))
        patient_close = make_state_change(centroid=(110.0, 100.0))
        patient_far = make_state_change(centroid=(700.0, 100.0))
        assert engine._proximity_score(agent, patient_close) > engine._proximity_score(agent, patient_far)

    def test_motion_alignment(self) -> None:
        engine = CausalInferenceEngine()
        towards = make_agent(centroid=(0.0, 0.0), velocity=(10.0, 0.0), speed=10.0)
        away = make_agent(centroid=(0.0, 0.0), velocity=(-10.0, 0.0), speed=10.0)
        patient = make_state_change(centroid=(20.0, 0.0))
        assert engine._motion_alignment_score(towards, patient) > 0.0
        assert engine._motion_alignment_score(away, patient) == 0.0

    def test_temporal_score_decay(self) -> None:
        engine = CausalInferenceEngine()
        agent = make_agent(timestamp=0.0)
        patient_near = make_state_change(timestamp=0.5)
        patient_far = make_state_change(timestamp=10.0)
        assert engine._temporal_score(agent, patient_near) > engine._temporal_score(agent, patient_far)

    def test_calculate_cis_bounds(self) -> None:
        engine = CausalInferenceEngine()
        agent = make_agent(centroid=(0.0, 0.0), timestamp=0.0, velocity=(15.0, 0.0), speed=15.0)
        patient = make_state_change(timestamp=0.2, centroid=(10.0, 0.0))
        score = engine.calculate_cis(agent, patient)
        assert 0.0 <= score <= 1.0

    def test_calculate_cis_low_when_far(self) -> None:
        engine = CausalInferenceEngine()
        agent = make_agent(centroid=(0.0, 0.0), timestamp=0.0, velocity=(0.0, 0.0), speed=0.0)
        patient = make_state_change(timestamp=15.0, centroid=(900.0, 900.0))
        assert engine.calculate_cis(agent, patient) < 0.3

    def test_score_all_agents_limits_and_order(self) -> None:
        engine = CausalInferenceEngine()
        agents = [
            make_agent(
                entity_id=f"agent_{i}",
                timestamp=1.0 + i * 0.2,
                centroid=(float(i) * 20.0, 0.0),
                velocity=(15.0 - i, 0.0),
                speed=15.0 - i,
            )
            for i in range(5)
        ]
        patient = make_state_change(timestamp=2.0, centroid=(30.0, 0.0))
        links = engine.score_all_agents(agents, patient)
        assert len(links) <= engine.config.top_k_per_event
        scores = [link.cis_score for link in links]
        assert scores == sorted(scores, reverse=True)

    def test_filter_temporal_window(self) -> None:
        engine = CausalInferenceEngine()
        observations = [make_agent(entity_id=f"agent_{i}", timestamp=i * 0.5) for i in range(10)]
        state_change = make_state_change(timestamp=2.5)
        filtered = engine.filter_temporal_window(observations, state_change, window_size=1.0)
        assert all(1.5 <= agent.timestamp <= 2.5 for agent in filtered)
        assert len(filtered) > 0


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        vec = [1.0, 0.0, 0.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        assert cosine_similarity(vec1, vec2) == pytest.approx(-1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
            timestamp=5.0,
            frame_number=150,
            old_description="closed",
            new_description="open",
            centroid=(110.0, 100.0),
            bounding_box=[105, 95, 115, 105]
        )
        
        # Filter with 5-second window
        filtered = engine.filter_temporal_window(
            candidates,
            state_change,
            window_size=5.0
        )
        
        # Should include timestamps within [t-window, t] = [0.0, 5.0]
        filtered_times = [c.timestamp for c in filtered]
        assert 0.0 in filtered_times
        assert 2.0 in filtered_times
        assert 4.9 in filtered_times
        assert 5.0 in filtered_times
        # 5.1 is after state change, should NOT be included
        assert 5.1 not in filtered_times
        assert 7.0 not in filtered_times
        assert 10.0 not in filtered_times


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_cosine_similarity_identical(self):
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        
        sim = cosine_similarity(vec1, vec2)
        assert abs(sim - 1.0) < 0.01
    
    def test_cosine_similarity_orthogonal(self):
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        
        sim = cosine_similarity(vec1, vec2)
        assert abs(sim - 0.0) < 0.01
    
    def test_cosine_similarity_opposite(self):
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        
        sim = cosine_similarity(vec1, vec2)
        assert abs(sim - (-1.0)) < 0.01
    
    def test_cosine_similarity_zero_vector(self):
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        
        sim = cosine_similarity(vec1, vec2)
        assert sim == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
