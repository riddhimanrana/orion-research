"""
Unit Tests for Causal Inference Engine
=======================================

Tests for CIS calculation, agent scoring, and filtering.

Author: Orion Research Team
Date: October 2025
"""

import math
import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.causal_inference import (
    CausalInferenceEngine,
    CausalConfig,
    AgentCandidate,
    StateChange as CISStateChange,
    CausalLink,
    cosine_similarity,
)
from orion.motion_tracker import MotionData


class TestCausalConfig:
    """Test CausalConfig dataclass"""
    
    def test_default_config(self):
        config = CausalConfig()
        
        # Check weights sum to ~1.0
        total_weight = (
            config.proximity_weight +
            config.motion_weight +
            config.temporal_weight +
            config.embedding_weight
        )
        assert abs(total_weight - 1.0) < 0.01
        
        # Check reasonable defaults
        assert config.min_score > 0.0
        assert config.max_pixel_distance > 0.0
        assert config.top_k_per_event > 0
    
    def test_custom_config(self):
        config = CausalConfig(
            proximity_weight=0.5,
            motion_weight=0.3,
            min_score=0.7
        )
        
        assert config.proximity_weight == 0.5
        assert config.motion_weight == 0.3
        assert config.min_score == 0.7


class TestCausalInferenceEngine:
    """Test CausalInferenceEngine"""
    
    def test_initialization(self):
        engine = CausalInferenceEngine()
        assert engine.config is not None
        assert isinstance(engine.config, CausalConfig)
    
    def test_proximity_score_close_objects(self):
        engine = CausalInferenceEngine()
        
        # Create agent and patient very close together
        agent = AgentCandidate(
            entity_id="agent1",
            temp_id="det_001",
            timestamp=1.0,
            centroid=(100.0, 100.0),
            bounding_box=[95, 95, 105, 105],
            motion_data=None,
            visual_embedding=[0.1] * 512,
            object_class="person",
            description="a person"
        )
        
        patient = CISStateChange(
            entity_id="patient1",
            timestamp=1.1,
            frame_number=30,
            old_description="closed door",
            new_description="open door",
            centroid=(110.0, 100.0),  # 10 pixels away
            bounding_box=[105, 95, 115, 105]
        )
        
        score = engine._proximity_score(agent, patient)
        
        # Very close objects should have high proximity score
        assert score > 0.9
    
    def test_proximity_score_far_objects(self):
        engine = CausalInferenceEngine()
        
        agent = AgentCandidate(
            entity_id="agent1",
            temp_id="det_001",
            timestamp=1.0,
            centroid=(100.0, 100.0),
            bounding_box=[95, 95, 105, 105],
            motion_data=None,
            visual_embedding=[0.1] * 512,
            object_class="person",
            description="a person"
        )
        
        patient = CISStateChange(
            entity_id="patient1",
            timestamp=1.1,
            frame_number=30,
            old_description="closed door",
            new_description="open door",
            centroid=(1000.0, 1000.0),  # Very far away
            bounding_box=[995, 995, 1005, 1005]
        )
        
        score = engine._proximity_score(agent, patient)
        
        # Far objects should have low proximity score
        assert score < 0.1
    
    def test_motion_score_moving_towards(self):
        engine = CausalInferenceEngine()
        
        # Agent moving towards patient
        motion_data = MotionData(
            centroid=(100.0, 100.0),
            velocity=(10.0, 0.0),  # Moving right
            speed=10.0,
            direction=0.0,
            timestamp=1.0
        )
        
        agent = AgentCandidate(
            entity_id="agent1",
            temp_id="det_001",
            timestamp=1.0,
            centroid=(100.0, 100.0),
            bounding_box=[95, 95, 105, 105],
            motion_data=motion_data,
            visual_embedding=[0.1] * 512,
            object_class="person",
            description="a person"
        )
        
        # Patient to the right
        patient = CISStateChange(
            entity_id="patient1",
            timestamp=1.1,
            frame_number=30,
            old_description="closed door",
            new_description="open door",
            centroid=(150.0, 100.0),
            bounding_box=[145, 95, 155, 105]
        )
        
        score = engine._motion_score(agent, patient)
        
        # Moving towards should give positive score
        assert score > 0.0
    
    def test_motion_score_moving_away(self):
        engine = CausalInferenceEngine()
        
        # Agent moving away from patient
        motion_data = MotionData(
            centroid=(100.0, 100.0),
            velocity=(-10.0, 0.0),  # Moving left
            speed=10.0,
            direction=math.pi,  # 180 degrees
            timestamp=1.0
        )
        
        agent = AgentCandidate(
            entity_id="agent1",
            temp_id="det_001",
            timestamp=1.0,
            centroid=(100.0, 100.0),
            bounding_box=[95, 95, 105, 105],
            motion_data=motion_data,
            visual_embedding=[0.1] * 512,
            object_class="person",
            description="a person"
        )
        
        # Patient to the right (opposite direction)
        patient = CISStateChange(
            entity_id="patient1",
            timestamp=1.1,
            frame_number=30,
            old_description="closed door",
            new_description="open door",
            centroid=(150.0, 100.0),
            bounding_box=[145, 95, 155, 105]
        )
        
        score = engine._motion_score(agent, patient)
        
        # Moving away should give zero score
        assert score == 0.0
    
    def test_motion_score_no_motion_data(self):
        engine = CausalInferenceEngine()
        
        agent = AgentCandidate(
            entity_id="agent1",
            temp_id="det_001",
            timestamp=1.0,
            centroid=(100.0, 100.0),
            bounding_box=[95, 95, 105, 105],
            motion_data=None,  # No motion data
            visual_embedding=[0.1] * 512,
            object_class="person",
            description="a person"
        )
        
        patient = CISStateChange(
            entity_id="patient1",
            timestamp=1.1,
            frame_number=30,
            old_description="closed door",
            new_description="open door",
            centroid=(150.0, 100.0),
            bounding_box=[145, 95, 155, 105]
        )
        
        score = engine._motion_score(agent, patient)
        assert score == 0.0
    
    def test_temporal_score_recent(self):
        engine = CausalInferenceEngine()
        
        # Agent observed just before state change
        agent = AgentCandidate(
            entity_id="agent1",
            temp_id="det_001",
            timestamp=1.0,
            centroid=(100.0, 100.0),
            bounding_box=[95, 95, 105, 105],
            motion_data=None,
            visual_embedding=[0.1] * 512,
            object_class="person",
            description="a person"
        )
        
        patient = CISStateChange(
            entity_id="patient1",
            timestamp=1.1,  # 0.1 seconds later
            frame_number=30,
            old_description="closed door",
            new_description="open door",
            centroid=(110.0, 100.0),
            bounding_box=[105, 95, 115, 105]
        )
        
        score = engine._temporal_score(agent, patient)
        
        # Very recent should have high temporal score
        assert score > 0.9
    
    def test_temporal_score_distant(self):
        engine = CausalInferenceEngine()
        
        agent = AgentCandidate(
            entity_id="agent1",
            temp_id="det_001",
            timestamp=1.0,
            centroid=(100.0, 100.0),
            bounding_box=[95, 95, 105, 105],
            motion_data=None,
            visual_embedding=[0.1] * 512,
            object_class="person",
            description="a person"
        )
        
        patient = CISStateChange(
            entity_id="patient1",
            timestamp=10.0,  # 9 seconds later (beyond decay)
            frame_number=300,
            old_description="closed door",
            new_description="open door",
            centroid=(110.0, 100.0),
            bounding_box=[105, 95, 115, 105]
        )
        
        score = engine._temporal_score(agent, patient)
        
        # Distant in time should have low temporal score
        assert score < 0.1
    
    def test_calculate_cis_high_score(self):
        engine = CausalInferenceEngine()
        
        # Create scenario with high CIS:
        # - Close proximity
        # - Moving towards
        # - Recent in time
        
        motion_data = MotionData(
            centroid=(100.0, 100.0),
            velocity=(10.0, 0.0),
            speed=10.0,
            direction=0.0,
            timestamp=1.0
        )
        
        agent = AgentCandidate(
            entity_id="agent1",
            temp_id="det_001",
            timestamp=1.0,
            centroid=(100.0, 100.0),
            bounding_box=[95, 95, 105, 105],
            motion_data=motion_data,
            visual_embedding=[0.1] * 512,
            object_class="person",
            description="a person"
        )
        
        patient = CISStateChange(
            entity_id="patient1",
            timestamp=1.1,
            frame_number=30,
            old_description="closed door",
            new_description="open door",
            centroid=(110.0, 100.0),
            bounding_box=[105, 95, 115, 105]
        )
        
        cis = engine.calculate_cis(agent, patient)
        
        # Should have high CIS score (relaxed threshold due to weights)
        assert cis > 0.65
    
    def test_calculate_cis_low_score(self):
        engine = CausalInferenceEngine()
        
        # Create scenario with low CIS:
        # - Far apart
        # - No motion
        # - Distant in time
        
        agent = AgentCandidate(
            entity_id="agent1",
            temp_id="det_001",
            timestamp=1.0,
            centroid=(100.0, 100.0),
            bounding_box=[95, 95, 105, 105],
            motion_data=None,
            visual_embedding=[0.1] * 512,
            object_class="person",
            description="a person"
        )
        
        patient = CISStateChange(
            entity_id="patient1",
            timestamp=10.0,
            frame_number=300,
            old_description="closed door",
            new_description="open door",
            centroid=(1000.0, 1000.0),
            bounding_box=[995, 995, 1005, 1005]
        )
        
        cis = engine.calculate_cis(agent, patient)
        
        # Should have low CIS score
        assert cis < 0.3
    
    def test_score_all_agents(self):
        engine = CausalInferenceEngine()
        
        # Create multiple agent candidates
        agents = []
        for i in range(5):
            motion_data = MotionData(
                centroid=(100.0 + i * 20, 100.0),
                velocity=(10.0, 0.0),
                speed=10.0,
                direction=0.0,
                timestamp=1.0
            )
            
            agent = AgentCandidate(
                entity_id=f"agent{i}",
                temp_id=f"det_{i:03d}",
                timestamp=1.0,
                centroid=(100.0 + i * 20, 100.0),
                bounding_box=[95 + i*20, 95, 105 + i*20, 105],
                motion_data=motion_data,
                visual_embedding=[0.1 * i] * 512,
                object_class="person",
                description=f"person {i}"
            )
            agents.append(agent)
        
        # Patient close to first agent
        patient = CISStateChange(
            entity_id="patient1",
            timestamp=1.1,
            frame_number=30,
            old_description="closed door",
            new_description="open door",
            centroid=(110.0, 100.0),
            bounding_box=[105, 95, 115, 105]
        )
        
        # Score all agents
        links = engine.score_all_agents(agents, patient)
        
        # Should return links
        assert len(links) > 0
        
        # Links should be sorted by CIS score (descending)
        for i in range(len(links) - 1):
            assert links[i].cis_score >= links[i+1].cis_score
        
        # First link should be the closest agent
        assert links[0].agent.entity_id == "agent0"
    
    def test_score_all_agents_filtering(self):
        engine = CausalInferenceEngine(
            config=CausalConfig(min_score=0.8)  # High threshold
        )
        
        # Create agents far from patient
        agents = []
        for i in range(3):
            agent = AgentCandidate(
                entity_id=f"agent{i}",
                temp_id=f"det_{i:03d}",
                timestamp=1.0,
                centroid=(1000.0 + i * 100, 1000.0),  # Far away
                bounding_box=[995 + i*100, 995, 1005 + i*100, 1005],
                motion_data=None,
                visual_embedding=[0.1] * 512,
                object_class="person",
                description=f"person {i}"
            )
            agents.append(agent)
        
        patient = CISStateChange(
            entity_id="patient1",
            timestamp=1.1,
            frame_number=30,
            old_description="closed door",
            new_description="open door",
            centroid=(100.0, 100.0),
            bounding_box=[95, 95, 105, 105]
        )
        
        links = engine.score_all_agents(agents, patient)
        
        # Should filter out all low-scoring agents
        assert len(links) == 0
    
    def test_filter_temporal_window(self):
        engine = CausalInferenceEngine()
        
        # Create candidates at various times
        candidates = []
        timestamps = [0.0, 2.0, 4.9, 5.0, 5.1, 7.0, 10.0]
        
        for i, ts in enumerate(timestamps):
            candidate = AgentCandidate(
                entity_id=f"agent{i}",
                temp_id=f"det_{i:03d}",
                timestamp=ts,
                centroid=(100.0, 100.0),
                bounding_box=[95, 95, 105, 105],
                motion_data=None,
                visual_embedding=[0.1] * 512,
                object_class="person",
                description=f"person {i}"
            )
            candidates.append(candidate)
        
        # State change at t=5.0
        state_change = CISStateChange(
            entity_id="patient1",
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
