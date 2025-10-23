"""
Causal Inference Engine
========================

This module implements the two-stage causal inference process:
1. Mathematical Causal Influence Score (CIS) calculation
2. LLM-based event verification and semantic labeling

The CIS function mathematically scores potential causal relationships
before passing high-scoring pairs to the LLM for verification.

Author: Orion Research Team
Date: October 2025
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from .motion_tracker import MotionData, MotionTracker, calculate_distance
except ImportError:
    from motion_tracker import MotionData, MotionTracker, calculate_distance


logger = logging.getLogger("CausalInference")


@dataclass
class CausalConfig:
    """
    Configuration for Causal Influence Score calculation.
    
    These weights should be learned from data using HPO (see orion.hpo.cis_optimizer).
    Default values are provided as starting points but should be optimized.
    """
    
    # === CIS Component Weights ===
    # CONSTRAINT: Should sum to ~1.0 for interpretability
    # These determine the relative importance of each factor
    
    temporal_proximity_weight: float = 0.3
    """Weight for temporal proximity component (when did it happen?)"""
    
    spatial_proximity_weight: float = 0.3
    """Weight for spatial proximity component (how close?)"""
    
    motion_alignment_weight: float = 0.2
    """Weight for motion alignment component (was it moving toward?)"""
    
    semantic_similarity_weight: float = 0.2
    """Weight for semantic similarity component (are they related?)"""
    
    # === Threshold Parameters ===
    # These control when causal links are created
    
    min_score: float = 0.55
    """Minimum CIS score to consider a causal relationship (learned via HPO)"""
    
    top_k_per_event: int = 5
    """Maximum number of causal agents to consider per state change"""
    
    # === Distance/Decay Parameters ===
    # These control how quickly influence drops off with distance/time
    
    max_pixel_distance: float = 600.0
    """Maximum spatial distance for influence (pixels, resolution-dependent)"""
    
    temporal_decay: float = 4.0
    """Temporal decay constant (seconds, affects how long influence persists)"""
    
    # === Motion Parameters ===
    # These control motion-based causality detection
    
    motion_angle_threshold: float = math.pi / 4  # 45 degrees
    """Maximum angle deviation to consider 'moving towards' (radians)"""
    
    min_motion_speed: float = 5.0
    """Minimum speed to consider as significant motion (pixels/second)"""
    
    # === Semantic Similarity Parameters ===
    
    use_semantic_similarity: bool = True
    """Whether to use embedding similarity in CIS calculation"""
    
    semantic_similarity_threshold: float = 0.7
    """Minimum embedding similarity to consider semantically related"""
    
    def __post_init__(self):
        """Validate configuration"""
        # Check that weights are reasonable
        total_weight = (
            self.temporal_proximity_weight +
            self.spatial_proximity_weight +
            self.motion_alignment_weight +
            self.semantic_similarity_weight
        )
        
        if abs(total_weight - 1.0) > 0.1:
            logger.warning(
                f"CIS weights sum to {total_weight:.3f}, not 1.0. "
                f"Consider normalizing for interpretability."
            )
    
    @classmethod
    def from_hpo_result(cls, hpo_result_path: str) -> "CausalConfig":
        """
        Load configuration from HPO optimization results.
        
        Args:
            hpo_result_path: Path to JSON file from CISOptimizer
            
        Returns:
            CausalConfig with optimized parameters
        """
        import json
        with open(hpo_result_path) as f:
            result = json.load(f)
        
        weights = result["best_weights"]
        threshold = result["best_threshold"]
        
        return cls(
            temporal_proximity_weight=weights["temporal"],
            spatial_proximity_weight=weights["spatial"],
            motion_alignment_weight=weights.get("motion", 0.2),
            semantic_similarity_weight=weights["semantic"],
            min_score=threshold
        )


@dataclass
class StateChange:
    """
    Represents a detected state change in an object (the "patient")
    """
    
    entity_id: str
    timestamp: float
    frame_number: int
    old_description: str
    new_description: str
    centroid: Tuple[float, float]
    bounding_box: List[int]
    
    def __repr__(self) -> str:
        return (
            f"StateChange(entity={self.entity_id}, "
            f"time={self.timestamp:.2f}s, "
            f"'{self.old_description}' → '{self.new_description}')"
        )


@dataclass
class AgentCandidate:
    """
    Represents a potential causal agent for a state change
    """
    
    entity_id: str
    temp_id: str
    timestamp: float
    centroid: Tuple[float, float]
    bounding_box: List[int]
    motion_data: Optional[MotionData]
    visual_embedding: List[float]
    object_class: str
    description: Optional[str]


@dataclass
class CausalLink:
    """
    Represents a scored causal relationship between agent and patient
    """
    
    agent: AgentCandidate
    patient: StateChange
    cis_score: float
    proximity_score: float
    motion_score: float
    temporal_score: float
    embedding_score: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for LLM prompt"""
        return {
            "agent_id": self.agent.entity_id,
            "agent_class": self.agent.object_class,
            "agent_description": self.agent.description or self.agent.object_class,
            "patient_id": self.patient.entity_id,
            "patient_old_state": self.patient.old_description,
            "patient_new_state": self.patient.new_description,
            "cis_score": round(self.cis_score, 3),
            "timestamp": self.patient.timestamp,
        }
    
    def __repr__(self) -> str:
        return (
            f"CausalLink(agent={self.agent.entity_id}, "
            f"patient={self.patient.entity_id}, "
            f"CIS={self.cis_score:.3f})"
        )


class CausalInferenceEngine:
    """
    Implements the Causal Influence Score (CIS) calculation and
    manages the two-stage causal inference process
    """
    
    def __init__(self, config: Optional[CausalConfig] = None):
        """
        Args:
            config: Configuration parameters for CIS calculation
        """
        self.config = config or CausalConfig()
        logger.info(f"Initialized CausalInferenceEngine with config: {self.config}")
    
    def _entity_overlap_score(self, agent: AgentCandidate, patient: StateChange) -> float:
        """
        f_overlap: Entity overlap function
        
        This is a placeholder. In a real scenario, this would check for shared parts
        or other forms of overlap. For now, we'll return a neutral score.
        """
        return 0.5

    def calculate_cis(
        self,
        agent: AgentCandidate,
        patient: StateChange,
    ) -> float:
        """
        Calculate the Causal Influence Score for an agent-patient pair
        
        CIS = w1·f_temporal + w2·f_spatial + w3·f_overlap + w4·f_semantic
        
        Args:
            agent: Potential causal agent
            patient: Object that underwent state change
            
        Returns:
            CIS score in [0, 1]
        """
        # Component scores
        temporal_score = self._temporal_score(agent, patient)
        spatial_score = self._proximity_score(agent, patient)
        motion_score = self._motion_alignment_score(agent, patient)
        semantic_score = self._embedding_score(agent, patient)
        
        # Weighted combination (weights should be learned via HPO)
        cis = (
            self.config.temporal_proximity_weight * temporal_score +
            self.config.spatial_proximity_weight * spatial_score +
            self.config.motion_alignment_weight * motion_score +
            self.config.semantic_similarity_weight * semantic_score
        )
        
        return max(0.0, min(1.0, cis))  # Clamp to [0, 1]
    
    def _proximity_score(
        self,
        agent: AgentCandidate,
        patient: StateChange
    ) -> float:
        """
        f_prox: Spatio-temporal proximity function
        
        Measures inverse distance between agent and patient at the time
        of the state change. Closer objects have higher influence.
        
        Returns:
            Score in [0, 1]
        """
        distance = calculate_distance(agent.centroid, patient.centroid)
        
        # Inverse distance with configurable max distance
        if distance >= self.config.max_pixel_distance:
            return 0.0
        
        # Normalized inverse distance
        normalized = 1.0 - (distance / self.config.max_pixel_distance)
        return normalized ** 2  # Quadratic falloff for sharper locality
    
    def _motion_alignment_score(
        self,
        agent: AgentCandidate,
        patient: StateChange
    ) -> float:
        """
        f_motion: Directed motion alignment function
        
        Measures if the agent was moving towards the patient before
        the state change occurred. Higher score for direct approach.
        
        Returns:
            Score in [0, 1]
        """
        if agent.motion_data is None:
            return 0.0
        
        # Check if moving towards patient
        is_towards = agent.motion_data.is_moving_towards(
            patient.centroid,
            self.config.motion_angle_threshold
        )
        
        if not is_towards:
            return 0.0
        
        # Score based on speed (faster = stronger influence)
        # Normalize by a reasonable max speed (e.g., 200 px/s)
        max_speed = 200.0
        speed_factor = min(agent.motion_data.speed / max_speed, 1.0)
        
        # Also consider if speed is above minimum threshold
        if agent.motion_data.speed < self.config.min_motion_speed:
            return 0.0
        
        return speed_factor
    
    def _temporal_score(
        self,
        agent: AgentCandidate,
        patient: StateChange
    ) -> float:
        """
        f_temporal: Temporal proximity function
        
        Measures how close in time the agent observation was to the
        state change. Recent observations have higher influence.
        
        Returns:
            Score in [0, 1]
        """
        time_diff = abs(patient.timestamp - agent.timestamp)
        
        if time_diff >= self.config.temporal_decay:
            return 0.0
        
        # Exponential decay
        decay_factor = math.exp(-time_diff / self.config.temporal_decay)
        return decay_factor
    
    def _embedding_score(
        self,
        agent: AgentCandidate,
        patient: StateChange
    ) -> float:
        """
        f_embedding: Visual/semantic similarity component
        
        Measures semantic relatedness between agent and patient using
        CLIP embeddings. Higher similarity suggests potential interaction
        (e.g., hand and cup, person and door).
        
        Returns:
            Score in [0, 1]
        """
        if not self.config.use_semantic_similarity:
            return 0.5  # Neutral score if disabled
        
        # Get agent embedding
        if not hasattr(agent, 'visual_embedding') or agent.visual_embedding is None:
            return 0.5
        
        agent_emb = np.array(agent.visual_embedding)
        
        # For patient, we don't have embedding stored directly
        # Instead, use a semantic similarity threshold
        # This is a simplification - in full implementation, we'd store
        # patient embeddings from state change detection
        
        # For now, return neutral score
        # TODO: Store patient embeddings during state change detection
        return 0.5
    
    def score_all_agents(
        self,
        agent_candidates: List[AgentCandidate],
        patient: StateChange
    ) -> List[CausalLink]:
        """
        Score all potential agents for a single state change event
        
        Args:
            agent_candidates: List of potential causal agents
            patient: The state change to explain
            
        Returns:
            List of CausalLink objects, sorted by CIS score (descending)
        """
        links = []
        
        for agent in agent_candidates:
            # Skip if agent is the patient itself
            if agent.entity_id == patient.entity_id:
                continue
            
            cis = self.calculate_cis(agent, patient)
            
            # Only include if above threshold
            if cis >= self.config.min_score:
                link = CausalLink(
                    agent=agent,
                    patient=patient,
                    cis_score=cis,
                    proximity_score=self._proximity_score(agent, patient),
                    motion_score=self._motion_alignment_score(agent, patient),
                    temporal_score=self._temporal_score(agent, patient),
                    embedding_score=self._embedding_score(agent, patient)
                )
                links.append(link)
        
        # Sort by CIS score (descending)
        links.sort(key=lambda x: x.cis_score, reverse=True)
        
        # Keep only top-K
        return links[:self.config.top_k_per_event]
    
    def filter_temporal_window(
        self,
        all_observations: List[AgentCandidate],
        state_change: StateChange,
        window_size: float = 5.0
    ) -> List[AgentCandidate]:
        """
        Filter observations to those within temporal window of state change
        
        Args:
            all_observations: All available object observations
            state_change: The state change event
            window_size: Time window in seconds (before state change)
            
        Returns:
            Filtered list of candidates within window
        """
        t_change = state_change.timestamp
        t_start = t_change - window_size
        
        candidates = [
            obs for obs in all_observations
            if t_start <= obs.timestamp <= t_change
        ]
        
        return candidates


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity in [-1, 1]
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(v1, v2) / (norm1 * norm2))
