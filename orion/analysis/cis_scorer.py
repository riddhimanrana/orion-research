"""
Causal Influence Scorer (CIS)
=============================

Computes causal influence scores between entities based on:
1. Temporal proximity
2. Spatial proximity (3D)
3. Motion alignment
4. Semantic compatibility (CLIP + Heuristics)
5. Interaction bonuses (Hand-Object)
"""

import numpy as np
import math
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass

@dataclass
class CISComponents:
    """Breakdown of CIS score components"""
    temporal: float
    spatial: float
    motion: float
    semantic: float
    hand_bonus: float
    total: float


class CausalInfluenceScorer:
    """
    Enhanced CIS computation with 3D and hand interaction signals.
    
    Formula:
    CIS = w_t * T + w_s * S + w_m * M + w_se * Se + H
    """
    
    def __init__(
        self,
        weight_temporal: float = 0.30,
        weight_spatial: float = 0.44,
        weight_motion: float = 0.21,
        weight_semantic: float = 0.06,
        temporal_decay_tau: float = 4.0,  # seconds
        max_spatial_distance_mm: float = 600.0,  # 600mm = 60cm
        hand_grasping_bonus: float = 0.30,
        hand_touching_bonus: float = 0.15,
        hand_near_bonus: float = 0.05,
        cis_threshold: float = 0.50,
    ):
        self.weight_temporal = weight_temporal
        self.weight_spatial = weight_spatial
        self.weight_motion = weight_motion
        self.weight_semantic = weight_semantic
        self.temporal_decay_tau = temporal_decay_tau
        self.max_spatial_distance_mm = max_spatial_distance_mm
        self.hand_grasping_bonus = hand_grasping_bonus
        self.hand_touching_bonus = hand_touching_bonus
        self.hand_near_bonus = hand_near_bonus
        self.cis_threshold = cis_threshold
        
        # Type compatibility table (heuristic)
        self.type_compatibility = self._build_type_compatibility()
    
    def calculate_cis(
        self,
        agent_entity: Any, # PerceptionEntity
        patient_entity: Any, # PerceptionEntity
        time_delta: float,
        scene_context: Optional[Dict] = None
    ) -> Tuple[float, CISComponents]:
        """
        Compute CIS between two entities.
        """
        # === 1. TEMPORAL SCORE ===
        T = self._temporal_score(time_delta)
        
        # === 2. SPATIAL SCORE (3D) ===
        S = self._spatial_score_3d(agent_entity, patient_entity)
        
        # === 3. MOTION SCORE (3D velocity alignment) ===
        M = self._motion_score_3d(agent_entity, patient_entity)
        
        # === 4. SEMANTIC SCORE ===
        Se = self._semantic_score(agent_entity, patient_entity)
        
        # === 5. HAND INTERACTION BONUS ===
        H = self._hand_bonus(agent_entity, patient_entity)
        
        # === COMBINE ===
        cis = (
            self.weight_temporal * T +
            self.weight_spatial * S +
            self.weight_motion * M +
            self.weight_semantic * Se +
            H
        )
        
        # Clip to [0, 1]
        cis = max(0.0, min(1.0, cis))
        
        # Apply scene context boost
        if scene_context:
            cis = self._apply_scene_context(cis, agent_entity, patient_entity, scene_context)
        
        components = CISComponents(
            temporal=T,
            spatial=S,
            motion=M,
            semantic=Se,
            hand_bonus=H,
            total=cis
        )
        
        return cis, components
    
    def _temporal_score(self, time_delta: float) -> float:
        """Exponential temporal decay."""
        score = np.exp(-abs(time_delta) / self.temporal_decay_tau)
        return float(score)
    
    def _spatial_score_3d(self, agent: Any, patient: Any) -> float:
        """3D Euclidean distance-based scoring."""
        # Try to get 3D centroids
        # Assuming PerceptionEntity has a method or property for this, or we use the last observation
        agent_pos = self._get_centroid_3d(agent)
        patient_pos = self._get_centroid_3d(patient)
        
        if agent_pos is None or patient_pos is None:
            return 0.5  # Unknown
        
        dist_3d = np.linalg.norm(np.array(agent_pos) - np.array(patient_pos))
        
        # Quadratic falloff
        score = max(0.0, 1.0 - (dist_3d / self.max_spatial_distance_mm) ** 2)
        
        # Boost if very close
        if dist_3d < 50.0: # 5cm
            score = min(1.0, score * 1.2)
            
        return float(score)

    def _motion_score_3d(self, agent: Any, patient: Any) -> float:
        """3D velocity alignment."""
        # Placeholder: assume entities might have velocity vectors in future
        # For now, return neutral
        return 0.5
    
    def _semantic_score(self, agent: Any, patient: Any) -> float:
        """Semantic similarity + type compatibility."""
        # Embedding similarity
        emb_sim = 0.5
        if agent.average_embedding is not None and patient.average_embedding is not None:
             # Cosine similarityz
            dot = np.dot(agent.average_embedding, patient.average_embedding)
            norm = np.linalg.norm(agent.average_embedding) * np.linalg.norm(patient.average_embedding)
            if norm > 1e-6:
                emb_sim = (dot / norm + 1.0) / 2.0
                
        # Type compatibility
        agent_cls = str(agent.object_class)
        patient_cls = str(patient.object_class)
        type_compat = self._check_type_compatibility(agent_cls, patient_cls)
        
        return 0.6 * emb_sim + 0.4 * type_compat

    def _hand_bonus(self, agent: Any, patient: Any) -> float:
        """Bonus if agent is hand."""
        # Simple check if "hand" is in class name
        agent_cls = str(agent.object_class).lower()
        if "hand" in agent_cls:
            # If we had interaction state, we'd use it. For now, small bonus if spatial is high
            return self.hand_near_bonus
        return 0.0

    def _get_centroid_3d(self, entity: Any) -> Optional[Tuple[float, float, float]]:
        """Helper to extract 3D centroid from entity."""
        # Use the most recent observation with depth
        if not entity.observations:
            return None
        # Sort by timestamp desc
        sorted_obs = sorted(entity.observations, key=lambda x: x.timestamp, reverse=True)
        for obs in sorted_obs:
            if hasattr(obs, 'centroid_3d') and obs.centroid_3d is not None:
                return obs.centroid_3d
        return None

    def _check_type_compatibility(self, class_a: str, class_b: str) -> float:
        pair = tuple(sorted([class_a.lower(), class_b.lower()]))
        return self.type_compatibility.get(pair, 0.5)

    def _build_type_compatibility(self) -> Dict[Tuple[str, str], float]:
        return {
            ('cup', 'hand'): 1.0,
            ('hand', 'keyboard'): 0.9,
            ('hand', 'mouse'): 0.9,
            ('hand', 'phone'): 0.9,
            ('book', 'hand'): 0.8,
            ('chair', 'person'): 0.8,
            ('cup', 'table'): 0.7,
        }

    def _apply_scene_context(self, cis: float, agent: Any, patient: Any, context: Dict) -> float:
        # Placeholder for scene context logic
        return cis
