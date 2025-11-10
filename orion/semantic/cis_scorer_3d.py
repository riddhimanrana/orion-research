"""
Enhanced Causal Influence Scorer (Phase 3)

CIS with 3D spatial/motion and hand interaction signals.
Extends the base CIS with:
- 3D distance scoring
- 3D velocity alignment
- Hand interaction bonuses
- Scene context modulation

Author: Orion Research Team
Date: November 2025
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from orion.semantic.types import StateChange, CausalLink


@dataclass
class CISComponents:
    """Breakdown of CIS score components"""
    temporal: float
    spatial: float
    motion: float
    semantic: float
    hand_bonus: float
    total: float


class CausalInfluenceScorer3D:
    """
    Enhanced CIS computation with 3D and hand interaction signals.
    
    Formula:
    CIS = w_t × T + w_s × S + w_m × M + w_se × Se + H
    
    Where:
    - T: Temporal decay (exp(-Δt/τ))
    - S: 3D spatial proximity
    - M: 3D velocity alignment
    - Se: Semantic similarity (CLIP embeddings)
    - H: Hand interaction bonus (additive)
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
        """
        Initialize enhanced CIS scorer.
        
        Args:
            weight_temporal: Weight for temporal component
            weight_spatial: Weight for spatial component
            weight_motion: Weight for motion component
            weight_semantic: Weight for semantic component
            temporal_decay_tau: Time constant for exponential decay (seconds)
            max_spatial_distance_mm: Max distance for full spatial score
            hand_grasping_bonus: Bonus for hand grasping interaction
            hand_touching_bonus: Bonus for hand touching interaction
            hand_near_bonus: Bonus for hand near interaction
            cis_threshold: Minimum CIS to create causal link
        """
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
        
        # Type compatibility table
        self.type_compatibility = self._build_type_compatibility()
    
    def calculate_cis(
        self,
        agent_entity: Dict,
        patient_entity: Dict,
        time_delta: float,
        scene_context: Optional[Dict] = None
    ) -> Tuple[float, CISComponents]:
        """
        Compute CIS between two entities.
        
        Args:
            agent_entity: Agent entity dict with keys:
                - centroid_3d_mm: (x, y, z) position
                - velocity_3d: (vx, vy, vz) velocity
                - embedding: CLIP embedding
                - class_label: entity class
                - is_hand: bool
            patient_entity: Patient entity dict (same structure)
            time_delta: Time difference in seconds (absolute value)
            scene_context: Optional scene context dict with:
                - scene_type: str
                - boost_multiplier: float
        
        Returns:
            Tuple of (cis_score, components)
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
        """
        Exponential temporal decay.
        
        Score = exp(-Δt / τ)
        
        Args:
            time_delta: Time difference in seconds
        
        Returns:
            Temporal score [0, 1]
        """
        score = np.exp(-abs(time_delta) / self.temporal_decay_tau)
        return float(score)
    
    def _spatial_score_3d(
        self,
        agent_entity: Dict,
        patient_entity: Dict
    ) -> float:
        """
        3D Euclidean distance-based scoring with quadratic falloff.
        
        Score = max(0, 1 - (d / d_max)²)
        
        Args:
            agent_entity: Agent entity
            patient_entity: Patient entity
        
        Returns:
            Spatial score [0, 1]
        """
        agent_pos = agent_entity.get('centroid_3d_mm')
        patient_pos = patient_entity.get('centroid_3d_mm')
        
        if agent_pos is None or patient_pos is None:
            return 0.5  # Unknown, neutral score
        
        # 3D Euclidean distance
        dist_3d = np.linalg.norm(
            np.array(agent_pos) - np.array(patient_pos)
        )
        
        # Quadratic falloff
        score = max(0.0, 1.0 - (dist_3d / self.max_spatial_distance_mm) ** 2)
        
        # Boost if very close (touching)
        if dist_3d < 10.0:  # 10mm = 1cm
            score = min(1.0, score * 1.3)
        
        return float(score)
    
    def _motion_score_3d(
        self,
        agent_entity: Dict,
        patient_entity: Dict
    ) -> float:
        """
        3D velocity alignment scoring.
        
        Computes cosine similarity of velocity vectors and checks if
        agent is moving toward patient.
        
        Args:
            agent_entity: Agent entity
            patient_entity: Patient entity
        
        Returns:
            Motion score [0, 1]
        """
        vel_agent = agent_entity.get('velocity_3d')
        vel_patient = patient_entity.get('velocity_3d')
        
        if vel_agent is None or vel_patient is None:
            return 0.5  # Unknown, neutral
        
        vel_agent = np.array(vel_agent)
        vel_patient = np.array(vel_patient)
        
        vel_mag_a = np.linalg.norm(vel_agent)
        vel_mag_p = np.linalg.norm(vel_patient)
        
        if vel_mag_a < 1e-6 or vel_mag_p < 1e-6:
            return 0.5  # Neither moving strongly
        
        # Normalize and compute cosine similarity
        vel_a_norm = vel_agent / vel_mag_a
        vel_p_norm = vel_patient / vel_mag_p
        
        cos_angle = np.dot(vel_a_norm, vel_p_norm)
        
        # Map [-1, 1] → [0, 1]
        score = (cos_angle + 1.0) / 2.0
        
        # Boost if agent moving toward patient
        agent_pos = agent_entity.get('centroid_3d_mm')
        patient_pos = patient_entity.get('centroid_3d_mm')
        
        if agent_pos is not None and patient_pos is not None:
            to_patient = np.array(patient_pos) - np.array(agent_pos)
            if np.dot(vel_agent, to_patient) > 0:
                score *= 1.2
        
        return float(min(1.0, score))
    
    def _semantic_score(
        self,
        agent_entity: Dict,
        patient_entity: Dict
    ) -> float:
        """
        Semantic similarity using CLIP embeddings + type compatibility.
        
        Score = 0.6 × embedding_sim + 0.4 × type_compat
        
        Args:
            agent_entity: Agent entity
            patient_entity: Patient entity
        
        Returns:
            Semantic score [0, 1]
        """
        agent_emb = agent_entity.get('embedding')
        patient_emb = patient_entity.get('embedding')
        
        if agent_emb is None or patient_emb is None:
            emb_sim = 0.5
        else:
            # Cosine similarity
            emb_sim = np.dot(agent_emb, patient_emb) / (
                np.linalg.norm(agent_emb) * np.linalg.norm(patient_emb) + 1e-6
            )
            # Map [-1, 1] → [0, 1]
            emb_sim = (emb_sim + 1.0) / 2.0
        
        # Type compatibility
        agent_class = agent_entity.get('class_label', 'unknown')
        patient_class = patient_entity.get('class_label', 'unknown')
        type_compat = self._check_type_compatibility(agent_class, patient_class)
        
        score = 0.6 * emb_sim + 0.4 * type_compat
        
        return float(score)
    
    def _hand_bonus(
        self,
        agent_entity: Dict,
        patient_entity: Dict
    ) -> float:
        """
        Bonus if agent is hand and interacting with patient.
        
        Args:
            agent_entity: Agent entity
            patient_entity: Patient entity
        
        Returns:
            Hand bonus [0, 0.3]
        """
        is_hand = agent_entity.get('is_hand', False)
        
        if not is_hand:
            return 0.0
        
        interaction_type = patient_entity.get('hand_interaction_type')
        
        if interaction_type == 'GRASPING':
            return self.hand_grasping_bonus
        elif interaction_type == 'TOUCHING':
            return self.hand_touching_bonus
        elif interaction_type == 'NEAR':
            return self.hand_near_bonus
        else:
            return 0.0
    
    def _check_type_compatibility(
        self,
        class_a: str,
        class_b: str
    ) -> float:
        """
        Heuristic type compatibility score.
        
        Args:
            class_a: First class
            class_b: Second class
        
        Returns:
            Compatibility score [0, 1]
        """
        pair = tuple(sorted([class_a, class_b]))
        return self.type_compatibility.get(pair, 0.5)
    
    def _build_type_compatibility(self) -> Dict[Tuple[str, str], float]:
        """Build type compatibility lookup table."""
        return {
            # Hand interactions
            ('cup', 'hand'): 1.0,
            ('door', 'hand'): 1.0,
            ('hand', 'keyboard'): 0.9,
            ('hand', 'mouse'): 0.9,
            ('hand', 'phone'): 0.9,
            ('hand', 'knife'): 0.9,
            ('book', 'hand'): 0.8,
            ('hand', 'remote'): 0.9,
            ('hand', 'pen'): 0.9,
            
            # Person interactions
            ('door', 'person'): 0.9,
            ('chair', 'person'): 0.8,
            
            # Object-object
            ('cup', 'table'): 0.7,
            ('food', 'knife'): 0.8,
            ('book', 'table'): 0.7,
            ('keyboard', 'mouse'): 0.6,
            ('phone', 'table'): 0.7,
            
            # Default for same class
            # (added dynamically if needed)
        }
    
    def _apply_scene_context(
        self,
        cis: float,
        agent_entity: Dict,
        patient_entity: Dict,
        context: Dict
    ) -> float:
        """
        Apply scene-specific boosting to CIS.
        
        Args:
            cis: Current CIS score
            agent_entity: Agent entity
            patient_entity: Patient entity
            context: Scene context dict
        
        Returns:
            Boosted CIS score
        """
        scene_type = context.get('scene_type', 'unknown')
        
        # Scene-specific interaction boosts
        scene_boosts = {
            'kitchen': {
                ('hand', 'knife'): 1.3,
                ('hand', 'oven'): 1.2,
                ('food', 'knife'): 1.3,
                ('cup', 'hand'): 1.2,
            },
            'office': {
                ('hand', 'keyboard'): 1.3,
                ('hand', 'mouse'): 1.3,
                ('hand', 'phone'): 1.2,
                ('keyboard', 'mouse'): 1.1,
            },
            'living_room': {
                ('hand', 'remote'): 1.3,
                ('hand', 'tv'): 1.2,
                ('remote', 'tv'): 1.2,
            },
            'bedroom': {
                ('book', 'hand'): 1.2,
                ('hand', 'phone'): 1.2,
                ('hand', 'lamp'): 1.2,
            },
        }
        
        agent_class = agent_entity.get('class_label', 'unknown')
        patient_class = patient_entity.get('class_label', 'unknown')
        pair = (agent_class, patient_class)
        
        boost = scene_boosts.get(scene_type, {}).get(pair, 1.0)
        
        return float(min(1.0, cis * boost))
    
    def compute_causal_links(
        self,
        state_changes: List,  # List[StateChange]
        embeddings: Dict[str, "np.ndarray"],
    ) -> List:  # List[CausalLink]
        """
        Compute causal links for state changes (compatibility wrapper for 2D CIS interface).
        
        Args:
            state_changes: List of StateChange objects
            embeddings: Dict mapping entity_id → CLIP embedding
        
        Returns:
            List of CausalLink objects
        """
        from orion.semantic.types import CausalLink
        
        if len(state_changes) < 2:
            return []
        
        causal_links = []
        sorted_changes = sorted(state_changes, key=lambda c: c.timestamp_after)
        
        # For each pair of state changes, compute 3D CIS
        for idx, agent_change in enumerate(sorted_changes):
            for patient_change in sorted_changes[idx + 1:]:
                # Build entity dicts
                agent_entity = {
                    'centroid_3d_mm': agent_change.centroid_3d_after,
                    'velocity_3d': agent_change.velocity_3d,
                    'embedding': embeddings.get(agent_change.entity_id),
                    'class_label': agent_change.entity_id.split('_')[0],  # Extract class from entity_id
                    'is_hand': 'hand' in agent_change.entity_id.lower(),
                }
                
                patient_entity = {
                    'centroid_3d_mm': patient_change.centroid_3d_after,
                    'velocity_3d': patient_change.velocity_3d,
                    'embedding': embeddings.get(patient_change.entity_id),
                    'class_label': patient_change.entity_id.split('_')[0],
                    'is_hand': 'hand' in patient_change.entity_id.lower(),
                }
                
                # Skip if 3D data not available (fall back to 0.5 default for legacy support)
                if agent_entity['centroid_3d_mm'] is None or patient_entity['centroid_3d_mm'] is None:
                    continue
                
                time_delta = abs(patient_change.timestamp_after - agent_change.timestamp_after)
                
                # Compute 3D CIS
                cis_score, components = self.calculate_cis(
                    agent_entity,
                    patient_entity,
                    time_delta
                )
                
                # Skip if below threshold
                if cis_score < self.cis_threshold:
                    continue
                
                # Create justification string
                justification = (
                    f"T={components.temporal:.2f}, S={components.spatial:.2f}, "
                    f"M={components.motion:.2f}, Se={components.semantic:.2f}, "
                    f"H={components.hand_bonus:.2f}"
                )
                
                # Create CausalLink
                link = CausalLink(
                    agent_id=agent_change.entity_id,
                    patient_id=patient_change.entity_id,
                    agent_change=agent_change,
                    patient_change=patient_change,
                    influence_score=cis_score,
                    features={
                        'temporal': components.temporal,
                        'spatial': components.spatial,
                        'motion': components.motion,
                        'semantic': components.semantic,
                        'hand_bonus': components.hand_bonus,
                    },
                    justification=justification,
                )
                
                causal_links.append(link)
        
        return causal_links
