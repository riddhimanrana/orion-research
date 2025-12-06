"""Causal Influence Score computation with 3D + motion cues."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import CausalConfig
from .types import CausalLink, StateChange

logger = logging.getLogger(__name__)


@dataclass
class CISComponents:
    temporal: float
    spatial: float
    motion: float
    semantic: float
    hand_bonus: float

    def to_features(self) -> Dict[str, float]:
        return {
            "temporal": self.temporal,
            "spatial": self.spatial,
            "motion": self.motion,
            "semantic": self.semantic,
            "hand_bonus": self.hand_bonus,
        }


class CausalInfluenceScorer3D:
    """Compute pairwise causal influence using spatial, motion, and semantic cues."""

    def __init__(self, config: Optional[CausalConfig] = None):
        self.config = config or CausalConfig()
        self.type_compatibility = self._build_type_compatibility()

    def compute_causal_links(
        self,
        state_changes: List[StateChange],
        embeddings: Dict[str, np.ndarray],
        scene_context: Optional[Dict[str, Any]] = None,
    ) -> List[CausalLink]:
        if len(state_changes) < 2:
            return []

        sorted_changes = sorted(state_changes, key=lambda change: change.timestamp_after)
        links: List[CausalLink] = []

        for idx, agent_change in enumerate(sorted_changes):
            for patient_change in sorted_changes[idx + 1 :]:
                if agent_change.entity_id == patient_change.entity_id:
                    continue

                agent_entity = self._state_change_to_entity(agent_change, embeddings)
                patient_entity = self._state_change_to_entity(patient_change, embeddings)

                time_delta = abs(patient_change.timestamp_after - agent_change.timestamp_after)
                cis_score, components = self.calculate_cis(
                    agent_entity,
                    patient_entity,
                    time_delta,
                    scene_context=scene_context,
                )

                if cis_score < self.config.cis_threshold:
                    continue

                justification = (
                    f"T={components.temporal:.2f}, S={components.spatial:.2f}, "
                    f"M={components.motion:.2f}, Se={components.semantic:.2f}, "
                    f"H={components.hand_bonus:.2f}"
                )
                link = CausalLink(
                    agent_id=agent_change.entity_id,
                    patient_id=patient_change.entity_id,
                    agent_change=agent_change,
                    patient_change=patient_change,
                    influence_score=cis_score,
                    justification=justification,
                    features=components.to_features(),
                )
                links.append(link)

        logger.info("Computed %d CIS links", len(links))
        return links

    def calculate_cis(
        self,
        agent_entity: Dict[str, Any],
        patient_entity: Dict[str, Any],
        time_delta: float,
        scene_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, CISComponents]:
        temporal = self._temporal_score(time_delta)
        spatial, _ = self._spatial_score(agent_entity, patient_entity)
        motion = self._motion_score(agent_entity, patient_entity)
        semantic = self._semantic_score(agent_entity, patient_entity)
        hand_bonus = self._hand_bonus(agent_entity, patient_entity)

        cis = (
            self.config.weight_temporal * temporal
            + self.config.weight_spatial * spatial
            + self.config.weight_motion * motion
            + self.config.weight_semantic * semantic
            + hand_bonus
        )
        cis = max(0.0, min(1.0, cis))

        if scene_context and self.config.enable_scene_boosts:
            cis = self._apply_scene_context(cis, agent_entity, patient_entity, scene_context)

        return cis, CISComponents(temporal, spatial, motion, semantic, hand_bonus)

    # ------------------------------------------------------------------
    # Component scorers
    # ------------------------------------------------------------------

    def _temporal_score(self, time_delta: float) -> float:
        return float(math.exp(-abs(time_delta) / max(1e-6, self.config.temporal_decay_tau)))

    def _spatial_score(self, agent: Dict[str, Any], patient: Dict[str, Any]) -> Tuple[float, Optional[float]]:
        agent_pos_3d = agent.get("centroid_3d_mm")
        patient_pos_3d = patient.get("centroid_3d_mm")
        if agent_pos_3d is not None and patient_pos_3d is not None:
            dist = float(np.linalg.norm(np.array(agent_pos_3d) - np.array(patient_pos_3d)))
            score = max(0.0, 1.0 - (dist / self.config.max_spatial_distance_mm) ** 2)
            if dist < 10.0:
                score = min(1.0, score * 1.2)
            return score, dist

        agent_pos = agent.get("centroid_2d_px")
        patient_pos = patient.get("centroid_2d_px")
        if agent_pos is not None and patient_pos is not None:
            dist = float(math.dist(agent_pos, patient_pos))
            norm = max(0.0, 1.0 - min(dist / self.config.max_spatial_distance_px, 1.0))
            return norm**2, dist

        return 0.5, None

    def _motion_score(self, agent: Dict[str, Any], patient: Dict[str, Any]) -> float:
        velocity = agent.get("velocity_3d") or agent.get("velocity_2d")
        patient_velocity = patient.get("velocity_3d") or patient.get("velocity_2d")
        if velocity is None or patient_velocity is None:
            return 0.4

        vel_a = np.array(velocity, dtype=np.float32)
        vel_b = np.array(patient_velocity, dtype=np.float32)
        if np.linalg.norm(vel_a) < 1e-6 or np.linalg.norm(vel_b) < 1e-6:
            return 0.3

        vel_a_norm = vel_a / np.linalg.norm(vel_a)
        vel_b_norm = vel_b / np.linalg.norm(vel_b)
        cos_angle = float(np.clip(np.dot(vel_a_norm, vel_b_norm), -1.0, 1.0))
        alignment = (cos_angle + 1.0) / 2.0
        if math.acos(max(-1.0, min(1.0, cos_angle))) > self.config.motion_alignment_threshold:
            alignment *= 0.5
        speed_norm = min(1.0, float(np.linalg.norm(vel_a)) / 1000.0)
        return min(1.0, 0.6 * alignment + 0.4 * speed_norm)

    def _semantic_score(self, agent: Dict[str, Any], patient: Dict[str, Any]) -> float:
        emb_a = agent.get("embedding")
        emb_b = patient.get("embedding")
        if emb_a is None or emb_b is None:
            return 0.5
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.5
        cosine = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
        proximity = (cosine + 1.0) / 2.0
        type_bonus = self._check_type_compatibility(agent.get("class_label"), patient.get("class_label"))
        return float(0.6 * proximity + 0.4 * type_bonus)

    def _hand_bonus(self, agent: Dict[str, Any], patient: Dict[str, Any]) -> float:
        if not agent.get("is_hand"):
            return 0.0
        state = patient.get("hand_interaction_type")
        if state == "GRASPING":
            return self.config.hand_grasping_bonus
        if state == "TOUCHING":
            return self.config.hand_touching_bonus
        if state == "NEAR":
            return self.config.hand_near_bonus
        return 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _state_change_to_entity(self, change: StateChange, embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        return {
            "centroid_3d_mm": change.centroid_3d_after,
            "centroid_2d_px": change.centroid_after,
            "velocity_3d": change.velocity_3d,
            "velocity_2d": change.velocity_2d,
            "embedding": embeddings.get(change.entity_id),
            "class_label": change.class_label,
            "is_hand": "hand" in change.class_label.lower(),
            "hand_interaction_type": change.metadata.get("hand_interaction_type"),
        }

    def _apply_scene_context(
        self,
        cis: float,
        agent: Dict[str, Any],
        patient: Dict[str, Any],
        context: Dict[str, Any],
    ) -> float:
        scene_type = context.get("scene_type")
        if not scene_type:
            return cis

        boost_map = {
            "kitchen": {("hand", "knife"): 1.3, ("hand", "cup"): 1.2},
            "office": {("hand", "keyboard"): 1.2, ("hand", "mouse"): 1.2},
            "living_room": {("hand", "remote"): 1.2},
        }
        agent_class = agent.get("class_label", "unknown")
        patient_class = patient.get("class_label", "unknown")
        multiplier = boost_map.get(scene_type, {}).get((agent_class, patient_class))
        if multiplier is None:
            return cis
        return float(min(1.0, cis * multiplier))

    def _check_type_compatibility(self, class_a: Optional[str], class_b: Optional[str]) -> float:
        if not class_a or not class_b:
            return 0.5
        pair = tuple(sorted((class_a.lower(), class_b.lower())))
        return self.type_compatibility.get(pair, 0.5)

    def _build_type_compatibility(self) -> Dict[Tuple[str, str], float]:
        pairs = {
            ("cup", "hand"): 1.0,
            ("door", "hand"): 1.0,
            ("keyboard", "hand"): 0.9,
            ("mouse", "hand"): 0.9,
            ("phone", "hand"): 0.9,
            ("keyboard", "mouse"): 0.7,
            ("remote", "hand"): 0.9,
            ("book", "hand"): 0.8,
        }
        # Ensure both orderings are covered
        mirrored = {tuple(sorted(k)): v for k, v in pairs.items()}
        return mirrored


__all__ = ["CausalInfluenceScorer3D", "CISComponents"]
