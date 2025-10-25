"""
Causal Influence Scorer (CIS)
===============================

Computes Causal Influence Scores for event relationships.

Responsibilities:
- Calculate CIS scores using temporal, spatial, semantic proximity
- Apply weighted scoring formula
- Identify strong causal relationships
- Support hyperparameter optimization

CIS Formula:
CIS(A→B) = w₁×P_temporal + w₂×P_spatial + w₃×P_motion + w₄×P_semantic

Author: Orion Research Team
Date: October 2025
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from orion.semantic.config import CausalConfig
from orion.semantic.types import CausalLink, StateChange

logger = logging.getLogger(__name__)


class CausalInfluenceScorer:
    """Compute weighted Causal Influence Scores (CIS) between state changes."""

    def __init__(self, config: CausalConfig):
        self.config = config
        try:
            weight_info = self.config.to_dict()
        except AttributeError:
            weight_info = {
                "temporal": self.config.weight_temporal,
                "spatial": self.config.weight_spatial,
                "motion": self.config.weight_motion,
                "semantic": self.config.weight_semantic,
                "threshold": self.config.cis_threshold,
            }
        logger.debug(
            "CausalInfluenceScorer initialized with weights T=%.3f S=%.3f M=%.3f Se=%.3f, threshold=%.3f",
            weight_info.get("temporal", self.config.weight_temporal),
            weight_info.get("spatial", self.config.weight_spatial),
            weight_info.get("motion", self.config.weight_motion),
            weight_info.get("semantic", self.config.weight_semantic),
            weight_info.get("threshold", self.config.cis_threshold),
        )

    def compute_causal_links(
        self,
        state_changes: List[StateChange],
        embeddings: Dict[str, np.ndarray],
    ) -> List[CausalLink]:
        logger.info("=" * 80)
        logger.info("PHASE 2E: CAUSAL REASONING (CIS)")
        logger.info("=" * 80)

        if len(state_changes) < 2:
            logger.warning("Need at least 2 state changes for causal reasoning")
            return []

        logger.info(
            "Computing causal links for %d state changes (T=%.3f, S=%.3f, M=%.3f, Se=%.3f, Th=%.3f)",
            len(state_changes),
            self.config.weight_temporal,
            self.config.weight_spatial,
            self.config.weight_motion,
            self.config.weight_semantic,
            self.config.cis_threshold,
        )

        causal_links: List[CausalLink] = []
        component_log: Dict[str, List[float]] = {"temporal": [], "spatial": [], "motion": [], "semantic": []}
        time_deltas: List[float] = []
        distances: List[float] = []

        sorted_changes = sorted(state_changes, key=lambda change: change.timestamp_after)

        for idx, agent_change in enumerate(sorted_changes):
            for patient_change in sorted_changes[idx + 1 :]:
                cis_score, details = self._compute_cis(agent_change, patient_change, embeddings)
                if cis_score < self.config.cis_threshold:
                    continue

                time_deltas.append(details["time_delta"])
                if details["distance_available"]:
                    distances.append(details["distance"])

                for key in component_log:
                    component_log[key].append(details[key])

                justification = (
                    f"T={details['temporal']:.2f}, S={details['spatial']:.2f}, "
                    f"M={details['motion']:.2f}, Se={details['semantic']:.2f}"
                )

                link = CausalLink(
                    agent_id=agent_change.entity_id,
                    patient_id=patient_change.entity_id,
                    agent_change=agent_change,
                    patient_change=patient_change,
                    influence_score=cis_score,
                    features={
                        "temporal": details["temporal"],
                        "spatial": details["spatial"],
                        "motion": details["motion"],
                        "semantic": details["semantic"],
                        "time_delta": details["time_delta"],
                        "distance": details["distance"],
                        "distance_available": details["distance_available"],
                        "alignment": details["alignment"],
                    },
                    justification=justification,
                )
                causal_links.append(link)

        logger.info("\n✓ Identified %d causal links", len(causal_links))

        if causal_links:
            avg_score = float(np.mean([link.influence_score for link in causal_links]))
            max_score = max(link.influence_score for link in causal_links)
            strong_links = sum(1 for link in causal_links if link.influence_score > 0.7)
            logger.info("  Average CIS: %.3f", avg_score)
            logger.info("  Maximum CIS: %.3f", max_score)
            logger.info("  Strong links (>0.7): %d", strong_links)

            logger.info(
                "  Component means: T=%.3f S=%.3f M=%.3f Se=%.3f",
                float(np.mean(component_log["temporal"]) if component_log["temporal"] else 0.0),
                float(np.mean(component_log["spatial"]) if component_log["spatial"] else 0.0),
                float(np.mean(component_log["motion"]) if component_log["motion"] else 0.0),
                float(np.mean(component_log["semantic"]) if component_log["semantic"] else 0.0),
            )
            if distances:
                logger.info("  Median spatial distance (px): %.1f", float(np.median(distances)))
            if time_deltas:
                logger.info("  Median temporal delta (s): %.2f", float(np.median(time_deltas)))

        logger.info("=" * 80 + "\n")
        return causal_links

    def _compute_cis(
        self,
        change_a: StateChange,
        change_b: StateChange,
        embeddings: Dict[str, np.ndarray],
    ) -> Tuple[float, Dict[str, float]]:
        time_delta = float(change_b.timestamp_after - change_a.timestamp_after)
        temporal = self._temporal_proximity(change_a, change_b, time_delta)
        spatial, distance = self._spatial_proximity(change_a, change_b)
        motion, alignment = self._motion_alignment(change_a, change_b, time_delta)
        semantic = self._semantic_proximity(change_a.entity_id, change_b.entity_id, embeddings)

        cis_score = (
            self.config.weight_temporal * temporal
            + self.config.weight_spatial * spatial
            + self.config.weight_motion * motion
            + self.config.weight_semantic * semantic
        )

        distance_value = float(distance) if distance is not None else 0.0
        distance_flag = 1.0 if distance is not None else 0.0

        return self._clamp(cis_score), {
            "temporal": temporal,
            "spatial": spatial,
            "motion": motion,
            "semantic": semantic,
            "time_delta": time_delta,
            "distance": distance_value,
            "distance_available": distance_flag,
            "alignment": alignment,
        }

    def _temporal_proximity(
        self,
        change_a: StateChange,
        change_b: StateChange,
        time_delta: float,
    ) -> float:
        if time_delta <= 0:
            return 0.0

        decay = math.exp(-time_delta / self.config.temporal_decay_seconds)
        magnitude = (change_a.change_magnitude + change_b.change_magnitude) * 0.5
        magnitude = max(0.0, min(1.0, magnitude))
        weighted = decay * (0.5 + 0.5 * magnitude)
        return self._clamp(weighted)

    def _spatial_proximity(
        self,
        change_a: StateChange,
        change_b: StateChange,
    ) -> Tuple[float, Optional[float]]:
        centroid_a = self._get_change_centroid(change_a, prefer_after=True)
        centroid_b = self._get_change_centroid(change_b, prefer_after=True)

        if centroid_a is None or centroid_b is None:
            return 0.5, None

        dx = centroid_a[0] - centroid_b[0]
        dy = centroid_a[1] - centroid_b[1]
        distance = math.hypot(dx, dy)

        normalized = 1.0 - min(distance / self.config.max_spatial_distance, 1.0)
        normalized = max(0.0, normalized)
        normalized = normalized ** 2

        if change_a.location_after and change_b.location_after:
            if change_a.location_after == change_b.location_after:
                normalized = self._clamp(normalized + 0.2)
            elif change_a.location_after.split(":")[0] == change_b.location_after.split(":")[0]:
                normalized = self._clamp(normalized + 0.1)

        if change_a.scene_after and change_b.scene_after and change_a.scene_after == change_b.scene_after:
            normalized = self._clamp(normalized + 0.05)

        return normalized, float(distance)

    def _motion_alignment(
        self,
        change_a: StateChange,
        change_b: StateChange,
        time_delta: float,
    ) -> Tuple[float, float]:
        if time_delta <= 0:
            return 0.0, 0.0

        before = self._get_change_centroid(change_a, prefer_after=False)
        after = self._get_change_centroid(change_a, prefer_after=True)
        patient_center = self._get_change_centroid(change_b, prefer_after=True)

        if before is None or after is None or patient_center is None:
            return 0.25, 0.0

        motion_vec = (after[0] - before[0], after[1] - before[1])
        motion_norm = math.hypot(*motion_vec)

        to_patient = (patient_center[0] - after[0], patient_center[1] - after[1])
        to_patient_norm = math.hypot(*to_patient)

        if motion_norm == 0.0 or to_patient_norm == 0.0:
            return 0.25, 0.0

        alignment = (motion_vec[0] * to_patient[0] + motion_vec[1] * to_patient[1]) / (motion_norm * to_patient_norm)
        alignment = max(0.0, min(1.0, alignment))

        # Penalize wide angles beyond the configured threshold
        try:
            angle = math.acos(max(-1.0, min(1.0, alignment)))
        except ValueError:
            angle = math.pi / 2
        if angle > self.config.motion_alignment_threshold:
            alignment *= 0.5

        speed = max(change_a.velocity, change_a.displacement)
        if speed <= 0 and motion_norm > 0 and time_delta > 0:
            speed = motion_norm / max(time_delta, 1e-6)

        speed_norm = min(1.0, speed / self.config.motion_max_speed)
        if speed < self.config.motion_min_speed:
            speed_norm *= 0.3

        score = (0.6 * alignment) + (0.4 * speed_norm)
        return self._clamp(score), alignment

    def _semantic_proximity(
        self,
        entity_a: str,
        entity_b: str,
        embeddings: Dict[str, np.ndarray],
    ) -> float:
        emb_a = embeddings.get(entity_a)
        emb_b = embeddings.get(entity_b)
        if emb_a is None or emb_b is None:
            return 0.5

        norm_a = float(np.linalg.norm(emb_a))
        norm_b = float(np.linalg.norm(emb_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.5

        similarity = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
        similarity = max(-1.0, min(1.0, similarity))
        proximity = (similarity + 1.0) / 2.0
        if proximity < self.config.semantic_min_similarity:
            proximity *= 0.5
        return self._clamp(proximity)

    def _get_change_centroid(
        self,
        change: StateChange,
        *,
        prefer_after: bool,
    ) -> Optional[Tuple[float, float]]:
        if prefer_after:
            if change.centroid_after is not None:
                return float(change.centroid_after[0]), float(change.centroid_after[1])
            if change.bounding_box_after:
                x1, y1, x2, y2 = change.bounding_box_after
                return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)
        else:
            if change.centroid_before is not None:
                return float(change.centroid_before[0]), float(change.centroid_before[1])
            if change.bounding_box_before:
                x1, y1, x2, y2 = change.bounding_box_before
                return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)
        # Fall back to opposite temporal side if requested data missing
        if prefer_after:
            if change.centroid_before is not None:
                return float(change.centroid_before[0]), float(change.centroid_before[1])
            if change.bounding_box_before:
                x1, y1, x2, y2 = change.bounding_box_before
                return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)
        else:
            if change.centroid_after is not None:
                return float(change.centroid_after[0]), float(change.centroid_after[1])
            if change.bounding_box_after:
                x1, y1, x2, y2 = change.bounding_box_after
                return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)
        return None

    @staticmethod
    def _clamp(value: float) -> float:
        if math.isnan(value):
            return 0.0
        return max(0.0, min(1.0, float(value)))
