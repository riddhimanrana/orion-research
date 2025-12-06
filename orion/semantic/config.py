"""Semantic configuration dataclasses for causal reasoning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class StateChangeConfig:
    """Heuristics for deriving state changes from perception outputs."""

    embedding_similarity_threshold: float = 0.80
    min_duration_frames: int = 2
    min_velocity_pixels: float = 4.0


@dataclass
class CausalConfig:
    """Tunable weights for Causal Influence Score computation."""

    weight_temporal: float = 0.30
    weight_spatial: float = 0.44
    weight_motion: float = 0.21
    weight_semantic: float = 0.05
    temporal_decay_tau: float = 4.0
    max_spatial_distance_mm: float = 600.0
    max_spatial_distance_px: float = 250.0
    motion_alignment_threshold: float = math.pi / 3
    hand_grasping_bonus: float = 0.30
    hand_touching_bonus: float = 0.15
    hand_near_bonus: float = 0.05
    cis_threshold: float = 0.50
    enable_scene_boosts: bool = True

    def to_dict(self) -> dict:
        return {
            "weight_temporal": self.weight_temporal,
            "weight_spatial": self.weight_spatial,
            "weight_motion": self.weight_motion,
            "weight_semantic": self.weight_semantic,
            "temporal_decay_tau": self.temporal_decay_tau,
            "max_spatial_distance_mm": self.max_spatial_distance_mm,
            "max_spatial_distance_px": self.max_spatial_distance_px,
            "motion_alignment_threshold": self.motion_alignment_threshold,
            "hand_grasping_bonus": self.hand_grasping_bonus,
            "hand_touching_bonus": self.hand_touching_bonus,
            "hand_near_bonus": self.hand_near_bonus,
            "cis_threshold": self.cis_threshold,
            "enable_scene_boosts": self.enable_scene_boosts,
        }


@dataclass
class SemanticConfig:
    """Grouping of semantic sub-configs used beyond perception."""

    state_change: StateChangeConfig = field(default_factory=StateChangeConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)


def get_fast_semantic_config() -> SemanticConfig:
    """Return a semantic config tuned for quick experimentation."""
    return SemanticConfig(
        causal=CausalConfig(
            weight_temporal=0.25,
            weight_spatial=0.50,
            weight_motion=0.20,
            weight_semantic=0.05,
            cis_threshold=0.45,
        )
    )


__all__ = [
    "StateChangeConfig",
    "CausalConfig",
    "SemanticConfig",
    "get_fast_semantic_config",
]
