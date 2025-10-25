"""
Semantic Uplift Engine Configuration
====================================

Validated configuration for Phase 2 (Semantic Uplift & Event Composition).

Manages state change detection, temporal windowing, event composition, and causal inference.
All fields are validated with meaningful error messages.

Author: Orion Research Team
Date: October 2025
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class StateChangeConfig:
    """State change detection configuration"""
    
    # Detection threshold
    embedding_similarity_threshold: float = 0.85
    """Cosine similarity threshold for detecting state changes (0-1)"""
    
    # Embedding model
    embedding_model: Literal["clip", "sentence-transformer"] = "clip"
    """Model used for state comparison embeddings"""
    
    # Temporal constraints
    min_time_between_changes: float = 0.5
    """Minimum seconds between consecutive state changes"""
    
    def __post_init__(self):
        """Validate state change config"""
        if not (0 <= self.embedding_similarity_threshold <= 1):
            raise ValueError(
                f"embedding_similarity_threshold must be in [0, 1], "
                f"got {self.embedding_similarity_threshold}"
            )
        
        valid_models = {"clip", "sentence-transformer"}
        if self.embedding_model not in valid_models:
            raise ValueError(
                f"embedding_model must be one of {valid_models}, got {self.embedding_model}"
            )
        
        if self.min_time_between_changes < 0:
            raise ValueError(
                f"min_time_between_changes must be >= 0, got {self.min_time_between_changes}"
            )
        
        logger.debug(
            f"StateChangeConfig validated: threshold={self.embedding_similarity_threshold}, "
            f"model={self.embedding_model}"
        )


@dataclass
class TemporalWindowConfig:
    """Temporal windowing for state change grouping"""
    
    # Window sizing
    max_duration_seconds: float = 5.0
    """Maximum duration of a temporal window"""
    
    max_gap_between_changes: float = 1.5
    """Maximum gap between changes within a window"""
    
    max_changes_per_window: int = 20
    """Maximum state changes grouped into one window"""
    
    # Significance thresholds
    min_confidence_for_significance: float = 0.6
    """Minimum average confidence for window to be significant"""
    
    def __post_init__(self):
        """Validate temporal window config"""
        if self.max_duration_seconds <= 0:
            raise ValueError(
                f"max_duration_seconds must be > 0, got {self.max_duration_seconds}"
            )
        
        if self.max_gap_between_changes <= 0:
            raise ValueError(
                f"max_gap_between_changes must be > 0, got {self.max_gap_between_changes}"
            )
        
        if self.max_changes_per_window < 1:
            raise ValueError(
                f"max_changes_per_window must be >= 1, got {self.max_changes_per_window}"
            )
        
        if not (0 <= self.min_confidence_for_significance <= 1):
            raise ValueError(
                f"min_confidence_for_significance must be in [0, 1], "
                f"got {self.min_confidence_for_significance}"
            )
        
        logger.debug(
            f"TemporalWindowConfig validated: max_duration={self.max_duration_seconds}s, "
            f"max_gap={self.max_gap_between_changes}s"
        )


@dataclass
class EventCompositionConfig:
    """Event composition and LLM generation configuration"""
    
    # LLM settings
    model: str = "gemma3:4b"
    """LLM model name (Ollama)"""
    
    temperature: float = 0.3
    """LLM sampling temperature"""
    
    max_tokens: int = 200
    """Maximum tokens per event description"""
    
    timeout_seconds: float = 30.0
    """LLM request timeout"""
    
    # Event filtering
    min_participants: int = 1
    """Minimum entities required for an event"""
    
    min_duration_seconds: float = 0.1
    """Minimum event duration"""
    
    def __post_init__(self):
        """Validate event composition config"""
        if not self.model:
            raise ValueError("model cannot be empty")
        
        if not (0 <= self.temperature <= 2):
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")
        
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds}")
        
        if self.min_participants < 1:
            raise ValueError(f"min_participants must be >= 1, got {self.min_participants}")
        
        if self.min_duration_seconds < 0:
            raise ValueError(f"min_duration_seconds must be >= 0, got {self.min_duration_seconds}")
        
        logger.debug(
            f"EventCompositionConfig validated: model={self.model}, "
            f"max_tokens={self.max_tokens}"
        )


@dataclass
class CausalConfig:
    """Causal Influence Score (CIS) configuration"""
    
    # CIS weights (must sum to 1.0)
    weight_temporal: float = 0.35
    """Weight for temporal proximity"""
    
    weight_spatial: float = 0.25
    """Weight for spatial proximity"""
    
    weight_motion: float = 0.20
    """Weight for motion correlation"""
    
    weight_semantic: float = 0.20
    """Weight for semantic correlation"""
    
    # Decision threshold
    cis_threshold: float = 0.55
    """CIS threshold for considering causality"""
    
    # Distance/decay parameters
    max_spatial_distance: float = 600.0
    """Maximum pixel distance considered causal before decaying to zero"""
    
    temporal_decay_seconds: float = 4.0
    """Temporal decay horizon in seconds"""
    
    # Motion parameters
    motion_min_speed: float = 5.0
    """Minimum speed (pixels/sec) to consider motion meaningful"""
    
    motion_max_speed: float = 200.0
    """Speed at which motion contribution saturates"""
    
    motion_alignment_threshold: float = math.pi / 4
    """Maximum angle (radians) treated as "towards" the patient"""
    
    # Semantic similarity parameters
    semantic_min_similarity: float = 0.3
    """Minimum cosine similarity mapped to non-zero semantic score"""
    
    # Learning (Bayesian Optimization)
    enable_learning: bool = True
    """Enable weight optimization via Bayesian optimization"""
    
    learning_rate: float = 0.05
    """Step size for weight updates during learning"""
    
    auto_load_hpo: bool = True
    """Automatically load latest HPO weights when available"""
    
    hpo_result_path: Optional[str] = None
    """Optional explicit path to HPO JSON result"""
    
    def __post_init__(self):
        """Validate causal config and optionally load HPO-tuned weights"""
        self._validate_weights()
        self._validate_thresholds()
        if self.auto_load_hpo:
            self._try_load_hpo_weights()
        logger.debug(
            "CausalConfig ready: weights=[T:%.3f, S:%.3f, M:%.3f, Se:%.3f], "
            "threshold=%.3f",
            self.weight_temporal,
            self.weight_spatial,
            self.weight_motion,
            self.weight_semantic,
            self.cis_threshold,
        )
    
    def _validate_weights(self) -> None:
        weights = [self.weight_temporal, self.weight_spatial, self.weight_motion, self.weight_semantic]
        for name, weight in zip(["temporal", "spatial", "motion", "semantic"], weights):
            if not (0 <= weight <= 1):
                raise ValueError(f"weight_{name} must be in [0, 1], got {weight}")
        total_weight = sum(weights)
        if abs(total_weight - 1.0) > 0.01 and total_weight > 0:
            logger.warning(
                "CIS weights sum to %.2f, expected 1.0. Normalizing in-memory weights.",
                total_weight,
            )
            scale = 1.0 / total_weight
            self.weight_temporal *= scale
            self.weight_spatial *= scale
            self.weight_motion *= scale
            self.weight_semantic *= scale
    
    def _validate_thresholds(self) -> None:
        if not (0 <= self.cis_threshold <= 1):
            raise ValueError(f"cis_threshold must be in [0, 1], got {self.cis_threshold}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.learning_rate > 1.0:
            logger.warning("Very large learning_rate: %.3f. May diverge.", self.learning_rate)
        if self.temporal_decay_seconds <= 0:
            raise ValueError(
                f"temporal_decay_seconds must be > 0, got {self.temporal_decay_seconds}"
            )
        if self.max_spatial_distance <= 0:
            raise ValueError(
                f"max_spatial_distance must be > 0, got {self.max_spatial_distance}"
            )
        if self.motion_max_speed <= 0:
            raise ValueError(
                f"motion_max_speed must be > 0, got {self.motion_max_speed}"
            )
        if self.motion_min_speed < 0:
            raise ValueError(
                f"motion_min_speed must be >= 0, got {self.motion_min_speed}"
            )
        if not (0 <= self.semantic_min_similarity <= 1):
            raise ValueError(
                f"semantic_min_similarity must be in [0, 1], got {self.semantic_min_similarity}"
            )
    
    def _try_load_hpo_weights(self) -> None:
        candidate_paths = self._candidate_hpo_paths()
        for path in candidate_paths:
            if not path:
                continue
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                weights = payload.get("best_weights")
                threshold = payload.get("best_threshold")
                if not weights:
                    continue
                self.weight_temporal = float(weights.get("temporal", self.weight_temporal))
                self.weight_spatial = float(weights.get("spatial", self.weight_spatial))
                self.weight_motion = float(weights.get("motion", self.weight_motion))
                self.weight_semantic = float(weights.get("semantic", self.weight_semantic))
                if threshold is not None:
                    self.cis_threshold = float(threshold)
                self._validate_weights()
                logger.info(
                    "Loaded HPO CIS weights from %s (T=%.3f, S=%.3f, M=%.3f, Se=%.3f, Th=%.3f)",
                    path,
                    self.weight_temporal,
                    self.weight_spatial,
                    self.weight_motion,
                    self.weight_semantic,
                    self.cis_threshold,
                )
                return
            except Exception as exc:
                logger.debug("Failed to load HPO weights from %s: %s", path, exc)
        logger.debug("No HPO CIS weights found; using default configuration")
    
    def _candidate_hpo_paths(self) -> List[str]:
        paths: List[str] = []
        if self.hpo_result_path:
            paths.append(self.hpo_result_path)
        workspace_root = os.getcwd()
        default_path = os.path.join(workspace_root, "hpo_results", "optimization_latest.json")
        module_path = os.path.join(os.path.dirname(__file__), "..", "hpo_results", "optimization_latest.json")
        env_path = os.getenv("ORION_CIS_HPO_PATH")
        for candidate in (default_path, module_path, env_path):
            if candidate and candidate not in paths:
                paths.append(candidate)
        return paths
    
    def to_dict(self) -> Dict[str, float]:
        """Serialize weight configuration for logging or dashboards."""
        return {
            "temporal": self.weight_temporal,
            "spatial": self.weight_spatial,
            "motion": self.weight_motion,
            "semantic": self.weight_semantic,
            "threshold": self.cis_threshold,
        }


@dataclass
class SpatialConfig:
    """Spatial zone detection configuration (Phase 2)"""
    
    # HDBSCAN parameters
    min_cluster_size: int = 3
    """Minimum entities required to form a zone"""
    
    min_samples: int = 2
    """Minimum samples for HDBSCAN core point"""
    
    # Feature weights (must sum to 1.0)
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        'position': 0.6,
        'temporal': 0.2,
        'size': 0.2,
    })
    """Weights for spatial features: position, temporal, size"""
    
    # Zone labeling
    enable_zone_labeling: bool = True
    """Enable semantic zone labeling (desk_area, bedroom_area, etc.)"""
    
    # Adjacency detection
    adjacency_threshold: float = 0.15
    """Normalized distance threshold for adjacent zones"""
    
    def __post_init__(self):
        """Validate spatial config"""
        if self.min_cluster_size < 2:
            raise ValueError(f"min_cluster_size must be >= 2, got {self.min_cluster_size}")
        
        if self.min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {self.min_samples}")
        
        # Validate feature weights
        if self.feature_weights:
            total = sum(self.feature_weights.values())
            if abs(total - 1.0) > 0.01:
                logger.warning(
                    f"Spatial feature weights sum to {total:.2f}, expected 1.0. Normalizing..."
                )
                if total > 0:
                    scale = 1.0 / total
                    for key in self.feature_weights:
                        self.feature_weights[key] *= scale
        
        if not (0 <= self.adjacency_threshold <= 1):
            raise ValueError(
                f"adjacency_threshold must be in [0, 1], got {self.adjacency_threshold}"
            )
        
        logger.debug(
            f"SpatialConfig validated: min_cluster_size={self.min_cluster_size}, "
            f"feature_weights={self.feature_weights}"
        )


@dataclass
class SemanticConfig:
    """Complete semantic uplift configuration"""
    
    state_change: StateChangeConfig = field(default_factory=StateChangeConfig)
    temporal_window: TemporalWindowConfig = field(default_factory=TemporalWindowConfig)
    event_composition: EventCompositionConfig = field(default_factory=EventCompositionConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)  # NEW - Phase 2
    
    # General settings
    enable_graph_ingestion: bool = True
    """Ingest results into Neo4j knowledge graph"""
    
    verbose: bool = False
    """Enable verbose logging for all semantic components"""
    
    def __post_init__(self):
        """Validate semantic config"""
        logger.debug(
            f"SemanticConfig validated: graph_ingestion={self.enable_graph_ingestion}, "
            f"verbose={self.verbose}"
        )


# Preset configurations
def get_fast_semantic_config() -> SemanticConfig:
    """
    Fast mode: quick event generation (fewer state changes grouped)
    
    Use for real-time or demo scenarios
    """
    return SemanticConfig(
        state_change=StateChangeConfig(
            embedding_similarity_threshold=0.80,
            min_time_between_changes=0.3,
        ),
        temporal_window=TemporalWindowConfig(
            max_duration_seconds=3.0,
            max_gap_between_changes=1.0,
            max_changes_per_window=10,
        ),
        event_composition=EventCompositionConfig(
            temperature=0.5,
            max_tokens=150,
            timeout_seconds=15.0,
        ),
        causal=CausalConfig(
            cis_threshold=0.6,
            learning_rate=0.05,
        ),
        enable_graph_ingestion=False,
    )


def get_balanced_semantic_config() -> SemanticConfig:
    """
    Balanced mode: good event quality with reasonable speed
    
    Recommended for production use
    """
    return SemanticConfig(
        state_change=StateChangeConfig(
            embedding_similarity_threshold=0.85,
            min_time_between_changes=0.5,
        ),
        temporal_window=TemporalWindowConfig(
            max_duration_seconds=5.0,
            max_gap_between_changes=1.5,
            max_changes_per_window=20,
        ),
        event_composition=EventCompositionConfig(
            temperature=0.3,
            max_tokens=200,
            timeout_seconds=30.0,
        ),
        causal=CausalConfig(
            cis_threshold=0.55,
            learning_rate=0.05,
        ),
        enable_graph_ingestion=True,
    )


def get_accurate_semantic_config() -> SemanticConfig:
    """
    Accurate mode: maximum event quality (slowest)
    
    For research and evaluation
    """
    return SemanticConfig(
        state_change=StateChangeConfig(
            embedding_similarity_threshold=0.90,
            min_time_between_changes=1.0,
        ),
        temporal_window=TemporalWindowConfig(
            max_duration_seconds=8.0,
            max_gap_between_changes=2.0,
            max_changes_per_window=30,
            min_confidence_for_significance=0.7,
        ),
        event_composition=EventCompositionConfig(
            temperature=0.2,
            max_tokens=300,
            timeout_seconds=60.0,
        ),
        causal=CausalConfig(
            cis_threshold=0.50,
            learning_rate=0.03,
            enable_learning=True,
        ),
        enable_graph_ingestion=True,
        verbose=True,
    )
