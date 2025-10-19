"""
Part 2: The Semantic Uplift Engine
===================================

This module transforms raw perception logs into a structured knowledge graph by:
1. Tracking entities across time using visual embeddings (object permanence)
2. Detecting state changes through semantic analysis
3. Composing events using LLM reasoning
4. Building a queryable Neo4j knowledge graph

Author: Orion Research Team
Date: October 3, 2025
"""

# mypy: ignore-errors
# pyright: reportGeneralTypeIssues=false, reportOptionalMemberAccess=false, reportArgumentType=false

import hashlib
import json
import logging
import math
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import requests
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

# Suppress warnings
warnings.filterwarnings("ignore")

# Optional imports with graceful fallback
try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    hdbscan = None  # type: ignore[assignment]
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not available. Install with: pip install hdbscan")

try:
    from .embedding_model import create_embedding_model

    EMBEDDING_MODEL_AVAILABLE = True
except ImportError:
    create_embedding_model = None  # type: ignore[assignment]
    EMBEDDING_MODEL_AVAILABLE = False
    print(
        "Warning: embedding_model not available. Ensure embedding_model.py is present."
    )

try:
    from .causal_inference import (
        CausalInferenceEngine,
        CausalConfig,
        AgentCandidate,
        StateChange as CISStateChange,
    )
    CAUSAL_INFERENCE_AVAILABLE = True
except ImportError:
    CausalInferenceEngine = None  # type: ignore[assignment, misc]
    CausalConfig = None  # type: ignore[assignment, misc]
    AgentCandidate = None  # type: ignore[assignment, misc]
    CISStateChange = None  # type: ignore[assignment, misc]
    CAUSAL_INFERENCE_AVAILABLE = False
    print(
        "Warning: causal_inference not available. Using basic causal scoring."
    )

try:
    from .motion_tracker import MotionData
except ImportError:
    from motion_tracker import MotionData  # type: ignore[assignment]


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================


class Config:
    """Configuration parameters for semantic uplift engine"""

    # Entity Tracking (HDBSCAN)
    MIN_CLUSTER_SIZE = 3  # Minimum appearances to be tracked entity
    MIN_SAMPLES = 2
    CLUSTER_METRIC = "euclidean"  # or 'cosine'
    CLUSTER_SELECTION_METHOD = "eom"  # Excess of Mass
    CLUSTER_SELECTION_EPSILON = 0.15

    # State Change Detection
    STATE_CHANGE_THRESHOLD = 0.85  # Cosine similarity threshold
    EMBEDDING_MODEL_TYPE = (
        "embeddinggemma"  # 'embeddinggemma' (Ollama) or 'sentence-transformer'
    )
    SCENE_SIMILARITY_THRESHOLD = 0.82  # Cosine threshold for scene similarity links
    SCENE_SIMILARITY_TOP_K = 3  # Max similar scenes per scene
    SCENE_LOCATION_TOP_OBJECTS = 3  # Number of dominant objects to describe a location

    # Temporal Windowing
    TIME_WINDOW_SIZE = 30.0  # Legacy maximum duration for backward compatibility
    TIME_WINDOW_MAX_DURATION = 18.0  # seconds, hard ceiling per adaptive window
    TIME_WINDOW_MAX_GAP = 4.0  # seconds, gap between consecutive changes before split
    TIME_WINDOW_MAX_CHANGES = 12  # prevent bloated windows
    MIN_EVENTS_PER_WINDOW = 2  # Minimum state changes to trigger event composition

    # Causal Influence Scoring
    CAUSAL_MAX_PIXEL_DISTANCE = 600.0
    CAUSAL_TEMPORAL_DECAY = 4.0  # seconds before influence decays
    CAUSAL_PROXIMITY_WEIGHT = 0.45
    CAUSAL_MOTION_WEIGHT = 0.25
    CAUSAL_TEMPORAL_WEIGHT = 0.2
    CAUSAL_EMBEDDING_WEIGHT = 0.1
    CAUSAL_MIN_SCORE = 0.55
    CAUSAL_TOP_K_PER_WINDOW = 5

    # Event Composition / LLM Guardrails
    EVENT_COMPOSITION_MIN_STATE_CHANGES = 2
    EVENT_COMPOSITION_MAX_WINDOWS = 80
    EVENT_COMPOSITION_SKIP_NO_CAUSAL = True

    # LLM Event Composition (Ollama)
    USE_LLM_COMPOSITION = True  # Enable LLM composition with gemma3:4b
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = (
        "gemma3:4b"  # Use gemma3:4b for better Cypher generation (was gemma3:1b)
    )
    OLLAMA_TEMPERATURE = 0.3  # More deterministic for structured output
    OLLAMA_MAX_TOKENS = 2000
    OLLAMA_TIMEOUT = 60  # seconds

    # Neo4j Configuration
    NEO4J_URI = "neo4j://127.0.0.1:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "orion123"  # Local Neo4j password
    NEO4J_DATABASE = "neo4j"
    MAX_CONNECTION_LIFETIME = 3600
    MAX_CONNECTION_POOL_SIZE = 50
    CONNECTION_TIMEOUT = 30

    # Graph Schema
    VECTOR_DIMENSIONS = 512  # For visual embeddings
    VECTOR_SIMILARITY_FUNCTION = "cosine"

    # Performance
    BATCH_SIZE = 100  # For bulk operations
    LOG_LEVEL = logging.INFO
    PROGRESS_LOGGING = True


# ============================================================================
# LOGGING SETUP
# ============================================================================


def setup_logger(name: str, level: int = Config.LOG_LEVEL) -> logging.Logger:
    """Set up a logger with consistent formatting"""
    logger = logging.getLogger(name)

    suppress_logs = (
        os.getenv("ORION_SUPPRESS_ENGINE_LOGS", "").lower() in {"1", "true", "yes"}
    )

    if suppress_logs:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        logger.setLevel(max(level, logging.WARNING))
        logger.propagate = False
        return logger

    logger.setLevel(level)
    for handler in list(logger.handlers):
        if isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)

    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler) for handler in logger.handlers
    )
    if not has_stream_handler:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


logger = setup_logger("SemanticUplift")


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class Entity:
    """Represents a tracked entity across time"""

    entity_id: str
    object_class: str
    appearances: List[Dict[str, Any]] = field(default_factory=list)
    first_timestamp: float = 0.0
    last_timestamp: float = 0.0
    average_embedding: Optional[np.ndarray] = None
    scene_ids: Set[str] = field(default_factory=set)

    def add_appearance(self, perception_obj: Dict[str, Any]):
        """Add an appearance of this entity"""
        self.appearances.append(perception_obj)

        timestamp = perception_obj["timestamp"]
        if not self.first_timestamp or timestamp < self.first_timestamp:
            self.first_timestamp = timestamp
        if not self.last_timestamp or timestamp > self.last_timestamp:
            self.last_timestamp = timestamp

    def add_scene(self, scene_id: str) -> None:
        """Associate the entity with a scene."""
        if scene_id:
            self.scene_ids.add(scene_id)

    def compute_average_embedding(self):
        """Compute average embedding from all appearances"""
        if not self.appearances:
            return

        embeddings = [np.array(obj["visual_embedding"]) for obj in self.appearances]
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = np.asarray(avg_embedding)
        norm = float(np.linalg.norm(avg_embedding))
        if norm > 0:
            avg_embedding = avg_embedding / norm
        self.average_embedding = avg_embedding

    def get_timeline(self) -> List[Dict[str, Any]]:
        """Get chronological timeline of appearances"""
        return sorted(self.appearances, key=lambda x: x["timestamp"])

    def get_appearance_near(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """Return the appearance closest to a given timestamp."""
        timeline = self.get_timeline()
        if not timeline:
            return None
        return min(timeline, key=lambda app: abs(app.get("timestamp", 0.0) - timestamp))


@dataclass
class StateChange:
    """Represents a detected state change for an entity"""

    entity_id: str
    timestamp_before: float
    timestamp_after: float
    description_before: str
    description_after: str
    similarity_score: float
    change_magnitude: float  # 1 - similarity
    centroid_before: Optional[Tuple[float, float]] = None
    centroid_after: Optional[Tuple[float, float]] = None
    displacement: float = 0.0
    velocity: float = 0.0
    frame_before: Optional[int] = None
    frame_after: Optional[int] = None
    bounding_box_before: Optional[List[int]] = None
    bounding_box_after: Optional[List[int]] = None
    scene_before: Optional[str] = None
    scene_after: Optional[str] = None
    location_before: Optional[str] = None
    location_after: Optional[str] = None
    embedding_before: Optional[List[float]] = None
    embedding_after: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "timestamp_before": self.timestamp_before,
            "timestamp_after": self.timestamp_after,
            "description_before": self.description_before,
            "description_after": self.description_after,
            "similarity_score": self.similarity_score,
            "change_magnitude": self.change_magnitude,
            "centroid_before": self.centroid_before,
            "centroid_after": self.centroid_after,
            "displacement": self.displacement,
            "velocity": self.velocity,
            "frame_before": self.frame_before,
            "frame_after": self.frame_after,
            "bounding_box_before": self.bounding_box_before,
            "bounding_box_after": self.bounding_box_after,
            "scene_before": self.scene_before,
            "scene_after": self.scene_after,
            "location_before": self.location_before,
            "location_after": self.location_after,
            "embedding_before": self.embedding_before,
            "embedding_after": self.embedding_after,
        }


@dataclass
class TemporalWindow:
    """Represents a time window with active entities and state changes"""

    start_time: float
    end_time: float
    active_entities: Set[str] = field(default_factory=set)
    state_changes: List[StateChange] = field(default_factory=list)
    causal_links: List["CausalLink"] = field(default_factory=list)

    def add_state_change(self, change: StateChange):
        """Add a state change to this window"""
        self.state_changes.append(change)
        self.active_entities.add(change.entity_id)

    def is_significant(self) -> bool:
        """Check if window has enough activity to warrant event composition"""
        return len(self.state_changes) >= Config.MIN_EVENTS_PER_WINDOW

    def add_causal_link(self, link: "CausalLink") -> None:
        """Attach a causal influence link to the window."""
        self.causal_links.append(link)

    def top_causal_links(self, limit: int) -> List["CausalLink"]:
        """Return top scoring causal links for the window."""
        if not self.causal_links:
            return []
        return sorted(
            self.causal_links, key=lambda link: link.influence_score, reverse=True
        )[:limit]


@dataclass
class CausalLink:
    """Represents an inferred causal influence between two entities."""

    agent_id: str
    patient_id: str
    agent_change: StateChange
    patient_change: StateChange
    influence_score: float
    features: Dict[str, float]
    justification: str


@dataclass
class SceneSegment:
    """Represents an aggregated view of a video scene or room."""

    scene_id: str
    frame_number: int
    start_timestamp: float
    end_timestamp: float
    description: str
    object_classes: List[str]
    entity_ids: List[str]
    location_id: str
    embedding: Optional[np.ndarray] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end_timestamp - self.start_timestamp)


@dataclass
class LocationProfile:
    """Encapsulates a logical location inferred from dominant scene objects."""

    location_id: str
    signature: str
    label: str
    object_classes: List[str]
    scene_ids: List[str] = field(default_factory=list)


@dataclass
class SceneSimilarity:
    """Represents similarity between two scenes."""

    source_id: str
    target_id: str
    score: float


def _compute_centroid_from_bbox(
    bbox: Optional[Iterable[float]],
) -> Optional[Tuple[float, float]]:
    """Compute centroid from bounding box if available."""
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _euclidean_distance(
    p1: Optional[Tuple[float, float]], p2: Optional[Tuple[float, float]]
) -> Optional[float]:
    """Return Euclidean distance between two points if both exist."""
    if p1 is None or p2 is None:
        return None
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


class SceneAssembler:
    """Builds scene segments, inferred locations, and similarity links."""

    def __init__(self, tracker: "EntityTracker") -> None:
        self.tracker = tracker
        self.embedding_model = None
        self.embedding_dim: Optional[int] = None
        self._init_embedding_model()

    def _init_embedding_model(self) -> None:
        if not EMBEDDING_MODEL_AVAILABLE or create_embedding_model is None:
            logger.warning(
                "Scene embeddings unavailable; install embedding backends for richer links."
            )
            return
        try:
            self.embedding_model = create_embedding_model(prefer_ollama=True)
            self.embedding_dim = self.embedding_model.get_embedding_dimension()
            logger.info(
                "Scene assembler using %s embeddings (dim=%s)",
                self.embedding_model.get_model_info().get("model_name"),
                self.embedding_dim,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialize scene embedding model: %s", exc)
            self.embedding_model = None
            self.embedding_dim = None

    def _encode(self, text: str) -> Optional[np.ndarray]:
        if not text or self.embedding_model is None:
            return None
        try:
            vector = self.embedding_model.encode([text])[0]
            vector = np.asarray(vector, dtype=float)
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
            return vector
        except Exception as exc:  # noqa: BLE001
            logger.warning("Scene embedding generation failed: %s", exc)
            return None

    def build_scenes(
        self,
        perception_log: List[Dict[str, Any]],
    ) -> Tuple[List[SceneSegment], Dict[str, LocationProfile]]:
        """Aggregate perception objects into scene segments and locations."""

        frames: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for obj in perception_log:
            frame_number = obj.get("frame_number")
            if frame_number is None:
                continue
            frames[int(frame_number)].append(obj)

        scenes: List[SceneSegment] = []
        locations: Dict[str, LocationProfile] = {}

        for frame_number in sorted(frames.keys()):
            frame_objects = frames[frame_number]
            if not frame_objects:
                continue

            timestamps = [float(obj.get("timestamp", 0.0)) for obj in frame_objects]
            start_time = min(timestamps) if timestamps else 0.0
            end_time = max(timestamps) if timestamps else start_time

            entity_ids: List[str] = []
            raw_classes: List[str] = []
            description_parts: List[str] = []

            for obj in frame_objects:
                entity_id = obj.get("entity_id")
                if entity_id:
                    entity_ids.append(entity_id)
                obj_class = obj.get("object_class", "object")
                raw_classes.append(obj_class)
                desc = obj.get("rich_description")
                if desc:
                    description_parts.append(desc)
                else:
                    description_parts.append(
                        f"Observed {obj_class} in frame {frame_number}."
                    )

            description = (
                " ".join(description_parts).strip()
                or f"Scene captured at frame {frame_number}."
            )
            embedding = self._encode(description)
            scene_id = f"scene_{frame_number:06d}"

            for obj in frame_objects:
                obj["scene_id"] = scene_id

            unique_entities = sorted(set(entity_ids))
            class_counts = Counter(raw_classes)
            dominant_classes = [
                cls
                for cls, _ in class_counts.most_common(
                    Config.SCENE_LOCATION_TOP_OBJECTS
                )
            ]
            signature = (
                "|".join(dominant_classes) if dominant_classes else "unspecified"
            )
            label = (
                ", ".join(dominant_classes)
                if dominant_classes
                else "Uncategorized Space"
            )
            signature_hash = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
            location_id = f"location_{signature_hash}"

            if location_id not in locations:
                locations[location_id] = LocationProfile(
                    location_id=location_id,
                    signature=signature,
                    label=label,
                    object_classes=dominant_classes,
                )
            locations[location_id].scene_ids.append(scene_id)

            for obj in frame_objects:
                obj["location_id"] = location_id

            scene_segment = SceneSegment(
                scene_id=scene_id,
                frame_number=frame_number,
                start_timestamp=start_time,
                end_timestamp=end_time,
                description=description,
                object_classes=dominant_classes,
                entity_ids=unique_entities,
                location_id=location_id,
                embedding=embedding,
            )
            scenes.append(scene_segment)

            for entity_id in unique_entities:
                entity = self.tracker.get_entity(entity_id)
                if entity is not None:
                    entity.add_scene(scene_id)

        return scenes, locations

    @staticmethod
    def compute_similarities(
        scenes: List[SceneSegment],
        *,
        threshold: float = Config.SCENE_SIMILARITY_THRESHOLD,
        top_k: int = Config.SCENE_SIMILARITY_TOP_K,
    ) -> List[SceneSimilarity]:
        """Compute similarity edges between scenes."""

        valid_indices = [
            idx for idx, scene in enumerate(scenes) if scene.embedding is not None
        ]
        if not valid_indices:
            return []

        embeddings = np.stack(
            [np.asarray(scenes[idx].embedding) for idx in valid_indices]
        )
        similarity_matrix = embeddings @ embeddings.T

        pairs_seen: Set[Tuple[str, str]] = set()
        edges: List[SceneSimilarity] = []

        for local_idx, global_idx in enumerate(valid_indices):
            row = similarity_matrix[local_idx]
            row[local_idx] = -np.inf  # Avoid self-selection
            candidate_count = min(top_k, len(valid_indices) - 1)
            if candidate_count <= 0:
                continue
            candidate_index_buffer = np.argpartition(
                -row, list(range(candidate_count))
            )[:candidate_count]
            top_candidate_indices = sorted(
                candidate_index_buffer, key=lambda idx: row[idx], reverse=True
            )

            for candidate_idx in top_candidate_indices:
                score = float(row[candidate_idx])
                if score < threshold:
                    continue
                other_global_idx = valid_indices[candidate_idx]
                source_id = scenes[global_idx].scene_id
                target_id = scenes[other_global_idx].scene_id
                if source_id == target_id:
                    continue
                ordered = tuple(sorted((source_id, target_id)))
                if ordered in pairs_seen:
                    continue
                pairs_seen.add(ordered)
                edges.append(
                    SceneSimilarity(
                        source_id=source_id, target_id=target_id, score=score
                    )
                )

        return edges


# ============================================================================
# MODULE 1: ENTITY TRACKING (OBJECT PERMANENCE)
# ============================================================================


class EntityTracker:
    """Tracks entities across time using visual embedding clustering"""

    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.detection_to_entity: Dict[str, str] = {}  # temp_id -> entity_id

    def cluster_embeddings(self, perception_log: List[Dict[str, Any]]) -> np.ndarray:
        """
        Cluster visual embeddings using HDBSCAN

        Args:
            perception_log: List of perception objects

        Returns:
            Array of cluster labels (same length as perception_log)
        """
        if not HDBSCAN_AVAILABLE:
            logger.error("HDBSCAN not available. Cannot perform clustering.")
            # Return all -1 (noise) labels
            return np.full(len(perception_log), -1)

        logger.info("Extracting visual embeddings for clustering...")
        embeddings = []
        valid_indices = []

        for i, obj in enumerate(perception_log):
            emb = obj.get("visual_embedding")
            if emb is not None and len(emb) > 0:
                embeddings.append(np.array(emb))
                valid_indices.append(i)

        if not embeddings:
            logger.error("No valid embeddings found in perception log")
            return np.full(len(perception_log), -1)

        embeddings = np.array(embeddings)
        logger.info(f"Clustering {len(embeddings)} embeddings...")

        # Create HDBSCAN clusterer
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=Config.MIN_CLUSTER_SIZE,
            min_samples=Config.MIN_SAMPLES,
            metric=Config.CLUSTER_METRIC,
            cluster_selection_method=Config.CLUSTER_SELECTION_METHOD,
            cluster_selection_epsilon=Config.CLUSTER_SELECTION_EPSILON,
        )

        # Perform clustering
        cluster_labels = clusterer.fit_predict(embeddings)

        # Map back to full perception log
        full_labels = np.full(len(perception_log), -1)
        for i, idx in enumerate(valid_indices):
            full_labels[idx] = cluster_labels[i]

        # Log clustering statistics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)

        logger.info(f"Clustering complete:")
        logger.info(f"  Number of clusters (tracked entities): {n_clusters}")
        logger.info(f"  Noise points (unique objects): {n_noise}")
        logger.info(f"  Total objects: {len(embeddings)}")

        return full_labels

    def assign_entity_ids(
        self, perception_log: List[Dict[str, Any]], cluster_labels: np.ndarray
    ):
        """
        Assign entity IDs based on cluster labels

        Args:
            perception_log: List of perception objects
            cluster_labels: Cluster assignments from HDBSCAN
        """
        logger.info("Assigning entity IDs...")

        # Create entities for each cluster
        cluster_to_entity_id = {}
        noise_counter = 0

        for i, (obj, label) in enumerate(zip(perception_log, cluster_labels)):
            temp_id = obj.get("temp_id", f"det_{i:06d}")

            if label == -1:
                # Noise point - create unique entity
                entity_id = f"entity_unique_{noise_counter:06d}"
                noise_counter += 1
            else:
                # Part of cluster
                if label not in cluster_to_entity_id:
                    cluster_to_entity_id[label] = f"entity_cluster_{label:04d}"
                entity_id = cluster_to_entity_id[label]

            # Update perception object
            obj["entity_id"] = entity_id

            # Track mapping
            self.detection_to_entity[temp_id] = entity_id

            # Create or update entity
            if entity_id not in self.entities:
                self.entities[entity_id] = Entity(
                    entity_id=entity_id, object_class=obj.get("object_class", "unknown")
                )

            self.entities[entity_id].add_appearance(obj)

        # Compute average embeddings
        logger.info("Computing average embeddings for entities...")
        for entity in self.entities.values():
            entity.compute_average_embedding()

        logger.info(f"Created {len(self.entities)} entities")
        logger.info(f"  Tracked entities (clusters): {len(cluster_to_entity_id)}")
        logger.info(f"  Unique entities (noise): {noise_counter}")

    def track_entities(self, perception_log: List[Dict[str, Any]]):
        """
        Main entity tracking pipeline

        Args:
            perception_log: List of perception objects (modified in-place)
        """
        logger.info("\n" + "=" * 80)
        logger.info("ENTITY TRACKING - OBJECT PERMANENCE")
        logger.info("=" * 80)

        # Cluster embeddings
        cluster_labels = self.cluster_embeddings(perception_log)

        # Assign entity IDs
        self.assign_entity_ids(perception_log, cluster_labels)

        logger.info("=" * 80 + "\n")

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID"""
        return self.entities.get(entity_id)

    def get_all_entities(self) -> List[Entity]:
        """Get all tracked entities"""
        return list(self.entities.values())


# ============================================================================
# MODULE 2: STATE CHANGE DETECTION
# ============================================================================


class StateChangeDetector:
    """Detects state changes in entity descriptions over time"""

    def __init__(self):
        self.model = None
        self.state_changes: List[StateChange] = []

    def load_model(self):
        """Load embedding model (EmbeddingGemma or SentenceTransformer)"""
        if not EMBEDDING_MODEL_AVAILABLE:
            logger.error("Embedding model not available")
            return False

        try:
            logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL_TYPE}")
            if create_embedding_model is None:  # type: ignore[truthy-function]
                raise RuntimeError("Embedding model factory unavailable")
            self.model = create_embedding_model(
                prefer_ollama=(Config.EMBEDDING_MODEL_TYPE == "embeddinggemma")
            )
            model_info = self.model.get_model_info()
            logger.info(
                f"✓ Model loaded: {model_info['model_name']} (type={model_info['type']}, dim={model_info['dimension']})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False

    def compute_similarity(self, desc1: str, desc2: str) -> float:
        """
        Compute semantic similarity between two descriptions

        Args:
            desc1: First description
            desc2: Second description

        Returns:
            Cosine similarity score (0-1)
        """
        if self.model is None:
            return 1.0  # No change if model not available

        try:
            # Use the unified embedding model's similarity method
            return self.model.compute_similarity(desc1, desc2)
        except Exception as e:
            logger.warning(f"Failed to compute similarity: {e}")
            return 1.0

    def detect_state_changes_for_entity(self, entity: Entity) -> List[StateChange]:
        """
        Detect state changes for a single entity

        Args:
            entity: Entity to analyze

        Returns:
            List of detected state changes
        """
        timeline = entity.get_timeline()
        changes = []

        if len(timeline) < 2:
            return changes

        # Compare consecutive appearances
        for i in range(len(timeline) - 1):
            curr = timeline[i]
            next_app = timeline[i + 1]

            curr_desc = curr.get("rich_description", "")
            next_desc = next_app.get("rich_description", "")

            if not curr_desc or not next_desc:
                continue

            # Compute similarity
            similarity = self.compute_similarity(curr_desc, next_desc)

            # Detect state change
            if similarity < Config.STATE_CHANGE_THRESHOLD:
                centroid_before = _compute_centroid_from_bbox(curr.get("bounding_box"))
                centroid_after = _compute_centroid_from_bbox(
                    next_app.get("bounding_box")
                )
                displacement = (
                    _euclidean_distance(centroid_before, centroid_after) or 0.0
                )
                time_delta = float(
                    next_app.get("timestamp", 0.0) - curr.get("timestamp", 0.0)
                )
                velocity = displacement / time_delta if time_delta > 0 else 0.0

                bbox_before = curr.get("bounding_box")
                bbox_after = next_app.get("bounding_box")
                if bbox_before is not None:
                    bbox_before = [int(v) for v in bbox_before]
                if bbox_after is not None:
                    bbox_after = [int(v) for v in bbox_after]

                emb_before = curr.get("visual_embedding")
                emb_after = next_app.get("visual_embedding")
                if emb_before is not None:
                    emb_before = [float(v) for v in emb_before]
                if emb_after is not None:
                    emb_after = [float(v) for v in emb_after]

                change = StateChange(
                    entity_id=entity.entity_id,
                    timestamp_before=curr["timestamp"],
                    timestamp_after=next_app["timestamp"],
                    description_before=curr_desc,
                    description_after=next_desc,
                    similarity_score=similarity,
                    change_magnitude=1.0 - similarity,
                    centroid_before=centroid_before,
                    centroid_after=centroid_after,
                    displacement=displacement,
                    velocity=velocity,
                    frame_before=curr.get("frame_number"),
                    frame_after=next_app.get("frame_number"),
                    bounding_box_before=bbox_before,
                    bounding_box_after=bbox_after,
                    scene_before=curr.get("scene_id"),
                    scene_after=next_app.get("scene_id"),
                    location_before=curr.get("location_id"),
                    location_after=next_app.get("location_id"),
                    embedding_before=emb_before,
                    embedding_after=emb_after,
                )
                changes.append(change)

        return changes

    def detect_all_state_changes(self, entities: List[Entity]) -> List[StateChange]:
        """
        Detect state changes for all entities

        Args:
            entities: List of entities to analyze

        Returns:
            List of all detected state changes
        """
        logger.info("\n" + "=" * 80)
        logger.info("STATE CHANGE DETECTION")
        logger.info("=" * 80)

        if not self.load_model():
            logger.warning("Continuing without state change detection")
            return []

        logger.info(f"Analyzing {len(entities)} entities for state changes...")

        all_changes = []
        for entity in entities:
            changes = self.detect_state_changes_for_entity(entity)
            all_changes.extend(changes)

        self.state_changes = all_changes

        logger.info(f"Detected {len(all_changes)} state changes")

        # Log some statistics
        if all_changes:
            magnitudes = [c.change_magnitude for c in all_changes]
            logger.info(f"  Average change magnitude: {np.mean(magnitudes):.3f}")
            logger.info(f"  Max change magnitude: {np.max(magnitudes):.3f}")
            logger.info(f"  Min change magnitude: {np.min(magnitudes):.3f}")

        logger.info("=" * 80 + "\n")

        return all_changes


# ============================================================================
# MODULE 3: TEMPORAL WINDOWING
# ============================================================================


def create_temporal_windows(
    state_changes: List[StateChange], window_size: float = Config.TIME_WINDOW_SIZE
) -> List[TemporalWindow]:
    """
    Divide state changes into adaptive temporal windows.

    Args:
        state_changes: List of state changes
        window_size: Legacy window size hint (unused in adaptive mode)

    Returns:
        List of temporal windows
    """
    if not state_changes:
        return []

    max_duration = max(Config.TIME_WINDOW_MAX_DURATION, 0.1)
    max_gap = max(Config.TIME_WINDOW_MAX_GAP, 0.1)
    max_changes = max(Config.TIME_WINDOW_MAX_CHANGES, 1)

    ordered_changes = sorted(state_changes, key=lambda c: c.timestamp_after)

    windows: List[TemporalWindow] = []
    current_window = TemporalWindow(
        start_time=ordered_changes[0].timestamp_before,
        end_time=ordered_changes[0].timestamp_after,
    )
    current_window.add_state_change(ordered_changes[0])

    for change in ordered_changes[1:]:
        gap = change.timestamp_before - current_window.end_time
        projected_end = max(current_window.end_time, change.timestamp_after)
        projected_duration = projected_end - current_window.start_time
        overflow_gap = gap > max_gap
        overflow_duration = projected_duration > max_duration
        overflow_changes = len(current_window.state_changes) >= max_changes

        if overflow_gap or overflow_duration or overflow_changes:
            if current_window.state_changes:
                windows.append(current_window)
            current_window = TemporalWindow(
                start_time=change.timestamp_before,
                end_time=change.timestamp_after,
            )
        else:
            current_window.end_time = projected_end

        current_window.add_state_change(change)

    if current_window.state_changes:
        windows.append(current_window)

    significant_windows = [w for w in windows if w.is_significant()]

    logger.info(
        "Created %d adaptive windows, %d significant",
        len(windows),
        len(significant_windows),
    )

    return significant_windows


# ============================================================================
# MODULE 3B: CAUSAL INFLUENCE SCORING
# ============================================================================


class CausalInfluenceScorer:
    """Scores likely causal influences between entities inside temporal windows."""

    def __init__(self, tracker: EntityTracker):
        self.tracker = tracker
        self.engine = None
        self.causal_config = None
        self._stats = {}

        if (
            CAUSAL_INFERENCE_AVAILABLE
            and CausalInferenceEngine is not None
            and CausalConfig is not None
            and AgentCandidate is not None
            and CISStateChange is not None
        ):
            self.causal_config = CausalConfig(
                proximity_weight=Config.CAUSAL_PROXIMITY_WEIGHT,
                motion_weight=Config.CAUSAL_MOTION_WEIGHT,
                temporal_weight=Config.CAUSAL_TEMPORAL_WEIGHT,
                embedding_weight=Config.CAUSAL_EMBEDDING_WEIGHT,
                max_pixel_distance=Config.CAUSAL_MAX_PIXEL_DISTANCE,
                temporal_decay=Config.CAUSAL_TEMPORAL_DECAY,
                min_score=Config.CAUSAL_MIN_SCORE,
                top_k_per_event=Config.CAUSAL_TOP_K_PER_WINDOW,
            )
            self.engine = CausalInferenceEngine(self.causal_config)
            logger.info(
                "Causal inference engine enabled (top_k=%d, min_score=%.2f)",
                self.causal_config.top_k_per_event,
                self.causal_config.min_score,
            )
        else:
            logger.warning(
                "Causal inference engine unavailable; falling back to heuristic scoring."
            )

    def score_windows(self, windows: List[TemporalWindow]) -> Dict[str, int]:
        """Populate each window with high-confidence causal influence links."""

        if not windows:
            return {
                "total_links": 0,
                "windows_with_links": 0,
                "engine_calls": 0,
                "candidate_pairs": 0,
                "accepted_pairs": 0,
            }

        self._stats = {
            "engine_calls": 0.0,
            "candidate_pairs": 0.0,
            "accepted_pairs": 0.0,
            "fallback_used": 1.0 if self.engine is None else 0.0,
        }

        logger.info("\n" + "=" * 80)
        logger.info("CAUSAL INFLUENCE SCORING")
        logger.info("=" * 80)

        total_links = 0
        windows_with_links = 0

        for window in windows:
            links = self._score_window(window)
            window.causal_links = links
            if links:
                windows_with_links += 1
                total_links += len(links)

        logger.info(
            "Computed %d causal links across %d/%d windows",
            total_links,
            windows_with_links,
            len(windows),
        )
        logger.info("=" * 80 + "\n")

        return {
            "total_links": total_links,
            "windows_with_links": windows_with_links,
            "engine_calls": int(self._stats.get("engine_calls", 0.0)),
            "candidate_pairs": int(self._stats.get("candidate_pairs", 0.0)),
            "accepted_pairs": int(self._stats.get("accepted_pairs", 0.0)),
        }

    def _score_window(self, window: TemporalWindow) -> List[CausalLink]:
        if self.engine is None:
            return self._score_window_fallback(window)
        return self._score_window_cis(window)

    def _score_window_cis(self, window: TemporalWindow) -> List[CausalLink]:
        if len(window.state_changes) < 2:
            return []

        best_links: Dict[Tuple[str, str], CausalLink] = {}

        for patient_change in window.state_changes:
            patient_entity = self.tracker.get_entity(patient_change.entity_id)
            if patient_entity is None:
                continue

            cis_patient = self._build_cis_state_change(patient_change, patient_entity)
            if cis_patient is None:
                continue

            agent_candidates: List[Any] = []
            agent_state_lookup: Dict[str, StateChange] = {}

            for agent_change in window.state_changes:
                if agent_change.entity_id == patient_change.entity_id:
                    continue
                agent_entity = self.tracker.get_entity(agent_change.entity_id)
                if agent_entity is None:
                    continue
                candidate = self._build_agent_candidate(
                    agent_change, agent_entity, patient_change.timestamp_after
                )
                if candidate is None:
                    continue
                agent_candidates.append(candidate)
                agent_state_lookup[candidate.entity_id] = agent_change

            if not agent_candidates:
                continue

            self._stats["engine_calls"] += 1
            self._stats["candidate_pairs"] += len(agent_candidates)

            cis_links = self.engine.score_all_agents(agent_candidates, cis_patient)
            if not cis_links:
                continue
            self._stats["accepted_pairs"] += len(cis_links)

            for cis_link in cis_links:
                agent_state = agent_state_lookup.get(cis_link.agent.entity_id)
                if agent_state is None:
                    continue
                features = {
                    "proximity": cis_link.proximity_score,
                    "motion": cis_link.motion_score,
                    "temporal": cis_link.temporal_score,
                    "embedding": cis_link.embedding_score,
                }
                agent_location = agent_state.location_after or agent_state.location_before
                patient_location = (
                    patient_change.location_after or patient_change.location_before
                )
                location_fragment = ""
                if agent_location or patient_location:
                    location_fragment = (
                        f" | locations agent:{agent_location or '?'}"
                        f" patient:{patient_location or '?'}"
                    )
                justification = (
                    f"{cis_link.agent.entity_id} → {cis_link.patient.entity_id}"
                    f" (CIS={cis_link.cis_score:.2f}, prox={cis_link.proximity_score:.2f},"
                    f" motion={cis_link.motion_score:.2f}, dt={cis_link.temporal_score:.2f},"
                    f" embed={cis_link.embedding_score:.2f}){location_fragment}"
                )
                semantic_link = CausalLink(
                    agent_id=cis_link.agent.entity_id,
                    patient_id=cis_link.patient.entity_id,
                    agent_change=agent_state,
                    patient_change=patient_change,
                    influence_score=cis_link.cis_score,
                    features=features,
                    justification=justification,
                )
                key = (semantic_link.agent_id, semantic_link.patient_id)
                stored = best_links.get(key)
                if stored is None or semantic_link.influence_score > stored.influence_score:
                    best_links[key] = semantic_link

        if not best_links:
            return []

        sorted_links = sorted(
            best_links.values(),
            key=lambda item: item.influence_score,
            reverse=True,
        )

        return sorted_links[: Config.CAUSAL_TOP_K_PER_WINDOW]

    def _score_window_fallback(self, window: TemporalWindow) -> List[CausalLink]:
        if len(window.state_changes) < 2:
            return []

        candidates: List[CausalLink] = []
        for agent_change in window.state_changes:
            for patient_change in window.state_changes:
                if agent_change.entity_id == patient_change.entity_id:
                    continue
                link = self._evaluate_pair(agent_change, patient_change)
                if link is not None:
                    candidates.append(link)

        self._stats["candidate_pairs"] += len(window.state_changes) * max(
            len(window.state_changes) - 1, 0
        )
        self._stats["accepted_pairs"] += len(candidates)

        if not candidates:
            return []

        dedup: Dict[Tuple[str, str], CausalLink] = {}
        for link in candidates:
            key = (link.agent_id, link.patient_id)
            stored = dedup.get(key)
            if stored is None or link.influence_score > stored.influence_score:
                dedup[key] = link

        sorted_links = sorted(
            dedup.values(),
            key=lambda item: item.influence_score,
            reverse=True,
        )

        return sorted_links[: Config.CAUSAL_TOP_K_PER_WINDOW]

    def _build_cis_state_change(
        self, change: StateChange, entity: Entity
    ) -> Optional[Any]:
        if CISStateChange is None:
            return None

        centroid = change.centroid_after or change.centroid_before
        bounding_box = change.bounding_box_after or change.bounding_box_before
        frame_number = change.frame_after or change.frame_before or 0

        if centroid is None or bounding_box is None:
            appearance = entity.get_appearance_near(change.timestamp_after)
            if appearance:
                centroid = centroid or _compute_centroid_from_bbox(
                    appearance.get("bounding_box")
                )
                bbox_val = appearance.get("bounding_box")
                if bounding_box is None and bbox_val is not None:
                    bounding_box = [int(v) for v in bbox_val]

        if centroid is None or bounding_box is None:
            return None

        return CISStateChange(
            entity_id=change.entity_id,
            timestamp=change.timestamp_after,
            frame_number=int(frame_number),
            old_description=change.description_before,
            new_description=change.description_after,
            centroid=(float(centroid[0]), float(centroid[1])),
            bounding_box=[int(v) for v in bounding_box],
        )

    def _build_agent_candidate(
        self,
        change: StateChange,
        entity: Entity,
        reference_time: float,
    ) -> Optional[Any]:
        if AgentCandidate is None:
            return None

        centroid = change.centroid_after or change.centroid_before
        bounding_box = change.bounding_box_after or change.bounding_box_before
        embedding = change.embedding_after or change.embedding_before

        appearance = entity.get_appearance_near(reference_time)
        if appearance:
            if centroid is None:
                centroid = _compute_centroid_from_bbox(appearance.get("bounding_box"))
            if bounding_box is None:
                bbox_val = appearance.get("bounding_box")
                if bbox_val is not None:
                    bounding_box = [int(v) for v in bbox_val]
            if embedding is None:
                emb_val = appearance.get("visual_embedding")
                if emb_val is not None:
                    embedding = [float(v) for v in emb_val]

        if embedding is None and entity.average_embedding is not None:
            embedding = [float(v) for v in entity.average_embedding.tolist()]

        if centroid is None or bounding_box is None or embedding is None:
            return None

        motion = self._estimate_motion(change)
        description = (
            change.description_after
            or change.description_before
            or entity.object_class
        )
        temp_id = appearance.get("temp_id") if appearance else None

        return AgentCandidate(
            entity_id=entity.entity_id,
            temp_id=temp_id or entity.entity_id,
            timestamp=change.timestamp_after,
            centroid=(float(centroid[0]), float(centroid[1])),
            bounding_box=[int(v) for v in bounding_box],
            motion_data=motion,
            visual_embedding=[float(v) for v in embedding],
            object_class=entity.object_class,
            description=description,
        )

    @staticmethod
    def _estimate_motion(change: StateChange) -> Optional[MotionData]:
        if MotionData is None:
            return None
        if change.centroid_before is None or change.centroid_after is None:
            return None
        time_delta = change.timestamp_after - change.timestamp_before
        if time_delta <= 0:
            return None
        dx = float(change.centroid_after[0] - change.centroid_before[0])
        dy = float(change.centroid_after[1] - change.centroid_before[1])
        vx = dx / time_delta
        vy = dy / time_delta
        speed = math.hypot(vx, vy)
        direction = math.atan2(vy, vx) if speed > 1e-6 else 0.0
        return MotionData(
            centroid=(float(change.centroid_after[0]), float(change.centroid_after[1])),
            velocity=(vx, vy),
            speed=speed,
            direction=direction,
            timestamp=change.timestamp_after,
        )

    def _evaluate_pair(
        self,
        agent_change: StateChange,
        patient_change: StateChange,
    ) -> Optional[CausalLink]:
        agent_entity = self.tracker.get_entity(agent_change.entity_id)
        patient_entity = self.tracker.get_entity(patient_change.entity_id)

        if agent_entity is None or patient_entity is None:
            return None

        proximity = self._compute_proximity(agent_change, patient_change)
        motion = self._compute_motion(agent_change, patient_change)
        temporal = self._compute_temporal(agent_change, patient_change)
        embedding_sim = self._compute_embedding_similarity(agent_entity, patient_entity)

        score = (
            Config.CAUSAL_PROXIMITY_WEIGHT * proximity
            + Config.CAUSAL_MOTION_WEIGHT * motion
            + Config.CAUSAL_TEMPORAL_WEIGHT * temporal
            + Config.CAUSAL_EMBEDDING_WEIGHT * embedding_sim
        )

        if score < Config.CAUSAL_MIN_SCORE:
            return None

        features = {
            "proximity": proximity,
            "motion": motion,
            "temporal": temporal,
            "embedding": embedding_sim,
        }

        justification = (
            f"Proximity={proximity:.2f}, Motion={motion:.2f}, "
            f"Temporal={temporal:.2f}, Embedding={embedding_sim:.2f}"
        )

        return CausalLink(
            agent_id=agent_change.entity_id,
            patient_id=patient_change.entity_id,
            agent_change=agent_change,
            patient_change=patient_change,
            influence_score=score,
            features=features,
            justification=justification,
        )

    @staticmethod
    def _compute_proximity(
        agent_change: StateChange,
        patient_change: StateChange,
    ) -> float:
        centroid_pairs = [
            (agent_change.centroid_after, patient_change.centroid_after),
            (agent_change.centroid_after, patient_change.centroid_before),
            (agent_change.centroid_before, patient_change.centroid_after),
            (agent_change.centroid_before, patient_change.centroid_before),
        ]

        distances = [
            dist
            for dist in (_euclidean_distance(a, b) for a, b in centroid_pairs)
            if dist is not None
        ]

        if not distances:
            return 0.0

        min_distance = min(distances)
        clamped = min(min_distance, Config.CAUSAL_MAX_PIXEL_DISTANCE)
        return max(0.0, 1.0 - clamped / Config.CAUSAL_MAX_PIXEL_DISTANCE)

    @staticmethod
    def _compute_motion(
        agent_change: StateChange, patient_change: StateChange
    ) -> float:
        agent_motion = (
            float(agent_change.displacement)
            + float(agent_change.velocity) * Config.CAUSAL_TEMPORAL_DECAY
        )
        patient_motion = (
            float(patient_change.displacement)
            + float(patient_change.velocity) * Config.CAUSAL_TEMPORAL_DECAY
        )

        total_motion = agent_motion + patient_motion
        if total_motion <= 0:
            return 0.0

        ratio = agent_motion / total_motion
        return max(0.0, min(1.0, ratio))

    @staticmethod
    def _compute_temporal(
        agent_change: StateChange, patient_change: StateChange
    ) -> float:
        delta = abs(agent_change.timestamp_after - patient_change.timestamp_after)
        clamped = min(delta, Config.CAUSAL_TEMPORAL_DECAY)
        return max(0.0, 1.0 - clamped / Config.CAUSAL_TEMPORAL_DECAY)

    @staticmethod
    def _compute_embedding_similarity(
        agent_entity: Entity, patient_entity: Entity
    ) -> float:
        emb_a = agent_entity.average_embedding
        emb_b = patient_entity.average_embedding

        if emb_a is None or emb_b is None:
            return 0.0

        similarity = float(np.dot(emb_a, emb_b))
        similarity = max(-1.0, min(1.0, similarity))
        return 0.5 * (similarity + 1.0)


# ============================================================================
# MODULE 4: EVENT COMPOSITION (LLM REASONING)
# ============================================================================


class EventComposer:
    """Composes events from state changes using LLM reasoning"""

    def __init__(
        self,
        scene_lookup: Optional[Dict[str, SceneSegment]] = None,
        location_lookup: Optional[Dict[str, LocationProfile]] = None,
    ):
        self.generated_queries: List[str] = []
        self.scene_lookup = scene_lookup or {}
        self.location_lookup = location_lookup or {}
        self.metrics: Dict[str, Any] = {
            "windows_total": 0,
            "windows_composed": 0,
            "windows_skipped": 0,
            "windows_capped": 0,
            "llm_calls": 0,
            "llm_latency": 0.0,
        }

    def _resolve_location_label(self, location_id: Optional[str]) -> Optional[str]:
        if not location_id:
            return None
        profile = self.location_lookup.get(location_id)
        if profile:
            return f"{profile.label} ({location_id})"
        return location_id

    def _collect_window_locations(self, window: TemporalWindow) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}
        for change in window.state_changes:
            scene_id = change.scene_after or change.scene_before
            if not scene_id:
                continue
            scene = self.scene_lookup.get(scene_id)
            if scene is None:
                continue
            loc_id = scene.location_id
            if not loc_id:
                continue
            if loc_id not in summary:
                summary[loc_id] = {
                    "label": self._resolve_location_label(loc_id) or loc_id,
                    "scenes": set(),
                }
            summary[loc_id]["scenes"].add(scene_id)
        return summary

    def _should_skip_window(self, window: TemporalWindow) -> Optional[str]:
        change_count = len(window.state_changes)
        if change_count == 0:
            return "no state changes"
        if (
            Config.EVENT_COMPOSITION_SKIP_NO_CAUSAL
            and not window.causal_links
            and change_count < Config.EVENT_COMPOSITION_MIN_STATE_CHANGES
        ):
            return "low activity without causal links"
        return None

    def query_ollama(self, prompt: str) -> str:
        """
        Query local Ollama instance

        Args:
            prompt: Prompt for the LLM

        Returns:
            Generated text
        """
        try:
            payload = {
                "model": Config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": Config.OLLAMA_TEMPERATURE,
                    "num_predict": Config.OLLAMA_MAX_TOKENS,
                },
            }

            response = requests.post(
                Config.OLLAMA_API_URL, json=payload, timeout=Config.OLLAMA_TIMEOUT
            )

            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return ""

        except requests.exceptions.ConnectionError:
            logger.error(
                "Could not connect to Ollama. Is it running at localhost:11434?"
            )
            return ""
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}")
            return ""

    def create_prompt_for_window(
        self, window: TemporalWindow, entity_tracker: EntityTracker
    ) -> str:
        """
        Create structured prompt for event composition

        Args:
            window: Temporal window with state changes
            entity_tracker: Entity tracker for getting entity info

        Returns:
            Formatted prompt
        """
        prompt = f"""You are a knowledge graph builder. Given entity tracking data and state changes, generate ONLY valid Cypher queries to represent this information in Neo4j.

SCHEMA:
- Node types: :Entity, :Event, :State
- Relationship types: [:PARTICIPATED_IN], [:CHANGED_TO], [:OCCURRED_AT], [:INFLUENCED]
- Entity properties: id (STRING), label (STRING), embedding (LIST<FLOAT>), first_description (TEXT)
- Event properties: type (STRING), timestamp (DATETIME), description (TEXT)
- State properties: description (TEXT), timestamp (FLOAT)

TIME WINDOW: {window.start_time:.2f}s - {window.end_time:.2f}s

ACTIVE ENTITIES:
"""

        for entity_id in window.active_entities:
            entity = entity_tracker.get_entity(entity_id)
            if entity:
                prompt += f"- {entity_id} ({entity.object_class}): {len(entity.appearances)} appearances\n"

        spatial_summary = self._collect_window_locations(window)
        if spatial_summary:
            prompt += "\nSPATIAL CONTEXT:\n"
            for loc_id, meta in spatial_summary.items():
                scenes = meta.get("scenes", set())
                scene_list = ", ".join(sorted(scenes)) if scenes else "n/a"
                prompt += (
                    f"- {meta.get('label', loc_id)} | id={loc_id} | scenes: {scene_list}\n"
                )

        prompt += "\nSTATE CHANGES:\n"

        for i, change in enumerate(window.state_changes, 1):
            prompt += f"{i}. Entity {change.entity_id} at {change.timestamp_before:.2f}s → {change.timestamp_after:.2f}s\n"
            prompt += f"   Before: {change.description_before}\n"
            prompt += f"   After: {change.description_after}\n"
            prompt += f"   Change magnitude: {change.change_magnitude:.3f}\n\n"
            prior_loc = self._resolve_location_label(change.location_before)
            post_loc = self._resolve_location_label(change.location_after)
            if prior_loc or post_loc:
                if prior_loc and post_loc and prior_loc != post_loc:
                    prompt += f"   Location change: {prior_loc} → {post_loc}\n"
                else:
                    prompt += f"   Location: {post_loc or prior_loc}\n"
            scene_id = change.scene_after or change.scene_before
            if scene_id:
                prompt += f"   Scene: {scene_id}\n"

        if window.causal_links:
            prompt += "CAUSAL INFLUENCE CANDIDATES (Agent → Patient):\n"
            for link in window.causal_links:
                agent_change = link.agent_change
                patient_change = link.patient_change
                prompt += (
                    f"- {link.agent_id} → {link.patient_id} | score={link.influence_score:.2f} | "
                    f"prox={link.features['proximity']:.2f} | motion={link.features['motion']:.2f} | "
                    f"dt={link.features['temporal']:.2f} | embed={link.features['embedding']:.2f}\n"
                    f"  Agent after: {agent_change.description_after}\n"
                    f"  Patient after: {patient_change.description_after}\n"
                )
        else:
            prompt += "CAUSAL INFLUENCE CANDIDATES: None detected in this window.\n"

        prompt += """
Generate Cypher queries to:
1. Create or merge entity nodes with their labels
2. Create event nodes for significant state changes
3. Create relationships between entities and events, including agent → patient influence
4. When causal links exist, emit an event with type 'causal_influence' capturing score, proximity, motion, temporal, and embedding metrics

CRITICAL SYNTAX RULES:
- Use MERGE for entities to avoid duplicates
- Use MERGE for events with unique IDs
- NEVER use WHERE after SET (invalid syntax!)
- Use literal values, not parameters
- Each query must end with semicolon
- No explanations, only Cypher code

CORRECT EXAMPLES:
MERGE (e:Entity {id: 'entity_cluster_0001'}) SET e.label = 'laptop';
MERGE (ev:Event {id: 'event_001'}) SET ev.type = 'movement', ev.timestamp = datetime({epochSeconds: 5}), ev.description = 'Laptop moved across desk';
MATCH (e:Entity {id: 'entity_cluster_0001'}), (ev:Event {id: 'event_001'}) MERGE (e)-[:PARTICIPATED_IN]->(ev);
MATCH (agent:Entity {id: 'entity_cluster_0001'}), (patient:Entity {id: 'entity_cluster_0002'}), (ev:Event {id: 'event_001'})
MERGE (agent)-[:PARTICIPATED_IN]->(ev)
MERGE (ev)-[:INFLUENCED {score: 0.78}]->(patient);

WRONG EXAMPLES (DO NOT DO THIS):
MERGE (e:Entity {id: '1'}) SET e.label = 'Laptop' WHERE e.type = 'Laptop';  // WRONG: WHERE after SET
CREATE (e:Entity {id: '1'}) SET e.label = 'Laptop';  // WRONG: Use MERGE not CREATE for entities
MERGE (e:Entity {id: '1'})  // WRONG: Missing semicolon

Now generate Cypher queries for the state changes above:
"""

        return prompt

    def parse_cypher_queries(self, llm_output: str) -> List[str]:
        """
        Parse Cypher queries from LLM output

        Args:
            llm_output: Raw output from LLM

        Returns:
            List of valid Cypher queries
        """
        queries = []

        lines = llm_output.strip().split("\n")
        for line in lines:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("//") or line.startswith("#"):
                continue

            # Look for Cypher keywords
            if any(
                keyword in line.upper()
                for keyword in ["MERGE", "CREATE", "MATCH", "SET"]
            ):
                # Ensure semicolon
                if not line.endswith(";"):
                    line += ";"
                queries.append(line)

        return queries

    def validate_cypher_queries(self, queries: List[str]) -> List[str]:
        """
        Validate Cypher queries for common syntax errors

        Args:
            queries: List of Cypher queries to validate

        Returns:
            List of valid queries (invalid ones filtered out)
        """
        valid_queries = []

        for query in queries:
            query_upper = query.upper()

            # Check for invalid patterns
            invalid = False

            # Pattern 1: WHERE after SET (invalid in Cypher)
            if "SET" in query_upper and "WHERE" in query_upper:
                set_pos = query_upper.index("SET")
                where_pos = query_upper.index("WHERE")
                if where_pos > set_pos:
                    logger.warning(f"Invalid query (WHERE after SET): {query[:50]}...")
                    invalid = True

            # Pattern 2: Missing required keywords
            if not any(
                kw in query_upper
                for kw in ["MERGE", "CREATE", "MATCH", "SET", "RETURN"]
            ):
                logger.warning(f"Invalid query (no valid keywords): {query[:50]}...")
                invalid = True

            if not invalid:
                valid_queries.append(query)

        return valid_queries

    def compose_events_for_window(
        self, window: TemporalWindow, entity_tracker: EntityTracker
    ) -> List[str]:
        """
        Compose events for a single window

        Args:
            window: Temporal window
            entity_tracker: Entity tracker

        Returns:
            List of Cypher queries
        """
        logger.info(
            f"Composing events for window {window.start_time:.1f}s - {window.end_time:.1f}s"
        )

        # Use fallback if LLM is disabled
        if not Config.USE_LLM_COMPOSITION:
            logger.info("Using fallback query generation (LLM disabled)")
            return self.generate_fallback_queries(window, entity_tracker)

        # Create prompt
        prompt = self.create_prompt_for_window(window, entity_tracker)

        llm_output = ""

        if Config.USE_LLM_COMPOSITION:
            call_started = time.time()
            llm_output = self.query_ollama(prompt)
            self.metrics["llm_calls"] += 1
            self.metrics["llm_latency"] += max(time.time() - call_started, 0.0)

        if not llm_output:
            logger.warning("No output from LLM, generating basic queries")
            return self.generate_fallback_queries(window, entity_tracker)

        # Parse queries
        queries = self.parse_cypher_queries(llm_output)

        # Validate queries before returning
        validated_queries = self.validate_cypher_queries(queries)

        if not validated_queries:
            logger.warning("LLM generated invalid queries, using fallback")
            return self.generate_fallback_queries(window, entity_tracker)

        logger.info(f"Generated {len(validated_queries)} valid Cypher queries")

        return validated_queries

    def generate_fallback_queries(
        self, window: TemporalWindow, entity_tracker: EntityTracker
    ) -> List[str]:
        """
        Generate basic Cypher queries without LLM

        Args:
            window: Temporal window
            entity_tracker: Entity tracker

        Returns:
            List of basic Cypher queries
        """
        queries = []

        # Create entity nodes (skip descriptions in fallback mode to avoid syntax errors)
        for entity_id in window.active_entities:
            entity = entity_tracker.get_entity(entity_id)
            if entity:
                # Escape apostrophes and special characters in label
                safe_label = entity.object_class.replace("'", "\\'").replace('"', '\\"')
                queries.append(
                    f"MERGE (e:Entity {{id: '{entity_id}'}}) "
                    f"SET e.label = '{safe_label}';"
                )

        # Create event nodes for state changes (use counter to ensure unique IDs)
        for idx, change in enumerate(window.state_changes):
            event_id = f"event_{change.entity_id}_{int(change.timestamp_after)}_{idx}"
            queries.append(
                f"MERGE (ev:Event {{id: '{event_id}'}}) "
                f"SET ev.type = 'state_change', "
                f"ev.timestamp = datetime({{epochSeconds: {int(change.timestamp_after)}}}), "
                f"ev.description = 'Entity changed state';"
            )

            # Link entity to event
            queries.append(
                f"MATCH (e:Entity {{id: '{change.entity_id}'}}), "
                f"(ev:Event {{id: '{event_id}'}}) "
                f"MERGE (e)-[:PARTICIPATED_IN]->(ev);"
            )

        # Create events for causal influence links
        for idx, link in enumerate(window.causal_links):
            event_id = f"causal_{link.agent_id}_{link.patient_id}_{int(window.start_time)}_{idx}"
            score = f"{link.influence_score:.3f}"
            prox = f"{link.features.get('proximity', 0.0):.3f}"
            motion = f"{link.features.get('motion', 0.0):.3f}"
            temporal = f"{link.features.get('temporal', 0.0):.3f}"
            embedding = f"{link.features.get('embedding', 0.0):.3f}"
            justification = link.justification.replace("'", "\\'")[:450]

            queries.append(
                f"MERGE (ev:Event {{id: '{event_id}'}}) "
                f"SET ev.type = 'causal_influence', "
                f"ev.timestamp = datetime({{epochSeconds: {int(window.end_time)}}}), "
                f"ev.score = {score}, "
                f"ev.proximity = {prox}, "
                f"ev.motion = {motion}, "
                f"ev.temporal = {temporal}, "
                f"ev.embedding_similarity = {embedding}, "
                f"ev.description = '{justification}';"
            )

            queries.append(
                f"MATCH (agent:Entity {{id: '{link.agent_id}'}}), "
                f"(patient:Entity {{id: '{link.patient_id}'}}), "
                f"(ev:Event {{id: '{event_id}'}}) "
                f"MERGE (agent)-[:PARTICIPATED_IN {{role: 'agent'}}]->(ev);"
            )
            queries.append(
                f"MATCH (agent:Entity {{id: '{link.agent_id}'}}), "
                f"(patient:Entity {{id: '{link.patient_id}'}}), "
                f"(ev:Event {{id: '{event_id}'}}) "
                f"MERGE (patient)-[:PARTICIPATED_IN {{role: 'patient'}}]->(ev);"
            )
            queries.append(
                f"MATCH (ev:Event {{id: '{event_id}'}}), "
                f"(patient:Entity {{id: '{link.patient_id}'}}) "
                f"MERGE (ev)-[:INFLUENCED {{score: {score}}}]->(patient);"
            )

        return queries

    def compose_all_events(
        self, windows: List[TemporalWindow], entity_tracker: EntityTracker
    ) -> List[str]:
        """
        Compose events for all windows

        Args:
            windows: List of temporal windows
            entity_tracker: Entity tracker

        Returns:
            List of all Cypher queries
        """
        logger.info("\n" + "=" * 80)
        logger.info("EVENT COMPOSITION - LLM REASONING")
        logger.info("=" * 80)

        self.metrics["llm_calls"] = 0
        self.metrics["llm_latency"] = 0.0
        self.metrics["windows_total"] = len(windows)
        self.metrics["windows_composed"] = 0
        self.metrics["windows_skipped"] = 0
        self.metrics["windows_capped"] = 0

        all_queries = []
        budget = Config.EVENT_COMPOSITION_MAX_WINDOWS or 0

        for i, window in enumerate(windows, 1):
            if budget and self.metrics["windows_composed"] >= budget:
                remaining = len(windows) - (i - 1)
                self.metrics["windows_capped"] += max(remaining, 0)
                logger.info(
                    "Reached LLM window budget (%d processed, %d remaining skipped)",
                    Config.EVENT_COMPOSITION_MAX_WINDOWS,
                    max(remaining, 0),
                )
                break

            skip_reason = self._should_skip_window(window)
            if skip_reason:
                self.metrics["windows_skipped"] += 1
                logger.info(
                    "Skipping window %d/%d (%s)", i, len(windows), skip_reason
                )
                continue

            self.metrics["windows_composed"] += 1
            logger.info(
                "Processing window %d/%d (state changes=%d, causal_links=%d)",
                i,
                len(windows),
                len(window.state_changes),
                len(window.causal_links),
            )
            queries = self.compose_events_for_window(window, entity_tracker)
            all_queries.extend(queries)

        self.generated_queries = all_queries

        logger.info(f"Generated {len(all_queries)} total Cypher queries")
        logger.info("=" * 80 + "\n")

        return all_queries


# ============================================================================
# MODULE 5: NEO4J KNOWLEDGE GRAPH INGESTION
# ============================================================================


class KnowledgeGraphBuilder:
    """Builds and manages Neo4j knowledge graph"""

    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or Config.NEO4J_URI
        self.user = user or Config.NEO4J_USER
        self.password = password or Config.NEO4J_PASSWORD
        self.driver = None

    def connect(self) -> bool:
        """
        Connect to Neo4j database

        Returns:
            True if connection successful
        """
        try:
            logger.info(f"Connecting to Neo4j at {self.uri}...")

            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=Config.MAX_CONNECTION_LIFETIME,
                max_connection_pool_size=Config.MAX_CONNECTION_POOL_SIZE,
                connection_timeout=Config.CONNECTION_TIMEOUT,
            )

            # Verify connection
            self.driver.verify_connectivity()

            logger.info("Connected to Neo4j successfully")
            return True

        except ServiceUnavailable:
            logger.error(f"Could not connect to Neo4j at {self.uri}")
            logger.error("Make sure Neo4j is running and credentials are correct")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def initialize_schema(self, *, scene_embedding_dim: Optional[int] = None):
        """Initialize Neo4j schema with constraints and indexes"""
        logger.info("Initializing Neo4j schema...")

        with self.driver.session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (ev:Event) REQUIRE ev.id IS UNIQUE",
                "CREATE CONSTRAINT scene_id IF NOT EXISTS FOR (s:Scene) REQUIRE s.id IS UNIQUE",
                "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (loc:Location) REQUIRE loc.id IS UNIQUE",
            ]

            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(
                        f"  Created constraint: {constraint.split('FOR')[1].split('REQUIRE')[0].strip()}"
                    )
                except Exception as e:
                    logger.warning(f"  Constraint may already exist: {e}")

            # Create indexes
            indexes = [
                "CREATE INDEX entity_label IF NOT EXISTS FOR (e:Entity) ON (e.label)",
                "CREATE INDEX event_timestamp IF NOT EXISTS FOR (ev:Event) ON (ev.timestamp)",
                "CREATE INDEX event_type IF NOT EXISTS FOR (ev:Event) ON (ev.type)",
                "CREATE INDEX scene_frame IF NOT EXISTS FOR (s:Scene) ON (s.frame_number)",
                "CREATE INDEX location_label IF NOT EXISTS FOR (loc:Location) ON (loc.label)",
            ]

            for index in indexes:
                try:
                    session.run(index)
                    logger.info(f"  Created index: {index.split('ON')[1].strip()}")
                except Exception as e:
                    logger.warning(f"  Index may already exist: {e}")

            # Create vector index (if supported)
            try:
                vector_index = f"""
                CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
                FOR (e:Entity) ON (e.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {Config.VECTOR_DIMENSIONS},
                        `vector.similarity_function`: '{Config.VECTOR_SIMILARITY_FUNCTION}'
                    }}
                }}
                """
                session.run(vector_index)
                logger.info("  Created vector index for embeddings")
            except Exception as e:
                logger.warning(f"  Vector index not supported or already exists: {e}")

            if scene_embedding_dim:
                try:
                    scene_vector_index = f"""
                    CREATE VECTOR INDEX scene_embedding IF NOT EXISTS
                    FOR (s:Scene) ON (s.embedding)
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {scene_embedding_dim},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                    """
                    session.run(scene_vector_index)
                    logger.info("  Created vector index for scene embeddings")
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"  Scene vector index not created: {exc}")

        logger.info("Schema initialization complete")

    def execute_query(self, query: str, parameters: Dict = None) -> bool:
        """
        Execute a single Cypher query

        Args:
            query: Cypher query
            parameters: Query parameters

        Returns:
            True if successful
        """
        try:
            with self.driver.session() as session:
                session.run(query, parameters or {})
            return True
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.debug(f"Query: {query}")
            return False

    def execute_queries_batch(
        self, queries: List[str], batch_size: int = Config.BATCH_SIZE
    ) -> int:
        """
        Execute queries in batches

        Args:
            queries: List of Cypher queries
            batch_size: Number of queries per batch

        Returns:
            Number of successfully executed queries
        """
        logger.info(f"Executing {len(queries)} queries in batches of {batch_size}...")

        successful = 0
        failed = 0

        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]

            try:
                with self.driver.session() as session:
                    with session.begin_transaction() as tx:
                        for query in batch:
                            try:
                                tx.run(query)
                                successful += 1
                            except Exception as e:
                                failed += 1
                                logger.warning(f"Query failed: {e}")
                                logger.debug(f"Failed query: {query[:100]}...")

                        tx.commit()

            except Exception as e:
                logger.error(f"Batch transaction failed: {e}")
                failed += len(batch)

            if Config.PROGRESS_LOGGING and (i + batch_size) % (batch_size * 5) == 0:
                logger.info(f"  Processed {i + batch_size}/{len(queries)} queries...")

        logger.info(
            f"Query execution complete: {successful} successful, {failed} failed"
        )

        return successful

    def ingest_entities(self, entities: List[Entity]) -> int:
        """
        Ingest entity nodes into Neo4j

        Args:
            entities: List of entities

        Returns:
            Number of entities ingested
        """
        logger.info(f"Ingesting {len(entities)} entities...")

        successful = 0

        with self.driver.session() as session:
            for entity in entities:
                try:
                    # Prepare embedding
                    embedding = entity.average_embedding
                    if embedding is not None:
                        embedding = embedding.tolist()
                    else:
                        embedding = []

                    # Get first description safely
                    first_desc = ""
                    if entity.appearances and len(entity.appearances) > 0:
                        first_appearance = entity.appearances[0]
                        if isinstance(first_appearance, dict):
                            first_desc = first_appearance.get("rich_description", "")
                        elif hasattr(first_appearance, "rich_description"):
                            first_desc = first_appearance.rich_description

                    scene_memberships = sorted(entity.scene_ids)

                    # Create/merge entity node
                    query = """
                    MERGE (e:Entity {id: $id})
                    SET e.label = $label,
                        e.first_seen = $first_seen,
                        e.last_seen = $last_seen,
                        e.appearance_count = $appearance_count,
                        e.embedding = $embedding,
                        e.first_description = $first_description,
                        e.scenes = $scenes
                    """

                    session.run(
                        query,
                        {
                            "id": entity.entity_id,
                            "label": entity.object_class,
                            "first_seen": entity.first_timestamp,
                            "last_seen": entity.last_timestamp,
                            "appearance_count": (
                                len(entity.appearances) if entity.appearances else 0
                            ),
                            "embedding": embedding,
                            "first_description": (
                                first_desc[:500] if first_desc else ""
                            ),  # Truncate long descriptions
                            "scenes": scene_memberships,
                        },
                    )

                    successful += 1

                except Exception as e:
                    logger.error(f"Failed to ingest entity {entity.entity_id}: {e}")

        logger.info(f"Ingested {successful}/{len(entities)} entities")

        return successful

    def ingest_locations(self, locations: Iterable[LocationProfile]) -> int:
        """Create or update location nodes."""

        location_list = list(locations)
        if not location_list:
            return 0

        logger.info(f"Ingesting {len(location_list)} inferred locations...")
        created = 0

        with self.driver.session() as session:
            for location in location_list:
                try:
                    session.run(
                        """
                        MERGE (loc:Location {id: $id})
                        SET loc.label = $label,
                            loc.signature = $signature,
                            loc.object_classes = $object_classes,
                            loc.scene_count = size($scene_ids)
                        """,
                        {
                            "id": location.location_id,
                            "label": location.label,
                            "signature": location.signature,
                            "object_classes": location.object_classes,
                            "scene_ids": location.scene_ids,
                        },
                    )
                    created += 1
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        f"Failed to ingest location {location.location_id}: {exc}"
                    )

        return created

    def ingest_scenes(self, scenes: Iterable[SceneSegment]) -> int:
        """Create or update scene nodes."""

        scene_list = list(scenes)
        if not scene_list:
            return 0

        logger.info(f"Ingesting {len(scene_list)} scene segments...")
        created = 0

        with self.driver.session() as session:
            for scene in scene_list:
                try:
                    embedding = (
                        scene.embedding.tolist() if scene.embedding is not None else []
                    )
                    session.run(
                        """
                        MERGE (s:Scene {id: $id})
                        SET s.frame_number = $frame_number,
                            s.start_timestamp = $start,
                            s.end_timestamp = $end,
                            s.description = $description,
                            s.object_classes = $object_classes,
                            s.entity_ids = $entity_ids,
                            s.location_id = $location_id,
                            s.embedding = $embedding
                        """,
                        {
                            "id": scene.scene_id,
                            "frame_number": scene.frame_number,
                            "start": scene.start_timestamp,
                            "end": scene.end_timestamp,
                            "description": scene.description[:1000],
                            "object_classes": scene.object_classes,
                            "entity_ids": scene.entity_ids,
                            "location_id": scene.location_id,
                            "embedding": embedding,
                        },
                    )
                    created += 1
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Failed to ingest scene {scene.scene_id}: {exc}")

        return created

    def link_scenes_to_locations(self, scenes: Iterable[SceneSegment]) -> int:
        """Attach scenes to their inferred locations."""

        scene_list = list(scenes)
        if not scene_list:
            return 0

        linked = 0
        with self.driver.session() as session:
            for scene in scene_list:
                try:
                    session.run(
                        """
                        MATCH (s:Scene {id: $scene_id})
                        MATCH (loc:Location {id: $location_id})
                        MERGE (s)-[rel:IN_LOCATION]->(loc)
                        SET rel.object_classes = $object_classes
                        """,
                        {
                            "scene_id": scene.scene_id,
                            "location_id": scene.location_id,
                            "object_classes": scene.object_classes,
                        },
                    )
                    linked += 1
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed to link scene %s to location %s: %s",
                        scene.scene_id,
                        scene.location_id,
                        exc,
                    )

        return linked

    def link_entities_to_scenes(self, perception_log: Iterable[Dict[str, Any]]) -> int:
        """Create appearance relationships between entities and scenes."""

        appearance_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for obj in perception_log:
            entity_id = obj.get("entity_id")
            scene_id = obj.get("scene_id")
            if not entity_id or not scene_id:
                continue
            key = (entity_id, scene_id)
            entry = appearance_map.setdefault(
                key,
                {
                    "count": 0,
                    "first": None,
                    "last": None,
                    "object_classes": set(),
                },
            )
            entry["count"] += 1
            timestamp = float(obj.get("timestamp", 0.0))
            entry["first"] = (
                timestamp if entry["first"] is None else min(entry["first"], timestamp)
            )
            entry["last"] = (
                timestamp if entry["last"] is None else max(entry["last"], timestamp)
            )
            entry["object_classes"].add(obj.get("object_class", "object"))

        if not appearance_map:
            return 0

        with self.driver.session() as session:
            for (entity_id, scene_id), payload in appearance_map.items():
                try:
                    session.run(
                        """
                        MATCH (e:Entity {id: $entity_id})
                        MATCH (s:Scene {id: $scene_id})
                        MERGE (e)-[rel:APPEARS_IN]->(s)
                        SET rel.count = $count,
                            rel.first_timestamp = $first,
                            rel.last_timestamp = $last,
                            rel.object_classes = $object_classes
                        """,
                        {
                            "entity_id": entity_id,
                            "scene_id": scene_id,
                            "count": payload["count"],
                            "first": payload["first"],
                            "last": payload["last"],
                            "object_classes": sorted(payload["object_classes"]),
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed to create APPEARS_IN for entity %s scene %s: %s",
                        entity_id,
                        scene_id,
                        exc,
                    )

        return len(appearance_map)

    def link_scene_transitions(self, scenes: Iterable[SceneSegment]) -> int:
        """Link scenes in chronological order."""

        scene_list = sorted(
            list(scenes),
            key=lambda scene: (scene.start_timestamp, scene.frame_number),
        )
        if len(scene_list) < 2:
            return 0

        transitions = 0
        with self.driver.session() as session:
            for first, second in zip(scene_list, scene_list[1:]):
                try:
                    gap = max(0.0, second.start_timestamp - first.end_timestamp)
                    session.run(
                        """
                        MATCH (a:Scene {id: $from})
                        MATCH (b:Scene {id: $to})
                        MERGE (a)-[rel:TRANSITIONS_TO]->(b)
                        SET rel.gap = $gap,
                            rel.order = $order
                        """,
                        {
                            "from": first.scene_id,
                            "to": second.scene_id,
                            "gap": gap,
                            "order": transitions,
                        },
                    )
                    transitions += 1
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed to link transition %s -> %s: %s",
                        first.scene_id,
                        second.scene_id,
                        exc,
                    )

        return transitions

    def link_scene_similarities(self, similarities: Iterable[SceneSimilarity]) -> int:
        """Create similarity relationships between scenes."""

        similarity_list = list(similarities)
        if not similarity_list:
            return 0

        created = 0
        with self.driver.session() as session:
            for similarity in similarity_list:
                try:
                    session.run(
                        """
                        MATCH (a:Scene {id: $source})
                        MATCH (b:Scene {id: $target})
                        MERGE (a)-[rel:SIMILAR_TO]->(b)
                        SET rel.score = $score
                        """,
                        {
                            "source": similarity.source_id,
                            "target": similarity.target_id,
                            "score": similarity.score,
                        },
                    )
                    session.run(
                        """
                        MATCH (a:Scene {id: $target})
                        MATCH (b:Scene {id: $source})
                        MERGE (a)-[rel:SIMILAR_TO]->(b)
                        SET rel.score = $score
                        """,
                        {
                            "source": similarity.source_id,
                            "target": similarity.target_id,
                            "score": similarity.score,
                        },
                    )
                    created += 2
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed to link similarity between %s and %s: %s",
                        similarity.source_id,
                        similarity.target_id,
                        exc,
                    )

        return created

    def get_graph_statistics(self) -> Dict[str, int]:
        """
        Get graph statistics

        Returns:
            Dictionary of statistics
        """
        stats = {}

        with self.driver.session() as session:
            # Count nodes
            result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            record = result.single()
            stats["entity_nodes"] = record["count"] if record else 0

            result = session.run("MATCH (ev:Event) RETURN count(ev) as count")
            record = result.single()
            stats["event_nodes"] = record["count"] if record else 0

            result = session.run("MATCH (s:Scene) RETURN count(s) as count")
            record = result.single()
            stats["scene_nodes"] = record["count"] if record else 0

            result = session.run("MATCH (loc:Location) RETURN count(loc) as count")
            record = result.single()
            stats["location_nodes"] = record["count"] if record else 0

            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            record = result.single()
            stats["relationships"] = record["count"] if record else 0

            result = session.run(
                "MATCH (:Entity)-[r:APPEARS_IN]->(:Scene) RETURN count(r) as count"
            )
            record = result.single()
            stats["entity_scene_relationships"] = record["count"] if record else 0

            result = session.run(
                "MATCH (:Scene)-[r:IN_LOCATION]->(:Location) RETURN count(r) as count"
            )
            record = result.single()
            stats["scene_location_relationships"] = record["count"] if record else 0

            result = session.run(
                "MATCH (:Scene)-[r:TRANSITIONS_TO]->(:Scene) RETURN count(r) as count"
            )
            record = result.single()
            stats["scene_transition_relationships"] = record["count"] if record else 0

            result = session.run(
                "MATCH (:Scene)-[r:SIMILAR_TO]->(:Scene) RETURN count(r) as count"
            )
            record = result.single()
            stats["scene_similarity_relationships"] = record["count"] if record else 0

        return stats


# ============================================================================
# MAIN SEMANTIC UPLIFT PIPELINE
# ============================================================================


def run_semantic_uplift(
    perception_log: List[Dict[str, Any]],
    neo4j_driver=None,
    neo4j_uri: str = None,
    neo4j_user: str = None,
    neo4j_password: str = None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Main semantic uplift pipeline

    Args:
        perception_log: List of RichPerceptionObject dictionaries from Part 1
        neo4j_driver: Optional pre-configured Neo4j driver
        neo4j_uri: Neo4j URI (if driver not provided)
        neo4j_user: Neo4j username (if driver not provided)
        neo4j_password: Neo4j password (if driver not provided)
        progress_callback: Optional callable to receive progress events

    Returns:
        Dictionary with uplift results and statistics
    """
    logger.info("\n" + "=" * 80)
    logger.info("SEMANTIC UPLIFT ENGINE - PART 2")
    logger.info("=" * 80)
    logger.info(f"Processing {len(perception_log)} perception objects")

    def emit(event: str, payload: Dict[str, Any]) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(event, payload)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Progress callback error (%s): %s", event, exc)

    total_steps = 15
    emit(
        "semantic.start",
        {
            "total": total_steps,
            "message": f"Processing {len(perception_log)} perception objects",
        },
    )

    step_index = 0
    current_step_name: Optional[str] = None

    def start_step(name: str, message: str) -> int:
        nonlocal step_index, current_step_name
        step_index += 1
        current_step_name = name
        emit(
            "semantic.step.start",
            {
                "name": name,
                "message": message,
                "index": step_index,
                "total": total_steps,
            },
        )
        logger.info("\n[%d/%d] %s", step_index, total_steps, message)
        return step_index

    def complete_step(index: int, name: str, detail: Optional[str] = None) -> None:
        nonlocal current_step_name
        emit(
            "semantic.step.complete",
            {
                "name": name,
                "detail": detail,
                "index": index,
                "total": total_steps,
            },
        )
        current_step_name = None

    start_time = time.time()
    results = {
        "success": False,
        "num_entities": 0,
        "num_scenes": 0,
        "num_locations": 0,
        "num_scene_similarity_links": 0,
        "num_state_changes": 0,
        "num_windows": 0,
        "num_causal_links": 0,
        "windows_with_causal": 0,
        "causal_engine_calls": 0,
        "causal_candidate_pairs": 0,
        "causal_pairs_retained": 0,
        "num_queries": 0,
        "llm_windows_total": 0,
        "llm_windows_composed": 0,
        "llm_windows_skipped": 0,
        "llm_windows_capped": 0,
        "llm_calls": 0,
        "llm_latency_seconds": 0.0,
        "graph_stats": {},
    }

    graph_builder: Optional[KnowledgeGraphBuilder] = None
    close_builder = False
    scene_embedding_dim: Optional[int] = None
    cypher_queries: List[str] = []
    scene_lookup: Dict[str, SceneSegment] = {}

    try:
        step_id = start_step("entity_tracking", "Tracking entities across time")
        tracker = EntityTracker()
        tracker.track_entities(perception_log)
        entities = tracker.get_all_entities()
        results["num_entities"] = len(entities)
        detail_msg = f"Tracked {results['num_entities']} entities"
        logger.info(detail_msg)
        complete_step(step_id, "entity_tracking", detail_msg)

        step_id = start_step("scene_assembly", "Assembling scenes and locations")
        scene_assembler = SceneAssembler(tracker)
        scenes, locations = scene_assembler.build_scenes(perception_log)
        results["num_scenes"] = len(scenes)
        results["num_locations"] = len(locations)
        scene_embedding_dim = scene_assembler.embedding_dim
        scene_lookup = {scene.scene_id: scene for scene in scenes}
        detail_msg = (
            f"Assembled {results['num_scenes']} scenes and {results['num_locations']} locations"
        )
        logger.info(detail_msg)
        complete_step(step_id, "scene_assembly", detail_msg)

        step_id = start_step("scene_similarity", "Computing scene similarity links")
        scene_similarities = SceneAssembler.compute_similarities(
            scenes,
            threshold=Config.SCENE_SIMILARITY_THRESHOLD,
            top_k=Config.SCENE_SIMILARITY_TOP_K,
        )
        results["num_scene_similarity_links"] = len(scene_similarities)
        detail_msg = (
            f"Computed {results['num_scene_similarity_links']} scene similarity links"
        )
        logger.info(detail_msg)
        complete_step(step_id, "scene_similarity", detail_msg)

        step_id = start_step("state_changes", "Detecting entity state changes")
        detector = StateChangeDetector()
        state_changes = detector.detect_all_state_changes(entities)
        results["num_state_changes"] = len(state_changes)
        detail_msg = f"Detected {results['num_state_changes']} state changes"
        logger.info(detail_msg)
        complete_step(step_id, "state_changes", detail_msg)

        step_id = start_step("temporal_windows", "Grouping activity into temporal windows")
        windows = create_temporal_windows(state_changes)
        results["num_windows"] = len(windows)
        detail_msg = f"Windowed activity into {results['num_windows']} intervals"
        logger.info(detail_msg)
        complete_step(step_id, "temporal_windows", detail_msg)

        step_id = start_step("causal_scoring", "Scoring causal links")
        causal_scorer = CausalInfluenceScorer(tracker)
        causal_stats = causal_scorer.score_windows(windows)
        results["num_causal_links"] = causal_stats.get("total_links", 0)
        results["windows_with_causal"] = causal_stats.get("windows_with_links", 0)
        results["causal_engine_calls"] = causal_stats.get("engine_calls", 0)
        results["causal_candidate_pairs"] = causal_stats.get("candidate_pairs", 0)
        results["causal_pairs_retained"] = causal_stats.get("accepted_pairs", 0)
        detail_msg = (
            f"Scored {results['num_causal_links']} causal links across "
            f"{results['windows_with_causal']} windows"
        )
        logger.info(detail_msg)
        complete_step(step_id, "causal_scoring", detail_msg)

        step_id = start_step(
            "event_composition", "Composing events and Cypher queries"
        )
        composer = EventComposer(scene_lookup=scene_lookup, location_lookup=locations)
        cypher_queries = composer.compose_all_events(windows, tracker)
        results["num_queries"] = len(cypher_queries)
        results["llm_windows_total"] = composer.metrics.get("windows_total", 0)
        results["llm_windows_composed"] = composer.metrics.get("windows_composed", 0)
        results["llm_windows_skipped"] = composer.metrics.get("windows_skipped", 0)
        results["llm_windows_capped"] = composer.metrics.get("windows_capped", 0)
        results["llm_calls"] = composer.metrics.get("llm_calls", 0)
        results["llm_latency_seconds"] = round(
            composer.metrics.get("llm_latency", 0.0), 2
        )
        detail_msg = f"Prepared {results['num_queries']} Cypher queries"
        detail_msg += (
            f" (windows composed: {results['llm_windows_composed']}, "
            f"skipped: {results['llm_windows_skipped']}, "
            f"budgeted: {results['llm_windows_capped']})"
        )
        logger.info(detail_msg)
        complete_step(step_id, "event_composition", detail_msg)

        step_id = start_step("neo4j_connection", "Connecting to Neo4j")
        if neo4j_driver:
            graph_builder = KnowledgeGraphBuilder()
            graph_builder.driver = neo4j_driver
            connection_detail = "Using provided Neo4j driver"
        else:
            graph_builder = KnowledgeGraphBuilder(
                neo4j_uri or Config.NEO4J_URI,
                neo4j_user or Config.NEO4J_USER,
                neo4j_password or Config.NEO4J_PASSWORD,
            )
            if not graph_builder.connect():
                message = "Failed to connect to Neo4j"
                emit("semantic.error", {"message": message, "step": "neo4j_connection"})
                raise RuntimeError(message)
            close_builder = True
            connection_detail = f"Connected to Neo4j at {graph_builder.uri}"
        logger.info(connection_detail)
        complete_step(step_id, "neo4j_connection", connection_detail)

        step_id = start_step("neo4j_schema", "Initializing Neo4j schema")
        if graph_builder is None:
            raise RuntimeError("KnowledgeGraphBuilder not initialized")
        graph_builder.initialize_schema(scene_embedding_dim=scene_embedding_dim)
        if scene_embedding_dim:
            detail_msg = f"Schema initialized (embedding dim {scene_embedding_dim})"
        else:
            detail_msg = "Schema initialized"
        logger.info(detail_msg)
        complete_step(step_id, "neo4j_schema", detail_msg)

        step_id = start_step("neo4j_ingest_entities", "Ingesting entity profiles")
        emit(
            "semantic.progress",
            {"message": f"Ingesting {results['num_entities']} entities into Neo4j"},
        )
        graph_builder.ingest_entities(entities)
        detail_msg = f"Ingested {results['num_entities']} entities"
        logger.info(detail_msg)
        complete_step(step_id, "neo4j_ingest_entities", detail_msg)

        step_id = start_step("neo4j_ingest_locations", "Ingesting inferred locations")
        emit(
            "semantic.progress",
            {"message": f"Ingesting {results['num_locations']} locations"},
        )
        graph_builder.ingest_locations(locations.values())
        detail_msg = f"Ingested {results['num_locations']} locations"
        logger.info(detail_msg)
        complete_step(step_id, "neo4j_ingest_locations", detail_msg)

        step_id = start_step("neo4j_ingest_scenes", "Ingesting scene segments")
        emit(
            "semantic.progress",
            {"message": f"Ingesting {results['num_scenes']} scenes"},
        )
        graph_builder.ingest_scenes(scenes)
        detail_msg = f"Ingested {results['num_scenes']} scenes"
        logger.info(detail_msg)
        complete_step(step_id, "neo4j_ingest_scenes", detail_msg)

        step_id = start_step(
            "neo4j_link_relationships", "Linking scenes, entities, and transitions"
        )
        emit("semantic.progress", {"message": "Linking scenes to inferred locations"})
        graph_builder.link_scenes_to_locations(scenes)
        emit("semantic.progress", {"message": "Linking entities to scenes"})
        graph_builder.link_entities_to_scenes(perception_log)
        emit(
            "semantic.progress", {"message": "Linking scene transitions and similarities"}
        )
        graph_builder.link_scene_transitions(scenes)
        graph_builder.link_scene_similarities(scene_similarities)
        detail_msg = "Scene relationships linked"
        logger.info(detail_msg)
        complete_step(step_id, "neo4j_link_relationships", detail_msg)

        step_id = start_step("neo4j_execute_queries", "Executing generated Cypher queries")
        if cypher_queries:
            emit(
                "semantic.progress",
                {"message": f"Executing {len(cypher_queries)} Cypher queries"},
            )
            graph_builder.execute_queries_batch(cypher_queries)
        else:
            emit("semantic.progress", {"message": "No Cypher queries to execute"})
        detail_msg = f"Executed {len(cypher_queries)} generated queries"
        logger.info(detail_msg)
        complete_step(step_id, "neo4j_execute_queries", detail_msg)

        step_id = start_step("neo4j_graph_stats", "Collecting graph statistics")
        graph_stats = graph_builder.get_graph_statistics()
        results["graph_stats"] = graph_stats
        detail_msg = "Graph statistics collected"
        logger.info(detail_msg)
        if graph_stats:
            for key, value in graph_stats.items():
                logger.info("  %s: %s", key, value)
        complete_step(step_id, "neo4j_graph_stats", detail_msg)

        results["success"] = True

    except Exception as exc:
        emit(
            "semantic.error",
            {
                "message": str(exc),
                "step": current_step_name or "semantic",
            },
        )
        raise
    finally:
        if graph_builder is not None and close_builder:
            graph_builder.close()

    elapsed_time = time.time() - start_time

    logger.info("\n" + "=" * 80)
    logger.info("SEMANTIC UPLIFT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Entities tracked: {results['num_entities']}")
    logger.info(f"Scenes segmented: {results['num_scenes']}")
    logger.info(f"Locations inferred: {results['num_locations']}")
    logger.info(f"Scene similarity links: {results['num_scene_similarity_links']}")
    logger.info(f"State changes detected: {results['num_state_changes']}")
    logger.info(f"Temporal windows: {results['num_windows']}")
    logger.info(
        f"Causal links scored: {results['num_causal_links']} "
        f"across {results['windows_with_causal']} windows"
    )
    logger.info(
        "Causal engine calls: %s (pairs considered: %s, retained: %s)",
        results["causal_engine_calls"],
        results["causal_candidate_pairs"],
        results["causal_pairs_retained"],
    )
    logger.info(
        "Event windows composed: %s/%s (skipped: %s, capped: %s)",
        results["llm_windows_composed"],
        results["llm_windows_total"],
        results["llm_windows_skipped"],
        results["llm_windows_capped"],
    )
    logger.info(
        "LLM calls: %s (latency %.2fs)",
        results["llm_calls"],
        results["llm_latency_seconds"],
    )
    logger.info(f"Cypher queries generated: {results['num_queries']}")
    logger.info(f"Total time: {elapsed_time:.2f}s")
    logger.info("=" * 80 + "\n")

    emit(
        "semantic.complete",
        {
            "message": f"Semantic uplift complete ({results['num_entities']} entities)",
            "duration": elapsed_time,
            "statistics": {
                "entities": results["num_entities"],
                "scenes": results["num_scenes"],
                "locations": results["num_locations"],
                "similarities": results["num_scene_similarity_links"],
                "state_changes": results["num_state_changes"],
                "causal_links": results["num_causal_links"],
                "causal_engine_calls": results["causal_engine_calls"],
                "queries": results["num_queries"],
                "llm_calls": results["llm_calls"],
                "llm_windows_composed": results["llm_windows_composed"],
            },
        },
    )

    return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Load perception log from Part 1
    PERCEPTION_LOG_PATH = "/Users/riddhiman.rana/Desktop/Coding/Orion/orion-research/data/testing/perception_log.json"

    if not os.path.exists(PERCEPTION_LOG_PATH):
        logger.error(f"Perception log not found: {PERCEPTION_LOG_PATH}")
        logger.info("Please run Part 1 first to generate a perception log")
        sys.exit(1)

    try:
        with open(PERCEPTION_LOG_PATH, "r") as f:
            perception_log = json.load(f)

        logger.info(f"Loaded perception log with {len(perception_log)} objects")

        # Run semantic uplift
        results = run_semantic_uplift(
            perception_log,
            neo4j_uri=Config.NEO4J_URI,
            neo4j_user=Config.NEO4J_USER,
            neo4j_password=Config.NEO4J_PASSWORD,
        )

        # Save results
        results_path = os.path.join(
            os.path.dirname(PERCEPTION_LOG_PATH), "uplift_results.json"
        )
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {results_path}")

        if results["success"]:
            logger.info("\n✅ Semantic uplift completed successfully!")
        else:
            logger.warning("\n⚠️ Semantic uplift completed with errors")

    except Exception as e:
        logger.error(f"Semantic uplift failed: {e}", exc_info=True)
        sys.exit(1)
