"""
Semantic Engine
================

High-level orchestrator for Phase 2 (Semantic Understanding).

Coordinates all semantic analysis components:
- Entity tracking
- State change detection
- Scene assembly
- Temporal windowing
- Causal reasoning
- Event composition
- Graph ingestion

Author: Orion Research Team
Date: October 2025
"""

import logging
import time
from collections import Counter
from typing import Dict, List, Optional

from orion.perception.types import PerceptionResult
from orion.semantic.types import SemanticResult, SemanticEntity, Event
from orion.semantic.config import SemanticConfig

from orion.semantic.entity_tracker import SemanticEntityTracker
from orion.semantic.state_detector import StateChangeDetector
from orion.semantic.scene_assembler import SceneAssembler
from orion.semantic.temporal_windows import TemporalWindowManager
from orion.semantic.causal_scorer import CausalInfluenceScorer
from orion.semantic.event_composer import EventComposer
from orion.graph.builder import GraphBuilder
from orion.ollama_client import OllamaClient
from orion.semantic.spatial_utils import (
    extract_spatial_features,
    cluster_entities_hdbscan,
    label_zone,
    compute_zone_centroid,
    compute_zone_bbox,
    compute_zone_relationships,
    SpatialZone,
)

logger = logging.getLogger(__name__)


class SemanticEngine:
    """
    High-level semantic understanding engine.
    
    Orchestrates the complete Phase 2 pipeline:
    Perception Results → Semantic Understanding → Knowledge Graph
    """
    
    def __init__(
        self,
        config: Optional[SemanticConfig] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        """
        Initialize semantic engine.
        
        Args:
            config: Semantic configuration (uses defaults if None)
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.config = config or SemanticConfig()
        
        # Get shared model manager
        from orion.managers.model_manager import ModelManager
        self.model_manager = ModelManager.get_instance()
        
        # Initialize components with shared models
        self.entity_tracker = SemanticEntityTracker(self.config.state_change)
        
        self.ollama_client = OllamaClient()
        
        # Pass CLIP to state detector for text embeddings
        self.state_detector = StateChangeDetector(
            embedding_model=self.model_manager.clip,  # Use shared CLIP instance
            config=self.config.state_change,
        )
        
        self.scene_assembler = SceneAssembler(self.config.temporal_window)
        self.window_manager = TemporalWindowManager(self.config.temporal_window)
        self.causal_scorer = CausalInfluenceScorer(self.config.causal)
        self.event_composer = EventComposer(self.config.event_composition)
        self.graph_builder: Optional[GraphBuilder] = None
        
        if self.config.enable_graph_ingestion:
            self.graph_builder = GraphBuilder(
                uri=neo4j_uri, user=neo4j_user, password=neo4j_password
            )
        
        logger.info("SemanticEngine initialized")
        if self.config.verbose:
            logger.info(f"  State change threshold: {self.config.state_change.embedding_similarity_threshold}")
            logger.info(f"  Temporal window max duration: {self.config.temporal_window.max_duration_seconds}s")
            logger.info(f"  CIS threshold: {self.config.causal.cis_threshold}")
            logger.info(f"  LLM model: {self.config.event_composition.model}")
            logger.info(f"  Graph ingestion: {self.config.enable_graph_ingestion}")
    
    def process(
        self,
        perception_result: PerceptionResult,
        video_path: Optional[str] = None,
    ) -> SemanticResult:
        """
        Run complete semantic understanding pipeline.
        
        Args:
            perception_result: Output from perception engine
            video_path: Optional path to source video
            
        Returns:
            Semantic understanding result
        """
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: SEMANTIC UNDERSTANDING ENGINE")
        logger.info("="*80)
        
        start_time = time.time()
        
        # Log input statistics
        logger.info(f"\nInput: {perception_result.total_detections} detections across {perception_result.total_frames} frames")
        
        # Step 1: Track entities across time
        logger.info("\n[1/8] Consolidating semantic entities...")
        step_start = time.time()
        entities = self.entity_tracker.consolidate_entities(perception_result)
        elapsed = time.time() - step_start
        logger.info(f"  ✓ Consolidated {len(entities)} semantic entities ({elapsed:.2f}s)")
        
        if self.config.verbose and len(entities) > 0:
            class_counts = Counter(e.class_label for e in entities)
            logger.info(f"    Classes detected: {dict(class_counts)}")
            logger.info(f"    Avg detections/entity: {perception_result.total_detections/len(entities):.1f}")
        
        # Step 1b: Generate descriptions with Ollama
        logger.info("\n[1b/8] Generating entity descriptions with Ollama...")
        step_start = time.time()
        for entity in entities:
            prompt = f"Describe the following entity: {entity.class_label}"
            description = self.ollama_client.generate(prompt)
            entity.description = description
        elapsed = time.time() - step_start
        logger.info(f"  ✓ Generated descriptions for {len(entities)} entities ({elapsed:.2f}s)")
        
        # Step 2: Detect spatial zones (NEW - Phase 2)
        logger.info("\n[2/8] Detecting spatial zones...")
        step_start = time.time()
        spatial_zones = self._detect_spatial_zones(entities)
        elapsed = time.time() - step_start
        logger.info(f"  ✓ Detected {len(spatial_zones)} spatial zones ({elapsed:.2f}s)")
        
        # Step 3: Detect state changes
        logger.info("\n[3/8] Detecting state changes...")
        step_start = time.time()
        state_changes = self.state_detector.detect_changes(entities)
        elapsed = time.time() - step_start
        logger.info(f"  ✓ Detected {len(state_changes)} state changes ({elapsed:.2f}s)")
        
        if self.config.verbose and state_changes:
            logger.info(f"    State changes will be used for temporal windowing")
        
        # Step 4: Assemble scenes
        logger.info("\n[4/8] Assembling scenes...")
        step_start = time.time()
        # Create temporal windows first (needed for scenes)
        windows = self.window_manager.create_windows(
            state_changes,
            total_duration=perception_result.duration_seconds,
        )
        scenes = self.scene_assembler.assemble_scenes(windows)
        elapsed = time.time() - step_start
        logger.info(f"  ✓ Assembled {len(scenes)} scenes from {len(windows)} windows ({elapsed:.2f}s)")
        
        # Step 5: Compute causal links
        logger.info("\n[5/8] Computing causal influence scores...")
        step_start = time.time()
        # Build embedding lookup
        embeddings = {
            e.entity_id: e.average_embedding
            for e in entities
            if e.average_embedding is not None
        }
        causal_links = self.causal_scorer.compute_causal_links(
            state_changes,
            embeddings,
        )
        elapsed = time.time() - step_start
        logger.info(f"  ✓ Identified {len(causal_links)} causal links ({elapsed:.2f}s)")
        
        # Add causal links to windows
        for link in causal_links:
            for window in windows:
                # Check if both state changes are in this window
                change_ids = {change.entity_id for change in window.state_changes}
                if link.agent_id in change_ids and link.patient_id in change_ids:
                    window.causal_links.append(link)
        
        if self.config.verbose and causal_links:
            logger.info(f"    Added causal links to scenes")
        
        # Step 6: Compose events
        logger.info("\n[6/8] Composing events...")
        step_start = time.time()
        events = self.event_composer.compose_events(windows)
        elapsed = time.time() - step_start
        logger.info(f"  ✓ Composed {len(events)} events ({elapsed:.2f}s)")
        
        # Link events to scenes
        for event in events:
            for scene in scenes:
                if scene.start_time <= event.start_time <= scene.end_time:
                    event.scene_id = scene.segment_id
                    break
        
        # Step 7: Build knowledge graph (if enabled)
        graph_stats = {}
        if self.config.enable_graph_ingestion and self.graph_builder:
            logger.info("\n[7/8] Ingesting into knowledge graph...")
            step_start = time.time()
            try:
                # Build locations dict
                locations = {
                    scene.segment_id: scene.location_profile
                    for scene in scenes
                    if scene.location_profile
                }
                graph_stats = self.graph_builder.ingest_semantic_results(
                    entities,
                    events,
                    scenes,
                    locations,
                    causal_links,
                )
                elapsed = time.time() - step_start
                logger.info(f"  ✓ Created {graph_stats.get('entities', 0)} entity nodes, "
                           f"{graph_stats.get('events', 0)} event nodes ({elapsed:.2f}s)")
            except Exception as e:
                logger.error(f"  Graph ingestion failed: {e}")
        else:
            logger.info("\n[7/8] Skipping knowledge graph ingestion (disabled)")

        
        # Step 8: Package results
        logger.info("\n[8/8] Packaging results...")
        
        # Build locations dict
        locations = {
            scene.segment_id: scene.location_profile
            for scene in scenes
            if scene.location_profile
        }
        
        result = SemanticResult(
            entities=entities,
            state_changes=state_changes,
            temporal_windows=windows,
            scenes=scenes,
            locations=locations,
            events=events,
            causal_links=causal_links,
            graph_stats=graph_stats,
        )
        
        elapsed = time.time() - start_time
        
        logger.info("\n" + "="*80)
        logger.info("SEMANTIC UNDERSTANDING COMPLETE")
        logger.info("="*80)
        logger.info(f"Entities: {len(entities)}")
        logger.info(f"Spatial Zones: {len(spatial_zones)}")  # NEW
        logger.info(f"State Changes: {len(state_changes)}")
        logger.info(f"Temporal Windows: {len(windows)}")
        logger.info(f"Scenes: {len(scenes)}")
        logger.info(f"Causal Links: {len(causal_links)}")
        logger.info(f"Events: {len(events)}")
        logger.info(f"Processing Time: {elapsed:.2f}s")
        logger.info("="*80 + "\n")
        
        return result
    
    def _detect_spatial_zones(self, entities: List[SemanticEntity]) -> List[SpatialZone]:
        """
        Detect spatial zones using HDBSCAN clustering.
        
        Args:
            entities: Consolidated semantic entities
            
        Returns:
            List of detected spatial zones
        """
        if len(entities) < self.config.spatial.min_cluster_size:
            logger.warning(
                f"Not enough entities ({len(entities)}) for spatial zone detection "
                f"(min={self.config.spatial.min_cluster_size})"
            )
            return []
        
        # Extract features
        features = extract_spatial_features(
            entities,
            feature_weights=self.config.spatial.feature_weights,
        )
        
        if len(features) == 0:
            return []
        
        # Cluster entities
        cluster_labels, cluster_probabilities = cluster_entities_hdbscan(
            features,
            min_cluster_size=self.config.spatial.min_cluster_size,
            min_samples=self.config.spatial.min_samples,
        )
        cluster_labels = list(cluster_labels)
        cluster_probabilities = list(cluster_probabilities)
        if len(cluster_probabilities) < len(cluster_labels):
            cluster_probabilities.extend([0.0] * (len(cluster_labels) - len(cluster_probabilities)))
        
        # Build zones from clusters
        zones: List[SpatialZone] = []
        label_to_zone: Dict[int, SpatialZone] = {}
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)  # Remove noise label
        
        for label_idx in unique_labels:
            indices = [i for i, label in enumerate(cluster_labels) if label == label_idx]
            cluster_entities = [entities[i] for i in indices]
            
            if not cluster_entities:
                continue
            
            # Compute zone properties
            entity_ids = [e.entity_id for e in cluster_entities]
            entity_classes = [str(e.class_label) for e in cluster_entities]
            class_counter = Counter(entity_classes)
            dominant_classes = [label for label, _ in class_counter.most_common(3)]
            zone_probability_values = [cluster_probabilities[i] for i in indices]
            zone_confidence = (
                float(sum(zone_probability_values) / len(zone_probability_values))
                if zone_probability_values
                else 0.0
            )
            
            centroid = compute_zone_centroid(cluster_entities)
            bbox = compute_zone_bbox(cluster_entities)
            zone_label = label_zone(entity_classes, centroid)
            summary = f"{zone_label.replace('_', ' ')} ({len(entity_ids)} entities)"
            
            # Create zone
            zone = SpatialZone(
                zone_id=f"zone_{label_idx}",
                label=zone_label,
                entity_ids=entity_ids,
                centroid=centroid,
                bounding_box=bbox,
                confidence=zone_confidence,
                dominant_classes=dominant_classes,
                summary=summary,
            )
            
            zones.append(zone)
            label_to_zone[label_idx] = zone
            
            logger.debug(
                f"  Zone {zone.zone_id} ({zone.label}): "
                f"{len(entity_ids)} entities - {', '.join(entity_classes[:3])}"
                + ("..." if len(entity_classes) > 3 else "")
            )
        
        # Compute relationships between zones
        if len(zones) > 1:
            compute_zone_relationships(zones)
        
        # Enrich entities with zone information
        for idx, entity in enumerate(entities):
            label = cluster_labels[idx]
            confidence = cluster_probabilities[idx]
            if label == -1 or label not in label_to_zone:
                continue
            zone = label_to_zone[label]
            entity.zone_id = zone.zone_id
            entity.zone_label = zone.label
            entity.zone_confidence = confidence
            neighbors = [
                e_id for e_id in zone.entity_ids if e_id != entity.entity_id
            ]
            entity.neighbor_entity_ids = neighbors

        return zones
    
    def close(self) -> None:
        """Clean up resources"""
        if self.graph_builder:
            self.graph_builder.close()


def run_semantic(
    perception_result: PerceptionResult,
    config: Optional[SemanticConfig] = None,
    video_path: Optional[str] = None,
) -> SemanticResult:
    """
    Convenience function to run semantic understanding.
    
    Args:
        perception_result: Output from perception engine
        config: Semantic configuration (optional)
        video_path: Path to source video (optional)
        
    Returns:
        Semantic understanding result
    """
    engine = SemanticEngine(config)
    try:
        return engine.process(perception_result, video_path)
    finally:
        engine.close()
