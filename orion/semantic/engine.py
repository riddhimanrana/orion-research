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

# Initialize logger early so optional import warnings work
logger = logging.getLogger(__name__)

from orion.perception.types import PerceptionResult
from orion.semantic.types import SemanticResult, SemanticEntity, Event
from orion.semantic.config import SemanticConfig

try:
    from orion.semantic.entity_tracker import SemanticEntityTracker  # archived optional
except Exception:
    SemanticEntityTracker = None  # type: ignore
try:
    from orion.semantic.state_detector import StateChangeDetector  # archived optional
except Exception:
    StateChangeDetector = None  # type: ignore
try:
    from orion.semantic.scene_assembler import SceneAssembler  # archived optional
except Exception:
    SceneAssembler = None  # type: ignore
try:
    from orion.semantic.temporal_windows import TemporalWindowManager  # archived optional
except Exception:
    TemporalWindowManager = None  # type: ignore
try:
    from orion.semantic.causal_scorer import CausalInfluenceScorer  # archived optional
except Exception:
    CausalInfluenceScorer = None  # type: ignore
try:
    from orion.semantic.cis_scorer_3d import CausalInfluenceScorer3D  # archived optional
except Exception:
    CausalInfluenceScorer3D = None  # type: ignore
try:
    from orion.semantic.event_composer import EventComposer  # archived optional
except Exception:
    EventComposer = None  # type: ignore
try:
    from orion.semantic.temporal_description_generator import TemporalDescriptionGenerator  # archived optional
except Exception:
    TemporalDescriptionGenerator = None  # type: ignore
from orion.graph.builder import GraphBuilder
from orion.semantic.spatial_utils import (
    extract_spatial_features,
    cluster_entities_hdbscan,
    label_zone,
    compute_zone_centroid,
    compute_zone_bbox,
    compute_zone_relationships,
    SpatialZone,
)

# (logger already initialized above for early import warnings)


class SemanticEngine:
    """
    High-level semantic understanding engine.
    
    Orchestrates the complete Phase 2 pipeline:
    Perception Results → Semantic Understanding → Knowledge Graph
    """
    
    def __init__(self, config: Optional[SemanticConfig] = None):
        """
        Initialize semantic engine.
        
        Args:
            config: Semantic configuration (uses defaults if None)
        """
        self.config = config or SemanticConfig()
        
        # Get shared model manager
        from orion.managers.model_manager import ModelManager
        self.model_manager = ModelManager.get_instance()
        
        # Initialize components with shared models
        self.entity_tracker = (
            SemanticEntityTracker(self.config.state_change)
            if SemanticEntityTracker is not None else None
        )
        
        # Pass CLIP and FastVLM to description generator
        self.description_generator = (
            TemporalDescriptionGenerator(
                clip_model=self.model_manager.clip,
                vlm_model=self.model_manager.fastvlm,
                sample_interval=1.0,
                min_samples_per_entity=2,
            ) if TemporalDescriptionGenerator is not None else None
        )
        
        # Pass CLIP to state detector for text embeddings
        self.state_detector = (
            StateChangeDetector(
                embedding_model=self.model_manager.clip,
                config=self.config.state_change,
            ) if StateChangeDetector is not None else None
        )
        
        self.scene_assembler = SceneAssembler(self.config.temporal_window) if SceneAssembler is not None else None
        self.window_manager = TemporalWindowManager(self.config.temporal_window) if TemporalWindowManager is not None else None
        
        # Initialize CIS scorer (2D or 3D based on config)
        if CausalInfluenceScorer3D is not None and self.config.causal.use_3d_cis:
            self.causal_scorer = CausalInfluenceScorer3D(
                weight_temporal=self.config.causal.weight_temporal,
                weight_spatial=self.config.causal.weight_spatial,
                weight_motion=self.config.causal.weight_motion,
                weight_semantic=self.config.causal.weight_semantic,
                temporal_decay_tau=self.config.causal.temporal_decay_tau,
                max_spatial_distance_mm=self.config.causal.max_spatial_distance_mm,
                hand_grasping_bonus=self.config.causal.hand_grasping_bonus,
                hand_touching_bonus=self.config.causal.hand_touching_bonus,
                hand_near_bonus=self.config.causal.hand_near_bonus,
                cis_threshold=self.config.causal.cis_threshold,
            )
            logger.info("Using 3D CIS scorer with SLAM + depth")
        elif CausalInfluenceScorer is not None:
            self.causal_scorer = CausalInfluenceScorer(self.config.causal)
            logger.info("Using 2D CIS scorer (legacy)")
        else:
            self.causal_scorer = None
            logger.warning("No causal scorer available; causal links disabled")
        
        self.event_composer = EventComposer(self.config.event_composition) if EventComposer is not None else None
        self.graph_builder: Optional[GraphBuilder] = None
        
        if self.config.enable_graph_ingestion:
            self.graph_builder = GraphBuilder()
        
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
        logger.info(f"\n[1/8] Consolidating semantic entities...")
        step_start = time.time()
        if self.entity_tracker is not None:
            entities = self.entity_tracker.consolidate_entities(perception_result)
        else:
            entities = []
            logger.warning("Entity tracker unavailable; skipping consolidation")
        elapsed = time.time() - step_start
        logger.info(f"  ✓ Consolidated {len(entities)} semantic entities ({elapsed:.2f}s)")
        
        if self.config.verbose and len(entities) > 0:
            class_counts = Counter(e.object_class for e in entities)
            logger.info(f"    Classes detected: {dict(class_counts)}")
            logger.info(f"    Avg detections/entity: {perception_result.total_detections/len(entities):.1f}")
        
        # Step 1b: Generate temporal descriptions (NEW - critical for state changes)
        logger.info(f"\n[1b/8] Generating temporal descriptions...")
        step_start = time.time()
        if self.description_generator is not None and entities:
            self.description_generator.generate_temporal_descriptions(entities)
        else:
            logger.warning("Description generator unavailable or no entities; skipping descriptions")
        elapsed = time.time() - step_start
        total_descriptions = sum(len(e.descriptions) for e in entities) if entities else 0
        logger.info(f"  ✓ Generated {total_descriptions} temporal descriptions ({elapsed:.2f}s)")
        
        if self.config.verbose and len(entities) > 0:
            avg_desc = total_descriptions / len(entities)
            logger.info(f"    Avg descriptions/entity: {avg_desc:.1f}")
        
        # Step 2: Detect spatial zones (NEW - Phase 2)
        logger.info("\n[2/8] Detecting spatial zones...")
        step_start = time.time()
        spatial_zones = self._detect_spatial_zones(entities) if entities else []
        elapsed = time.time() - step_start
        logger.info(f"  ✓ Detected {len(spatial_zones)} spatial zones ({elapsed:.2f}s)")
        
        # Step 3: Detect state changes
        logger.info("\n[3/8] Detecting state changes...")
        step_start = time.time()
        if self.state_detector is not None and entities:
            state_changes = self.state_detector.detect_changes(entities)
        else:
            state_changes = []
            logger.warning("State detector unavailable or no entities; skipping state change detection")
        elapsed = time.time() - step_start
        logger.info(f"  ✓ Detected {len(state_changes)} state changes ({elapsed:.2f}s)")
        
        if self.config.verbose and state_changes:
            logger.info(f"    State changes will be used for temporal windowing")
        
        # Step 4: Assemble scenes
        logger.info("\n[4/8] Assembling scenes...")
        step_start = time.time()
        # Create temporal windows first (needed for scenes)
        if self.window_manager is not None:
            windows = self.window_manager.create_windows(
                state_changes,
                total_duration=perception_result.duration_seconds,
            )
        else:
            windows = []
            logger.warning("Temporal window manager unavailable; skipping window creation")
        if self.scene_assembler is not None and windows:
            scenes = self.scene_assembler.assemble_scenes(windows)
        else:
            scenes = []
            logger.warning("Scene assembler unavailable or no windows; skipping scene assembly")
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
        if self.causal_scorer is not None and state_changes:
            causal_links = self.causal_scorer.compute_causal_links(
                state_changes,
                embeddings,
            )
        else:
            causal_links = []
            logger.warning("Causal scorer unavailable or no state changes; skipping causal link computation")
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
        if self.event_composer is not None and windows:
            events = self.event_composer.compose_events(windows)
        else:
            events = []
            logger.warning("Event composer unavailable or no windows; skipping event composition")
        elapsed = time.time() - step_start
        logger.info(f"  ✓ Composed {len(events)} events ({elapsed:.2f}s)")
        
        # Link events to scenes
        for event in events:
            for scene in scenes:
                if scene.start_time <= event.start_timestamp <= scene.end_time:
                    event.scene_id = scene.scene_id
                    break
        
        # Step 7: Build knowledge graph (if enabled)
        graph_stats = {}
        if self.config.enable_graph_ingestion and self.graph_builder:
            logger.info("\n[7/8] Ingesting into knowledge graph...")
            step_start = time.time()
            try:
                locations = {
                    scene.scene_id: scene.location_profile
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
            scene.scene_id: scene.location_profile
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
            spatial_zones=spatial_zones,  # NEW - Phase 2
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
            entity_classes = [str(e.object_class) for e in cluster_entities]
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
