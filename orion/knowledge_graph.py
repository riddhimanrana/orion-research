"""
Dynamic Knowledge Graph Builder
================================

Builds a rich, contextual knowledge graph with:
- Scene/Room detection and classification
- Spatial relationships (proximity, containment, orientation)
- Contextual entity embeddings (object + surroundings)
- Causal reasoning chains (state changes + potential causes)
- Multi-level indexing (entity, scene, spatial)

This integrates seamlessly with the refactored tracking engine.

Author: Orion Research Team
Date: October 2025
"""

import hashlib
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from .config import OrionConfig
from .config_manager import ConfigManager
from .model_manager import ModelManager
from .neo4j_manager import Neo4jManager

logger = logging.getLogger('orion.knowledge_graph')


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SceneContext:
    """Represents a scene/room with spatial context"""
    scene_id: str
    frame_range: Tuple[int, int]
    timestamp_range: Tuple[float, float]
    
    # Objects present
    entity_ids: List[str]
    object_classes: List[str]
    dominant_objects: List[str]  # Top 3-5 most common
    
    # Scene classification
    scene_type: str  # 'office', 'kitchen', 'outdoor', etc.
    confidence: float
    
    # Spatial layout
    spatial_zones: Dict[str, List[str]]  # 'left', 'center', 'right', 'top', 'bottom'
    
    # Contextual description
    description: str
    embedding: Optional[np.ndarray] = None


@dataclass
class SpatialRelationship:
    """Represents spatial relationship between two entities"""
    entity_a: str
    entity_b: str
    relationship_type: str  # 'near', 'on', 'under', 'left_of', 'right_of', etc.
    confidence: float
    frame_range: Tuple[int, int]
    co_occurrence_count: int
    avg_distance: float  # Normalized 0-1


@dataclass
class CausalChain:
    """Represents potential causal relationship"""
    cause_entity: str
    effect_entity: str
    cause_state: str
    effect_state: str
    temporal_gap: float  # seconds
    spatial_proximity: float  # 0-1
    confidence: float  # Combined score
    evidence: Dict[str, Any]


@dataclass
class ContextualEntityProfile:
    """Enhanced entity profile with full context"""
    entity_id: str
    object_class: str
    description: str
    
    # Temporal
    appearance_count: int
    first_seen: float
    last_seen: float
    
    # Spatial context
    typical_locations: List[Tuple[float, float]]  # (x, y) centroids
    spatial_zone: str  # 'left', 'center', 'right'
    
    # Scene context
    scenes: List[str]
    scene_types: Set[str]
    
    # Relationships
    nearby_objects: List[Tuple[str, float]]  # (entity_id, co_occurrence_score)
    spatial_relationships: List[SpatialRelationship]
    
    # Causal involvement
    caused_changes: List[str]  # Entity IDs affected
    affected_by: List[str]  # Entity IDs that affected this
    
    # Embeddings
    visual_embedding: np.ndarray
    contextual_embedding: Optional[np.ndarray] = None  # Visual + spatial + scene

    # Fine-grained canonical label (optional, non-COCO), e.g., 'knob'
    canonical_label: Optional[str] = None


# ============================================================================
# SCENE CLASSIFIER
# ============================================================================

class SceneClassifier:
    """Classifies scenes/rooms based on object composition"""
    
    # Scene signatures based on common object patterns
    SCENE_PATTERNS = {
        'office': {
            'required': {'laptop', 'keyboard', 'mouse', 'monitor', 'computer', 'desk'},
            'common': {'chair', 'book', 'cup', 'phone', 'bottle'},
            'weight': 1.0
        },
        'kitchen': {
            'required': {'oven', 'refrigerator', 'sink', 'microwave'},
            'common': {'bowl', 'cup', 'bottle', 'fork', 'knife', 'spoon'},
            'weight': 1.0
        },
        'living_room': {
            'required': {'couch', 'tv', 'chair'},
            'common': {'remote', 'book', 'vase', 'potted plant', 'clock'},
            'weight': 0.9
        },
        'bedroom': {
            'required': {'bed'},
            'common': {'clock', 'book', 'lamp', 'chair'},
            'weight': 0.9
        },
        'bathroom': {
            'required': {'toilet', 'sink'},
            'common': {'bottle', 'toothbrush'},
            'weight': 1.0
        },
        'dining_room': {
            'required': {'dining table', 'chair'},
            'common': {'bowl', 'cup', 'fork', 'knife', 'spoon', 'wine glass'},
            'weight': 0.8
        },
        'outdoor': {
            'required': {'tree', 'car', 'person', 'bicycle'},
            'common': {'bench', 'traffic light', 'stop sign', 'bird'},
            'weight': 0.7
        },
        'workspace': {
            'required': {'laptop', 'keyboard'},
            'common': {'mouse', 'cup', 'book', 'phone', 'monitor'},
            'weight': 0.8
        }
    }
    
    @classmethod
    def classify_scene(cls, object_classes: List[str]) -> Tuple[str, float]:
        """
        Classify scene type based on objects present
        
        Returns:
            (scene_type, confidence)
        """
        if not object_classes:
            return 'unknown', 0.0
        
        object_set = set(obj.lower() for obj in object_classes)
        scores = {}
        
        for scene_type, pattern in cls.SCENE_PATTERNS.items():
            required = pattern['required']
            common = pattern['common']
            weight = pattern['weight']
            
            # Check required objects
            required_matches = len(required & object_set)
            required_ratio = required_matches / len(required) if required else 0
            
            # Check common objects
            common_matches = len(common & object_set)
            common_ratio = common_matches / len(common) if common else 0
            
            # Combined score
            score = (required_ratio * 0.7 + common_ratio * 0.3) * weight
            scores[scene_type] = score
        
        if not scores:
            return 'unknown', 0.0
        
        best_scene = max(scores.items(), key=lambda x: x[1])[0]
        confidence = scores[best_scene]
        
        # Require minimum confidence
        if confidence < 0.3:
            return 'unknown', confidence
        
        return best_scene, confidence


# ============================================================================
# SPATIAL ANALYZER
# ============================================================================

class SpatialAnalyzer:
    """Analyzes spatial relationships between entities"""
    
    @staticmethod
    def compute_spatial_zone(bbox: List[float], frame_width: int, frame_height: int) -> str:
        """Determine which zone of frame the object is in"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Normalize to 0-1
        nx = cx / frame_width
        ny = cy / frame_height
        
        # Determine zone
        h_zone = 'left' if nx < 0.33 else ('right' if nx > 0.66 else 'center')
        v_zone = 'top' if ny < 0.33 else ('bottom' if ny > 0.66 else 'middle')
        
        return f"{v_zone}_{h_zone}"
    
    @staticmethod
    def compute_relationship(
        bbox_a: List[float],
        bbox_b: List[float],
        frame_width: int,
        frame_height: int
    ) -> Tuple[str, float]:
        """
        Determine spatial relationship between two bounding boxes
        
        Returns:
            (relationship_type, confidence)
        """
        x1a, y1a, x2a, y2a = bbox_a
        x1b, y1b, x2b, y2b = bbox_b
        
        # Compute centers
        cxa, cya = (x1a + x2a) / 2, (y1a + y2a) / 2
        cxb, cyb = (x1b + x2b) / 2, (y1b + y2b) / 2
        
        # Compute normalized distance
        distance = np.sqrt((cxa - cxb)**2 + (cya - cyb)**2)
        frame_diagonal = np.sqrt(frame_width**2 + frame_height**2)
        norm_distance = distance / frame_diagonal
        
        # Very close - specific relationships
        if norm_distance < 0.15:
            # Check vertical relationship
            if y2a < y1b - 10:  # A is above B
                return 'above', 0.9
            elif y2b < y1a - 10:  # B is above A (A is below B)
                return 'below', 0.9
            
            # Check horizontal relationship
            if x2a < x1b - 10:  # A is left of B
                return 'left_of', 0.85
            elif x2b < x1a - 10:  # B is left of A (A is right of B)
                return 'right_of', 0.85
            
            # Check containment
            if x1a <= x1b and y1a <= y1b and x2a >= x2b and y2a >= y2b:
                return 'contains', 0.95
            elif x1b <= x1a and y1b <= y1a and x2b >= x2a and y2b >= y2a:
                return 'inside', 0.95
            
            return 'very_near', 0.9
        
        elif norm_distance < 0.3:
            return 'near', 0.7
        
        elif norm_distance < 0.5:
            return 'same_region', 0.5
        
        else:
            return 'distant', 0.3
    
    @staticmethod
    def cluster_spatial_locations(locations: List[Tuple[float, float]], eps: float = 0.1) -> Dict[int, List[int]]:
        """Cluster spatial locations to find typical positions"""
        if not locations:
            return {}
        
        # Normalize locations
        locations_array = np.array(locations)
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=2).fit(locations_array)
        
        # Group by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            clusters[int(label)].append(idx)
        
        return dict(clusters)


# ============================================================================
# CONTEXTUAL EMBEDDING GENERATOR
# ============================================================================

class ContextualEmbeddingGenerator:
    """Generates rich contextual embeddings for entities"""
    
    def __init__(self, config: OrionConfig):
        self.config = config
        self.model_manager = ModelManager.get_instance()
    
    def generate_contextual_embedding(
        self,
        entity_profile: ContextualEntityProfile,
        scene_context: Optional[SceneContext] = None
    ) -> np.ndarray:
        """
        Generate contextual embedding combining:
        - Visual embedding (from CLIP)
        - Textual context (description + scene + relationships)
        
        Returns:
            Combined embedding vector
        """
        # Start with visual embedding
        visual_emb = entity_profile.visual_embedding
        
        # Build rich textual context
        context_parts = [
            f"A {entity_profile.object_class}",
            entity_profile.description[:200],  # Truncate
        ]
        
        # Add scene context
        if scene_context:
            context_parts.append(f"Located in {scene_context.scene_type}")
            context_parts.append(f"Surrounded by {', '.join(scene_context.dominant_objects[:3])}")
        
        # Add spatial context
        context_parts.append(f"Typically found in {entity_profile.spatial_zone}")
        
        # Add relationship context
        if entity_profile.nearby_objects:
            nearby_names = [obj[0] for obj in entity_profile.nearby_objects[:3]]
            context_parts.append(f"Often near {', '.join(nearby_names)}")
        
        context_text = ". ".join(context_parts)
        
        # Generate text embedding
        try:
            text_emb = self.model_manager.clip.encode_text([context_text])[0]
            
            # Normalize
            visual_emb = visual_emb / (np.linalg.norm(visual_emb) + 1e-8)
            text_emb = text_emb / (np.linalg.norm(text_emb) + 1e-8)
            
            # Combine (weighted average)
            combined = 0.6 * visual_emb + 0.4 * text_emb
            combined = combined / (np.linalg.norm(combined) + 1e-8)
            
            return combined
            
        except Exception as e:
            logger.warning(f"Failed to generate contextual embedding: {e}")
            return visual_emb


# ============================================================================
# CAUSAL REASONING ENGINE
# ============================================================================

class CausalReasoningEngine:
    """Infers potential causal relationships from state changes"""
    
    def __init__(self, config: OrionConfig):
        self.config = config
    
    def find_causal_chains(
        self,
        state_changes: List[Dict[str, Any]],
        entity_profiles: Dict[str, ContextualEntityProfile],
        spatial_relationships: List[SpatialRelationship]
    ) -> List[CausalChain]:
        """
        Find potential causal chains between state changes
        
        Logic:
        - Temporal ordering (cause before effect)
        - Spatial proximity (cause near effect)
        - Semantic plausibility (based on object types)
        """
        causal_chains = []
        
        # Sort state changes by time
        sorted_changes = sorted(state_changes, key=lambda x: x['from_time'])
        
        # Build spatial proximity map
        proximity_map = self._build_proximity_map(spatial_relationships)
        
        # Check each pair of changes
        for i, change_a in enumerate(sorted_changes):
            for change_b in sorted_changes[i+1:]:
                # Temporal constraint: A before B
                temporal_gap = change_b['from_time'] - change_a['to_time']
                
                # Must be within reasonable window (e.g., 5 seconds)
                if temporal_gap < 0 or temporal_gap > 5.0:
                    continue
                
                entity_a = change_a['entity_id']
                entity_b = change_b['entity_id']
                
                # Skip same entity
                if entity_a == entity_b:
                    continue
                
                # Check spatial proximity
                proximity = proximity_map.get((entity_a, entity_b), 0.0)
                
                # Must be reasonably close
                if proximity < 0.3:
                    continue
                
                # Compute causal confidence
                # Factors: temporal proximity, spatial proximity, semantic plausibility
                temporal_score = 1.0 / (1.0 + temporal_gap)  # Closer in time = higher
                spatial_score = proximity
                
                # Semantic plausibility (simple heuristic for now)
                semantic_score = self._compute_semantic_plausibility(
                    entity_profiles.get(entity_a),
                    entity_profiles.get(entity_b)
                )
                
                confidence = (
                    0.3 * temporal_score +
                    0.4 * spatial_score +
                    0.3 * semantic_score
                )
                
                # Threshold
                if confidence < 0.5:
                    continue
                
                # Create causal chain
                chain = CausalChain(
                    cause_entity=entity_a,
                    effect_entity=entity_b,
                    cause_state=change_a.get('new_description', '')[:100],
                    effect_state=change_b.get('new_description', '')[:100],
                    temporal_gap=temporal_gap,
                    spatial_proximity=proximity,
                    confidence=confidence,
                    evidence={
                        'temporal_score': temporal_score,
                        'spatial_score': spatial_score,
                        'semantic_score': semantic_score
                    }
                )
                
                causal_chains.append(chain)
        
        logger.info(f"Found {len(causal_chains)} potential causal chains")
        return causal_chains
    
    @staticmethod
    def _build_proximity_map(
        spatial_relationships: List[SpatialRelationship]
    ) -> Dict[Tuple[str, str], float]:
        """Build map of spatial proximity scores"""
        proximity_map = {}
        
        for rel in spatial_relationships:
            key = (rel.entity_a, rel.entity_b)
            
            # Score based on relationship type
            if rel.relationship_type in ['very_near', 'contains', 'inside']:
                score = 1.0
            elif rel.relationship_type in ['near', 'above', 'below', 'left_of', 'right_of']:
                score = 0.7
            elif rel.relationship_type == 'same_region':
                score = 0.5
            else:
                score = 0.2
            
            # Weight by co-occurrence
            score *= min(1.0, rel.co_occurrence_count / 10.0)
            
            proximity_map[key] = score
            proximity_map[(rel.entity_b, rel.entity_a)] = score  # Symmetric
        
        return proximity_map
    
    @staticmethod
    def _compute_semantic_plausibility(
        entity_a: Optional[ContextualEntityProfile],
        entity_b: Optional[ContextualEntityProfile]
    ) -> float:
        """
        Compute how semantically plausible it is for A to affect B
        
        Simple heuristic for now - can be enhanced with learned models
        """
        if not entity_a or not entity_b:
            return 0.5
        
        # Agents (things that move/act) more likely to cause changes
        agent_classes = {'person', 'dog', 'cat', 'bird', 'car', 'truck', 'robot'}
        
        class_a = entity_a.object_class.lower()
        class_b = entity_b.object_class.lower()
        
        # Person acting on objects - high plausibility
        if class_a in agent_classes:
            return 0.9
        
        # Both static objects - low plausibility
        if class_a not in agent_classes and class_b not in agent_classes:
            return 0.3
        
        # Default
        return 0.5


# ============================================================================
# MAIN KNOWLEDGE GRAPH BUILDER
# ============================================================================

class KnowledgeGraphBuilder:
    """Main class for building the knowledge graph"""
    
    def __init__(
        self,
        config: Optional[OrionConfig] = None,
        neo4j_manager: Optional[Neo4jManager] = None
    ):
        self.config = config or ConfigManager.get_config()
        self.neo4j_manager: Optional[Neo4jManager] = neo4j_manager
        self.driver: Optional[Any] = None
        
        self.scene_classifier = SceneClassifier()
        self.spatial_analyzer = SpatialAnalyzer()
        self.contextual_embedder = ContextualEmbeddingGenerator(self.config)
        self.causal_engine = CausalReasoningEngine(self.config)
    
    def connect(self) -> bool:
        """Connect to Neo4j"""
        try:
            if self.neo4j_manager is None:
                neo4j_config = self.config.neo4j
                self.neo4j_manager = Neo4jManager(
                    uri=neo4j_config.uri,
                    user=neo4j_config.user,
                    password=neo4j_config.password,
                )

            if not self.neo4j_manager.connect():
                return False

            self.driver = self.neo4j_manager.driver
            logger.info("✓ Connected to Neo4j for knowledge graph")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to connect to Neo4j: {e}")
            return False
    
    def build_from_tracking_results(
        self,
        tracking_results: Dict[str, Any]
    ) -> Dict[str, int]:
        """Build the knowledge graph from tracking results."""

        logger.info("\n" + "=" * 80)
        logger.info("KNOWLEDGE GRAPH CONSTRUCTION")
        logger.info("=" * 80)
        
        if not self.driver:
            if not self.connect():
                raise RuntimeError("Cannot connect to Neo4j")
        
        stats = {
            'entities': 0,
            'scenes': 0,
            'spatial_relationships': 0,
            'causal_chains': 0,
            'scene_transitions': 0
        }
        
        # Extract data
        entities_data = tracking_results.get('entities', [])
        
        # Phase 1: Build scene contexts
        logger.info("\n[1/5] Building scene contexts...")
        scene_contexts = self._build_scene_contexts(entities_data)
        stats['scenes'] = len(scene_contexts)
        
        # Phase 2: Analyze spatial relationships
        logger.info("[2/5] Analyzing spatial relationships...")
        spatial_relationships = self._analyze_spatial_relationships(entities_data)
        stats['spatial_relationships'] = len(spatial_relationships)
        
        # Phase 3: Build entity profiles with context
        logger.info("[3/5] Building contextual entity profiles...")
        entity_profiles = self._build_entity_profiles(
            entities_data,
            scene_contexts,
            spatial_relationships
        )
        stats['entities'] = len(entity_profiles)
        
        # Phase 4: Find causal chains
        logger.info("[4/5] Finding causal relationships...")
        state_changes = self._extract_state_changes(entities_data)
        causal_chains = self.causal_engine.find_causal_chains(
            state_changes,
            entity_profiles,
            spatial_relationships
        )
        stats['causal_chains'] = len(causal_chains)
        
        # Phase 5: Ingest into Neo4j
        logger.info("[5/5] Ingesting into Neo4j...")
        driver = self.driver
        if driver is None:
            raise RuntimeError("Neo4j driver not initialized")
        with driver.session() as session:
            self._initialize_schema(session)
            self._ingest_scenes(session, scene_contexts)
            self._ingest_entities(session, entity_profiles, scene_contexts)
            self._ingest_spatial_relationships(session, spatial_relationships)
            self._ingest_causal_chains(session, causal_chains)
            # Infer events after entities and relationships exist
            self._infer_and_ingest_events(session)
            scene_trans = self._ingest_scene_transitions(session, scene_contexts)
            stats['scene_transitions'] = scene_trans
        
        logger.info("\n" + "="*80)
        logger.info("KNOWLEDGE GRAPH COMPLETE")
        logger.info("="*80)
        logger.info(f"Entities: {stats['entities']}")
        logger.info(f"Scenes: {stats['scenes']}")
        logger.info(f"Spatial Relationships: {stats['spatial_relationships']}")
        logger.info(f"Causal Chains: {stats['causal_chains']}")
        logger.info(f"Scene Transitions: {stats['scene_transitions']}")
        logger.info("="*80 + "\n")
        
        return stats
    
    def _build_scene_contexts(self, entities_data: List[Dict]) -> List[SceneContext]:
        """Build scene contexts from entity data"""
        # Group entities by frame ranges
        frame_to_entities = defaultdict(list)
        
        for entity in entities_data:
            for frame_num in entity.get('frame_numbers', []):
                frame_to_entities[frame_num].append(entity)
        
        # Cluster consecutive frames into scenes
        sorted_frames = sorted(frame_to_entities.keys())
        scenes = []
        
        if not sorted_frames:
            return scenes
        
        current_scene_frames = [sorted_frames[0]]
        current_scene_entities = set(e['id'] for e in frame_to_entities[sorted_frames[0]])
        
        for frame_num in sorted_frames[1:]:
            frame_entities = set(e['id'] for e in frame_to_entities[frame_num])
            
            # If similar entities or consecutive frame, extend scene
            overlap = len(current_scene_entities & frame_entities) / max(len(current_scene_entities), 1)
            
            if frame_num - current_scene_frames[-1] <= 10 and overlap > 0.5:
                current_scene_frames.append(frame_num)
                current_scene_entities.update(frame_entities)
            else:
                # Save current scene and start new one
                if len(current_scene_frames) >= 5:  # Minimum scene length
                    scenes.append(self._create_scene_context(
                        current_scene_frames,
                        current_scene_entities,
                        frame_to_entities
                    ))
                
                current_scene_frames = [frame_num]
                current_scene_entities = frame_entities
        
        # Add final scene
        if len(current_scene_frames) >= 5:
            scenes.append(self._create_scene_context(
                current_scene_frames,
                current_scene_entities,
                frame_to_entities
            ))
        
        logger.info(f"Created {len(scenes)} scene contexts")
        return scenes
    
    def _create_scene_context(
        self,
        frames: List[int],
        entity_ids: Set[str],
        frame_to_entities: Dict
    ) -> SceneContext:
        """Create a scene context from frame data"""
        frame_range = (min(frames), max(frames))
        
        # Collect object classes
        object_classes = []
        for frame_num in frames:
            for entity in frame_to_entities[frame_num]:
                if entity['id'] in entity_ids:
                    object_classes.append(entity['class'])
        
        # Count occurrences
        class_counts = Counter(object_classes)
        dominant = [cls for cls, _ in class_counts.most_common(5)]
        
        # Classify scene
        scene_type, confidence = self.scene_classifier.classify_scene(object_classes)
        
        # Generate description
        if scene_type != 'unknown':
            desc = f"A {scene_type} scene with {', '.join(dominant[:3])}"
        else:
            desc = f"A scene with {', '.join(dominant[:3])}"
        
        # Get timestamp range (approximate based on frame rate)
        fps = 30  # Default assumption
        timestamp_range = (frame_range[0] / fps, frame_range[1] / fps)
        
        scene_id = f"scene_{frame_range[0]:06d}_{frame_range[1]:06d}"
        
        return SceneContext(
            scene_id=scene_id,
            frame_range=frame_range,
            timestamp_range=timestamp_range,
            entity_ids=list(entity_ids),
            object_classes=object_classes,
            dominant_objects=dominant,
            scene_type=scene_type,
            confidence=confidence,
            spatial_zones={},  # TODO: Implement spatial zone analysis
            description=desc,
            embedding=None  # Will be generated later if needed
        )
    
    def _analyze_spatial_relationships(
        self,
        entities_data: List[Dict]
    ) -> List[SpatialRelationship]:
        """Analyze spatial relationships between entities"""
        relationships = []
        
        # Build co-occurrence matrix
        co_occurrence: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(lambda: {'count': 0, 'distances': []})
        
        # Group by frame
        frame_to_entities = defaultdict(list)
        for entity in entities_data:
            for frame_num in entity.get('frame_numbers', []):
                frame_to_entities[frame_num].append(entity)
        
        # Check each frame for co-occurrences
        for frame_num, frame_entities in frame_to_entities.items():
            if len(frame_entities) < 2:
                continue
            
            # Check each pair
            for i, entity_a in enumerate(frame_entities):
                for entity_b in frame_entities[i+1:]:
                    key = tuple(sorted([entity_a['id'], entity_b['id']]))
                    co_occurrence[key]['count'] += 1
                    
                    # TODO: If we had bbox info per frame, compute distance here
                    # For now, just track co-occurrence
        
        # Create relationships for significant co-occurrences
        for (entity_a_id, entity_b_id), data in co_occurrence.items():
            count = int(data['count'])
            if count >= 3:  # Minimum co-occurrence threshold
                # Determine relationship type (simplified without bbox data)
                rel_type = 'near' if count > 10 else 'same_region'
                confidence = min(1.0, count / 20.0)
                
                relationships.append(SpatialRelationship(
                    entity_a=entity_a_id,
                    entity_b=entity_b_id,
                    relationship_type=rel_type,
                    confidence=confidence,
                    frame_range=(0, 0),  # TODO: Track actual frame range
                    co_occurrence_count=count,
                    avg_distance=0.5  # Placeholder
                ))
        
        logger.info(f"Found {len(relationships)} spatial relationships")
        return relationships
    
    def _build_entity_profiles(
        self,
        entities_data: List[Dict],
        scene_contexts: List[SceneContext],
        spatial_relationships: List[SpatialRelationship]
    ) -> Dict[str, ContextualEntityProfile]:
        """Build rich contextual profiles for entities"""
        profiles = {}
        
        # Build scene map
        entity_to_scenes = defaultdict(list)
        for scene in scene_contexts:
            for entity_id in scene.entity_ids:
                entity_to_scenes[entity_id].append(scene)
        
        # Build spatial relationship map
        entity_to_spatial = defaultdict(list)
        for rel in spatial_relationships:
            entity_to_spatial[rel.entity_a].append(rel)
            entity_to_spatial[rel.entity_b].append(rel)
        
        for entity in entities_data:
            entity_id = entity['id']
            
            # Get scenes
            scenes = entity_to_scenes[entity_id]
            scene_types = set(s.scene_type for s in scenes)
            
            # Get spatial relationships
            spatial_rels = entity_to_spatial[entity_id]
            
            # Compute nearby objects
            nearby = {}
            for rel in spatial_rels:
                other_id = rel.entity_b if rel.entity_a == entity_id else rel.entity_a
                nearby[other_id] = nearby.get(other_id, 0) + rel.co_occurrence_count
            
            nearby_list = sorted(nearby.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Create profile
            profile = ContextualEntityProfile(
                entity_id=entity_id,
                object_class=entity['class'],
                description=entity.get('description', ''),
                canonical_label=entity.get('canonical_label'),
                appearance_count=entity['appearance_count'],
                first_seen=entity['first_seen'],
                last_seen=entity['last_seen'],
                typical_locations=[],  # TODO: Compute from bbox centroids
                spatial_zone='center',  # TODO: Compute actual zone
                scenes=[s.scene_id for s in scenes],
                scene_types=scene_types,
                nearby_objects=nearby_list,
                spatial_relationships=spatial_rels,
                caused_changes=[],  # Will be filled by causal engine
                affected_by=[],
                visual_embedding=np.zeros(512)  # Placeholder - would load actual embedding
            )
            
            profiles[entity_id] = profile
        
        logger.info(f"Built profiles for {len(profiles)} entities")
        return profiles
    
    @staticmethod
    def _extract_state_changes(entities_data: List[Dict]) -> List[Dict[str, Any]]:
        """Extract state changes from entity data"""
        state_changes = []
        
        for entity in entities_data:
            # Check if entity has state_changes field (from tracking engine)
            if 'state_changes' in entity and isinstance(entity['state_changes'], list):
                for change in entity['state_changes']:
                    state_changes.append({
                        'entity_id': entity['id'],
                        'from_frame': change.get('from_frame'),
                        'to_frame': change.get('to_frame'),
                        'from_time': change.get('from_time'),
                        'to_time': change.get('to_time'),
                        'similarity': change.get('similarity'),
                        'new_description': change.get('new_description', '')
                    })
        
        return state_changes
    
    def _initialize_schema(self, session):
        """Initialize Neo4j schema"""
        # Constraints
        constraints = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT scene_id IF NOT EXISTS FOR (s:Scene) REQUIRE s.id IS UNIQUE",
        ]
        
        for constraint in constraints:
            try:
                session.run(constraint)
            except Exception:
                pass  # Already exists
        
        # Indexes
        indexes = [
            "CREATE INDEX entity_class IF NOT EXISTS FOR (e:Entity) ON (e.class)",
            "CREATE INDEX scene_type IF NOT EXISTS FOR (s:Scene) ON (s.scene_type)",
        ]
        
        for index in indexes:
            try:
                session.run(index)
            except Exception:
                pass
        
        logger.info("Schema initialized")
    
    def _ingest_scenes(self, session, scene_contexts: List[SceneContext]):
        """Ingest scene nodes"""
        for scene in scene_contexts:
            session.run("""
                MERGE (s:Scene {id: $id})
                SET s.scene_type = $type,
                    s.confidence = $confidence,
                    s.frame_start = $frame_start,
                    s.frame_end = $frame_end,
                    s.timestamp_start = $ts_start,
                    s.timestamp_end = $ts_end,
                    s.description = $description,
                    s.dominant_objects = $dominant
            """, {
                'id': scene.scene_id,
                'type': scene.scene_type,
                'confidence': scene.confidence,
                'frame_start': scene.frame_range[0],
                'frame_end': scene.frame_range[1],
                'ts_start': scene.timestamp_range[0],
                'ts_end': scene.timestamp_range[1],
                'description': scene.description,
                'dominant': scene.dominant_objects
            })
    
    def _ingest_entities(
        self,
        session,
        entity_profiles: Dict[str, ContextualEntityProfile],
        scene_contexts: List[SceneContext]
    ):
        """Ingest entity nodes and link to scenes"""
        for entity_id, profile in entity_profiles.items():
            # Create entity node
            session.run("""
                MERGE (e:Entity {id: $id})
                SET e.class = $class,
                    e.description = $description,
                    e.appearance_count = $count,
                    e.first_seen = $first_seen,
                    e.last_seen = $last_seen,
                    e.spatial_zone = $zone,
                    e.scene_types = $scene_types,
                    e.canonical_label = coalesce($canonical_label, e.canonical_label)
            """, {
                'id': entity_id,
                'class': profile.object_class,
                'description': profile.description,
                'count': profile.appearance_count,
                'first_seen': profile.first_seen,
                'last_seen': profile.last_seen,
                'zone': profile.spatial_zone,
                'scene_types': list(profile.scene_types),
                'canonical_label': profile.canonical_label
            })
            
            # Link to scenes
            for scene_id in profile.scenes:
                session.run("""
                    MATCH (e:Entity {id: $entity_id})
                    MATCH (s:Scene {id: $scene_id})
                    MERGE (e)-[:APPEARS_IN]->(s)
                """, {
                    'entity_id': entity_id,
                    'scene_id': scene_id
                })

        # After entities are upserted, infer PART_OF for common patterns
        if getattr(self.config, 'correction', None) and self.config.correction.infer_part_of:
            # knob/handle part of door if both co-appear in a scene
            session.run("""
                MATCH (k:Entity)
                WHERE toLower(coalesce(k.canonical_label, '')) IN ['knob','handle']
                MATCH (d:Entity)
                WHERE toLower(d.class) = 'refrigerator' OR toLower(d.class) = 'oven' OR toLower(d.class) = 'door'
                // Co-appearance: share a scene
                MATCH (k)-[:APPEARS_IN]->(s:Scene)<-[:APPEARS_IN]-(d)
                MERGE (k)-[r:PART_OF]->(d)
                ON CREATE SET r.confidence = 0.8
                ON MATCH SET r.confidence = coalesce(r.confidence, 0.8)
            """)
    
    def _ingest_spatial_relationships(self, session, relationships: List[SpatialRelationship]):
        """Ingest spatial relationships"""
        for rel in relationships:
            session.run(f"""
                MATCH (a:Entity {{id: $id_a}})
                MATCH (b:Entity {{id: $id_b}})
                MERGE (a)-[r:SPATIAL_REL {{type: $rel_type}}]->(b)
                SET r.confidence = $confidence,
                    r.co_occurrence = $co_occur,
                    r.avg_distance = $distance
            """, {
                'id_a': rel.entity_a,
                'id_b': rel.entity_b,
                'rel_type': rel.relationship_type,
                'confidence': rel.confidence,
                'co_occur': rel.co_occurrence_count,
                'distance': rel.avg_distance
            })
    
    def _ingest_causal_chains(self, session, causal_chains: List[CausalChain]):
        """Ingest causal relationships"""
        for chain in causal_chains:
            session.run("""
                MATCH (cause:Entity {id: $cause_id})
                MATCH (effect:Entity {id: $effect_id})
                MERGE (cause)-[r:POTENTIALLY_CAUSED]->(effect)
                SET r.temporal_gap = $temporal_gap,
                    r.spatial_proximity = $spatial,
                    r.confidence = $confidence,
                    r.cause_state = $cause_state,
                    r.effect_state = $effect_state
            """, {
                'cause_id': chain.cause_entity,
                'effect_id': chain.effect_entity,
                'temporal_gap': chain.temporal_gap,
                'spatial': chain.spatial_proximity,
                'confidence': chain.confidence,
                'cause_state': chain.cause_state,
                'effect_state': chain.effect_state
            })
    
    def _ingest_scene_transitions(self, session, scene_contexts: List[SceneContext]) -> int:
        """Create transitions between consecutive scenes"""
        count = 0
        sorted_scenes = sorted(scene_contexts, key=lambda s: s.frame_range[0])
        
        for i in range(len(sorted_scenes) - 1):
            scene_a = sorted_scenes[i]
            scene_b = sorted_scenes[i + 1]
            
            session.run("""
                MATCH (a:Scene {id: $id_a})
                MATCH (b:Scene {id: $id_b})
                MERGE (a)-[r:TRANSITIONS_TO]->(b)
                SET r.frame_gap = $frame_gap
            """, {
                'id_a': scene_a.scene_id,
                'id_b': scene_b.scene_id,
                'frame_gap': scene_b.frame_range[0] - scene_a.frame_range[1]
            })
            count += 1
        
        return count

    def _infer_and_ingest_events(self, session) -> None:
        """Infer simple events like OPENS_DOOR and ENTERS_ROOM from patterns"""
        # OPENS_DOOR: person near knob/handle that is part of a door-like object in same scene
        session.run("""
            MATCH (p:Entity)-[pr:SPATIAL_REL]->(k:Entity)
            WHERE toLower(p.class) = 'person'
              AND toLower(coalesce(k.canonical_label, '')) IN ['knob','handle']
            MATCH (k)-[:PART_OF]->(d:Entity)
            WHERE toLower(d.class) IN ['refrigerator','oven','door']
            MATCH (p)-[:APPEARS_IN]->(s:Scene)<-[:APPEARS_IN]-(k)
            WITH p,k,d,s,pr
            MERGE (ev:Event {type: 'OPENS_DOOR', scene_id: s.id})
            ON CREATE SET ev.confidence = 0.7, ev.ts_start = s.timestamp_start, ev.ts_end = s.timestamp_end
            MERGE (ev)-[:INVOLVES]->(p)
            MERGE (ev)-[:TARGETS]->(d)
                        // Also create causal link from person to door-like entity
                        MERGE (p)-[c:POTENTIALLY_CAUSED]->(d)
                        ON CREATE SET c.confidence = 0.6, c.temporal_gap = 0.0, c.cause_state = 'opened', c.effect_state = 'door opened'
        """)
        
        # ENTERS_ROOM: a person appears in next scene but not previous
        session.run("""
            MATCH (a:Scene)-[:TRANSITIONS_TO]->(b:Scene)
            MATCH (p:Entity)-[:APPEARS_IN]->(b)
            WHERE toLower(p.class) = 'person' AND NOT (p)-[:APPEARS_IN]->(a)
            MERGE (ev:Event {type: 'ENTERS_ROOM', scene_id: b.id})
            ON CREATE SET ev.confidence = 0.6, ev.ts_start = b.timestamp_start, ev.ts_end = b.timestamp_end
            MERGE (ev)-[:INVOLVES]->(p)
            MERGE (ev)-[:TARGETS]->(b)
            // Causal link from person to scene change proxy node (optional)
        """)

    
    
    def close(self):
        """Close Neo4j connection"""
        if self.neo4j_manager:
            self.neo4j_manager.close()
            self.driver = None
            logger.info("Neo4j connection closed")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import json
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)
    
    # Load tracking results
    results_path = Path("data/testing/tracking_results_save1.json")
    
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        logger.info("Run tracking_engine.py first to generate results")
        exit(1)
    
    with open(results_path) as f:
        tracking_results = json.load(f)
    
    logger.info(f"Loaded {len(tracking_results['entities'])} entities")
    
    # Build knowledge graph
    builder = KnowledgeGraphBuilder()
    stats = builder.build_from_tracking_results(tracking_results)
    builder.close()
    
    logger.info("\nKnowledge graph ready for querying!")
    logger.info("Try: python -m orion.video_qa")
