# Orion: Technical Architecture & Comprehensive Pipeline Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Layers](#architecture-layers)
3. [Configuration System](#configuration-system)
4. [Core Modules](#core-modules)
5. [Pipeline Execution Flow](#pipeline-execution-flow)
6. [Mathematical Foundations](#mathematical-foundations)
7. [Database Schema](#database-schema)
8. [Integration Points](#integration-points)

---

## System Overview

### Purpose
Orion is a multi-phase video understanding system that performs intelligent analysis of video content through perception, semantic understanding, and knowledge graph construction. The system bridges low-level vision (object detection, embedding generation) with high-level reasoning (causality, temporal relationships, contextual understanding).

### Core Phases
```
VIDEO INPUT
    ↓
[1] PERCEPTION PHASE
    - Object Detection (YOLO11x)
    - Spatial Analysis (bounding boxes)
    - Embedding Generation (CLIP)
    ↓
[2] TRACKING & SEMANTIC UPLIFT PHASE
    - Entity Clustering (HDBSCAN)
    - State Change Detection
    - Temporal Windowing
    - Event Composition (LLM)
    ↓
[3] KNOWLEDGE GRAPH CONSTRUCTION
    - Scene/Entity/Event node creation
    - Spatial relationship analysis
    - Causal reasoning
    - Temporal sequencing
    ↓
[4] STORAGE & INDEXING
    - Neo4j graph persistence
    - Vector indexing
    - Relationship constraints
    ↓
[5] QUERY & Q&A
    - Knowledge retrieval
    - Contextual reasoning
    - LLM-based answer generation
```

### Key Design Principles
- **Modular Architecture**: Each phase is independently testable and can be run in isolation
- **Centralized Configuration**: All parameters managed through `config.py` with environment-aware presets
- **Credential Management**: Secure handling via `ConfigManager` singleton and environment variables
- **Hardware Abstraction**: Seamless switching between MLX (Apple Silicon) and Torch backends
- **Neo4j Integration**: Rich graph representation with comprehensive schema and constraints

---

## Architecture Layers

### Layer 1: Configuration & Credential Management
**Files**: `config.py`, `config_manager.py`

```python
# Three-tier configuration system:
┌─────────────────────────────────────────┐
│ Environment Variables (.env)            │
│ (ORION_NEO4J_PASSWORD, etc.)            │
└──────────────────────┬──────────────────┘
                       ↓
┌─────────────────────────────────────────┐
│ ConfigManager Singleton                 │
│ - Loads env vars                        │
│ - Manages credential lifecycle          │
│ - Provides lazy initialization          │
└──────────────────────┬──────────────────┘
                       ↓
┌─────────────────────────────────────────┐
│ OrionConfig Instance                    │
│ - All system parameters                 │
│ - Model configurations                  │
│ - Neo4j connection details              │
└─────────────────────────────────────────┘
```

**Configuration Presets**:
- `get_fast_config()`: Minimal latency, lower accuracy (YOLO11n, 512-dim embeddings, 100 entity limit)
- `get_balanced_config()`: Recommended for production (YOLO11m, 1024-dim embeddings, 500 entity limit)
- `get_accurate_config()`: Maximum accuracy, higher resource usage (YOLO11x, 2048-dim embeddings, 1000 entity limit)

### Layer 2: Data Models & Persistence
**Files**: `neo4j_manager.py`, `model_manager.py` (AssetManager)

**Neo4jManager** (Singleton-like):
```python
# Manages driver lifecycle, connection pooling, utility operations
class Neo4jManager:
    - __init__(uri, user, password): Connection initialization
    - connect(): Establish driver connection
    - close(): Graceful shutdown
    - get_stats(): Query performance metrics
    - clear_database(): Clean state for new runs
    - clear_neo4j_for_new_run(): Pre-pipeline cleanup
```

**AssetManager** (Model Management):
```python
# Handles model loading, caching, backend selection
class AssetManager:
    - get_detection_model(): Load YOLO (device-aware)
    - get_embedding_model(): Load CLIP (with caching)
    - get_description_model(): Load FastVLM or comparable
    - Auto-select MLX vs Torch based on hardware
```

### Layer 3: Pipeline Engines
**Files**: `perception_engine.py`, `tracking_engine.py`, `semantic_uplift.py`

#### Perception Engine
```python
# Detects objects in video frames
class PerceptionEngine:
    - process_frame(): Run YOLO detection on single frame
    - extract_embeddings(): Generate CLIP embeddings for detections
    - build_spatial_context(): Compute bounding box relationships
    
    Input: Video frame (H×W×3)
    Output: {
        'detections': List[Detection],  # bbox, confidence, class
        'embeddings': ndarray[N×embed_dim],
        'spatial_graph': Dict[str, List[Tuple[int, str]]]  # spatial rels
    }
```

#### Tracking Engine
```python
# Maintains temporal entity continuity across frames
class TrackingEngine:
    - update_tracks(): Match current detections to known entities
    - register_entity(): Create new entity on first appearance
    - get_entity_trajectory(): Retrieve temporal sequence for entity
    
    Input: Detection list for frame t
    Output: Entity ID assignments with confidence scores
    
    Cosine similarity computation for detection matching:
    similarity(det1, det2) = cos_dist(embed1, embed2)
    match_threshold = 0.7 (configurable)
```

#### Semantic Uplift Engine
```python
# Detects state changes and composes high-level events
class SemanticUplift:
    - cluster_entities(): HDBSCAN on embedding space
    - detect_state_changes(): Identify behavioral transitions
    - compose_events(): LLM-based event description
    - ingest_to_graph(): Store in Neo4j
    
    Key Components:
    1. EntityTracker: Maintains entity states across temporal windows
    2. StateChangeDetector: Monitors position, velocity, attributes
    3. EventComposer: Uses Ollama to generate natural descriptions
```

### Layer 4: Knowledge Representation
**Files**: `knowledge_graph.py`, `temporal_graph_builder.py`, `semantic_config.py`

#### Knowledge Graph Builder
```python
# Rich graph construction with multi-faceted analysis
class KnowledgeGraphBuilder:
    
    Core Operations:
    1. build_graph(semantic_data): Main ingestion pipeline
    2. _create_scene_nodes(): Scene classification and persistence
    3. _create_entity_nodes(): Entity properties and relationships
    4. _analyze_spatial_relationships(): Spatial graph construction
    5. _perform_causal_reasoning(): Event causality inference
    6. _create_temporal_sequences(): Temporal ordering
    
    Neo4j Node Types:
    - Scene: scene_type, timestamp_start, timestamp_end, duration
    - Entity: class, appearance_count, average_confidence, description
    - Event: event_type, description, confidence, timestamp
    
    Relationship Types:
    - APPEARS_IN: Entity → Scene
    - CONTAINS: Scene → Entity, Scene → Event
    - SPATIAL_*: Spatial relationships (left_of, above, inside, etc.)
    - CAUSES: Event → Event (causal relationships)
    - FOLLOWS: Event → Event (temporal ordering)
    - HAS_STATE: Entity → State changes
```

---

## Configuration System

### Structure of `OrionConfig`
```python
class OrionConfig:
    # Detection configuration (YOLO)
    detection: DetectionConfig
        - model_variant: str  # "11x", "11m", "11s", "11n"
        - confidence_threshold: float  # 0.45
        - nms_threshold: float  # 0.45
        - image_size: int  # 640
    
    # Embedding configuration (CLIP)
    embedding: EmbeddingConfig
        - model_name: str  # "openai/clip-vit-base-patch32"
        - embedding_dimension: int  # 512, 1024, or 2048
        - batch_size: int
    
    # Clustering configuration (HDBSCAN)
    clustering: ClusteringConfig
        - min_cluster_size: int
        - min_samples: int
        - metric: str  # "euclidean", "cosine"
        - cluster_selection_epsilon: float
    
    # Description generation (FastVLM)
    description: DescriptionConfig
        - model_name: str
        - max_tokens: int
        - temperature: float
    
    # Neo4j configuration
    neo4j: Neo4jConfig
        - uri: str  # "bolt://localhost:7687"
        - user: str
        - password: str  # From environment
    
    # LLM configuration (Ollama)
    ollama: OllamaConfig
        - base_url: str
        - model: str
        - timeout: int
    
    # Performance parameters
    temporal_window_size: int  # frames
    state_change_threshold: float
    spatial_distance_threshold: float
```

### Usage Pattern
```python
from orion.config_manager import ConfigManager

# Get singleton instance
config = ConfigManager.get_config()

# Access subsystems
yolo_confidence = config.detection.confidence_threshold
embedding_dim = config.embedding.embedding_dimension
neo4j_uri = config.neo4j.uri

# Use preset configurations
from orion.config import get_fast_config, get_balanced_config, get_accurate_config
fast = get_fast_config()  # For quick testing
balanced = get_balanced_config()  # Production recommended
accurate = get_accurate_config()  # Maximum quality
```

---

## Core Modules

### 1. Perception Engine (`perception_engine.py`)

**Responsibility**: Convert raw video frames to semantic representations

```python
class PerceptionEngine:
    def __init__(self, config: DetectionConfig, embedding_config: EmbeddingConfig):
        self.detection_model = AssetManager.get_detection_model(config.model_variant)
        self.embedding_model = AssetManager.get_embedding_model(
            embedding_config.model_name, 
            embedding_config.embedding_dimension
        )
    
    def process_frame(self, frame: np.ndarray) -> PerceptionResult:
        """
        Process single video frame through detection and embedding pipeline
        
        Args:
            frame: Image array [H, W, 3] in RGB format
        
        Returns:
            PerceptionResult containing:
            - detections: List[Detection] with bbox, class, confidence
            - embeddings: ndarray [N, embedding_dim] - normalized L2
            - processing_time_ms: float
        
        Process:
            1. Resize to YOLO input size (640×640)
            2. Run inference through YOLO11x backbone
            3. Apply NMS with IoU threshold 0.45
            4. Extract patches for detected objects
            5. Generate CLIP embeddings for each patch
            6. Normalize embeddings to unit L2 norm
        """
        
        # Implementation details
        yolo_results = self.detection_model(frame)
        detections = self._parse_yolo_output(yolo_results)
        
        # Batch embedding generation for efficiency
        if detections:
            patches = [self._crop_patch(frame, d.bbox) for d in detections]
            embeddings = self.embedding_model.encode_batch(patches)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        else:
            embeddings = np.array([])
        
        return PerceptionResult(detections, embeddings)
    
    def _parse_yolo_output(self, results) -> List[Detection]:
        """Extract bounding boxes and confidence scores"""
        # YOLO output format: [x_center, y_center, width, height]
        # Convert to [x_min, y_min, x_max, y_max]
    
    def _crop_patch(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract region from frame with padding"""
        # Add 10% padding to capture context
```

### 2. Tracking Engine (`tracking_engine.py`)

**Responsibility**: Maintain entity identity across frames

```python
class TrackingEngine:
    def __init__(self, config: TrackingConfig):
        self.entities: Dict[int, Entity] = {}
        self.next_id = 0
        self.embedding_similarity_threshold = 0.7
        self.temporal_consistency_weight = 0.3
    
    def update_tracks(self, frame_idx: int, detections: List[Detection], 
                     embeddings: np.ndarray) -> List[Tuple[int, Detection]]:
        """
        Match detections to existing entities using Hungarian algorithm
        
        Args:
            frame_idx: Current frame number
            detections: List of detections in this frame
            embeddings: Matrix [N, embed_dim] of detection embeddings
        
        Returns:
            List of (entity_id, detection) tuples with assignments
        
        Algorithm:
            1. Compute distance matrix between current embeddings and entity histories
            2. Use Hungarian algorithm to find optimal assignment
            3. Register new entities for unmatched detections
            4. Update existing entities with new detections
        
        Distance computation:
            For detection d and entity e:
            distance(d, e) = 
                (1 - embedding_similarity(d.embedding, e.latest_embedding)) * 0.7 +
                spatial_distance(d.bbox, e.latest_bbox) / max_distance * 0.3
        """
        
        if not self.entities:
            # First frame: register all detections as new entities
            return [(self._register_entity(frame_idx, d, emb), d) 
                   for d, emb in zip(detections, embeddings)]
        
        # Compute cost matrix
        cost_matrix = self._compute_cost_matrix(detections, embeddings)
        
        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        assignments = []
        matched_detection_indices = set(col_ind)
        
        for entity_idx, detection_idx in zip(row_ind, col_ind):
            entity_id = list(self.entities.keys())[entity_idx]
            detection = detections[detection_idx]
            
            # Update entity
            self.entities[entity_id].update(
                frame_idx, detection, embeddings[detection_idx]
            )
            assignments.append((entity_id, detection))
        
        # Register new entities for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detection_indices:
                entity_id = self._register_entity(frame_idx, detection, embeddings[i])
                assignments.append((entity_id, detection))
        
        return assignments
    
    def _compute_cost_matrix(self, detections: List[Detection], 
                            embeddings: np.ndarray) -> np.ndarray:
        """Compute distance matrix for assignment problem"""
        num_entities = len(self.entities)
        num_detections = len(detections)
        cost_matrix = np.full((num_entities, num_detections), 1e6)
        
        entity_ids = list(self.entities.keys())
        
        for i, entity_id in enumerate(entity_ids):
            entity = self.entities[entity_id]
            entity_embedding = entity.get_representative_embedding()
            entity_bbox = entity.get_latest_bbox()
            
            for j, detection in enumerate(detections):
                # Embedding similarity (cosine distance)
                embedding_sim = 1 - np.dot(entity_embedding, embeddings[j])
                
                # Spatial distance (normalized)
                spatial_dist = self._bbox_iou_distance(entity_bbox, detection.bbox)
                
                # Combined cost
                cost = 0.7 * embedding_sim + 0.3 * spatial_dist
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def _bbox_iou_distance(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Compute 1 - IoU as distance metric"""
        iou = self._compute_iou(bbox1, bbox2)
        return 1.0 - iou
    
    def _register_entity(self, frame_idx: int, detection: Detection, 
                        embedding: np.ndarray) -> int:
        """Create new entity from detection"""
        entity_id = self.next_id
        self.next_id += 1
        
        self.entities[entity_id] = Entity(
            id=entity_id,
            class_name=detection.class_name,
            first_appearance_frame=frame_idx,
            embeddings=[embedding],
            bboxes=[detection.bbox],
            confidences=[detection.confidence]
        )
        
        return entity_id
```

### 3. Semantic Uplift Engine (`semantic_uplift.py`)

**Responsibility**: Detect state changes and compose high-level events

```python
class EntityTracker:
    """Maintains entity state across temporal windows"""
    
    def __init__(self, clustering_config: ClusteringConfig):
        self.clusterer = HDBSCAN(
            min_cluster_size=clustering_config.min_cluster_size,
            min_samples=clustering_config.min_samples,
            metric=clustering_config.metric
        )
        self.entity_states: Dict[int, EntityState] = {}
    
    def update_states(self, entities: List[Entity], frame_idx: int):
        """Update entity states based on current observations"""
        for entity in entities:
            state = EntityState(
                entity_id=entity.id,
                position=entity.get_latest_bbox().center,
                velocity=entity.compute_velocity(),  # Δposition/Δframe
                confidence=entity.latest_confidence,
                class_name=entity.class_name,
                temporal_consistency=self._compute_consistency(entity)
            )
            self.entity_states[entity.id] = state


class StateChangeDetector:
    """Identifies behavioral transitions"""
    
    def __init__(self, temporal_window_size: int = 30):
        self.window_size = temporal_window_size
        self.state_history: Dict[int, List[EntityState]] = {}
    
    def detect_changes(self, entity_states: Dict[int, EntityState]) -> List[StateChange]:
        """
        Detect state transitions across temporal window
        
        Returns:
            List of StateChange events with:
            - entity_id
            - change_type: "position", "velocity", "class", "appearance"
            - confidence: float in [0, 1]
            - timestamp
        
        Detection Algorithm:
            1. Maintain rolling window of entity states (last 30 frames)
            2. Compute velocity: v_t = (pos_t - pos_{t-1}) / dt
            3. Detect change if ||v_t|| > velocity_threshold OR
                   Δconfidence > 0.2 OR
                   ||accel|| > acceleration_threshold
            4. Score confidence based on temporal consistency
        """
        changes = []
        
        for entity_id, state in entity_states.items():
            if entity_id not in self.state_history:
                self.state_history[entity_id] = []
            
            # Maintain window
            self.state_history[entity_id].append(state)
            if len(self.state_history[entity_id]) > self.window_size:
                self.state_history[entity_id].pop(0)
            
            # Analyze state sequence
            history = self.state_history[entity_id]
            
            if len(history) >= 2:
                prev_state = history[-2]
                curr_state = history[-1]
                
                # Check for changes
                if self._is_significant_position_change(prev_state, curr_state):
                    changes.append(StateChange(
                        entity_id=entity_id,
                        change_type="motion",
                        description="significant position change",
                        confidence=self._compute_change_confidence(history)
                    ))
                
                if curr_state.confidence - prev_state.confidence > 0.15:
                    changes.append(StateChange(
                        entity_id=entity_id,
                        change_type="appearance",
                        description="confidence increase",
                        confidence=abs(curr_state.confidence - prev_state.confidence)
                    ))
        
        return changes


class EventComposer:
    """Generates natural language event descriptions using LLM"""
    
    def __init__(self, ollama_config: OllamaConfig):
        self.client = httpx.Client(base_url=ollama_config.base_url)
        self.model = ollama_config.model
    
    def compose_event(self, state_changes: List[StateChange], 
                      entities: Dict[int, Entity]) -> Event:
        """
        Generate natural language description of event
        
        Args:
            state_changes: List of detected state transitions
            entities: Entity information for context
        
        Returns:
            Event with:
            - event_type: classified category
            - description: natural language text
            - confidence: composite score
            - affected_entities: List of entity IDs
        
        LLM Prompt Construction:
            System: You are analyzing video understanding output. Describe events concisely.
            User: [Entity descriptions] [State changes] → What happened?
        """
        
        # Build context
        context = self._build_event_context(state_changes, entities)
        
        # LLM generation
        prompt = f"""Given these video analysis observations:
        
        {context}
        
        Describe the event in one sentence focusing on what changed and which entities were involved."""
        
        response = self.client.post(
            "/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": 0.7,
                "stream": False
            }
        )
        
        description = response.json()["response"].strip()
        
        return Event(
            event_type=self._classify_event_type(description),
            description=description,
            affected_entities=[change.entity_id for change in state_changes],
            confidence=self._compute_event_confidence(state_changes),
            timestamp=time.time()
        )
```

### 4. Knowledge Graph Builder (`knowledge_graph.py`)

**Responsibility**: Construct rich Neo4j representation

```python
class KnowledgeGraphBuilder:
    
    def __init__(self, neo4j_manager: Optional[Neo4jManager] = None):
        if neo4j_manager is None:
            config = ConfigManager.get_config()
            neo4j_manager = Neo4jManager(
                uri=config.neo4j.uri,
                user=config.neo4j.user,
                password=config.neo4j.password
            )
        self.manager = neo4j_manager
        self.driver = self.manager.driver
    
    def build_graph(self, semantic_data: Dict[str, Any]) -> bool:
        """
        Main ingestion pipeline
        
        Input Structure:
        {
            'scenes': List[Scene],
            'entities': List[Entity],
            'events': List[Event],
            'state_changes': List[StateChange],
            'spatial_relationships': List[SpatialRelationship],
            'causal_links': List[CausalLink],
            'temporal_ordering': List[TemporalOrdering]
        }
        
        Execution Flow:
            1. Initialize schema (constraints, indexes)
            2. Create scene nodes
            3. Create entity nodes
            4. Link entities to scenes
            5. Add spatial relationships
            6. Compose and link events
            7. Add causal relationships
            8. Add temporal ordering
        """
        
        try:
            self.manager.connect()
            self._initialize_schema()
            
            self._create_scene_nodes(semantic_data['scenes'])
            self._create_entity_nodes(semantic_data['entities'])
            self._link_entities_to_scenes(semantic_data)
            self._create_spatial_relationships(semantic_data['spatial_relationships'])
            self._create_event_nodes(semantic_data['events'])
            self._create_causal_relationships(semantic_data['causal_links'])
            self._create_temporal_ordering(semantic_data['temporal_ordering'])
            
            return True
        finally:
            self.manager.close()
    
    def _initialize_schema(self):
        """Create constraints and indexes for performance"""
        with self.driver.session() as session:
            # Uniqueness constraints
            session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            session.run("CREATE CONSTRAINT scene_id IF NOT EXISTS FOR (s:Scene) REQUIRE s.id IS UNIQUE")
            session.run("CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE")
            
            # Indexes for query performance
            session.run("CREATE INDEX scene_type IF NOT EXISTS FOR (s:Scene) ON (s.scene_type)")
            session.run("CREATE INDEX entity_class IF NOT EXISTS FOR (e:Entity) ON (e.class)")
            session.run("CREATE INDEX event_type IF NOT EXISTS FOR (e:Event) ON (e.event_type)")
            session.run("CREATE INDEX entity_appearance IF NOT EXISTS FOR (e:Entity) ON (e.appearance_count)")
    
    def _create_scene_nodes(self, scenes: List[Scene]):
        """Create Scene nodes with classification"""
        scene_classifier = SceneClassifier()
        
        with self.driver.session() as session:
            for scene in scenes:
                scene_type = scene_classifier.classify(scene)
                
                session.run("""
                    CREATE (s:Scene {
                        id: $id,
                        scene_type: $type,
                        timestamp_start: $start,
                        timestamp_end: $end,
                        duration_ms: $duration,
                        description: $desc
                    })
                """, {
                    'id': scene.id,
                    'type': scene_type,
                    'start': scene.timestamp_start,
                    'end': scene.timestamp_end,
                    'duration': scene.timestamp_end - scene.timestamp_start,
                    'desc': scene.description
                })
    
    def _create_entity_nodes(self, entities: List[Entity]):
        """Create Entity nodes with properties"""
        with self.driver.session() as session:
            for entity in entities:
                session.run("""
                    CREATE (e:Entity {
                        id: $id,
                        class: $class,
                        appearance_count: $count,
                        average_confidence: $conf,
                        first_seen: $first,
                        last_seen: $last,
                        trajectory_length: $length,
                        description: $desc
                    })
                """, {
                    'id': entity.id,
                    'class': entity.class_name,
                    'count': entity.appearance_count,
                    'conf': entity.average_confidence,
                    'first': entity.first_appearance_frame,
                    'last': entity.last_appearance_frame,
                    'length': entity.trajectory_length,
                    'desc': entity.description
                })
    
    def _create_spatial_relationships(self, relationships: List[SpatialRelationship]):
        """Create spatial relationship edges"""
        with self.driver.session() as session:
            for rel in relationships:
                rel_type = f"SPATIAL_{rel.relationship_type.upper()}"
                
                session.run(f"""
                    MATCH (e1:Entity {{id: $e1_id}})
                    MATCH (e2:Entity {{id: $e2_id}})
                    CREATE (e1)-[:{rel_type} {{
                        distance: $dist,
                        confidence: $conf,
                        frame: $frame
                    }}]->(e2)
                """, {
                    'e1_id': rel.entity1_id,
                    'e2_id': rel.entity2_id,
                    'dist': rel.distance,
                    'conf': rel.confidence,
                    'frame': rel.frame
                })
    
    def _perform_causal_reasoning(self, events: List[Event]):
        """
        Perform causal analysis between events
        
        Algorithm:
            For each pair of events (e1, e2):
            1. Check temporal ordering: e1.timestamp < e2.timestamp
            2. Check spatial proximity: affected entities within spatial_threshold
            3. Compute conditional probability: P(e2 | e1)
            4. If P(e2 | e1) > causal_threshold, create CAUSES relationship
            
        Causal Scoring:
            causal_score = 
                0.3 * temporal_proximity_score +
                0.3 * spatial_proximity_score +
                0.2 * entity_overlap_score +
                0.2 * semantic_similarity_score
        """
        
        causal_relationships = []
        
        for i, e1 in enumerate(events):
            for e2 in events[i+1:]:
                if e2.timestamp - e1.timestamp <= self.max_causality_window:
                    score = self._compute_causality_score(e1, e2)
                    if score > self.causality_threshold:
                        causal_relationships.append({
                            'source': e1.id,
                            'target': e2.id,
                            'score': score,
                            'type': self._classify_causal_type(e1, e2)
                        })
        
        # Store in Neo4j
        with self.driver.session() as session:
            for rel in causal_relationships:
                session.run("""
                    MATCH (e1:Event {id: $source})
                    MATCH (e2:Event {id: $target})
                    CREATE (e1)-[c:CAUSES {
                        confidence: $score,
                        causal_type: $type
                    }]->(e2)
                """, rel)
```

---

## Pipeline Execution Flow

### End-to-End Workflow

```python
def run_pipeline(
    video_path: str,
    config: Optional[OrionConfig] = None,
    neo4j_uri: Optional[str] = None,
    neo4j_password: Optional[str] = None
) -> PipelineResult:
    """
    Complete video analysis pipeline
    
    Phases:
        1. Initialize components with config resolution
        2. Load video and stream frames
        3. Perception: detect objects and generate embeddings
        4. Tracking: maintain entity continuity
        5. Semantic Uplift: detect state changes and compose events
        6. Knowledge Graph: build Neo4j representation
        7. Query: answer Q&A about video
    
    Returns:
        PipelineResult with:
        - graph_stats: Neo4j database statistics
        - entity_trajectories: Entity movement data
        - events_detected: List of Event objects
        - query_results: Q&A system outputs
    """
    
    # Phase 1: Configuration
    if config is None:
        config = ConfigManager.get_config()
    
    if neo4j_password is not None:
        config.neo4j.password = neo4j_password
    if neo4j_uri is not None:
        config.neo4j.uri = neo4j_uri
    
    # Phase 2: Component Initialization
    perception_engine = PerceptionEngine(
        config.detection,
        config.embedding
    )
    tracking_engine = TrackingEngine(config.temporal_window_size)
    semantic_uplift = SemanticUplift(config)
    kg_builder = KnowledgeGraphBuilder()
    
    # Phase 3: Video Processing
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    all_entities = {}
    all_events = []
    entity_trajectories = defaultdict(list)
    
    frame_idx = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        # Phase 4: Perception
        perception_result = perception_engine.process_frame(frame)
        detections = perception_result.detections
        embeddings = perception_result.embeddings
        
        # Phase 5: Tracking
        assignments = tracking_engine.update_tracks(
            frame_idx,
            detections,
            embeddings
        )
        
        # Phase 6: Entity trajectory accumulation
        for entity_id, detection in assignments:
            entity = tracking_engine.entities[entity_id]
            entity_trajectories[entity_id].append({
                'frame': frame_idx,
                'bbox': detection.bbox,
                'confidence': detection.confidence,
                'position': detection.bbox.center
            })
        
        frame_idx += 1
    
    # Phase 7: Semantic Uplift (post-frame processing)
    all_entities = tracking_engine.entities
    semantic_result = semantic_uplift.process_entities(
        all_entities,
        entity_trajectories
    )
    all_events = semantic_result.events
    
    # Phase 8: Knowledge Graph Construction
    semantic_data = {
        'scenes': segment_video_into_scenes(all_entities, entity_trajectories),
        'entities': list(all_entities.values()),
        'events': all_events,
        'spatial_relationships': compute_spatial_relationships(all_entities),
        'causal_links': detect_causal_links(all_events),
        'temporal_ordering': compute_temporal_ordering(all_events)
    }
    
    kg_builder.build_graph(semantic_data)
    
    # Phase 9: Results compilation
    neo4j_manager = Neo4jManager(
        uri=config.neo4j.uri,
        user=config.neo4j.user,
        password=config.neo4j.password
    )
    graph_stats = neo4j_manager.get_stats()
    
    return PipelineResult(
        graph_stats=graph_stats,
        entity_trajectories=entity_trajectories,
        events_detected=all_events,
        total_frames_processed=total_frames
    )
```

---

## Mathematical Foundations

### 1. Cosine Similarity for Embedding Matching

**Purpose**: Compare CLIP embeddings to maintain entity identity across frames

$$\text{similarity}(e_1, e_2) = \frac{e_1 \cdot e_2}{\|e_1\| \|e_2\|} \in [-1, 1]$$

For normalized embeddings (L2 norm = 1):
$$\text{similarity} = e_1 \cdot e_2$$

**Distance metric**:
$$\text{distance} = 1 - \text{similarity}$$

**Matching threshold**: 0.7 (configurable)

### 2. Hungarian Algorithm for Assignment

**Problem**: Assign M detections to N entities minimizing cost

**Cost matrix**:
$$C_{ij} = 0.7 \times (1 - \text{embedding\_sim}_{ij}) + 0.3 \times \text{spatial\_dist}_{ij}$$

**Solution**: $\text{Hungarian}(C)$ finds permutation minimizing $\sum_i C_{i,\pi(i)}$

**Complexity**: O(N³) for N×N matrix

### 3. HDBSCAN Clustering

**Purpose**: Group detections into entities based on embedding space density

**Key parameters**:
- `min_cluster_size`: Minimum points per cluster (default: 10)
- `min_samples`: Sparsity parameter (default: 5)
- `metric`: Distance function (default: "cosine" for embeddings)

**Algorithm**:
1. Compute k-NN graph with k = min_samples
2. Build mutual reachability graph
3. Compute MST of mutual reachability distances
4. Extract hierarchical clustering
5. Select stable clusters by lifetime criterion

### 4. Temporal State Change Detection

**Velocity computation** (Δposition/Δframe):
$$v_t = \frac{p_t - p_{t-1}}{\Delta t}$$

**Acceleration**:
$$a_t = \frac{v_t - v_{t-1}}{\Delta t}$$

**Change detection**:
$$\text{changed} = \|v_t\| > \text{velocity\_threshold} \text{ OR } \|a_t\| > \text{accel\_threshold}$$

### 5. Causal Scoring Function

$$\text{causal\_score}(e_1 \to e_2) = 0.3 \times T + 0.3 \times S + 0.2 \times O + 0.2 \times M$$

Where:
- $T$: Temporal proximity (1 if $\Delta t < \text{max\_window}$, else 0)
- $S$: Spatial proximity (1 if affected entities within threshold, else 0)
- $O$: Entity overlap (fraction of shared entities)
- $M$: Semantic/LLM similarity (0 to 1)

---

## Database Schema

### Neo4j Node Types

```cypher
// Scene Node
CREATE (s:Scene {
    id: String,                    // unique: "scene_001"
    scene_type: String,            // "office", "outdoor", "indoor"
    timestamp_start: Integer,      // milliseconds
    timestamp_end: Integer,
    duration: Integer,             // end - start
    description: String            // Generated description
})

// Entity Node
CREATE (e:Entity {
    id: String,                    // unique: "entity_042"
    class: String,                 // "person", "car", "desk"
    appearance_count: Integer,     // frames where visible
    average_confidence: Float,     // mean detection confidence
    first_seen: Integer,           // frame number
    last_seen: Integer,
    trajectory_length: Integer,    // number of positions
    description: String            // Generated description
})

// Event Node
CREATE (event:Event {
    id: String,                    // unique: "event_012"
    event_type: String,            // "motion", "interaction", "state_change"
    description: String,           // Natural language description
    confidence: Float,             // Composite confidence score
    timestamp: Integer,            // When detected
    affected_entities: [String]    // IDs of entities involved
})

// State Change Node
CREATE (sc:StateChange {
    id: String,
    entity_id: String,             // Reference to Entity
    change_type: String,           // "position", "velocity", "appearance"
    description: String,
    confidence: Float,
    timestamp: Integer
})
```

### Neo4j Relationship Types

```cypher
// Temporal relationships
(e:Entity)-[APPEARS_IN]->(s:Scene)
    Properties: {
        first_frame: Integer,
        last_frame: Integer,
        appearance_duration: Integer
    }

(s1:Scene)-[PRECEDES]->(s2:Scene)
    Properties: {
        gap_duration: Integer
    }

(e1:Event)-[FOLLOWS]->(e2:Event)
    Properties: {
        temporal_gap: Integer
    }

// Spatial relationships
(e1:Entity)-[SPATIAL_LEFT_OF]->(e2:Entity)
(e1:Entity)-[SPATIAL_RIGHT_OF]->(e2:Entity)
(e1:Entity)-[SPATIAL_ABOVE]->(e2:Entity)
(e1:Entity)-[SPATIAL_BELOW]->(e2:Entity)
(e1:Entity)-[SPATIAL_INSIDE]->(e2:Entity)
(e1:Entity)-[SPATIAL_CONTAINS]->(e2:Entity)
    Properties: {
        distance: Float,
        confidence: Float,
        frame: Integer
    }

// Causal relationships
(e1:Event)-[CAUSES]->(e2:Event)
    Properties: {
        confidence: Float,
        causal_type: String    // "direct", "indirect", "enabling"
    }

// State transitions
(e:Entity)-[HAS_STATE]->(sc:StateChange)
    Properties: {
        transition_index: Integer
    }

// Hierarchical relationships
(s:Scene)-[CONTAINS]->(e:Entity)
(s:Scene)-[CONTAINS]->(ev:Event)
```

### Indexes and Constraints

```cypher
// Uniqueness constraints
CREATE CONSTRAINT entity_unique FOR (e:Entity) REQUIRE e.id IS UNIQUE
CREATE CONSTRAINT scene_unique FOR (s:Scene) REQUIRE s.id IS UNIQUE
CREATE CONSTRAINT event_unique FOR (e:Event) REQUIRE e.id IS UNIQUE

// Indexes for query performance
CREATE INDEX entity_class FOR (e:Entity) ON (e.class)
CREATE INDEX scene_type FOR (s:Scene) ON (s.scene_type)
CREATE INDEX event_type FOR (e:Event) ON (e.event_type)
CREATE INDEX entity_appearance FOR (e:Entity) ON (e.appearance_count)
CREATE INDEX scene_timestamp FOR (s:Scene) ON (s.timestamp_start)
```

---

## Integration Points

### Component Communication Flow

```
VIDEO INPUT
    ↓
┌─────────────────────────────────────┐
│ PerceptionEngine                    │
│ - Input: Frame [H×W×3]              │
│ - Output: PerceptionResult          │
│   {detections[], embeddings[], ...} │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│ TrackingEngine                      │
│ - Input: Detections + Embeddings    │
│ - Maintains: Entity ID continuity   │
│ - Output: (entity_id, detection)[]  │
│ - State: self.entities dict         │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│ SemanticUplift                      │
│ - Input: Entity trajectories        │
│ - Operations:                       │
│   1. EntityTracker.update_states()  │
│   2. StateChangeDetector.detect()   │
│   3. EventComposer.compose()        │
│ - Output: Event[]                   │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│ KnowledgeGraphBuilder               │
│ - Input: Complete semantic data     │
│   {scenes, entities, events, ...}   │
│ - Operations:                       │
│   1. Create nodes (Scene/Entity)    │
│   2. Build relationships            │
│   3. Perform causal reasoning       │
│   4. Add temporal ordering          │
│ - Output: Neo4j graph               │
└────────────────┬────────────────────┘
                 ↓
           Neo4j Database
           (Persistent storage)
```

### Configuration Dependency Injection

```python
# All components receive config at initialization
config = ConfigManager.get_config()

# Each component has typed access to its subsystem
engine = PerceptionEngine(
    config.detection,      # DetectionConfig with YOLO params
    config.embedding       # EmbeddingConfig with CLIP params
)

tracker = TrackingEngine(
    config.temporal_window_size
)

uplift = SemanticUplift(
    config.clustering,     # ClusteringConfig for HDBSCAN
    config.description,    # DescriptionConfig for FastVLM
    config.ollama          # OllamaConfig for LLM
)

kg = KnowledgeGraphBuilder(
    # Neo4jManager automatically uses ConfigManager
    # if not explicitly provided
)
```

### Data Type Contracts

```python
# Perception → Tracking
@dataclass
class PerceptionResult:
    detections: List[Detection]        # [N] YOLO outputs
    embeddings: np.ndarray             # [N, embed_dim] normalized
    processing_time_ms: float

# Tracking → Semantic Uplift
@dataclass
class Entity:
    id: int
    class_name: str
    embeddings: List[np.ndarray]       # Temporal sequence
    bboxes: List[Tuple[int,int,int,int]]
    confidences: List[float]

# Semantic Uplift → Knowledge Graph
@dataclass
class Event:
    id: str
    event_type: str
    description: str
    confidence: float
    affected_entities: List[int]
    timestamp: float

# Knowledge Graph → Query/Q&A
@dataclass
class GraphQueryResult:
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    properties: Dict[str, Any]
```

---

## Performance Considerations

### Computational Bottlenecks

1. **Object Detection (YOLO)**
   - ~40-100 ms per frame (depending on model size)
   - Can batch frames for streaming scenarios
   - Model selection impacts detection quality vs speed

2. **Embedding Generation (CLIP)**
   - ~5-20 ms per frame (10-20 objects per frame typical)
   - Batch processing recommended
   - L2 normalization is essential for cosine similarity

3. **HDBSCAN Clustering**
   - O(N log N) complexity
   - Can be expensive with large entity populations
   - Consider parameter tuning for speed/quality trade-off

4. **Neo4j Ingestion**
   - Batch writes significantly faster than individual creates
   - Index creation adds overhead first time
   - Subsequent queries use indexes effectively

### Optimization Strategies

- **Frame Sampling**: Process every Nth frame for initial analysis
- **Adaptive Tracking**: Increase tracking frequency during high-motion scenes
- **Lazy Neo4j Connection**: Connect only when building graphs
- **Embedding Caching**: Cache common object embeddings
- **Parallel Processing**: Use multiprocessing for perception on multi-core systems

---

## Validation & Testing

### Unit Test Coverage

- `test_tracking.py`: Entity tracking and assignment
- `test_semantic_uplift.py`: State detection and event composition
- `test_knowledge_graph.py`: Neo4j schema and queries
- `test_integration.py`: End-to-end pipeline

### Evaluation Metrics

- **Tracking Quality**: Intersection over Union (IoU) with ground truth trajectories
- **Event Detection**: Precision, recall, F1 on event timing
- **Knowledge Graph**: Query correctness and completeness

---

This architecture document comprehensively covers the entire Orion pipeline from configuration through Neo4j persistence, with mathematical foundations and integration specifications suitable for research publication.
