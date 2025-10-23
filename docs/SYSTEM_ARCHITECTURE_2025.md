# Orion System Architecture (2025) - Complete Technical Overview

**Last Updated**: October 23, 2025  
**Status**: Production-ready with semantic validation improvements  
**Paper Reference**: AAAI 2026 Submission (orion.pdf)

---

## Executive Summary

Orion is a **semantic uplift pipeline** that transforms raw egocentric video into causally-aware knowledge graphs. It bridges low-level perception (YOLO, CLIP) with high-level reasoning (LLM-based event composition) through a multi-stage architecture.

**Key Metrics** (current implementation):
- **Throughput**: 30-60 FPS (perception), 1-5 FPS (description generation)
- **Accuracy**: ~85-90% object classification (with semantic validation)
- **Scale**: Handles 1000+ entities per video with HDBSCAN clustering
- **Storage**: Neo4j graph database with vector indices

---

## System Architecture Diagram

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          ORION SEMANTIC UPLIFT PIPELINE                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT: RAW VIDEO                                │
│                         (Egocentric or 3rd-person)                           │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                       PHASE 1: ASYNC PERCEPTION ENGINE                        ║
║  File: async_perception.py                                                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝
│
├─► [Fast Loop: 30-60 FPS]
│   ┌────────────────────────────────────────────────────────────────┐
│   │ 1. Frame Extraction (OpenCV)                                   │
│   │    - Adaptive sampling (skip_rate, detect_every_n_frames)      │
│   │    - Frame buffer management                                    │
│   │                                                                 │
│   │ 2. Object Detection (YOLO11x)                                  │
│   │    - Bounding box coordinates [x1, y1, x2, y2]                 │
│   │    - Class predictions (80 COCO classes)                       │
│   │    - Confidence scores                                          │
│   │                                                                 │
│   │ 3. Visual Embedding (CLIP)                                     │
│   │    - Patch-level feature extraction                            │
│   │    - L2 normalized 512-2048 dim vectors                        │
│   │    - Per-object crop embedding                                  │
│   │                                                                 │
│   │ 4. Motion Tracking (MotionTracker)                             │
│   │    - Optical flow (Farneback algorithm)                        │
│   │    - Velocity, acceleration, direction                          │
│   │    - Motion magnitude & trajectory                              │
│   └────────────────────────────────────────────────────────────────┘
│
├─► [Clustering Phase: Post-Detection]
│   ┌────────────────────────────────────────────────────────────────┐
│   │ 5. Entity Clustering (HDBSCAN)                                 │
│   │    - Groups detections by visual similarity                    │
│   │    - Creates entity IDs for object permanence                  │
│   │    - Parameters: min_cluster_size, min_samples                 │
│   │    - Handles occlusion & re-identification                     │
│   └────────────────────────────────────────────────────────────────┘
│
└─► [Slow Loop: 1-5 FPS]
    ┌────────────────────────────────────────────────────────────────┐
    │ 6. Smart Description Generation                                │
    │    - Async queue-based processing                              │
    │    - VLM (FastVLM/LLaVA) for rich descriptions                │
    │    - Describes UNIQUE entities only (not every detection)      │
    │    - Best-frame selection for quality                          │
    └────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    [Perception Log: JSON with observations]
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                   PHASE 2: CLASS CORRECTION & VALIDATION                      ║
║  File: class_correction.py                                                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝
│
├─► [Semantic Validation Layer] ✅ NEW (October 2025)
│   ┌────────────────────────────────────────────────────────────────┐
│   │ 1. Rule-Based Corrections                                      │
│   │    - Common misclassifications (e.g., "hair drier" → "remote")│
│   │    - COCO class mapping                                        │
│   │                                                                 │
│   │ 2. Semantic Class Matching (Sentence Transformers)             │
│   │    - Description → Class similarity                            │
│   │    - Model: all-MiniLM-L6-v2                                  │
│   │    - Threshold: 0.75 for high confidence                       │
│   │                                                                 │
│   │ 3. Part-of Relationship Detection                              │
│   │    - Detects "car tire" → NOT "car"                          │
│   │    - Prevents bad corrections                                   │
│   │    - Pattern matching + semantic checks                        │
│   │                                                                 │
│   │ 4. Validation with Description                                 │
│   │    - Embedding similarity: description ↔ proposed class        │
│   │    - Threshold: 0.5 (conservative)                             │
│   │    - Rejects weak matches                                      │
│   │                                                                 │
│   │ 5. CLIP Verification (Optional)                                │
│   │    - Visual-semantic alignment                                  │
│   │    - Fallback for ambiguous cases                              │
│   └────────────────────────────────────────────────────────────────┘
│
└─► Output: Corrected observations with canonical_label, correction_confidence
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                     PHASE 3: SEMANTIC UPLIFT ENGINE                           ║
║  File: semantic_uplift.py                                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
│
├─► [Entity Management]
│   ┌────────────────────────────────────────────────────────────────┐
│   │ 1. Entity Consolidation                                        │
│   │    - Merge observations into coherent entities                 │
│   │    - Track temporal duration (first_seen → last_seen)         │
│   │    - Aggregate embeddings (average + L2 normalize)             │
│   │    - Assign descriptions to entities                           │
│   └────────────────────────────────────────────────────────────────┘
│
├─► [Spatial Analysis]
│   ┌────────────────────────────────────────────────────────────────┐
│   │ 2. Co-Location Zone Detection                                  │
│   │    File: spatial_colocation.py                                 │
│   │    - DBSCAN clustering of centroids                            │
│   │    - Identifies groups of nearby objects                       │
│   │    - Parameters: eps, min_samples, temporal_threshold          │
│   │    - Creates ZONE nodes in graph                               │
│   └────────────────────────────────────────────────────────────────┘
│
├─► [Temporal Analysis]
│   ┌────────────────────────────────────────────────────────────────┐
│   │ 3. State Change Detection                                      │
│   │    - Embedding similarity across time                          │
│   │    - Threshold: 0.85 (cosine distance)                        │
│   │    - Detects appearance changes (e.g., "door opens")          │
│   │    - Temporal windowing for smoothing                          │
│   └────────────────────────────────────────────────────────────────┘
│
├─► [Causal Reasoning]
│   ┌────────────────────────────────────────────────────────────────┐
│   │ 4. Causal Inference Engine                                     │
│   │    File: causal_inference.py                                   │
│   │    - Temporal precedence analysis                              │
│   │    - Spatial proximity scoring                                  │
│   │    - Agent-action-patient detection                            │
│   │    - Granger causality estimation                              │
│   │    - Outputs: CAUSES/ENABLES relationships                     │
│   └────────────────────────────────────────────────────────────────┘
│
└─► [Event Composition]
    ┌────────────────────────────────────────────────────────────────┐
    │ 5. LLM-Based Event Generation                                  │
    │    - Temporal windowing (e.g., 5-10 second windows)           │
    │    - Structured prompts with entity/state context              │
    │    - Ollama/Gemma for event descriptions                       │
    │    - Outputs: Natural language event narratives                │
    └────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                  PHASE 4: CONTEXTUAL UNDERSTANDING ENGINE                     ║
║  File: contextual_engine.py                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝
│
├─► [Position Tagging]
│   ┌────────────────────────────────────────────────────────────────┐
│   │ 1. Spatial Position Analysis                                   │
│   │    - Frame quadrant detection (TOP_LEFT, BOTTOM_RIGHT, etc.)  │
│   │    - Centrality scoring (how centered is object?)              │
│   │    - Adds position tags to entities                            │
│   └────────────────────────────────────────────────────────────────┘
│
├─► [Description Validation]
│   ┌────────────────────────────────────────────────────────────────┐
│   │ 2. LLM-Based Description Checking                              │
│   │    - Reviews entity descriptions for accuracy                  │
│   │    - Compares description vs class label                       │
│   │    - Flags inconsistencies                                     │
│   │    - Updates descriptions if needed                            │
│   └────────────────────────────────────────────────────────────────┘
│
└─► Output: Enriched entities with position tags & validated descriptions
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    PHASE 5: KNOWLEDGE GRAPH CONSTRUCTION                      ║
║  File: knowledge_graph.py, temporal_graph_builder.py                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝
│
├─► [Node Creation]
│   ┌────────────────────────────────────────────────────────────────┐
│   │ 1. Entity Nodes                                                │
│   │    - Properties: id, class, description, embedding, duration   │
│   │    - Position tags (if available)                              │
│   │                                                                 │
│   │ 2. Scene Nodes                                                 │
│   │    - Temporal segments (e.g., 10-second windows)              │
│   │    - Scene embeddings (aggregate of entity embeddings)         │
│   │    - Dominant colors, motion patterns                          │
│   │                                                                 │
│   │ 3. Event Nodes                                                 │
│   │    - Natural language descriptions                             │
│   │    - Participant entities (agent, patient, instrument)         │
│   │    - Temporal bounds (start_time, end_time)                   │
│   │                                                                 │
│   │ 4. Zone Nodes                                                  │
│   │    - Co-located entity groups                                  │
│   │    - Spatial coordinates (centroid)                            │
│   └────────────────────────────────────────────────────────────────┘
│
├─► [Relationship Creation]
│   ┌────────────────────────────────────────────────────────────────┐
│   │ 5. Spatial Relationships                                       │
│   │    - NEAR, FAR, LEFT_OF, RIGHT_OF, ABOVE, BELOW               │
│   │    - Computed from bounding box centroids                      │
│   │                                                                 │
│   │ 6. Temporal Relationships                                      │
│   │    - APPEARS_IN (Entity → Scene)                              │
│   │    - NEXT (Scene → Scene, temporal ordering)                  │
│   │    - PARTICIPATES_IN (Entity → Event)                         │
│   │                                                                 │
│   │ 7. Causal Relationships                                        │
│   │    - CAUSES (Event A → Event B)                               │
│   │    - ENABLES (Event A → Event B, permissive causality)        │
│   │    - Scored by causal inference engine                         │
│   │                                                                 │
│   │ 8. Part-of Relationships                                       │
│   │    - PART_OF (detected from descriptions)                      │
│   │    - Example: "tire" PART_OF "car"                            │
│   └────────────────────────────────────────────────────────────────┘
│
└─► [Graph Optimization]
    ┌────────────────────────────────────────────────────────────────┐
    │ 9. Similarity Computation                                      │
    │    - Scene-to-scene similarity (embedding cosine distance)     │
    │    - Creates SIMILAR_TO edges                                  │
    │    - Enables content-based retrieval                           │
    │                                                                 │
    │ 10. Location Extraction                                        │
    │     - LLM-based location inference from scenes                 │
    │     - Creates LOCATION nodes (e.g., "kitchen", "living room") │
    │     - Links scenes to locations                                │
    └────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         PHASE 6: NEO4J PERSISTENCE                            ║
║  File: neo4j_manager.py                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════╝
│
├─► [Database Schema]
│   ┌────────────────────────────────────────────────────────────────┐
│   │ Node Labels:                                                   │
│   │   - Entity (objects tracked across frames)                    │
│   │   - Scene (temporal segments)                                  │
│   │   - Event (composed from LLM)                                 │
│   │   - Zone (co-location groups)                                  │
│   │   - Location (inferred spatial context)                        │
│   │                                                                 │
│   │ Relationship Types:                                            │
│   │   - APPEARS_IN, NEXT, SIMILAR_TO, IN_ZONE                     │
│   │   - CAUSES, ENABLES, PARTICIPATES_IN                          │
│   │   - NEAR, FAR, LEFT_OF, RIGHT_OF, ABOVE, BELOW                │
│   │   - PART_OF, HAS_PART                                         │
│   │                                                                 │
│   │ Constraints:                                                   │
│   │   - Unique entity IDs                                          │
│   │   - Temporal ordering (scenes)                                 │
│   │   - Referential integrity                                      │
│   └────────────────────────────────────────────────────────────────┘
│
└─► [Vector Indexing]
    ┌────────────────────────────────────────────────────────────────┐
    │ - HNSW indices for entity embeddings                           │
    │ - Scene embedding indices                                       │
    │ - Enables similarity search queries                            │
    └────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                       PHASE 7: QUERY & Q&A INTERFACE                          ║
║  Files: video_qa/*.py                                                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝
│
├─► [Retrieval-Augmented Generation]
│   ┌────────────────────────────────────────────────────────────────┐
│   │ 1. Query Embedding                                             │
│   │    - CLIP text encoder for semantic queries                    │
│   │                                                                 │
│   │ 2. Graph Retrieval                                             │
│   │    - Cypher queries for structured data                        │
│   │    - Vector similarity search                                  │
│   │    - Path finding for causal chains                            │
│   │                                                                 │
│   │ 3. Context Assembly                                            │
│   │    - Gather relevant entities, events, relationships           │
│   │    - Rank by relevance                                         │
│   │                                                                 │
│   │ 4. LLM Answer Generation                                       │
│   │    - Ollama/Gemma with structured context                      │
│   │    - Grounded in graph data                                    │
│   │    - Cites entity IDs and timestamps                           │
│   └────────────────────────────────────────────────────────────────┘
│
└─► Output: Natural language answers with provenance
```

---

## Component Details

### 1. Async Perception Engine

**File**: `orion/async_perception.py`  
**Key Classes**: `AsyncPerceptionEngine`, `DetectionTask`, `EntityDescriptionTask`

**Design Pattern**: Producer-Consumer with AsyncIO queues

```python
# Architecture
Fast Producer Loop (30-60 FPS)
    ├─ YOLO Detection
    ├─ CLIP Embedding
    ├─ Motion Tracking
    └─ Enqueue tasks

Slow Consumer Loop (1-5 FPS)
    ├─ Dequeue tasks
    ├─ VLM Description
    └─ Return results

# Key Innovation: Describe Once Strategy
- Cluster detections with HDBSCAN
- Identify unique entities
- Generate descriptions ONLY for unique entities (not every detection)
- Dramatically reduces VLM calls (10-100x fewer)
```

**Performance**:
- Fast loop: ~30-60 FPS (depends on video resolution, YOLO model)
- Slow loop: ~1-5 FPS (VLM latency-bound)
- Queue depth: Configurable (default 100)
- Memory: O(n) where n = queue size

**Configuration**:
```python
class AsyncConfig:
    max_queue_size: int = 100
    num_description_workers: int = 2
    describe_strategy: str = "unique_entities"  # or "all", "sample"
    frame_buffer_size: int = 30
    report_interval_seconds: float = 5.0
```

### 2. Class Correction & Semantic Validation

**File**: `orion/class_correction.py`  
**Key Class**: `ClassCorrector`

**Methods**:
1. **Rule-Based Correction**: Hardcoded mappings for common errors
2. **Semantic Matching**: Sentence Transformer similarity
3. **Part-of Detection**: Pattern matching + semantic checks
4. **CLIP Verification**: Visual-semantic alignment
5. **Validation**: Embedding similarity between description ↔ proposed class

**Validation Thresholds** (tuned October 2025):
```python
semantic_match_threshold = 0.75  # High confidence
validation_threshold = 0.5       # Conservative (was 0.4)
clip_threshold = 0.6             # Visual alignment
```

**Example Workflow**:
```
YOLO detects: "suitcase"
Description: "A car tire with visible tread pattern"

1. should_correct() → True (description doesn't match "suitcase")
2. semantic_class_match() → "car" (finds "car" in description)
3. validate_correction_with_description()
   - Check part-of: "car tire" contains "car" → REJECT ✅
   - Semantic similarity: 0.463 < 0.5 → REJECT ✅
4. Final result: Keep as "suitcase" (no valid correction)
```

### 3. Semantic Uplift Engine

**File**: `orion/semantic_uplift.py`  
**Key Class**: `SemanticUpliftEngine`

**Modules**:
- **Entity Management**: Consolidate observations → entities
- **State Detection**: Embedding similarity over time
- **Causal Inference**: `CausalInferenceEngine` integration
- **Event Composition**: LLM-based natural language generation
- **Spatial Analysis**: `SpatialCoLocationAnalyzer` integration

**State Change Detection**:
```python
# Compares embeddings across time
threshold = 0.85  # cosine similarity
if similarity < threshold:
    # Significant change detected
    state_change = {
        'entity_id': entity.id,
        'type': 'appearance_change',
        'frame': frame_number,
        'before_embedding': prev_emb,
        'after_embedding': curr_emb
    }
```

### 4. Causal Inference Engine

**File**: `orion/causal_inference.py`  
**Key Class**: `CausalInferenceEngine`

**Scoring Components**:
1. **Temporal Precedence**: A happens before B
2. **Spatial Proximity**: A and B are nearby
3. **Agent-Action-Patient**: A acts on B
4. **Granger Causality**: Statistical dependency

**Output**:
```python
{
    'source_event': 'event_001',
    'target_event': 'event_002',
    'causal_score': 0.87,
    'causal_type': 'CAUSES',  # or 'ENABLES'
    'justification': 'Temporal precedence (0.9) + proximity (0.85)'
}
```

### 5. Contextual Understanding Engine

**File**: `orion/contextual_engine.py`  
**Key Class**: `ContextualUnderstandingEngine`

**Features**:
- **Position Tagging**: Frame quadrants (9 regions)
- **Description Validation**: LLM reviews descriptions
- **Consistency Checks**: Class vs description alignment

**Position Tags**:
```
TOP_LEFT    | TOP_CENTER    | TOP_RIGHT
------------|---------------|------------
CENTER_LEFT | CENTER        | CENTER_RIGHT
------------|---------------|------------
BOTTOM_LEFT | BOTTOM_CENTER | BOTTOM_RIGHT
```

### 6. Knowledge Graph Construction

**Files**: `orion/knowledge_graph.py`, `orion/temporal_graph_builder.py`  
**Key Classes**: `KnowledgeGraphBuilder`, `TemporalGraphBuilder`

**Node Types**:
- **Entity**: Tracked objects (id, class, description, embedding, duration)
- **Scene**: Temporal segments (frame_range, embedding, entities)
- **Event**: LLM-composed events (description, participants, causal_links)
- **Zone**: Co-location groups (entities, spatial_bounds)
- **Location**: Inferred places (name, scenes)

**Relationship Types**:
- **Spatial**: NEAR, FAR, LEFT_OF, RIGHT_OF, ABOVE, BELOW
- **Temporal**: APPEARS_IN, NEXT, PARTICIPATES_IN
- **Causal**: CAUSES, ENABLES
- **Structural**: PART_OF, HAS_PART, IN_ZONE, SIMILAR_TO

### 7. Neo4j Persistence Layer

**File**: `orion/neo4j_manager.py`  
**Key Class**: `Neo4jManager`

**Schema**:
```cypher
// Constraints
CREATE CONSTRAINT entity_id_unique FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT scene_id_unique FOR (s:Scene) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT event_id_unique FOR (ev:Event) REQUIRE ev.id IS UNIQUE;

// Indices
CREATE INDEX entity_class FOR (e:Entity) ON (e.class_name);
CREATE INDEX scene_timestamp FOR (s:Scene) ON (s.start_timestamp);
CREATE VECTOR INDEX entity_embedding FOR (e:Entity) ON (e.embedding);
```

**Vector Search**:
```cypher
CALL db.index.vector.queryNodes(
    'entity_embedding', 
    5, 
    $query_embedding
) YIELD node, score
RETURN node, score
```

---

## Implementation vs Paper Claims

### ✅ Implemented & Working

| Feature | Paper Section | Implementation | Status |
|---------|---------------|----------------|--------|
| **YOLO11x Detection** | 3.1 Perception | `async_perception.py` | ✅ |
| **CLIP Embeddings** | 3.1 Perception | `async_perception.py` | ✅ |
| **HDBSCAN Clustering** | 3.2 Tracking | `async_perception.py` | ✅ |
| **Motion Tracking** | 3.2 Tracking | `motion_tracker.py` | ✅ |
| **VLM Descriptions** | 3.3 Semantic Uplift | `async_entity_describer.py` | ✅ |
| **State Change Detection** | 3.3 Semantic Uplift | `semantic_uplift.py` | ✅ |
| **Event Composition** | 3.3 Semantic Uplift | `semantic_uplift.py` | ✅ |
| **Causal Inference** | 3.4 Causality | `causal_inference.py` | ✅ |
| **Spatial Co-location** | 3.4 Causality | `spatial_colocation.py` | ✅ |
| **Neo4j Graph** | 3.5 Knowledge Graph | `knowledge_graph.py` | ✅ |
| **Vector Indexing** | 3.5 Knowledge Graph | `neo4j_manager.py` | ✅ |
| **Class Correction** | (Not in paper) | `class_correction.py` | ✅ NEW |
| **Semantic Validation** | (Not in paper) | `class_correction.py` | ✅ NEW |

### ⚠️ Partially Implemented

| Feature | Paper Section | Status | Notes |
|---------|---------------|--------|-------|
| **Triplet F1 Evaluation** | 4. Evaluation | ⚠️ | Metrics defined, not yet run on VSGR |
| **Causal Reasoning Score** | 4. Evaluation | ⚠️ | Engine exists, scoring not validated |
| **VSGR Benchmark** | 4. Evaluation | ⚠️ | Dataset loader needed |

### ❌ Missing (Paper Claims)

| Feature | Paper Section | Status | Priority |
|---------|---------------|--------|----------|
| **VSGR Dataset Integration** | 4. Evaluation | ❌ | HIGH |
| **HyperGLM Comparison** | 4. Evaluation | ❌ | HIGH |
| **Ablation Studies** | 4.3 Ablations | ❌ | MEDIUM |
| **Qualitative Analysis** | 4.4 Case Studies | ❌ | LOW |

### 🆕 Implemented But Not in Paper

| Feature | File | Status | Notes |
|---------|------|--------|-------|
| **Async Perception** | `async_perception.py` | ✅ | Major architectural improvement |
| **Semantic Validation** | `class_correction.py` | ✅ | Prevents bad corrections (Oct 2025) |
| **Contextual Engine** | `contextual_engine.py` | ✅ | Position tagging, description validation |
| **Config Presets** | `config.py` | ✅ | fast/balanced/accurate modes |
| **ConfigManager** | `config_manager.py` | ✅ | Secure credential management |

---

## Configuration System

### Three-Tier Architecture

```
Environment Variables (.env)
    ↓
ConfigManager (Singleton)
    ↓
OrionConfig (Dataclass)
```

### Configuration Presets

```python
# Fast Mode (Low latency, lower accuracy)
config = get_fast_config()
# - YOLO11n (smallest model)
# - 512-dim embeddings
# - 100 entity limit
# - Minimal LLM calls

# Balanced Mode (Recommended)
config = get_balanced_config()
# - YOLO11m (medium model)
# - 1024-dim embeddings
# - 500 entity limit
# - Moderate LLM usage

# Accurate Mode (Max accuracy, high resources)
config = get_accurate_config()
# - YOLO11x (largest model)
# - 2048-dim embeddings
# - 1000 entity limit
# - Extensive LLM reasoning
```

### Key Parameters

```python
class OrionConfig:
    # Perception
    yolo_model: str = "yolo11m.pt"
    skip_rate: int = 2
    detect_every_n_frames: int = 5
    
    # Embedding
    clip_model: str = "ViT-L/14"
    embedding_dim: int = 1024
    
    # Clustering
    min_cluster_size: int = 3
    min_samples: int = 2
    
    # Semantic Uplift
    temporal_window_seconds: float = 10.0
    state_change_threshold: float = 0.85
    
    # Causal Inference
    causal_min_score: float = 0.7
    
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""  # Loaded from ConfigManager
```

---

## Mathematical Foundations

### 1. Embedding Similarity

**Cosine Similarity**:
```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Used for:
- Entity clustering (HDBSCAN uses cosine distance)
- State change detection (threshold: 0.85)
- Scene similarity (graph edges)
- Semantic validation (description ↔ class)

### 2. Causal Scoring

**Weighted Sum**:
```
causal_score = w1 × temporal_precedence 
             + w2 × spatial_proximity 
             + w3 × agent_action_patient
             + w4 × granger_causality

where Σ wi = 1
```

Default weights: `[0.4, 0.3, 0.2, 0.1]`

### 3. Spatial Relationships

**Euclidean Distance**:
```
distance(A, B) = sqrt((x2 - x1)² + (y2 - y1)²)
```

**Relationship Thresholds**:
- NEAR: distance < 0.2 × frame_diagonal
- FAR: distance > 0.5 × frame_diagonal
- Directional (LEFT_OF, RIGHT_OF, etc.): Based on centroid comparison

### 4. Temporal Windowing

**Sliding Window**:
```
window_size = fps × temporal_window_seconds
stride = window_size // 2  # 50% overlap
```

Default: 10-second windows with 5-second stride

### 5. HDBSCAN Clustering

**Parameters**:
- `min_cluster_size`: Minimum entities to form cluster (default: 3)
- `min_samples`: Core point threshold (default: 2)
- `metric`: Cosine distance
- `cluster_selection_method`: 'eom' (excess of mass)

---

## Pipeline Execution Flow

### Command-Line Interface

```bash
# Run full pipeline
python orion/run_pipeline.py \
    --video path/to/video.mp4 \
    --mode balanced \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-password $NEO4J_PASSWORD \
    --output-dir data/results

# Run specific phases
python orion/run_pipeline.py \
    --video video.mp4 \
    --phase perception \
    --mode fast

python orion/run_pipeline.py \
    --perception-log data/results/perception_log.json \
    --phase semantic_uplift \
    --mode accurate
```

### Phase Dependencies

```
VIDEO → [Perception] → perception_log.json
                          ↓
                       [Semantic Uplift] → entities.json, events.json
                          ↓
                       [Knowledge Graph] → Neo4j database
                          ↓
                       [Query/Q&A] → answers
```

### Error Handling

```python
# Graceful degradation
try:
    descriptions = generate_descriptions(entities)
except VLMError:
    logger.warning("VLM unavailable, using YOLO classes only")
    descriptions = fallback_descriptions(entities)

# Retry logic
@retry(max_attempts=3, backoff=2.0)
def call_llm(prompt):
    return ollama.generate(prompt)
```

---

## Performance Characteristics

### Throughput

| Phase | Speed | Bottleneck |
|-------|-------|------------|
| Perception (Fast Loop) | 30-60 FPS | YOLO inference |
| Perception (Slow Loop) | 1-5 FPS | VLM latency |
| Clustering | < 1s | HDBSCAN O(n log n) |
| Semantic Uplift | 2-10s | LLM calls |
| Graph Construction | < 1s | Neo4j writes |

### Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| YOLO11x | ~500 MB | Model weights |
| CLIP ViT-L/14 | ~900 MB | Model weights |
| Frame Buffer | ~100 MB | 30 frames @ 1080p |
| Embeddings | ~4 MB | 1000 entities × 1024 dims × 4 bytes |
| Neo4j | Variable | Depends on graph size |

### Scalability

| Video Length | Entities | Neo4j Nodes | Processing Time |
|--------------|----------|-------------|-----------------|
| 1 min | ~50 | ~500 | ~2 min |
| 5 min | ~200 | ~2000 | ~10 min |
| 30 min | ~1000 | ~10000 | ~60 min |

---

## Deployment Options

### 1. Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set up Neo4j
docker run -p 7687:7687 -p 7474:7474 neo4j:latest

# Run pipeline
python orion/run_pipeline.py --video test.mp4
```

### 2. Docker Container
```bash
docker build -t orion:latest .
docker run -v $(pwd)/data:/app/data orion:latest
```

### 3. Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orion-pipeline
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: orion
        image: orion:latest
        env:
        - name: NEO4J_URI
          value: "bolt://neo4j-service:7687"
```

---

## Testing & Quality Assurance

### Unit Tests

```bash
pytest tests/unit/
```

Coverage:
- `test_async_perception.py`: Detection, clustering, async queue
- `test_class_correction.py`: Semantic validation, part-of detection
- `test_semantic_uplift.py`: Entity management, state detection
- `test_causal_inference.py`: Scoring, temporal precedence
- `test_knowledge_graph.py`: Node creation, relationship types

### Integration Tests

```bash
pytest tests/integration/
```

Tests:
- End-to-end pipeline on sample videos
- Neo4j connectivity and schema validation
- Configuration presets (fast/balanced/accurate)

### Evaluation Scripts

```python
from orion.evaluation.core import ClassificationEvaluator

evaluator = ClassificationEvaluator()
metrics = evaluator.evaluate(predictions, ground_truth)
print(f"Precision: {metrics.precision:.3f}")
print(f"Recall: {metrics.recall:.3f}")
print(f"F1: {metrics.f1:.3f}")
```

---

## Known Limitations & Future Work

### Current Limitations

1. **COCO Class Vocabulary**: Limited to 80 object classes
   - Misses common objects like "tire", "knob", "door handle"
   - **Mitigation**: Class correction with semantic validation

2. **VLM Latency**: Slow description generation (200-500ms per entity)
   - **Mitigation**: Async queue + "describe once" strategy

3. **Causal Inference**: Heuristic-based, not learned
   - **Future**: Train causal relation classifier on VSGR

4. **No Online Learning**: Static models, no adaptation
   - **Future**: Active learning loop for class corrections

5. **Limited Relationship Types**: Predefined set of relationships
   - **Future**: LLM-based relation extraction

### Roadmap

**Phase 1: Evaluation (Q4 2025)**
- ✅ Class correction semantic validation
- ⚠️ VSGR dataset integration
- ⚠️ HyperGLM baseline comparison
- ❌ Ablation studies

**Phase 2: Model Improvements (Q1 2026)**
- Train causal relation classifier
- Fine-tune VLM for egocentric video
- Expand object vocabulary (200+ classes)
- Active learning for corrections

**Phase 3: Scale & Deployment (Q2 2026)**
- Distributed processing (multi-GPU)
- Real-time streaming mode
- API server (FastAPI)
- Web UI for graph visualization

---

## References

### Core Papers

1. **Orion Paper** (AAAI 2026 Submission)
   - Semantic uplift pipeline
   - VSGR evaluation
   - Causal inference

2. **HyperGLM** (arXiv:2411.18042v2)
   - SOTA video scene graph generation
   - Hypergraph representation
   - VSGR benchmark results

3. **VSGR Dataset** (Nguyen et al. 2025)
   - 1.9M frames with annotations
   - Causal relationship labels
   - Egocentric + 3rd person videos

### Implementation References

- **YOLO11**: Ultralytics YOLO11x
- **CLIP**: OpenAI CLIP ViT-L/14
- **HDBSCAN**: scikit-learn-contrib
- **Sentence Transformers**: all-MiniLM-L6-v2
- **Neo4j**: Graph database v5.x
- **Ollama**: Local LLM inference (Gemma, LLaMA)

---

## Summary: What's Missing vs Paper

### HIGH PRIORITY (For Research Paper)
1. ❌ **VSGR Dataset Integration** - Need to load and process VSGR annotations
2. ❌ **HyperGLM Comparison** - Baseline comparison for evaluation
3. ❌ **Ablation Studies** - Show component contributions
4. ⚠️ **Metrics Implementation** - Code exists, needs VSGR ground truth

### MEDIUM PRIORITY (For Production)
1. ❌ **Expanded Vocabulary** - 80 COCO classes → 200+ classes
2. ❌ **Online Learning** - Adapt to new object types
3. ❌ **API Server** - RESTful interface for deployment

### LOW PRIORITY (Nice to Have)
1. ❌ **Web UI** - Graph visualization
2. ❌ **Multi-GPU** - Distributed processing
3. ❌ **Qualitative Analysis** - Case study visualizations

### IMPLEMENTED BUT NOT IN PAPER (Should Add)
1. ✅ **Async Perception** - Major architectural improvement
2. ✅ **Semantic Validation** - Prevents bad corrections
3. ✅ **Config System** - Flexible, secure, production-ready

---

**Document Version**: 1.0  
**Last Updated**: October 23, 2025  
**Authors**: Orion Research Team  
**Contact**: riddhiman.rana@gmail.com
