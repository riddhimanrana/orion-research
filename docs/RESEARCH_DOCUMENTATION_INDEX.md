# Orion: Complete Research Paper Documentation Index

## Overview

This directory contains comprehensive technical documentation for the **Orion video understanding pipeline**—a multi-phase system for intelligent video analysis combining object detection, entity tracking, semantic understanding, and knowledge graph construction.

## Documentation Structure

### 1. TECHNICAL_ARCHITECTURE.md
**Primary reference for system design**

- Complete system architecture with 5 processing phases
- Mathematical foundations (cosine similarity, Hungarian algorithm, HDBSCAN, causal scoring)
- Detailed component specifications (Perception Engine, Tracking Engine, Semantic Uplift, Knowledge Graph Builder)
- End-to-end pipeline execution flow
- Data type contracts and integration points
- Performance characteristics and optimization strategies

**Key Sections**:
- System Overview & Core Phases
- Architecture Layers (Configuration, Data Models, Pipeline Engines, Knowledge Representation)
- Pipeline Execution Flow with pseudocode
- Mathematical Foundations with equations
- Database Schema (Neo4j node/relationship definitions)
- Performance Considerations

**Best For**: Understanding overall system design and component interactions

### 2. SEMANTIC_UPLIFT_GUIDE.md
**Deep dive into entity tracking and event detection**

- Entity tracking algorithm using Hungarian assignment
- State change detection with temporal windowing
- Event composition via LLM (Ollama integration)
- Detailed mathematical explanations
- Configuration parameters and tuning guide
- Integration patterns with main pipeline

**Key Sections**:
- Entity Tracking System (detection-to-entity assignment)
- Semantic Uplift Pipeline (4 phases: state tracking, change detection, event composition, Neo4j ingestion)
- Configuration parameters and thresholds
- Performance characteristics

**Best For**: Understanding how the system detects behavioral changes and composes high-level events

### 3. KNOWLEDGE_GRAPH_SCHEMA.md
**Complete Neo4j schema and query reference**

- Detailed Neo4j node types with properties
- Relationship definitions with property schemas
- Constraints and indexes for performance
- Standard query patterns (8 common use cases)
- Data population pipeline
- Performance optimization strategies
- Example queries for common tasks

**Key Sections**:
- Database Schema (Scene, Entity, Event, StateChange, SpatialRelationship nodes)
- Relationship Types (temporal, spatial, causal, hierarchical)
- Query Patterns (entity timeline, scene composition, causal chains, spatial networks)
- Data Population Pipeline
- Performance Tuning

**Best For**: Writing queries against the knowledge graph and understanding data persistence

### 4. DEPLOYMENT_GUIDE.md
**Configuration, integration, and deployment scenarios**

- Three-tier configuration system (environment → ConfigManager → OrionConfig)
- Credential management best practices
- Component integration patterns
- Multiple deployment scenarios (local, Docker, Kubernetes)
- Performance tuning guide
- Testing and validation procedures
- Troubleshooting guide

**Key Sections**:
- Configuration System Overview
- Environment Variable Setup
- Component Integration Patterns
- Deployment Scenarios (Local, Docker, Kubernetes)
- Performance Tuning
- Testing & Validation
- Troubleshooting

**Best For**: Setting up the system for deployment and integrating components in production

---

## System Architecture at a Glance

```
VIDEO INPUT
    ↓
[PERCEPTION PHASE]
├─ YOLO11x object detection
├─ CLIP embedding generation
└─ Spatial context analysis

    ↓
[TRACKING PHASE]
├─ Entity-detection matching (Hungarian algorithm)
├─ Embedding similarity + spatial consistency
└─ Trajectory maintenance

    ↓
[SEMANTIC UPLIFT PHASE]
├─ Entity state tracking
├─ State change detection (motion, appearance, interaction)
├─ LLM-based event composition
└─ Causal relationship inference

    ↓
[KNOWLEDGE GRAPH CONSTRUCTION]
├─ Scene node creation
├─ Entity node creation
├─ Spatial relationship analysis
├─ Causal reasoning
└─ Neo4j persistence

    ↓
[STORAGE & INDEXING]
└─ Neo4j graph database with constraints and indexes

    ↓
[QUERY & Q&A]
├─ Knowledge retrieval
├─ Contextual reasoning
└─ LLM-based answer generation
```

---

## Key Technical Concepts

### 1. Configuration Management
- **Unified Configuration**: All system parameters through `config.py`
- **Preset Factories**: `get_fast_config()`, `get_balanced_config()`, `get_accurate_config()`
- **Environment Integration**: Credentials loaded via environment variables through `ConfigManager`
- **Three-Tier System**: Environment → ConfigManager → OrionConfig

### 2. Entity Tracking
- **Algorithm**: Hungarian algorithm with composite cost function
- **Similarity Metrics**: 70% embedding similarity + 30% spatial consistency
- **Threshold**: Matching cost < 0.5
- **State Representation**: Maintains position, velocity, acceleration, embeddings

### 3. Semantic Understanding
- **State Changes**: Motion (velocity/acceleration), appearance (confidence), interaction (proximity)
- **Event Composition**: LLM-based description generation via Ollama
- **Temporal Windows**: Rolling 30-frame windows for consistent change detection
- **Causal Analysis**: Temporal proximity + spatial proximity + entity overlap + semantic similarity

### 4. Knowledge Graph
- **Node Types**: Scene, Entity, Event, StateChange, SpatialRelationship
- **Relationships**: Temporal (APPEARS_IN, PRECEDES), Spatial (SPATIAL_*), Causal (CAUSES)
- **Constraints**: Uniqueness constraints on node IDs, composite indexes for query performance
- **Query Patterns**: Timeline, composition, causality, interaction networks

### 5. Deployment Flexibility
- **Credential Security**: All sensitive data from environment variables
- **Hardware Agnostic**: Supports MLX (Apple Silicon) and Torch backends
- **Modular Integration**: Each phase independently testable
- **Multiple Deployment Options**: Local, Docker, Kubernetes

---

## Core Modules Reference

| Module | Purpose | Key Classes |
|--------|---------|-----------|
| `config.py` | Unified configuration system | `OrionConfig`, `DetectionConfig`, `EmbeddingConfig`, preset factories |
| `config_manager.py` | Credential/config singleton | `ConfigManager` |
| `neo4j_manager.py` | Neo4j connection lifecycle | `Neo4jManager` |
| `perception_engine.py` | Object detection + embedding | `PerceptionEngine` |
| `tracking_engine.py` | Entity-detection assignment | `TrackingEngine`, `Entity`, `EntityState` |
| `semantic_uplift.py` | Event detection + composition | `EntityTracker`, `StateChangeDetector`, `EventComposer` |
| `knowledge_graph.py` | Graph construction + storage | `KnowledgeGraphBuilder`, `SceneClassifier`, `SpatialAnalyzer` |
| `temporal_graph_builder.py` | Temporal graph ingestion | `TemporalGraphBuilder` |

---

## Common Integration Patterns

### Pattern 1: Basic Pipeline Execution
```python
from orion.config_manager import ConfigManager
from orion.run_pipeline import run_pipeline

config = ConfigManager.get_config()
result = run_pipeline("video.mp4", config=config)
```

### Pattern 2: Component-Level Integration
```python
from orion.perception_engine import PerceptionEngine
from orion.tracking_engine import TrackingEngine
from orion.semantic_uplift import SemanticUplift
from orion.knowledge_graph import KnowledgeGraphBuilder

# Initialize components
perception = PerceptionEngine(config.detection, config.embedding)
tracker = TrackingEngine(config.temporal_window_size)
uplift = SemanticUplift(config)
kg_builder = KnowledgeGraphBuilder()

# Process video frame-by-frame
# ...
```

### Pattern 3: Custom Configuration
```python
from orion.config import get_accurate_config, OrionConfig, DetectionConfig

# Start with preset and customize
config = get_accurate_config()
config.detection.confidence_threshold = 0.3  # Lower threshold for edge cases
config.temporal_window_size = 60  # Longer temporal context

result = run_pipeline("video.mp4", config=config)
```

### Pattern 4: Secure Credentials
```bash
# In .env file
ORION_NEO4J_PASSWORD=secure_password_123
ORION_NEO4J_URI=bolt://neo4j-server:7687

# In Python
from orion.config_manager import ConfigManager
config = ConfigManager.get_config()
# Credentials automatically loaded from environment
```

---

## Mathematical Foundations

### 1. Embedding Similarity (Cosine Distance)
$$\text{distance} = 1 - \cos(\theta) = 1 - \frac{\mathbf{e_1} \cdot \mathbf{e_2}}{\|\mathbf{e_1}\| \|\mathbf{e_2}\|}$$

For normalized embeddings: $\text{distance} = 1 - \mathbf{e_1} \cdot \mathbf{e_2}$

### 2. Tracking Cost Function
$$\text{cost}_{ij} = 0.7 \times (1 - \text{embedding\_sim}_{ij}) + 0.3 \times \text{spatial\_dist}_{ij}$$

### 3. State Change Detection
- **Velocity**: $v_t = \frac{p_t - p_{t-1}}{\Delta t}$
- **Acceleration**: $a_t = \frac{v_t - v_{t-1}}{\Delta t}$
- **Detection**: Changed if $\|v_t\| > \text{threshold}$ OR $\|a_t\| > \text{threshold}$

### 4. Causal Scoring
$$\text{causal\_score} = 0.3T + 0.3S + 0.2O + 0.2M$$

Where T = temporal proximity, S = spatial proximity, O = entity overlap, M = semantic similarity

### 5. Cluster Quality (HDBSCAN)
- Minimum cluster size affects computational complexity: O(N log N)
- Stability measured by cluster lifetime in hierarchical dendrogram

---

## Performance Characteristics

### Computation Time per Frame (typical)
- **YOLO Detection**: 40-100 ms (model size dependent)
- **CLIP Embedding**: 5-20 ms (10-20 objects typical)
- **Tracking Assignment**: 2-5 ms (Hungarian algorithm)
- **State Change Detection**: 1-3 ms
- **Event Composition**: 100-500 ms (LLM dependent)

### Memory Usage
- **YOLO Model**: 200-500 MB (depending on variant)
- **CLIP Model**: 300-600 MB
- **Entity Embeddings**: N × embedding_dim × 4 bytes (N = entity count)
- **Neo4j Driver**: ~50 MB connection pool

### Optimization Strategies
1. **Frame Sampling**: Process every Nth frame
2. **Batch Processing**: 16-32 frames per batch
3. **Model Caching**: Load models once, reuse
4. **Neo4j Batch Writes**: 1000+ entities per transaction
5. **Async Processing**: Parallel perception while tracking

---

## Testing & Validation

### Unit Test Coverage
- Entity tracking and assignment verification
- State change detection accuracy
- Event composition consistency
- Neo4j schema validation
- Configuration loading and preset correctness

### Integration Testing
- End-to-end pipeline execution
- Multi-component data flow
- Neo4j query correctness
- Performance under load

### Validation Metrics
- **Tracking**: IoU with ground truth trajectories
- **Events**: Precision/recall on ground truth events
- **Graph**: Query result correctness and completeness
- **Performance**: Throughput (FPS), latency, memory usage

---

## Deployment Checklist

- [ ] Review Configuration System (DEPLOYMENT_GUIDE.md)
- [ ] Set up environment variables (.env file)
- [ ] Initialize Neo4j database with schema
- [ ] Deploy Ollama for event composition
- [ ] Select appropriate config preset (fast/balanced/accurate)
- [ ] Download and cache model weights
- [ ] Test pipeline with sample video
- [ ] Configure monitoring and logging
- [ ] Implement error handling
- [ ] Document any custom configurations
- [ ] Plan database backup strategy
- [ ] Schedule performance benchmarks

---

## Document Navigation Guide

**For System Design Understanding** → Start with TECHNICAL_ARCHITECTURE.md

**For Implementation Details** → Review specific component in TECHNICAL_ARCHITECTURE.md, then SEMANTIC_UPLIFT_GUIDE.md or KNOWLEDGE_GRAPH_SCHEMA.md

**For Neo4j Queries** → Go to KNOWLEDGE_GRAPH_SCHEMA.md (query patterns section)

**For Deployment** → Follow DEPLOYMENT_GUIDE.md step-by-step

**For Performance Tuning** → See configuration sections in DEPLOYMENT_GUIDE.md and optimization strategies in TECHNICAL_ARCHITECTURE.md

**For Troubleshooting** → Check DEPLOYMENT_GUIDE.md troubleshooting section

---

## Key Files in Codebase

### Core Components
- `orion/config.py` - Configuration definitions (508 lines)
- `orion/config_manager.py` - Configuration singleton (240 lines)
- `orion/neo4j_manager.py` - Neo4j lifecycle management (164 lines)
- `orion/perception_engine.py` - Object detection pipeline
- `orion/tracking_engine.py` - Entity tracking
- `orion/semantic_uplift.py` - Event detection (3778 lines)
- `orion/knowledge_graph.py` - Graph construction (1149 lines)
- `orion/temporal_graph_builder.py` - Temporal graphs (443 lines)
- `orion/run_pipeline.py` - Complete pipeline orchestration (1296 lines)

### Scripts
- `scripts/test_complete_pipeline.py` - Full pipeline test
- `scripts/explore_kg.py` - Neo4j explorer tool
- `scripts/run_evaluation.py` - Evaluation metrics
- `scripts/test_contextual_understanding.py` - Contextual reasoning tests

---

## Recent Architecture Improvements

### Configuration Consolidation
- Unified all config into single `config.py` with presets
- Centralized credential management via `ConfigManager` singleton
- Removed hardcoded passwords and per-module config

### Neo4j Integration Modernization
- Centralized connection management via `Neo4jManager`
- All modules use injection pattern (optional parameter)
- Automatic fallback to ConfigManager for credential resolution

### Credential Security
- All sensitive data from environment variables
- No hardcoded passwords in any module
- Secure pattern: Environment → ConfigManager → Module

### Code Cleanup
- All test scripts use proper sys.path handling
- Removed unused configuration modules
- Fixed type hints for Python 3.11+ compatibility

---

## Future Enhancement Opportunities

1. **Distributed Processing**: Implement Ray/Dask for frame-level parallelization
2. **Advanced Causality**: Add probabilistic graphical models for causal inference
3. **Temporal Reasoning**: Integrate temporal logic (Allen's interval algebra)
4. **Multi-Modal**: Support audio analysis and speaker diarization
5. **Real-time Streaming**: Implement online learning for concept drift
6. **Custom Models**: Fine-tuning pipeline for domain-specific detection
7. **Interactive Exploration**: Web UI for graph exploration and query building

---

## References & Related Work

- **YOLO11**: Ultralytics YOLOv8 detection architecture
- **CLIP**: Vision-language model for semantic embeddings
- **HDBSCAN**: Density-based hierarchical clustering
- **Neo4j**: Property graph database for knowledge representation
- **Ollama**: Local LLM inference framework
- **Hungarian Algorithm**: Optimal assignment in bipartite graphs

---

**This documentation suite provides complete technical specification for research publication, system deployment, and maintenance.**

For questions about specific components, refer to the relevant documentation file listed above.
