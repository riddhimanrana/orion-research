# Part 2: The Semantic Uplift Engine

## Overview

This module transforms raw perception logs from Part 1 into a structured, queryable knowledge graph in Neo4j. It implements sophisticated entity tracking, state change detection, and LLM-powered event composition to create a persistent memory system from video observations.

## Architecture

```
Perception Log (from Part 1)
        │
        ▼
┌────────────────────────────────────┐
│  ENTITY TRACKING                   │
│  - HDBSCAN clustering              │
│  - Visual embedding similarity     │
│  - Assign permanent entity IDs     │
└────────────┬───────────────────────┘
            │
            ▼
┌────────────────────────────────────┐
│  STATE CHANGE DETECTION            │
│  - Sentence transformer embeddings │
│  - Cosine similarity (< 0.85)      │
│  - Create state change events      │
└────────────┬───────────────────────┘
            │
            ▼
┌────────────────────────────────────┐
│  TEMPORAL WINDOWING                │
│  - 30-second windows               │
│  - Group state changes             │
│  - Identify significant events     │
└────────────┬───────────────────────┘
            │
            ▼
┌────────────────────────────────────┐
│  EVENT COMPOSITION (LLM)           │
│  - Ollama/Llama3 reasoning         │
│  - Generate Cypher queries         │
│  - Structured event creation       │
└────────────┬───────────────────────┘
            │
            ▼
┌────────────────────────────────────┐
│  NEO4J KNOWLEDGE GRAPH             │
│  - Entity nodes with embeddings    │
│  - Event nodes with timestamps     │
│  - Relationships and properties    │
└────────────────────────────────────┘
```

## Key Features

### 1. Entity Tracking (Object Permanence)
- **HDBSCAN Clustering**: Groups visual embeddings to identify the same object across frames
- **Noise Handling**: Objects appearing once are tracked as unique entities
- **Average Embeddings**: Computes representative embedding for each entity
- **Timeline Construction**: Creates chronological appearance history

### 2. State Change Detection
- **Semantic Analysis**: Uses sentence transformers to compare descriptions
- **Threshold-Based**: Detects changes when similarity < 0.85
- **Magnitude Tracking**: Quantifies how significant each change is
- **Temporal Ordering**: Maintains before/after state information

### 3. Event Composition
- **LLM Reasoning**: Uses local Ollama/Llama3 for structured thinking
- **Cypher Generation**: Produces valid Neo4j queries
- **Context-Aware**: Includes entity timelines and state changes in prompts
- **Fallback Mode**: Works without LLM using template-based queries

### 4. Knowledge Graph Construction
- **Schema Management**: Automatically creates constraints and indexes
- **Vector Search**: Supports embedding-based similarity queries
- **Batch Processing**: Efficient bulk ingestion
- **Relationship Modeling**: Links entities, events, and states

## Data Structures

### Entity
```python
{
    "entity_id": "entity_cluster_0042",
    "object_class": "person",
    "first_timestamp": 10.5,
    "last_timestamp": 45.2,
    "appearance_count": 12,
    "average_embedding": [512-dim vector],
    "appearances": [list of perception objects]
}
```

### StateChange
```python
{
    "entity_id": "entity_cluster_0042",
    "timestamp_before": 15.2,
    "timestamp_after": 18.7,
    "description_before": "Person standing still",
    "description_after": "Person walking forward",
    "similarity_score": 0.72,
    "change_magnitude": 0.28
}
```

### TemporalWindow
```python
{
    "start_time": 0.0,
    "end_time": 30.0,
    "active_entities": ["entity_cluster_0042", ...],
    "state_changes": [list of StateChange objects]
}
```

## Neo4j Schema

### Node Types

**:Entity**
- `id` (STRING, UNIQUE): Entity identifier
- `label` (STRING): Object class
- `first_seen` (FLOAT): First appearance timestamp
- `last_seen` (FLOAT): Last appearance timestamp
- `appearance_count` (INT): Number of appearances
- `embedding` (LIST<FLOAT>): 512-dim visual embedding (vector indexed)
- `first_description` (TEXT): Initial description

**:Event**
- `id` (STRING, UNIQUE): Event identifier
- `type` (STRING): Event type (e.g., "state_change")
- `timestamp` (DATETIME): When event occurred
- `description` (TEXT): Event description

**:State**
- `description` (TEXT): State description
- `timestamp` (FLOAT): State timestamp

### Relationships

- `(:Entity)-[:PARTICIPATED_IN]->(:Event)`: Entity involved in event
- `(:Entity)-[:CHANGED_TO]->(:State)`: Entity transitioned to state
- `(:Event)-[:OCCURRED_AT]->(timestamp)`: Temporal relationship

## Installation

### Prerequisites

1. **Python packages** (already installed from Part 1):
```bash
pip install hdbscan==0.8.39
pip install sentence-transformers==3.3.1
pip install neo4j==5.26.0
```

2. **Neo4j Database**:

**Option A: Neo4j Desktop** (Recommended for development)
- Download from https://neo4j.com/download/
- Create a new project and database
- Set password to `password` or update `Config.NEO4J_PASSWORD`
- Start the database

**Option B: Docker**
```bash
docker run \
    --name neo4j \
    -p7474:7474 -p7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    -v $HOME/neo4j/data:/data \
    neo4j:latest
```

3. **Ollama (Optional but recommended)**:
```bash
# macOS
brew install ollama

# Start server
ollama serve

# Pull model
ollama pull llama3
```

## Usage

### Basic Usage

```python
from production.part2_semantic_uplift import run_semantic_uplift
import json

# Load perception log from Part 1
with open('data/testing/perception_log.json', 'r') as f:
    perception_log = json.load(f)

# Run semantic uplift
results = run_semantic_uplift(
    perception_log,
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
)

# Check results
print(f"Entities: {results['num_entities']}")
print(f"State changes: {results['num_state_changes']}")
print(f"Graph nodes: {results['graph_stats']['entity_nodes']}")
```

### With Custom Configuration

```python
from production.part2_config import apply_config, ACCURATE_CONFIG
from production.part2_semantic_uplift import run_semantic_uplift

# Apply configuration
apply_config(ACCURATE_CONFIG)

# Run with custom settings
results = run_semantic_uplift(perception_log)
```

### Using Existing Neo4j Driver

```python
from neo4j import GraphDatabase
from production.part2_semantic_uplift import run_semantic_uplift

# Create driver
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)

# Run uplift
results = run_semantic_uplift(perception_log, neo4j_driver=driver)

# Close when done
driver.close()
```

### Running the Test Script

```bash
# Use Part 1 output
python production/test_part2.py --use-part1-output

# Use custom perception log
python production/test_part2.py --perception-log path/to/perception_log.json

# With custom Neo4j
python production/test_part2.py --use-part1-output --neo4j-uri bolt://localhost:7687
```

## Configuration

### Preset Configurations

| Preset | Clustering | State Sensitivity | Window Size | Use Case |
|--------|-----------|-------------------|-------------|----------|
| FAST_CONFIG | Loose | Low | 60s | Quick processing |
| BALANCED_CONFIG | Medium | Medium | 30s | Default |
| ACCURATE_CONFIG | Tight | High | 15s | Maximum detail |
| HIGH_PRECISION_TRACKING | Strict | Medium | 30s | Accurate entity IDs |
| SENSITIVE_STATE_DETECTION | Medium | Very High | 20s | Catch subtle changes |

### Key Parameters

```python
class Config:
    # Entity Tracking
    MIN_CLUSTER_SIZE = 3           # Minimum appearances for tracked entity
    CLUSTER_SELECTION_EPSILON = 0.15  # Clustering strictness
    
    # State Change Detection
    STATE_CHANGE_THRESHOLD = 0.85   # Similarity threshold (lower = more sensitive)
    SENTENCE_MODEL = 'all-MiniLM-L6-v2'  # Sentence transformer model
    
    # Temporal Windowing
    TIME_WINDOW_SIZE = 30.0         # Window size in seconds
    MIN_EVENTS_PER_WINDOW = 2       # Minimum state changes per window
    
    # LLM Event Composition
    OLLAMA_MODEL = "llama3"
    OLLAMA_TEMPERATURE = 0.3        # More deterministic output
    
    # Neo4j
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"
```

## Performance Characteristics

### Processing Speed
- **Clustering**: O(n log n) with HDBSCAN, ~1-5s for 1000 objects
- **State Detection**: O(n) sentence embedding, ~0.5s per 100 descriptions
- **Event Composition**: O(w) where w = number of windows, ~2-10s per window with LLM
- **Neo4j Ingestion**: ~1000 nodes/sec, ~500 relationships/sec

### Memory Usage
- **Base**: ~1-2 GB (models loaded)
- **Processing**: +100MB per 1000 entities
- **Neo4j**: Variable based on graph size

### Scalability
- Handles 100-10,000 perception objects efficiently
- Can process multiple videos sequentially or in parallel
- Neo4j can scale to millions of nodes

## Output Format

### Results Dictionary
```json
{
    "success": true,
    "num_entities": 145,
    "num_state_changes": 67,
    "num_windows": 8,
    "num_queries": 89,
    "graph_stats": {
        "entity_nodes": 145,
        "event_nodes": 52,
        "relationships": 178
    }
}
```

## Querying the Knowledge Graph

### Sample Cypher Queries

**Find all entities:**
```cypher
MATCH (e:Entity)
RETURN e.id, e.label, e.appearance_count
ORDER BY e.appearance_count DESC
```

**Find entities with state changes:**
```cypher
MATCH (e:Entity)-[:PARTICIPATED_IN]->(ev:Event {type: 'state_change'})
RETURN e.id, e.label, count(ev) as num_changes
ORDER BY num_changes DESC
```

**Find similar entities by embedding:**
```cypher
// Requires vector index
MATCH (e:Entity)
WHERE e.id = 'entity_cluster_0001'
CALL db.index.vector.queryNodes('entity_embedding', 5, e.embedding)
YIELD node, score
RETURN node.id, node.label, score
```

**Timeline of entity:**
```cypher
MATCH (e:Entity {id: 'entity_cluster_0042'})-[:PARTICIPATED_IN]->(ev:Event)
RETURN ev.timestamp, ev.description
ORDER BY ev.timestamp
```

## Error Handling

The system includes robust error handling:

### Missing Dependencies
- **HDBSCAN**: Falls back to treating each object as unique entity
- **Sentence Transformers**: Skips state change detection
- **Ollama**: Uses template-based Cypher generation

### Neo4j Connection Failures
- Clear error messages with troubleshooting steps
- Connection retry logic
- Graceful degradation

### Data Quality Issues
- Handles missing embeddings
- Validates perception log structure
- Logs warnings for incomplete data

## Troubleshooting

### Issue: "hdbscan not available"
**Solution**: Install HDBSCAN
```bash
pip install hdbscan==0.8.39
```

### Issue: "Could not connect to Neo4j"
**Solution**: 
1. Check Neo4j is running: http://localhost:7474
2. Verify credentials in config
3. Test connection:
```python
from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
driver.verify_connectivity()
```

### Issue: "Could not connect to Ollama"
**Solution**: Ollama is optional
- Install: `brew install ollama`
- Start: `ollama serve`
- Pull model: `ollama pull llama3`
- Or: System will use fallback Cypher generation

### Issue: "No state changes detected"
**Solution**: Lower the threshold
```python
from production.part2_config import create_custom_config, apply_config

config = create_custom_config(STATE_CHANGE_THRESHOLD=0.80)
apply_config(config)
```

### Issue: "Too many unique entities (no clustering)"
**Solution**: Adjust clustering parameters
```python
config = create_custom_config(
    MIN_CLUSTER_SIZE=2,
    CLUSTER_SELECTION_EPSILON=0.20
)
apply_config(config)
```

## Integration with Part 1 & 3

### From Part 1:
```python
# Part 1: Generate perception log
from production.part1_perception_engine import run_perception_engine
perception_log = run_perception_engine("video.mp4")
```

### To Part 3:
```python
# Part 2: Build knowledge graph
from production.part2_semantic_uplift import run_semantic_uplift
results = run_semantic_uplift(perception_log)

# Part 3: Query & evaluate (to be implemented)
from production.part3_query_evaluation import agent_augmented_sota
answer = agent_augmented_sota(query, video_clips, neo4j_driver)
```

## Advanced Features

### Custom Entity Tracking
```python
from production.part2_semantic_uplift import EntityTracker

tracker = EntityTracker()
tracker.track_entities(perception_log)

# Get specific entity
entity = tracker.get_entity("entity_cluster_0042")
print(f"Appearances: {len(entity.appearances)}")
print(f"First seen: {entity.first_timestamp}s")
```

### Manual State Change Detection
```python
from production.part2_semantic_uplift import StateChangeDetector

detector = StateChangeDetector()
detector.load_model()

# Detect for specific entity
changes = detector.detect_state_changes_for_entity(entity)
for change in changes:
    print(f"Change at {change.timestamp_after}s")
    print(f"  Magnitude: {change.change_magnitude}")
```

### Custom Event Composition
```python
from production.part2_semantic_uplift import EventComposer

composer = EventComposer()
queries = composer.compose_events_for_window(window, tracker)
```

## Limitations & Future Work

### Current Limitations
1. **Single-threaded processing**: No parallel entity tracking yet
2. **Memory-bound**: All embeddings loaded into memory
3. **Ollama dependency**: Best results require local LLM
4. **Fixed schema**: Graph schema not easily customizable

### Future Enhancements
1. Batch processing of multiple videos
2. Incremental graph updates
3. Spatial relationship modeling
4. Hierarchical event composition
5. Graph querying optimization
6. Support for other LLM backends

## License

Part of the Orion Research project. See LICENSE file for details.

## Contributors

- Riddhiman Rana
- Aryav Semwal
- Yogesh Atluru
- Jason Zhang

## Contact

For questions or issues, please contact the Orion Research team.
