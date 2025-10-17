# Perception Engine Improvements

## Current Issues Identified

### 1. **Too Many Unique Objects (436 from 283 frames)**
- **Problem**: OSNet embeddings (512-dim) are not discriminative enough for re-identification
- **Symptom**: 283 frames yielded 436 "unique" detections, meaning objects aren't being properly re-identified
- **Expected**: Should have far fewer unique objects (maybe 20-50) across the video

### 2. **Wrong Description Mode (OBJECT vs SCENE)**
- **Problem**: System was set to `OBJECT` mode, generating one FastVLM call per detection crop
- **Impact**: 436 separate FastVLM calls instead of 283 scene descriptions
- **Fix Applied**: Changed to `SCENE` mode - one description per frame shared by all objects

### 3. **Embedding Model Choice**
- **Old Working Code**: Used ResNet50 with 2048-dim features
- **Current Broken Code**: Uses OSNet with 512-dim features
- **Fix Applied**: Switched back to ResNet50 for better discrimination

## Applied Fixes

### ✅ 1. Changed Description Mode
```python
# Before:
DESCRIPTION_MODE = DescriptionMode.OBJECT  # 436 descriptions!

# After:
DESCRIPTION_MODE = DescriptionMode.SCENE  # 283 descriptions
```

### ✅ 2. Switched to ResNet50 Embeddings
```python
# Before:
EMBEDDING_MODEL = 'osnet'
EMBEDDING_DIM = 512

# After:
EMBEDDING_MODEL = 'resnet50'
EMBEDDING_DIM = 2048
EMBEDDING_POOLING = 'avg'
```

### ✅ 3. Verified Causal Inference
The CIS calculation is **actually correct**:
- ✅ Proximity score (spatial distance)
- ✅ Motion score (moving towards target)
- ✅ Temporal score (recent observations)
- ✅ Embedding score (returns 0.5 as neutral - correct for missing patient embeddings)

## Remaining Work Needed

### 1. **Better Clustering Parameters**
Current HDBSCAN settings may be too strict:
```python
MIN_CLUSTER_SIZE = 3  # Requires 3+ appearances to form cluster
MIN_SAMPLES = 2
CLUSTER_SELECTION_EPSILON = 0.15
```

With ResNet50's better embeddings, we might need to:
- Adjust `CLUSTER_SELECTION_EPSILON` based on 2048-dim space
- Consider using cosine distance instead of euclidean
- Tune `MIN_CLUSTER_SIZE` based on video length

### 2. **Spatial Scene Understanding**
Need to add scene/location tracking:
- Detect rooms/regions from scene descriptions
- Link objects to locations: "laptop on desk in room A"
- Track movement between locations
- Build spatial knowledge graph

Example structure:
```cypher
(:Scene {name: "Living Room", description: "..."})
(:Entity {id: "laptop_01"})-[:LOCATED_IN]->(:Scene)
(:Entity {id: "person_01"})-[:MOVED_FROM]->(:Scene {name: "Kitchen"})
(:Entity {id: "person_01"})-[:MOVED_TO]->(:Scene {name: "Living Room"})
```

### 3. **Event Composition Improvements**
Current LLM event composition needs better:
- **Temporal coherence**: Group state changes within time windows
- **Spatial coherence**: Link events in same location
- **Actor-action-object triples**: "Person picked up laptop"
- **Relationship inference**: "Person X approached Person Y"

### 4. **Description Quality Validation**
Add checks for:
- Description length (not too short/long)
- Duplicate descriptions (same text for different frames)
- Missing object mentions (YOLO detected but not in description)
- Confidence scoring on descriptions

### 5. **HPO (Hyperparameter Optimization)**
While the current hardcoded values work, could optimize:
```python
# Clustering
- HDBSCAN min_cluster_size: [2, 3, 4, 5]
- cluster_selection_epsilon: [0.1, 0.15, 0.2, 0.25]
- metric: ['euclidean', 'cosine']

# Causal Inference
- proximity_weight: [0.3, 0.4, 0.5]
- motion_weight: [0.2, 0.25, 0.3]
- temporal_weight: [0.15, 0.2, 0.25]

# Detection
- YOLO confidence: [0.2, 0.25, 0.3]
- MIN_OBJECT_SIZE: [24, 32, 40]
```

## Testing Plan

### Phase 1: Validate Re-ID (Current Priority)
```bash
# Run pipeline with ResNet50
orion analyze data/examples/video1.mp4 --runtime mlx --output data/testing/resnet50_test

# Check metrics:
# - Total frames vs total detections ratio (should be < 2:1)
# - Number of unique entities (should be 20-50 for typical video)
# - Average appearances per entity (should be > 5)
```

Expected improvements:
- **Before**: 283 frames → 436 detections (1.54:1)
- **After**: 283 frames → ~300-400 detections but only **20-50 unique entities**

### Phase 2: Validate Descriptions
Check that:
- Only 283 FastVLM calls made (not 436)
- Each frame gets one shared scene description
- Descriptions mention multiple detected objects
- No duplicate descriptions

### Phase 3: Validate Knowledge Graph
```cypher
// Count unique entities
MATCH (e:Entity)
RETURN count(e) as total_entities,
       avg(e.appearance_count) as avg_appearances

// Check event composition
MATCH (e:Entity)-[:PARTICIPATED_IN]->(ev:Event)
RETURN ev.type, count(distinct e) as num_participants
ORDER BY count(distinct e) DESC

// Verify temporal ordering
MATCH (ev:Event)
RETURN ev.timestamp, ev.description
ORDER BY ev.timestamp
```

### Phase 4: Interactive Q&A Testing
```python
qa = VideoQASystem()
qa.start_interactive_session()

# Test questions:
# - "How many people were in the video?"  # Should get accurate count
# - "What happened to the laptop?"        # Should track object across time
# - "Who interacted with whom?"           # Should find relationships
# - "What rooms/locations were shown?"    # Should extract spatial info
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     PERCEPTION ENGINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Video Input (30fps)                                           │
│         ↓                                                       │
│  Frame Sampling (4fps)                                         │
│         ↓                                                       │
│  ┌──────────────────────────────────────────────┐             │
│  │ TIER 1: Real-Time Processing                 │             │
│  │  • YOLO Object Detection (80 classes)       │             │
│  │  • ResNet50 Visual Embeddings (2048-dim)    │ ← FIXED!   │
│  │  • Motion Tracking (velocity, direction)    │             │
│  └──────────────────────────────────────────────┘             │
│         ↓                                                       │
│  ┌──────────────────────────────────────────────┐             │
│  │ TIER 2: Async Description (SCENE MODE)      │ ← FIXED!   │
│  │  • FastVLM-0.5B (MLX on Apple Silicon)      │             │
│  │  • One description per frame (not per object)│             │
│  │  • All objects share scene context           │             │
│  └──────────────────────────────────────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                    SEMANTIC UPLIFT ENGINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────┐             │
│  │ Entity Tracking (HDBSCAN Clustering)         │             │
│  │  • Input: 2048-dim ResNet50 embeddings       │ ← Better!  │
│  │  • Output: Unique entity IDs (20-50 typical) │             │
│  │  • Tracks object permanence across time      │             │
│  └──────────────────────────────────────────────┘             │
│         ↓                                                       │
│  ┌──────────────────────────────────────────────┐             │
│  │ State Change Detection                        │             │
│  │  • Compare consecutive descriptions           │             │
│  │  • EmbeddingGemma similarity (Ollama)        │             │
│  │  • Threshold: < 0.85 similarity = change     │             │
│  └──────────────────────────────────────────────┘             │
│         ↓                                                       │
│  ┌──────────────────────────────────────────────┐             │
│  │ Causal Inference (CIS Scoring)                │             │
│  │  • Proximity: spatial distance                │             │
│  │  • Motion: moving towards target              │             │
│  │  • Temporal: recent observations              │             │
│  │  • Threshold: CIS > 0.55 → LLM verification  │             │
│  └──────────────────────────────────────────────┘             │
│         ↓                                                       │
│  ┌──────────────────────────────────────────────┐             │
│  │ Event Composition (LLM Reasoning)             │             │
│  │  • Gemma3:4b (Ollama)                        │             │
│  │  • Generate Cypher queries for Neo4j          │             │
│  │  • Temporal windowing (30s windows)           │             │
│  └──────────────────────────────────────────────┘             │
│         ↓                                                       │
│  ┌──────────────────────────────────────────────┐             │
│  │ Neo4j Knowledge Graph                         │             │
│  │  • Entities with visual embeddings            │             │
│  │  • Events with timestamps                     │             │
│  │  • Relationships (PARTICIPATED_IN, etc)       │             │
│  │  • Vector index for semantic search           │             │
│  └──────────────────────────────────────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                     INTERACTIVE Q&A                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Question → Vector Search (optional) → Context Retrieval  │
│                          ↓                                      │
│                    LLM (Gemma3:4b)                             │
│                          ↓                                      │
│                    Natural Language Answer                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Success Criteria

### ✅ Short-term (Current Sprint)
1. [ ] Pipeline runs without errors with ResNet50
2. [ ] Detections:Frames ratio < 2:1
3. [ ] Number of unique entities is reasonable (20-50)
4. [ ] Only 283 FastVLM calls (not 436)
5. [ ] Scene descriptions mention multiple objects

### 🎯 Medium-term (Next Week)
1. [ ] Knowledge graph has accurate entity count
2. [ ] Events are properly linked to entities
3. [ ] Temporal ordering is correct
4. [ ] Q&A system answers basic questions accurately
5. [ ] No "same person appears as 10 different entities"

### 🚀 Long-term (Future)
1. [ ] Spatial scene understanding (rooms/locations)
2. [ ] Complex relationship inference
3. [ ] Multi-object event composition
4. [ ] Confidence scoring on all inferences
5. [ ] Benchmark against ground truth annotations

## References

- **Original Working Code**: Used ResNet50, SCENE mode, proper clustering
- **Paper**: Causal Influence Score (CIS) - proximity, motion, temporal components
- **MLX Backend**: Successfully integrated with CoreML vision tower
- **Semantic Uplift**: HDBSCAN clustering for entity tracking

## Next Steps

1. **Run full test** with ResNet50 + SCENE mode
2. **Analyze results**: Count unique entities, check descriptions
3. **If still seeing too many entities**: Tune HDBSCAN parameters
4. **Add spatial tracking**: Extract room/location info from descriptions
5. **Improve event composition**: Better LLM prompts with spatial/temporal context
