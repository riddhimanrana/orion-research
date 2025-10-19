# Orion Pipeline: Entity Tracking & Semantic Uplift

## Entity Tracking System

### Overview
The tracking engine maintains entity identity across video frames using embedding similarity and spatial consistency.

### Core Algorithm

#### Detection-to-Entity Assignment

**Problem**: Given detections in frame t with embeddings, assign them to known entities or create new ones.

**Solution**: Hungarian Algorithm with composite cost function

```
Cost Matrix: C[i,j] = cost(entity_i, detection_j)
    where cost = 0.7 * embedding_distance + 0.3 * spatial_distance

Steps:
  1. Extract recent entity embeddings (use latest 5 frames)
  2. Compute embedding similarity: cosine_distance(entity_emb, detection_emb)
  3. Compute spatial distance: 1 - IoU(entity_bbox, detection_bbox)
  4. Combine: cost = 0.7 * emb_dist + 0.3 * spatial_dist
  5. Apply Hungarian algorithm for optimal assignment
  6. Register unmatched detections as new entities
```

**Matching Threshold**: cost < 0.5 (empirically determined)

#### Entity State Representation

```python
class Entity:
    - id: int                           # Unique identifier
    - class_name: str                   # "person", "car", etc.
    - first_appearance_frame: int
    - last_appearance_frame: int
    - embeddings: List[ndarray]         # Historical embeddings
    - bboxes: List[Tuple]               # Bounding box history
    - confidences: List[float]          # Detection confidences
    
    Methods:
    - get_representative_embedding(): Average over recent frames
    - compute_velocity(): Change in position per frame
    - get_trajectory(): Sequence of positions
```

### Performance Characteristics

- **Computational Complexity**: O(N³) for N entities/detections (Hungarian algorithm)
- **Memory Usage**: O(N × embedding_dim) for storing embeddings
- **Typical Performance**:
  - 5-10 persons: <10ms assignment per frame
  - 50+ objects: 50-100ms per frame
  - Bottleneck: HDBSCAN clustering for embedding analysis

---

## Semantic Uplift Pipeline

### Phase 1: Entity State Tracking

**Purpose**: Maintain consistent state representation for each entity

```python
class EntityTracker:
    def update_states(frame_idx: int, entities: Dict[int, Entity]):
        """Update entity states based on current observations"""
        
        for entity in entities.values():
            state = EntityState(
                entity_id=entity.id,
                position=entity.latest_bbox.center,
                velocity=(pos_t - pos_{t-1}) / dt,
                acceleration=(vel_t - vel_{t-1}) / dt,
                confidence=entity.latest_confidence,
                temporal_consistency=consistency_score
            )
            
            # Compute temporal consistency from embedding stability
            recent_embeddings = entity.embeddings[-5:]
            pairwise_similarities = [
                cosine_sim(recent_embeddings[i], recent_embeddings[i+1])
                for i in range(len(recent_embeddings)-1)
            ]
            consistency_score = mean(pairwise_similarities)
```

**Output**: EntityState dictionary for each active entity

### Phase 2: State Change Detection

**Purpose**: Identify behavioral transitions (motion, interaction, appearance changes)

```python
class StateChangeDetector:
    def __init__(temporal_window_size: int = 30):  # frames
        self.state_history = {}
        self.window_size = temporal_window_size
    
    def detect_changes(entity_states: Dict) -> List[StateChange]:
        """Detect transitions across temporal window"""
        
        changes = []
        
        for entity_id, current_state in entity_states.items():
            if entity_id not in self.state_history:
                self.state_history[entity_id] = []
            
            # Maintain rolling window
            self.state_history[entity_id].append(current_state)
            if len(self.state_history[entity_id]) > self.window_size:
                self.state_history[entity_id].pop(0)
            
            history = self.state_history[entity_id]
            
            # Analyze state sequence
            if len(history) >= 3:
                prev_state = history[-2]
                curr_state = history[-1]
                
                # 1. Motion change detection
                if is_significant_velocity_change(prev_state, curr_state):
                    changes.append(StateChange(
                        entity_id=entity_id,
                        change_type="motion",
                        description="significant velocity change",
                        velocity_magnitude=norm(curr_state.velocity),
                        confidence=compute_change_confidence(history)
                    ))
                
                # 2. Appearance change detection
                confidence_delta = curr_state.confidence - prev_state.confidence
                if abs(confidence_delta) > 0.15:
                    changes.append(StateChange(
                        entity_id=entity_id,
                        change_type="appearance",
                        description="confidence shift",
                        confidence=abs(confidence_delta)
                    ))
                
                # 3. Spatial change detection
                position_delta = distance(curr_state.position, prev_state.position)
                if position_delta > spatial_distance_threshold:
                    changes.append(StateChange(
                        entity_id=entity_id,
                        change_type="spatial",
                        description="significant position change",
                        distance=position_delta,
                        confidence=0.8  # High confidence for spatial
                    ))
```

**Thresholds**:
- `velocity_threshold`: 10 pixels/frame
- `acceleration_threshold`: 5 pixels/frame²
- `confidence_delta_threshold`: 0.15
- `spatial_distance_threshold`: 50 pixels

### Phase 3: Event Composition

**Purpose**: Generate high-level natural language descriptions of events

```python
class EventComposer:
    def __init__(ollama_config: OllamaConfig):
        self.client = httpx.Client(base_url=ollama_config.base_url)
        self.model = ollama_config.model
        self.timeout = ollama_config.timeout
    
    def compose_event(state_changes: List[StateChange], 
                     entities: Dict[int, Entity]) -> Event:
        """Generate natural language event description"""
        
        # Step 1: Build event context
        context = self._build_event_context(state_changes, entities)
        
        # Step 2: Construct LLM prompt
        prompt = f"""Given these video analysis observations:

{context}

Describe the event in 1-2 sentences. Focus on:
1. What happened (action/state change)
2. Which entities were involved
3. Spatial/temporal relationships

Be concise and factual."""
        
        # Step 3: Call Ollama for description
        response = self.client.post(
            "/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": 0.7,
                "top_p": 0.9,
                "stream": False
            },
            timeout=self.timeout
        )
        
        description = response.json()["response"].strip()
        
        # Step 4: Classify event type
        event_type = self._classify_event_type(description, state_changes)
        
        # Step 5: Compute composite confidence
        confidence = self._compute_event_confidence(state_changes)
        
        return Event(
            id=generate_event_id(),
            event_type=event_type,
            description=description,
            affected_entities=[change.entity_id for change in state_changes],
            confidence=confidence,
            timestamp=time.time()
        )
    
    def _build_event_context(state_changes, entities) -> str:
        """Format state changes into readable context"""
        
        lines = []
        for change in state_changes:
            entity = entities[change.entity_id]
            lines.append(
                f"- {entity.class_name} (ID: {entity.id}): "
                f"{change.description} "
                f"(confidence: {change.confidence:.2f})"
            )
        
        return "\n".join(lines)
    
    def _classify_event_type(description: str, state_changes: List) -> str:
        """Classify event into category"""
        
        keywords = {
            'motion': ['moving', 'walks', 'runs', 'slides', 'accelerates'],
            'interaction': ['touches', 'pushes', 'pulls', 'grabs', 'holds'],
            'state': ['appears', 'disappears', 'stops', 'starts', 'changes'],
            'spatial': ['enters', 'exits', 'approaches', 'leaves', 'passes']
        }
        
        desc_lower = description.lower()
        for event_type, words in keywords.items():
            if any(word in desc_lower for word in words):
                return event_type
        
        return 'unknown'
    
    def _compute_event_confidence(state_changes: List[StateChange]) -> float:
        """Compute composite event confidence score"""
        
        if not state_changes:
            return 0.0
        
        # Weighted average of individual confidence scores
        weights = [1.0 / len(state_changes)] * len(state_changes)
        confidence = sum(
            change.confidence * weight 
            for change, weight in zip(state_changes, weights)
        )
        
        # Boost confidence if multiple entities involved
        if len(set(change.entity_id for change in state_changes)) > 1:
            confidence = min(1.0, confidence * 1.1)
        
        return confidence
```

**Event Types**:
- `motion`: Position, velocity, or acceleration changes
- `interaction`: Multiple entities in close proximity with state changes
- `state`: Appearance/disappearance, confidence changes
- `spatial`: Relative position changes

### Phase 4: Neo4j Ingestion

**Purpose**: Store semantic data in graph database

```python
def ingest_to_graph(events: List[Event], entities: Dict[int, Entity]):
    """Store events and entities in Neo4j"""
    
    kg_builder = KnowledgeGraphBuilder()
    
    with kg_builder.driver.session() as session:
        # 1. Create Event nodes
        for event in events:
            session.run("""
                CREATE (e:Event {
                    id: $id,
                    event_type: $type,
                    description: $desc,
                    confidence: $conf,
                    timestamp: $ts
                })
            """, {
                'id': event.id,
                'type': event.event_type,
                'desc': event.description,
                'conf': event.confidence,
                'ts': event.timestamp
            })
        
        # 2. Create Entity-Event relationships
        for event in events:
            for entity_id in event.affected_entities:
                session.run("""
                    MATCH (e:Event {id: $event_id})
                    MATCH (ent:Entity {id: $entity_id})
                    CREATE (ent)-[r:INVOLVED_IN]->(e)
                """, {
                    'event_id': event.id,
                    'entity_id': str(entity_id)
                })
        
        # 3. Create causal relationships between events
        for i, event1 in enumerate(events):
            for event2 in events[i+1:]:
                causal_score = compute_causality(event1, event2)
                if causal_score > CAUSALITY_THRESHOLD:
                    session.run("""
                        MATCH (e1:Event {id: $id1})
                        MATCH (e2:Event {id: $id2})
                        CREATE (e1)-[c:CAUSES {
                            confidence: $score,
                            causal_type: $type
                        }]->(e2)
                    """, {
                        'id1': event1.id,
                        'id2': event2.id,
                        'score': causal_score,
                        'type': 'temporal' if is_temporally_adjacent(event1, event2) else 'indirect'
                    })
```

---

## Configuration for Semantic Uplift

### Key Parameters

```python
# From config.py - SEMANTIC_UPLIFT_CONFIG preset
{
    "clustering": {
        "min_cluster_size": 10,
        "min_samples": 5,
        "metric": "cosine",
        "cluster_selection_epsilon": 0.5
    },
    "temporal_window_size": 30,              # frames
    "state_change_threshold": 0.7,           # confidence
    "spatial_distance_threshold": 50,        # pixels
    "velocity_threshold": 10,                # pixels/frame
    "acceleration_threshold": 5,             # pixels/frame²
    "embedding_similarity_threshold": 0.7,
    "causality_threshold": 0.6,
    "max_causality_window": 5000            # milliseconds
}
```

### Tuning Guide

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| `min_cluster_size` | Smaller → more clusters | 5-20 |
| `temporal_window_size` | Larger → more history | 10-60 frames |
| `velocity_threshold` | Higher → fewer motion events | 5-20 px/frame |
| `causality_threshold` | Higher → fewer causal links | 0.5-0.8 |

---

## Integration with Pipeline

```
Tracking Engine outputs: (entity_id, Detection) assignments
    ↓
SemanticUplift processes:
    1. EntityTracker.update_states()
    2. StateChangeDetector.detect_changes()
    3. EventComposer.compose_event()
    4. ingest_to_graph()
    ↓
Neo4j stores: Events + Relationships + Causality
```

The semantic uplift layer bridges low-level tracking with high-level reasoning, enabling the knowledge graph to represent not just entities but their behaviors and interactions.
