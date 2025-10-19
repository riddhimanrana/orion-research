# Orion Knowledge Graph: Schema & Query Patterns

## Database Schema Design

### Node Types

#### Scene Node
Represents a temporal segment of video with consistent context/activity

```
Scene {
    id: String (UNIQUE)                 # "scene_001"
    scene_type: String                  # "office", "outdoor", "hallway", "room"
    timestamp_start: Integer            # Milliseconds since video start
    timestamp_end: Integer              # Milliseconds since video start
    duration: Integer                   # end - start
    frame_start: Integer                # First frame index
    frame_end: Integer                  # Last frame index
    description: String                 # Generated scene description
    dominant_entities: List[String]     # Top entities by appearance count
    event_count: Integer                # Number of events in scene
}
```

#### Entity Node
Represents a persistent object detected across multiple frames

```
Entity {
    id: String (UNIQUE)                 # "entity_042"
    class: String                       # "person", "car", "desk", "phone"
    appearance_count: Integer           # Number of frames where visible
    average_confidence: Float           # Mean detection confidence
    first_seen: Integer                 # First frame number
    last_seen: Integer                  # Last frame number
    trajectory_length: Integer          # Number of distinct positions
    description: String                 # Generated description
    bounding_boxes: List[Float]         # Serialized bbox coordinates
    embeddings: List[Float]             # Average CLIP embedding
    state_transitions: Integer          # Number of detected state changes
    interaction_count: Integer          # Number of interactions with other entities
}
```

#### Event Node
Represents a detected behavioral event or state change

```
Event {
    id: String (UNIQUE)                 # "event_012"
    event_type: String                  # "motion", "interaction", "state", "spatial"
    description: String                 # Natural language description
    confidence: Float                   # Composite confidence [0, 1]
    timestamp: Integer                  # Milliseconds when detected
    duration: Integer                   # Event duration in milliseconds
    affected_entities: List[String]     # Entity IDs involved
    scene_context: String               # Scene where event occurred
    attributes: Map[String, Any]        # Event-specific attributes
    temporal_window: Integer            # Window size used for detection (frames)
}
```

#### StateChange Node
Represents a specific behavioral transition

```
StateChange {
    id: String (UNIQUE)                 # "statechange_089"
    entity_id: String                   # Reference to Entity
    change_type: String                 # "position", "velocity", "appearance", "interaction"
    description: String                 # Description of change
    confidence: Float                   # Confidence in detection
    timestamp: Integer                  # When change occurred
    magnitude: Float                    # Change magnitude (velocity, distance, etc.)
    previous_value: String              # Serialized previous state
    current_value: String               # Serialized current state
}
```

#### SpatialRelationship Node
Represents spatial configuration between entities

```
SpatialRelationship {
    id: String (UNIQUE)                 # "spatial_001_042_034"
    entity1_id: String                  # First entity
    entity2_id: String                  # Second entity
    relationship_type: String           # "left_of", "above", "inside", "touching"
    distance: Float                     # Euclidean distance between centroids
    confidence: Float                   # Detection confidence
    frame: Integer                      # Frame where measured
    timestamp: Integer                  # Time of measurement
}
```

### Relationship Types

#### Temporal Relationships

**APPEARS_IN**
```
(entity:Entity)-[APPEARS_IN {
    first_frame: Integer,
    last_frame: Integer,
    appearance_duration: Integer,       # Milliseconds
    average_confidence: Float
}]->(scene:Scene)
```
Indicates entity was present during scene

**PRECEDES**
```
(scene1:Scene)-[PRECEDES {
    gap_duration: Integer               # Milliseconds between scenes
    transition_type: String             # "continuous", "cut", "fade"
}]->(scene2:Scene)
```
Temporal ordering of scenes

**FOLLOWS**
```
(event1:Event)-[FOLLOWS {
    temporal_gap: Integer,              # Milliseconds
    gap_frames: Integer
}]->(event2:Event)
```
Event sequence

#### Spatial Relationships

**SPATIAL_LEFT_OF, SPATIAL_RIGHT_OF, SPATIAL_ABOVE, SPATIAL_BELOW**
```
(entity1:Entity)-[SPATIAL_LEFT_OF {
    distance: Float,
    confidence: Float,
    frame: Integer,
    relative_position: Float            # Percentage of entity1 width
}]->(entity2:Entity)
```

**SPATIAL_INSIDE, SPATIAL_CONTAINS**
```
(entity1:Entity)-[SPATIAL_INSIDE {
    containment_ratio: Float,           # Percentage overlap
    distance: Float,
    confidence: Float
}]->(entity2:Entity)
```

**SPATIAL_TOUCHES, SPATIAL_NEAR**
```
(entity1:Entity)-[SPATIAL_TOUCHES {
    distance: Float,
    confidence: Float
}]->(entity2:Entity)
```

#### Causal Relationships

**CAUSES**
```
(event1:Event)-[CAUSES {
    confidence: Float,                  # Causal confidence
    causal_type: String,                # "direct", "indirect", "enabling"
    temporal_lag: Integer,              # Milliseconds
    shared_entities: List[String]       # Entities in both events
    reasoning: String                   # Explanation of causality
}]->(event2:Event)
```

#### Containment/Hierarchical Relationships

**CONTAINS**
```
(scene:Scene)-[CONTAINS]->(entity:Entity)
(scene:Scene)-[CONTAINS]->(event:Event)
```

**INVOLVED_IN**
```
(entity:Entity)-[INVOLVED_IN {
    role: String                        # "subject", "object", "bystander"
}]->(event:Event)
```

#### State Relationships

**HAS_STATE**
```
(entity:Entity)-[HAS_STATE {
    transition_index: Integer
}]->(statechange:StateChange)
```

**TRIGGERED_BY**
```
(event:Event)-[TRIGGERED_BY]->(statechange:StateChange)
```

### Constraints & Indexes

```cypher
-- Uniqueness constraints
CREATE CONSTRAINT entity_id_unique 
    IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE

CREATE CONSTRAINT scene_id_unique 
    IF NOT EXISTS FOR (s:Scene) REQUIRE s.id IS UNIQUE

CREATE CONSTRAINT event_id_unique 
    IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE

-- Performance indexes
CREATE INDEX entity_class_idx 
    IF NOT EXISTS FOR (e:Entity) ON (e.class)

CREATE INDEX scene_type_idx 
    IF NOT EXISTS FOR (s:Scene) ON (s.scene_type)

CREATE INDEX event_type_idx 
    IF NOT EXISTS FOR (e:Event) ON (e.event_type)

CREATE INDEX entity_appearance_idx 
    IF NOT EXISTS FOR (e:Entity) ON (e.appearance_count)

CREATE INDEX scene_timestamp_idx 
    IF NOT EXISTS FOR (s:Scene) ON (s.timestamp_start)

CREATE INDEX event_timestamp_idx 
    IF NOT EXISTS FOR (e:Event) ON (e.timestamp)

-- Composite indexes for common queries
CREATE INDEX entity_scene_idx 
    IF NOT EXISTS FOR ()-[r:APPEARS_IN]-() ON (r.first_frame, r.last_frame)
```

---

## Standard Query Patterns

### Pattern 1: Entity Timeline

**Query**: Get all appearances of entity across scenes

```cypher
MATCH (e:Entity {id: "entity_042"})-[rel:APPEARS_IN]->(s:Scene)
RETURN s.id, s.scene_type, s.timestamp_start, s.timestamp_end, 
       rel.appearance_duration, rel.average_confidence
ORDER BY s.timestamp_start
```

### Pattern 2: Scene Composition

**Query**: Get all entities and events in a specific scene

```cypher
MATCH (s:Scene {id: "scene_001"})
RETURN 
    s,
    [(s)<-[r1:APPEARS_IN]-(e:Entity) | {entity: e, duration: r1.appearance_duration}] as entities,
    [(s)-[r2:CONTAINS]->(ev:Event) | {event: ev, type: ev.event_type}] as events
```

### Pattern 3: Causal Chains

**Query**: Find causal sequences of events

```cypher
MATCH path = (e1:Event)-[c:CAUSES*1..5]->(e2:Event)
WHERE e1.timestamp < e2.timestamp
RETURN path, length(path) as depth,
       [rel in relationships(path) | rel.confidence] as confidences,
       reduce(minConf = 1.0, rel in relationships(path) | min(minConf, rel.confidence)) as min_confidence
ORDER BY min_confidence DESC
LIMIT 10
```

### Pattern 4: Spatial Interaction Network

**Query**: Find entities in spatial relationships during specific timeframe

```cypher
MATCH (e1:Entity)-[rel:SPATIAL_LEFT_OF|SPATIAL_RIGHT_OF|SPATIAL_INSIDE|SPATIAL_CONTAINS]-(e2:Entity)
WHERE rel.confidence > 0.7
RETURN e1.class as class1, e2.class as class2, 
       type(rel) as relationship, COUNT(*) as frequency
ORDER BY frequency DESC
```

### Pattern 5: Entity Interaction Graph

**Query**: Find entities frequently involved in events together

```cypher
MATCH (e1:Entity)-[:INVOLVED_IN]->(ev:Event)<-[:INVOLVED_IN]-(e2:Entity)
WHERE e1.id < e2.id  -- Avoid duplicates
RETURN e1.id, e1.class, e2.id, e2.class, 
       COUNT(DISTINCT ev) as shared_events,
       COLLECT(ev.event_type) as event_types
ORDER BY shared_events DESC
```

### Pattern 6: Temporal Scene Sequences

**Query**: Find scenes with specific entity combinations

```cypher
MATCH (s1:Scene)-[:PRECEDES]->(s2:Scene)-[:PRECEDES]->(s3:Scene)
WHERE s1.scene_type = "office" AND s2.scene_type = "hallway"
RETURN s1, s2, s3
LIMIT 5
```

### Pattern 7: Most Influential Events

**Query**: Find events that caused the most downstream events

```cypher
MATCH (e:Event)-[c:CAUSES*1..]->(downstream:Event)
WITH e, COUNT(DISTINCT downstream) as causality_breadth,
     [rel in collect(c) | rel.confidence] as conf_scores
RETURN e.id, e.event_type, e.description, causality_breadth,
       reduce(avg = 0, score in conf_scores | avg + score) / size(conf_scores) as avg_causality
ORDER BY causality_breadth DESC
LIMIT 10
```

### Pattern 8: Frequently Co-Located Entities

**Query**: Find entity pairs that consistently appear together

```cypher
MATCH (e1:Entity)-[rel:SPATIAL_NEAR]->(e2:Entity)
WHERE e1.id < e2.id
WITH e1, e2, COLLECT(rel.distance) as distances, COLLECT(rel.confidence) as confidences
RETURN e1.class, e2.class, 
       COUNT(*) as co_location_count,
       AVG(distances) as avg_distance,
       AVG(confidences) as avg_confidence
WHERE co_location_count > 5
ORDER BY co_location_count DESC
```

---

## Data Population Pipeline

### Step 1: Entity Creation

```cypher
// Create entity nodes from tracking results
UNWIND $entities as entity_data
CREATE (e:Entity {
    id: entity_data.id,
    class: entity_data.class_name,
    appearance_count: entity_data.appearance_count,
    average_confidence: entity_data.average_confidence,
    first_seen: entity_data.first_appearance_frame,
    last_seen: entity_data.last_appearance_frame,
    trajectory_length: size(entity_data.trajectory),
    description: entity_data.description
})
```

### Step 2: Scene Creation

```cypher
// Create scene nodes from temporal segmentation
UNWIND $scenes as scene_data
CREATE (s:Scene {
    id: scene_data.id,
    scene_type: scene_data.scene_type,
    timestamp_start: scene_data.timestamp_start,
    timestamp_end: scene_data.timestamp_end,
    duration: scene_data.timestamp_end - scene_data.timestamp_start,
    description: scene_data.description
})
```

### Step 3: Entity-Scene Linking

```cypher
// Create APPEARS_IN relationships
UNWIND $appearances as appearance
MATCH (e:Entity {id: appearance.entity_id})
MATCH (s:Scene {id: appearance.scene_id})
CREATE (e)-[r:APPEARS_IN {
    first_frame: appearance.first_frame,
    last_frame: appearance.last_frame,
    appearance_duration: appearance.last_frame - appearance.first_frame,
    average_confidence: appearance.avg_confidence
}]->(s)
```

### Step 4: Event Creation & Linking

```cypher
// Create event nodes
UNWIND $events as event_data
CREATE (e:Event {
    id: event_data.id,
    event_type: event_data.event_type,
    description: event_data.description,
    confidence: event_data.confidence,
    timestamp: event_data.timestamp
})

// Create INVOLVED_IN relationships
UNWIND $events as event_data
MATCH (e:Event {id: event_data.id})
UNWIND event_data.affected_entities as entity_id
MATCH (ent:Entity {id: entity_id})
CREATE (ent)-[:INVOLVED_IN]->(e)
```

### Step 5: Spatial Relationships

```cypher
// Create spatial relationship edges
UNWIND $spatial_rels as spatial_data
MATCH (e1:Entity {id: spatial_data.entity1_id})
MATCH (e2:Entity {id: spatial_data.entity2_id})
CALL apoc.create.relationship(
    e1,
    "SPATIAL_" + apoc.text.upper(spatial_data.relationship_type),
    {
        distance: spatial_data.distance,
        confidence: spatial_data.confidence,
        frame: spatial_data.frame
    },
    e2
) YIELD rel
RETURN COUNT(rel)
```

### Step 6: Causal Relationships

```cypher
// Create CAUSES relationships between events
UNWIND $causal_links as causal_data
MATCH (e1:Event {id: causal_data.source_id})
MATCH (e2:Event {id: causal_data.target_id})
CREATE (e1)-[c:CAUSES {
    confidence: causal_data.confidence,
    causal_type: causal_data.causal_type,
    temporal_lag: causal_data.temporal_lag
}]->(e2)
```

---

## Performance Optimization

### Query Optimization Tips

1. **Always filter early**: Use WHERE clauses with indexed properties first
   ```cypher
   MATCH (e:Entity)
   WHERE e.class = "person"  -- Use index
   MATCH (e)-[:INVOLVED_IN]->(ev:Event)
   WHERE ev.timestamp > start_time  -- Indexed
   ```

2. **Use APOC for complex operations**:
   ```cypher
   CALL apoc.periodic.commit(
       "MATCH (e:Entity) LIMIT $limit SET e.processed = true",
       {limit: 1000}
   )
   ```

3. **Batch write operations**:
   ```python
   # Write 1000 entities at a time instead of one-by-one
   entities_batch = []
   for i, entity in enumerate(all_entities):
       entities_batch.append(entity)
       if (i + 1) % 1000 == 0:
           session.run(CREATE_ENTITIES_QUERY, {'entities': entities_batch})
           entities_batch = []
   ```

4. **Create appropriate indexes**:
   - Index on frequently searched properties
   - Composite indexes for multi-property WHERE clauses
   - Don't over-index; each write must update indexes

### Database Tuning

```python
# neo4j_manager.py connection parameters
driver = GraphDatabase.driver(
    uri,
    auth=(user, password),
    max_connection_pool_size=50,        # Tune based on concurrency
    connection_acquisition_timeout=30,   # Connection pool timeout
    connection_timeout=30,
    max_retry_time=30,
    trust=Trust.TRUST_SYSTEM_CA_SIGNED_CERTIFICATES  # Security
)
```

---

## Example Use Cases

### Use Case 1: Video Summarization

```cypher
MATCH (s:Scene)
WITH s ORDER BY s.timestamp_start
WITH COLLECT(s) as scenes
RETURN [scene in scenes | {
    type: scene.scene_type,
    time: scene.timestamp_start,
    description: scene.description,
    entity_count: size([()-[:APPEARS_IN]->(scene)])
}] as summary
```

### Use Case 2: Finding Key Moments

```cypher
MATCH (e:Event)
WHERE e.event_type IN ["interaction", "motion"]
RETURN e
ORDER BY e.confidence DESC, 
         (()-[:INVOLVED_IN]->(e)) as involvement_count DESC
LIMIT 20
```

### Use Case 3: Anomaly Detection

```cypher
MATCH (e:Entity)
WHERE e.appearance_count < 5 OR e.interaction_count > 10
RETURN e.id, e.class, e.appearance_count, e.interaction_count
```

These queries and patterns demonstrate the rich relational structure enabling complex video understanding tasks through graph-based reasoning.
