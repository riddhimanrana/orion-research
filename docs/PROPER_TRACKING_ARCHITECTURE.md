# Proper Object Tracking & Description Architecture

## Core Principle

**Track First, Describe Once, Link Always**

1. **Track objects across entire video** using visual embeddings
2. **Generate description ONCE per unique object** when first seen
3. **Update description ONLY if significant state change** detected
4. **Link all appearances** to build temporal knowledge graph

## Current Problems

### ❌ What We're Doing Wrong:
```python
# For each frame:
#   For each detection in frame:
#     Extract crop
#     Generate embedding
#     Run FastVLM on crop  ← WASTEFUL!
#     Add to queue

# Result: Same laptop gets described 100 times!
```

### ✅ What We Should Do:
```python
# Phase 1: DETECT & EMBED (Fast)
for frame in video:
    detections = detect_objects(frame)
    for detection in detections:
        embedding = extract_embedding(crop)
        tracker.add_observation(embedding, bbox, timestamp)

# Phase 2: CLUSTER & IDENTIFY (Once)
unique_entities = tracker.cluster_all_embeddings()
# Result: 283 frames → 20-50 unique entities

# Phase 3: DESCRIBE ONCE (Efficient)
for entity in unique_entities:
    best_appearance = entity.get_best_frame()  # clearest, largest
    description = fastvlm.describe(best_appearance)
    entity.description = description
# Result: Only 20-50 FastVLM calls!

# Phase 4: LINK TEMPORAL GRAPH (Smart)
for entity in unique_entities:
    for appearance in entity.timeline:
        graph.link(entity, appearance.frame, appearance.location)
```

## Architecture: 3-Phase Pipeline

### Phase 1: Fast Detection & Embedding

**Goal**: Process entire video quickly, collect all observations

```python
class ObservationCollector:
    def __init__(self):
        self.observations = []  # Raw detections
        
    def process_video(self, video_path):
        for frame_num, frame in video_frames(video_path):
            detections = yolo.detect(frame)
            
            for detection in detections:
                crop = extract_crop(frame, detection.bbox)
                embedding = resnet50.encode(crop)  # 2048-dim
                
                obs = Observation(
                    frame_number=frame_num,
                    timestamp=frame_num / fps,
                    bbox=detection.bbox,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    embedding=embedding,
                    crop=crop  # Keep for later description
                )
                self.observations.append(obs)
        
        return self.observations  # ~300-400 observations
```

**Output**: List of all detections with embeddings, NO descriptions yet

### Phase 2: Entity Clustering & Tracking

**Goal**: Group observations into unique entities

```python
class EntityTracker:
    def cluster_observations(self, observations):
        # Extract embeddings
        embeddings = [obs.embedding for obs in observations]
        
        # HDBSCAN clustering with ResNet50 embeddings
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=3,
            metric='euclidean',  # or 'cosine' for normalized
            cluster_selection_epsilon=0.15
        )
        labels = clusterer.fit_predict(embeddings)
        
        # Group observations by cluster
        entities = defaultdict(list)
        for obs, label in zip(observations, labels):
            if label == -1:  # Noise (unique object)
                entity_id = f"unique_{obs.frame_number}"
            else:
                entity_id = f"entity_{label:04d}"
            entities[entity_id].append(obs)
        
        # Create entity objects
        tracked_entities = []
        for entity_id, obs_list in entities.items():
            entity = Entity(
                id=entity_id,
                class_name=obs_list[0].class_name,
                observations=sorted(obs_list, key=lambda x: x.timestamp),
                first_seen=min(obs.timestamp for obs in obs_list),
                last_seen=max(obs.timestamp for obs in obs_list)
            )
            tracked_entities.append(entity)
        
        return tracked_entities  # 20-50 unique entities
```

**Output**: Unique entities with all their appearances linked

### Phase 3: Smart Description Generation

**Goal**: Describe each entity ONCE, using best available frame

```python
class SmartDescriber:
    def describe_entities(self, entities):
        for entity in entities:
            # Select best observation for description
            best_obs = self._select_best_observation(entity)
            
            # Generate description ONCE
            description = self._generate_description(
                image=best_obs.crop,
                entity=entity
            )
            
            entity.description = description
            entity.described_from_frame = best_obs.frame_number
        
        return entities
    
    def _select_best_observation(self, entity):
        """Pick clearest, largest, most centered frame"""
        scored_obs = []
        
        for obs in entity.observations:
            # Score based on:
            # 1. Size (larger = better)
            bbox_area = (obs.bbox[2] - obs.bbox[0]) * (obs.bbox[3] - obs.bbox[1])
            
            # 2. Position (centered = better)
            center_x = (obs.bbox[0] + obs.bbox[2]) / 2
            frame_width = 1920  # or get from video metadata
            distance_from_center = abs(center_x - frame_width/2)
            centrality = 1.0 - (distance_from_center / (frame_width/2))
            
            # 3. Confidence (higher = better)
            confidence = obs.confidence
            
            # Combined score
            score = (bbox_area * 0.5) + (centrality * 0.3) + (confidence * 0.2)
            scored_obs.append((score, obs))
        
        # Return best
        return max(scored_obs, key=lambda x: x[0])[1]
    
    def _generate_description(self, image, entity):
        """Context-aware description with temporal info"""
        
        # Build context prompt
        prompt = f"""Describe this {entity.class_name} in detail.

Context:
- Object appears {len(entity.observations)} times across video
- First seen at {entity.first_seen:.1f}s
- Last seen at {entity.last_seen:.1f}s
- Duration: {entity.last_seen - entity.first_seen:.1f}s

Describe its visual appearance, key characteristics, and any notable features."""

        description = fastvlm.generate(image, prompt)
        return description
```

**Output**: Each entity has ONE high-quality description

### Phase 4: Temporal Knowledge Graph

**Goal**: Link entities to frames, locations, and relationships

```python
class TemporalGraphBuilder:
    def build_graph(self, entities, neo4j_driver):
        with neo4j_driver.session() as session:
            # 1. Create entity nodes
            for entity in entities:
                session.run("""
                    MERGE (e:Entity {id: $id})
                    SET e.class = $class,
                        e.description = $description,
                        e.first_seen = $first_seen,
                        e.last_seen = $last_seen,
                        e.appearance_count = $count,
                        e.embedding = $embedding
                """, {
                    'id': entity.id,
                    'class': entity.class_name,
                    'description': entity.description,
                    'first_seen': entity.first_seen,
                    'last_seen': entity.last_seen,
                    'count': len(entity.observations),
                    'embedding': entity.average_embedding.tolist()
                })
            
            # 2. Create appearance relationships
            for entity in entities:
                for obs in entity.observations:
                    session.run("""
                        MATCH (e:Entity {id: $entity_id})
                        MERGE (f:Frame {number: $frame_num})
                        SET f.timestamp = $timestamp
                        MERGE (e)-[r:APPEARS_IN]->(f)
                        SET r.bbox = $bbox,
                            r.confidence = $confidence
                    """, {
                        'entity_id': entity.id,
                        'frame_num': obs.frame_number,
                        'timestamp': obs.timestamp,
                        'bbox': obs.bbox,
                        'confidence': obs.confidence
                    })
            
            # 3. Detect spatial relationships
            self._link_spatial_relationships(entities, session)
            
            # 4. Detect temporal patterns
            self._link_temporal_patterns(entities, session)
    
    def _link_spatial_relationships(self, entities, session):
        """Find objects that appear together in frames"""
        
        # Group by frame
        frame_entities = defaultdict(list)
        for entity in entities:
            for obs in entity.observations:
                frame_entities[obs.frame_number].append((entity, obs))
        
        # Find co-occurrences
        for frame_num, entity_obs_list in frame_entities.items():
            if len(entity_obs_list) < 2:
                continue
            
            # Check each pair
            for i, (e1, obs1) in enumerate(entity_obs_list):
                for e2, obs2 in entity_obs_list[i+1:]:
                    # Calculate spatial relationship
                    distance = self._calculate_distance(obs1.bbox, obs2.bbox)
                    
                    if distance < 200:  # Nearby threshold
                        session.run("""
                            MATCH (e1:Entity {id: $id1}),
                                  (e2:Entity {id: $id2})
                            MERGE (e1)-[r:NEAR {frame: $frame}]->(e2)
                            SET r.distance = $distance,
                                r.timestamp = $timestamp
                        """, {
                            'id1': e1.id,
                            'id2': e2.id,
                            'frame': frame_num,
                            'distance': distance,
                            'timestamp': obs1.timestamp
                        })
    
    def _link_temporal_patterns(self, entities, session):
        """Detect state changes and movements"""
        
        for entity in entities:
            observations = entity.observations
            
            for i in range(len(observations) - 1):
                curr_obs = observations[i]
                next_obs = observations[i + 1]
                
                # Detect movement
                curr_center = self._get_bbox_center(curr_obs.bbox)
                next_center = self._get_bbox_center(next_obs.bbox)
                distance_moved = self._distance(curr_center, next_center)
                
                if distance_moved > 50:  # Significant movement
                    session.run("""
                        MATCH (e:Entity {id: $entity_id})
                        MERGE (m:Movement {
                            entity_id: $entity_id,
                            from_frame: $from_frame,
                            to_frame: $to_frame
                        })
                        SET m.distance = $distance,
                            m.from_time = $from_time,
                            m.to_time = $to_time
                        MERGE (e)-[:HAS_MOVEMENT]->(m)
                    """, {
                        'entity_id': entity.id,
                        'from_frame': curr_obs.frame_number,
                        'to_frame': next_obs.frame_number,
                        'distance': distance_moved,
                        'from_time': curr_obs.timestamp,
                        'to_time': next_obs.timestamp
                    })
```

## State Change Detection

**Only re-describe if significant visual change detected:**

```python
class StateChangeDetector:
    def detect_changes(self, entity):
        """Compare embeddings across timeline to detect state changes"""
        
        observations = entity.observations
        changes = []
        
        for i in range(len(observations) - 1):
            curr_emb = observations[i].embedding
            next_emb = observations[i + 1].embedding
            
            # Cosine similarity of embeddings
            similarity = cosine_similarity(curr_emb, next_emb)
            
            if similarity < 0.85:  # Significant change threshold
                # Visual appearance changed significantly
                # Generate new description for this state
                new_description = self._describe_state_change(
                    entity,
                    observations[i],
                    observations[i + 1]
                )
                
                changes.append(StateChange(
                    entity_id=entity.id,
                    from_time=observations[i].timestamp,
                    to_time=observations[i + 1].timestamp,
                    old_embedding=curr_emb,
                    new_embedding=next_emb,
                    description=new_description
                ))
        
        return changes
```

## Performance Comparison

### Old Approach (Frame-Based):
```
283 frames × ~1.5 objects/frame = 436 FastVLM calls
Time: ~7 minutes (1s per call)
```

### New Approach (Entity-Based):
```
Phase 1: Detection & Embedding
  283 frames × 50ms = 14 seconds

Phase 2: Clustering
  436 observations → 20-50 entities
  Time: < 1 second

Phase 3: Smart Description
  50 entities × 1s = 50 seconds
  (+ maybe 10 state changes × 1s = 10 seconds)

Phase 4: Graph Building
  Time: < 5 seconds

Total: ~80 seconds (6x faster!)
```

## Key Innovations

1. **ResNet50 for Re-ID**: 2048-dim embeddings provide excellent discrimination
2. **HDBSCAN Clustering**: Automatically finds optimal number of entities
3. **Best-Frame Selection**: Describe from clearest, largest frame
4. **State Change Detection**: Only re-describe when appearance changes
5. **Temporal Graph**: Link all appearances to build complete timeline

## Next Steps

1. Implement `ObservationCollector` class
2. Update `EntityTracker` to collect first, cluster second
3. Add `SmartDescriber` with best-frame selection
4. Implement state change detection with embedding comparison
5. Build rich temporal knowledge graph with movements and relationships

This approach will be:
- ✅ **6x faster** (80s vs 7min)
- ✅ **More accurate** (better embeddings, context-aware descriptions)
- ✅ **Smarter** (tracks permanence, detects state changes)
- ✅ **Scalable** (works for any video length)
