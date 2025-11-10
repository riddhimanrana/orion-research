# Persistent Spatial Intelligence System 🧠

## Vision

Build a **persistent spatial memory system** that acts as a "historian" for robotics models like Gemini Robotics 1.5 - continuously learning and remembering everything about a space with perfect spatial and temporal context over hours, days, or even longer periods.

## The Goal

You want the system to:
- ✅ **Remember everything** - build semantic memory that persists
- ✅ **Smart semantic indexing** - spatial + semantic + temporal relationships
- ✅ **Reconstruct scenes** - 3D spatial understanding, depth maps, zones
- ✅ **Interactive queries** - ask clarifying questions, maintain context
- ✅ **Long-term memory** - work over hours/days, recognize same objects
- ✅ **Spatial computation** - depth, zones, 3D reconstruction for indoor scenes
- ✅ **Act as historian** - provide context to models like Gemini Robotics

## System Architecture

### Core Components

```
┌──────────────────────────────────────────────────────────────┐
│                  PERSISTENT SPATIAL INTELLIGENCE              │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. PROCESSING LAYER (Real-time)                             │
│     - YOLO detection                                          │
│     - Entity tracking with 3D + CLIP                          │
│     - SLAM pose estimation                                    │
│     - Depth maps + spatial zones                              │
│     - Export to Memgraph (real-time queries)                  │
│                                                               │
│  2. SEMANTIC MEMORY LAYER (Persistent)                        │
│     - SpatialMemorySystem: Remembers everything              │
│     - Entities with full history                              │
│     - Zones with semantic labels                              │
│     - Relationships (spatial, temporal, semantic)             │
│     - FastVLM captions (generated strategically)              │
│     - Conversation context                                    │
│                                                               │
│  3. INTELLIGENCE LAYER (Interactive)                          │
│     - SpatialIntelligenceAssistant                           │
│     - Contextual query understanding                          │
│     - Clarifying questions                                    │
│     - Rich multi-modal answers                                │
│     - Cross-session memory                                    │
│                                                               │
│  4. VISUALIZATION LAYER (Rerun)                               │
│     - 3D scene reconstruction                                 │
│     - Depth maps                                              │
│     - Entity trajectories                                     │
│     - Spatial zones                                           │
│     - SLAM pose graph                                         │
└──────────────────────────────────────────────────────────────┘
```

## What's Been Implemented

### 1. Spatial Memory System (`orion/graph/spatial_memory.py`)

**The persistent "brain" that remembers everything:**

```python
class SpatialMemorySystem:
    """
    Persistent knowledge base with:
    - All entities ever seen (with full history)
    - Semantic labels: "the red book on the desk near the laptop"
    - Spatial understanding: 3D positions, zones, movements
    - Temporal tracking: when, duration, sequences
    - Relationship graphs: spatial + semantic + temporal
    - Conversation context: understands "it", "that", etc.
    - Disk persistence: survives across sessions
    """
```

**Data Structures:**

**SpatialEntity**:
```python
{
    'entity_id': 42,
    'class_name': 'book',
    'semantic_label': 'the red book on the desk near the laptop',
    'first_seen': 10.5,
    'last_seen': 205.3,
    'observations_count': 47,
    'primary_zone': 2,
    'all_zones': [2, 5],
    'avg_position_3d': (1200, -300, 800),  # mm
    'movement_history': [(10.5, pos1), (11.0, pos2), ...],
    'captions': [
        'A red hardcover book with gold lettering',
        'Book lying flat on wooden desk'
    ],
    'activities': ['being_read', 'moved_to_shelf'],
    'relationships': [
        {'related_entity': 43, 'type': 'NEAR', 'confidence': 0.9},
        {'related_entity': 12, 'type': 'ON_TOP_OF', 'confidence': 0.95}
    ],
    'dominant_colors': ['red', 'gold'],
    'appearance_features': <CLIP embedding>
}
```

**SpatialZone**:
```python
{
    'zone_id': 2,
    'zone_type': 'desk area',
    'center_3d': (1000, -200, 700),
    'radius_mm': 500,
    'entity_ids': [42, 43, 44, 12],
    'permanent_entities': [43],  # desk itself
    'transient_entities': [42, 44, 12],  # book, cup, phone
    'typical_activities': ['working', 'reading', 'writing'],
    'scene_context': 'wooden desk with laptop and study materials'
}
```

### 2. Interactive Assistant (`scripts/spatial_intelligence_assistant.py`)

**Intelligent query interface with context:**

```python
class SpatialIntelligenceAssistant:
    """
    Your spatial intelligence historian:
    
    Features:
    - Understands context: "it", "that", "the same one"
    - Asks clarifying questions when ambiguous
    - Provides rich multi-modal answers
    - Remembers conversation history
    - Syncs with Memgraph for latest data
    - Persists memory across sessions
    """
```

**Example Interactions:**

```bash
You: What color was the book?
Assistant: The book was red. Specifically: A red hardcover book with 
           gold lettering. I saw it 47 times over 192.3 seconds.

You: Where was it?
Assistant: The book was in zone 2 (desk area). It was mostly stationary
           near the laptop.

You: What happened to it after that?
Assistant: At 15:20, a person picked up the book. It moved to zone 5
           (shelf area) at 15:23 where it remained.

You: Tell me everything about that area
Assistant: Zone 2 (desk area) is the central workspace. Permanent
           objects: wooden desk (entity #43), office chair (entity #67).
           Transient objects: laptop, book, cup, phone. Typical activities:
           working, reading, writing. Dimensions: ~1m radius at position
           (1.0m, -0.2m, 0.7m) from camera origin.
```

### 3. Integration Points

**Modified `scripts/run_slam_complete.py`** to feed into memory:

```python
# After processing frame
if self.spatial_memory:
    self.spatial_memory.add_entity_observation(
        entity_id=track.id,
        class_name=track.most_likely_class,
        timestamp=timestamp,
        position_3d=track.centroid_3d_mm,
        zone_id=track.zone_id,
        caption=caption if caption else None,
        confidence=track.confidence
    )

# After processing complete
self.spatial_memory.save()  # Persist to disk
```

## Usage

### 1. Process Video with Memory Building

```bash
# Process video - builds persistent memory
python scripts/run_slam_complete.py \
    --video data/examples/video.mp4 \
    --skip 30 \
    --yolo-model yolo11s \
    --export-memgraph \
    --rerun

# This creates:
# - Real-time graph in Memgraph
# - Persistent memory in memory/spatial_intelligence/
# - Rerun visualization (.rrd file)
# - Depth maps, zones, 3D reconstruction
```

### 2. Interactive Queries

```bash
# Start intelligent assistant
python scripts/spatial_intelligence_assistant.py -i

# Available commands:
#   sync     - Pull latest from Memgraph
#   stats    - Show memory statistics  
#   entities - List all entities
#   help     - Show help
#   exit     - Save and exit
```

### 3. Programmatic Access

```python
from orion.graph.spatial_memory import SpatialMemorySystem

# Load persistent memory
memory = SpatialMemorySystem(memory_dir=Path("memory/spatial_intelligence"))

# Query with context
result = memory.query_with_context("What color was the book?")
print(result['answer'])

# Get entity details
entity = memory.entities[42]
print(f"Semantic label: {memory.generate_semantic_label(42)}")
print(f"Observations: {entity.observations_count}")
print(f"Captions: {entity.captions}")
print(f"Movement: {len(entity.movement_history)} positions")

# Get zone details
zone = memory.zones[2]
print(f"Zone type: {zone.zone_type}")
print(f"Entities: {zone.entity_ids}")
print(f"Context: {zone.scene_context}")
```

## Long-Term Memory (Hours/Days)

### Cross-Session Intelligence

The system is designed for continuous learning:

```
SESSION 1 (Monday Morning):
- Process: morning_video.mp4
- Memory: 50 entities, 10 zones
- Storage: memory/spatial_intelligence/

SESSION 2 (Monday Afternoon - SAME SPACE):
- Process: afternoon_video.mp4
- Memory GROWS: 75 entities (25 new, 50 recognized)
- Recognizes: "book" is same as before (CLIP + spatial)
- Tracks: "book moved from zone 2 to zone 5"

SESSION 3 (Tuesday):
- Process: tuesday_video.mp4
- Memory: 95 entities, 12 zones
- Temporal understanding: "book has been on shelf since Monday 15:23"

QUERY (Wednesday):
You: "Where was the red book on Monday morning?"
Assistant: "The red book was in zone 2 (desk area) on Monday morning
            from 10:30-12:15. It was then moved to zone 5 (shelf) at
            15:23 and has remained there since."
```

### Storage Structure

```
memory/
└── spatial_intelligence/
    ├── entities.json          # All entities with full history
    ├── zones.json              # All spatial zones
    ├── metadata.json           # Session info, scene type
    └── conversation_history/
        ├── 2025-11-10.json    # Daily conversation logs
        ├── 2025-11-11.json
        └── ...
```

## Indoor Scene Reconstruction

### Depth Maps + Spatial Computation

The system uses MiDaS depth estimation + SLAM to:

1. **Generate depth maps** for each frame
2. **Backproject to 3D** using camera intrinsics
3. **Build spatial zones** based on 3D clustering
4. **Track entity movements** in 3D space
5. **Reconstruct room layout** from entity positions

```python
# Example: Reconstruct room from memory
memory = SpatialMemorySystem()

# Get all permanent entities (furniture)
furniture = [e for e in memory.entities.values() 
             if e.observations_count > 100 and e.movement_history_variance < 0.1]

# Reconstruct room boundaries
room_bounds = calculate_room_bounds(furniture)

# Identify functional areas
desk_area = identify_zone_by_entities(memory, ['desk', 'chair', 'laptop'])
kitchen_area = identify_zone_by_entities(memory, ['sink', 'stove', 'refrigerator'])

# 3D room model
room_model = {
    'bounds': room_bounds,
    'zones': memory.zones,
    'furniture': furniture,
    'layout_type': classify_room_type(furniture, memory.zones)
}
```

### Rerun Visualization

When using `--rerun`, the system logs:

```python
# 3D scene
- Camera poses (SLAM trajectory)
- Entity 3D positions and bboxes
- Depth maps (colored)
- Spatial zones (3D meshes)
- Entity trajectories over time
- Relationships as 3D arrows

# 2D overlays
- Bounding boxes with labels
- Zone boundaries
- Off-screen entity indicators
- Confidence scores

# Metrics
- FPS, entity count
- Zone occupancy
- SLAM statistics
```

## Acting as "Historian" for Gemini Robotics

### Integration Pattern

```python
# Your robotics model (e.g., Gemini Robotics 1.5)
from gemini_robotics import GeminiRobotics
from orion.graph.spatial_memory import SpatialMemorySystem

# Load Orion's spatial memory
spatial_memory = SpatialMemorySystem()

# Initialize robotics model
robot = GeminiRobotics()

# When robot needs context
robot_query = "I need to pick up the red book"

# Get rich context from Orion
context = spatial_memory.query_with_context(
    "Where is the red book and what's around it?"
)

# Provide to robot
robot.execute_task(
    task="pick_object",
    object_id=context['entities_mentioned'][0],
    spatial_context={
        'location': spatial_memory.entities[42].avg_position_3d,
        'zone': spatial_memory.zones[2],
        'nearby_obstacles': get_nearby_entities(42, radius=500),
        'approach_vector': calculate_safe_approach(42),
        'last_seen': spatial_memory.entities[42].last_seen
    }
)
```

### Benefits for Robotics

1. **Perfect Memory**: Robot knows where everything is, even if not currently visible
2. **Spatial Understanding**: 3D positions, zones, safe paths
3. **Temporal Context**: When things moved, typical states
4. **Semantic Labels**: "the red book on the desk" not just "entity #42"
5. **Relationship Awareness**: What's near, on top of, inside
6. **Cross-Session**: Remember from hours/days ago

## Next Steps to Complete Vision

### Current Status ✅

- ✅ Spatial memory system implemented
- ✅ Interactive assistant with context
- ✅ Persistent storage (JSON)
- ✅ Entity and zone tracking
- ✅ Conversation context
- ✅ Semantic labeling
- ✅ Integration hooks in pipeline

### To Fully Realize (Next Iterations) 🔄

1. **Auto-Integration with run_slam_complete.py**
   - Add `--use-spatial-memory` flag
   - Automatically feed all observations
   - Generate captions strategically

2. **Smart FastVLM Strategy**
   - Caption important frames only
   - Batch captioning at end
   - Priority: new entities, significant events

3. **Enhanced Relationship Extraction**
   - Spatial relationships from 3D positions
   - Activity detection (person HOLDING phone)
   - Temporal patterns (book ALWAYS_NEAR laptop)

4. **Room Reconstruction**
   - Build 3D mesh from depth + entities
   - Classify room type automatically
   - Export to standard formats (OBJ, PLY)

5. **Advanced Query Understanding**
   - NLP for complex queries
   - Multi-entity queries
   - Temporal queries ("what happened before...")

6. **Robotics API**
   - Standardized interface for robot models
   - Real-time pose queries
   - Safe navigation suggestions

## Testing the Current System

### Quick Test

```bash
# 1. Setup
bash scripts/setup_memgraph.sh

# 2. Process a video
python scripts/run_slam_complete.py \
    --video data/examples/video.mp4 \
    --skip 30 --yolo-model yolo11s \
    --export-memgraph --rerun

# 3. Sync into memory
python scripts/spatial_intelligence_assistant.py --sync

# 4. Interactive queries
python scripts/spatial_intelligence_assistant.py -i

# Try:
#   "What objects did you see?"
#   "Tell me about the book"
#   "Where was it?"
#   "What was near the laptop?"
```

### Full System Test

```bash
# Process multiple videos of same space
python scripts/run_slam_complete.py --video morning.mp4 ...
python scripts/spatial_intelligence_assistant.py --sync

python scripts/run_slam_complete.py --video afternoon.mp4 ...
python scripts/spatial_intelligence_assistant.py --sync

# Query across sessions
python scripts/spatial_intelligence_assistant.py -i
> "What changed between morning and afternoon?"
> "Where is the book now compared to this morning?"
```

## Summary

You now have:

✅ **Persistent spatial memory** - remembers everything  
✅ **Semantic indexing** - spatial + semantic + temporal  
✅ **3D reconstruction** - depth maps, zones, SLAM  
✅ **Interactive queries** - context-aware, clarifying questions  
✅ **Long-term memory** - cross-session intelligence  
✅ **Indoor scene understanding** - spatial computation  
✅ **Foundation for robotics** - historian for models like Gemini  

The system is designed to be the **perfect spatial memory** for robotics models - continuous understanding, perfect recall, rich context, over any time period.

**Next**: Integrate fully with the processing pipeline and test with real-world multi-hour/multi-day scenarios!
