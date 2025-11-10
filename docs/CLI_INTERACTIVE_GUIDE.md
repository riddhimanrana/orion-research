# Orion CLI - Interactive Mode Guide

**Last Updated**: January 2025

---

## 🎯 Overview

Orion CLI provides **two interactive modes** for querying video analysis results:

1. **Standard Q&A Mode** - Neo4j graph-based queries (semantic understanding)
2. **Spatial Intelligence Mode** - Persistent spatial memory with context awareness (experimental)

---

## 📊 Comparison

| Feature | Standard Q&A | Spatial Intelligence |
|---------|-------------|---------------------|
| **Backend** | Neo4j Graph | Memgraph + Spatial Memory |
| **Processing** | `orion analyze` | `orion research slam` |
| **Persistence** | Session only | Cross-session (days) |
| **Queries** | Semantic relations | Spatial + Temporal |
| **Context** | Limited | Full context awareness |
| **3D Tracking** | No | Yes (SLAM-based) |
| **Best For** | Semantic analysis | Robotics, scene memory |

---

## 🔵 Mode 1: Standard Q&A (Neo4j)

### When to Use
- Semantic video analysis
- Relationship queries ("What objects interact?")
- Scene understanding
- Quick video analysis

### Quick Start

```bash
# Process video
orion analyze video.mp4 -i

# After processing completes → Interactive Q&A starts automatically
# Ask questions about the video!
```

### Example Workflow

```bash
$ orion analyze kitchen_video.mp4 -i

# Processing...
# ✓ Video processed
# ✓ Graph built in Neo4j

════════════════════════════════════════════════
       Starting Interactive Q&A Mode
════════════════════════════════════════════════

Enter your question (or 'quit' to exit):
> What objects did you see in the kitchen?

> Which objects are on the table?

> What activities happened?

> quit
```

### Advanced Options

```bash
# Fast mode with Q&A
orion analyze video.mp4 --fast -i

# Keep existing Neo4j data
orion analyze video.mp4 --keep-db -i

# Specify QA model
orion analyze video.mp4 -i --qa-model llama3.2

# Custom output directory
orion analyze video.mp4 -i -o results/my_analysis
```

---

## 🟢 Mode 2: Spatial Intelligence (Experimental)

### When to Use
- Robotics applications
- Long-term spatial memory (hours/days)
- 3D scene reconstruction
- Cross-session queries
- Movement tracking
- Context-aware conversations

### Quick Start

```bash
# Step 1: Process video with spatial memory
orion research slam --video video.mp4 \
    --use-spatial-memory \
    --export-memgraph \
    -i

# Step 2: After processing → Spatial Intelligence Assistant starts
# Ask spatial and contextual questions!
```

### Example Workflow

```bash
$ orion research slam --video kitchen.mp4 \
    --use-spatial-memory \
    --export-memgraph \
    -i

# Processing with SLAM...
# ✓ Video processed (57s)
# ✓ Spatial memory saved
# ✓ Exported to Memgraph

════════════════════════════════════════════════
   Starting Spatial Intelligence Assistant
════════════════════════════════════════════════

✓ Loaded spatial memory: 15 entities, 142 observations

Commands: sync, stats, entities, help, exit

> stats
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Spatial Memory Statistics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Entities:      15
Total Captions:      8
Total Zones:         4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

> What did you see on the kitchen counter?

📍 Found 3 entities in kitchen counter area:
  • Entity 5 (coffee_mug): "Red ceramic coffee mug"
  • Entity 8 (laptop): "Silver laptop with glowing screen"
  • Entity 12 (book): "Cookbook with yellow cover"

> Tell me about the laptop

📦 Entity 8: laptop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Location: Kitchen counter (Zone 2)
Position: (850mm, 320mm, 1200mm)
Observations: 23 times
Caption: "Silver laptop with glowing screen"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

> Did it move?

📈 Movement Analysis for Entity 8 (laptop):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total distance: 12mm (minimal movement)
Status: STATIONARY (likely furniture)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

> exit
Goodbye!
```

### Advanced Features

#### Cross-Session Memory

```bash
# Day 1: Process morning video
orion research slam --video kitchen_morning.mp4 \
    --use-spatial-memory \
    --memory-dir memory/kitchen

# Day 2: Process afternoon video (SAME SPACE)
orion research slam --video kitchen_afternoon.mp4 \
    --use-spatial-memory \
    --memory-dir memory/kitchen

# Query accumulated memory
python scripts/spatial_intelligence_assistant.py -i
> What changed between morning and afternoon?
```

#### Multiple Environments

```bash
# Separate memories for different rooms
orion research slam --video kitchen.mp4 \
    --use-spatial-memory \
    --memory-dir memory/rooms/kitchen

orion research slam --video bedroom.mp4 \
    --use-spatial-memory \
    --memory-dir memory/rooms/bedroom

orion research slam --video living_room.mp4 \
    --use-spatial-memory \
    --memory-dir memory/rooms/living_room
```

#### Query Existing Memory (Without Processing)

```bash
# Just query existing spatial memory
python scripts/spatial_intelligence_assistant.py --interactive

# Or sync from Memgraph first
python scripts/spatial_intelligence_assistant.py --sync --interactive
```

---

## 🔄 Workflow Comparison

### Standard Q&A Workflow

```
┌─────────────────────────────────────────────┐
│  1. orion analyze video.mp4 -i              │
│     ↓                                       │
│  2. Video → Perception → Semantic → Neo4j   │
│     ↓                                       │
│  3. Interactive Q&A starts                  │
│     ↓                                       │
│  4. Ask semantic questions                  │
│     ↓                                       │
│  5. Quit → Memory lost                      │
└─────────────────────────────────────────────┘
```

### Spatial Intelligence Workflow

```
┌─────────────────────────────────────────────┐
│  1. orion research slam --video X           │
│     --use-spatial-memory -i                 │
│     ↓                                       │
│  2. Video → YOLO → Track → SLAM → Zones    │
│     → Spatial Memory (persistent)           │
│     ↓                                       │
│  3. Spatial Intelligence Assistant starts   │
│     ↓                                       │
│  4. Ask spatial/contextual questions        │
│     ↓                                       │
│  5. Quit → Memory persists to disk          │
│     ↓                                       │
│  6. Process more videos (same memory)       │
│     ↓                                       │
│  7. Query again anytime                     │
└─────────────────────────────────────────────┘
```

---

## 🎮 Interactive Commands

### Standard Q&A Commands

```
> [Your question]     - Ask any question
> quit               - Exit Q&A
> exit               - Exit Q&A
```

**Example Questions**:
- "What objects did you see?"
- "Which entities are on the table?"
- "What activities happened in the video?"
- "Describe the person wearing red"

### Spatial Intelligence Commands

```
> sync              - Pull latest from Memgraph
> stats             - Show memory statistics
> entities          - List all entities
> help              - Show commands
> exit              - Exit assistant

> [Your question]   - Ask spatial/contextual questions
```

**Example Questions**:
- "What's in zone 2?"
- "Where was the laptop?"
- "Show me entities that moved"
- "What changed between sessions?"
- "Did it move?" (context-aware)
- "Tell me about that" (uses previous context)

---

## 🚀 Complete Examples

### Example 1: Quick Video Analysis

```bash
# Standard Q&A - Quick and simple
orion analyze meeting.mp4 -i

# After processing...
> Who was in the meeting?
> What objects were on the conference table?
> quit
```

### Example 2: Robot Navigation Memory

```bash
# Build spatial memory for robot
orion research slam --video robot_exploration.mp4 \
    --use-spatial-memory \
    --memory-dir memory/robot_map \
    -i

# Query spatial memory
> Where did I see the charging station?
> What objects are in the living room?
> Show me the kitchen layout
> exit
```

### Example 3: Long-term Scene Monitoring

```bash
# Day 1 Morning
orion research slam --video office_morning.mp4 \
    --use-spatial-memory \
    --memory-dir memory/office_week

# Day 1 Afternoon
orion research slam --video office_afternoon.mp4 \
    --use-spatial-memory \
    --memory-dir memory/office_week

# Day 2 Morning
orion research slam --video office_day2.mp4 \
    --use-spatial-memory \
    --memory-dir memory/office_week

# Query accumulated knowledge
python scripts/spatial_intelligence_assistant.py -i
> What objects appeared this week?
> Which entities moved between Monday and Tuesday?
> Show me the desk area evolution
```

### Example 4: Multi-room Mapping

```bash
# Map entire house
for room in kitchen bedroom living_room bathroom; do
    orion research slam --video ${room}.mp4 \
        --use-spatial-memory \
        --memory-dir memory/house_map/${room}
done

# Query specific room
python scripts/spatial_intelligence_assistant.py \
    --memory-dir memory/house_map/kitchen -i
```

---

## 🛠️ Troubleshooting

### Standard Q&A Issues

**Problem**: "Q&A not available"
```bash
# Solution: Install Ollama
pip install ollama

# Or specify in config
orion config set qa_model llama3.2
```

**Problem**: Neo4j connection error
```bash
# Check Neo4j is running
orion services neo4j status

# Start if needed
orion services neo4j start
```

### Spatial Intelligence Issues

**Problem**: "No spatial memory found"
```bash
# You must process with 'orion research slam' first
orion research slam --video X --use-spatial-memory

# NOT 'orion analyze' (that's for Neo4j mode)
```

**Problem**: Empty memory after processing
```bash
# Check memory directory
ls -la memory/spatial_intelligence/

# Should see: entities.json, zones.json, metadata.json
```

**Problem**: Assistant not syncing from Memgraph
```bash
# Check Memgraph is running (Docker)
docker ps | grep memgraph

# Test connection
python -c "from orion.graph.memgraph_backend import MemgraphBackend; \
    backend = MemgraphBackend(); \
    print('✓ Connected!'); \
    backend.close()"
```

---

## 📚 Related Documentation

- **Spatial Memory System**: `docs/SPATIAL_MEMORY_QUICKSTART.md`
- **Complete Guide**: `docs/PERSISTENT_SPATIAL_INTELLIGENCE.md`
- **CLI Reference**: `orion --help`, `orion analyze --help`, `orion research slam --help`

---

## 🎯 Decision Guide

**Use Standard Q&A if you want**:
- ✅ Quick video analysis
- ✅ Semantic understanding
- ✅ One-time processing
- ✅ Simple setup

**Use Spatial Intelligence if you want**:
- ✅ Robotics applications
- ✅ Long-term memory (hours/days)
- ✅ 3D spatial tracking
- ✅ Cross-session queries
- ✅ Context-aware conversations
- ✅ Movement analysis

---

## 🔑 Key Takeaways

1. **Two modes, different purposes**:
   - `orion analyze -i` → Semantic Q&A (Neo4j)
   - `orion research slam --use-spatial-memory -i` → Spatial Intelligence

2. **Standard Q&A** is simpler and faster for one-off analysis

3. **Spatial Intelligence** is powerful for robotics and long-term applications

4. **Memory persistence**:
   - Standard Q&A: Lost after session
   - Spatial Intelligence: Persists forever

5. **Context awareness**:
   - Standard Q&A: Limited
   - Spatial Intelligence: Full context ("it", "that", "the same")

---

**Ready to start?**

```bash
# Quick semantic analysis
orion analyze video.mp4 -i

# OR

# Persistent spatial intelligence
orion research slam --video video.mp4 --use-spatial-memory -i
```
