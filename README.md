# Orion Video Analysis Pipeline

A high-performance video understanding system that builds rich knowledge graphs from video content.

## Quick Start

```bash
# Process a video
python -m orion.cli process path/to/video.mp4

# Query the knowledge graph  
python -m orion.cli query "What objects appeared in the kitchen?"
```

## Features

- **Object Detection:** YOLO11m for 80 object classes
- **Rich Descriptions:** FastVLM for detailed object descriptions  
- **Spatial Understanding:** Automatic spatial zone detection (walls, floor, ceiling)
- **Scene Recognition:** Kitchen, bedroom, office, living room detection
- **Object Tracking:** HDBSCAN clustering across frames
- **State Changes:** Automatic detection of object state changes
- **Causal Inference:** Mathematical scoring + LLM verification
- **Knowledge Graph:** Neo4j graph with temporal, spatial, and causal relationships
- **Q&A System:** Natural language queries over the knowledge graph

## Performance

**1-minute video processing:**
- Total time: ~110 seconds (1.8 minutes)
- Perception: 60s (object detection + descriptions)
- Contextual: 25s (spatial zones + corrections)
- Semantic: 25s (tracking + graph building)

**Key Optimizations:**
- Batch LLM processing (15x fewer calls)
- Smart filtering (skip 70% of obvious cases)
- 90%+ spatial zone accuracy
- Evidence-based scene inference
- Efficient graph operations

## Architecture

```
Video Input
    ↓
Perception Engine (YOLO + FastVLM)
    ↓
Contextual Analysis (spatial + scene)
    ↓
Semantic Uplift (tracking + states)
    ↓
Neo4j Knowledge Graph
    ↓
Q&A System
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Start Neo4j
neo4j start

# Ensure Ollama is running with gemma3:4b
ollama pull gemma3:4b
```

## Usage

### Process Video

```bash
# Basic usage
python -m orion.cli process video.mp4

# With options
python -m orion.cli process video.mp4 \
  --output-dir ./output \
  --config balanced \
  --verbose
```

### Query Results

```bash
# Ask questions
python -m orion.cli query "What happened in the video?"
python -m orion.cli query "Where was the person located?"
python -m orion.cli query "What objects were on the desk?"
```

## Project Structure

```
orion-research/
├── orion/
│   ├── perception_engine.py      # YOLO + FastVLM
│   ├── contextual_engine.py      # Spatial + scene analysis
│   ├── semantic_uplift.py        # Tracking + graph building
│   ├── knowledge_graph.py       # Neo4j ingestion + reasoning
│   ├── tracking_engine.py        # Object tracking
│   ├── causal_inference.py       # Causal relationships
│   ├── video_qa/                 # Q&A system package
│   └── run_pipeline.py           # Main pipeline
├── docs/
│   ├── SYSTEM_ARCHITECTURE.md    # Detailed architecture
│   └── EVALUATION_README.md      # Evaluation framework
└── README.md                      # This file
```

## Documentation

- **[SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md)** - Detailed system design
- **[EVALUATION_FRAMEWORK.md](docs/EVALUATION_FRAMEWORK.md)** - Testing and evaluation

## Requirements

- Python 3.8+
- Neo4j 5.x  
- Ollama with gemma3:4b model
- 8GB+ RAM recommended
- CUDA (optional, for GPU acceleration)

## Testing

```bash
# Run test suite
python3 scripts/test_optimizations.py

# Expected: ✅ All tests passed! (5/5)
```

---

**Status:** Production Ready  
**Version:** 1.0  
**Last Updated:** October 2025
