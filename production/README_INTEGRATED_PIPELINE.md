# Integrated Pipeline: Parts 1 + 2

## Overview

This is the complete "From Moments to Memory" pipeline that processes videos from raw frames to queryable knowledge graphs.

```
Video File
    ↓
┌─────────────────────────────────────┐
│  PART 1: PERCEPTION ENGINE          │
│  • Video frame selection            │
│  • Object detection (YOLO)          │
│  • Visual embeddings (ResNet50)     │
│  • Rich descriptions (FastVLM)      │
└────────────┬────────────────────────┘
            ↓
    Perception Log (JSON)
            ↓
┌─────────────────────────────────────┐
│  PART 2: SEMANTIC UPLIFT            │
│  • Entity tracking (HDBSCAN)        │
│  • State change detection           │
│  • Event composition (LLM)          │
│  • Knowledge graph (Neo4j)          │
└────────────┬────────────────────────┘
            ↓
    Neo4j Knowledge Graph
```

## Quick Start

### 1. Setup Prerequisites

```bash
# Install all dependencies
pip install -r requirements.txt

# Start Neo4j (choose one)
# Option A: Docker
docker run --name neo4j -p7474:7474 -p7687:7687 \
    -e NEO4J_AUTH=neo4j/password neo4j:latest

# Option B: Neo4j Desktop (download from neo4j.com/download)

# (Optional) Start Ollama for better event composition
ollama serve
ollama pull llama3
```

### 2. Run Complete Pipeline

```python
from production.integrated_pipeline import run_integrated_pipeline

# Run with real FastVLM
results = run_integrated_pipeline(
    video_path="path/to/video.mp4",
    use_fastvlm=True,  # Use real Apple FastVLM model
    part1_config="balanced",
    part2_config="balanced"
)

print(f"Part 1: {results['part1']['num_objects']} objects detected")
print(f"Part 2: {results['part2']['num_entities']} entities tracked")
```

### 3. Command Line Usage

```bash
# Basic usage with default settings
python production/integrated_pipeline.py path/to/video.mp4

# With FastVLM enabled (real model)
python production/integrated_pipeline.py path/to/video.mp4 --use-fastvlm

# With custom configurations
python production/integrated_pipeline.py path/to/video.mp4 \
    --use-fastvlm \
    --part1-config accurate \
    --part2-config sensitive \
    --output-dir data/my_results

# Skip Part 1 and use existing perception log
python production/integrated_pipeline.py path/to/video.mp4 \
    --skip-part1 \
    --perception-log data/testing/perception_log.json

# Run only Part 1 (no graph building)
python production/integrated_pipeline.py path/to/video.mp4 \
    --skip-part2 \
    --use-fastvlm
```

## FastVLM Integration

### What is FastVLM?

FastVLM is Apple's efficient vision-language model designed for fast, high-quality image understanding. We use the official **apple/FastVLM-0.5B** model from HuggingFace Hub.

### Why Use Real FastVLM vs Placeholder?

**Real FastVLM** (`use_fastvlm=True`):
- ✅ Actual semantic understanding of images
- ✅ Context-aware descriptions
- ✅ Detects actions, states, relationships
- ✅ Better downstream knowledge graph quality
- ❌ Requires GPU/Apple Silicon for speed
- ❌ Slower than placeholder (~2-5s per object)

**Placeholder** (`use_fastvlm=False`):
- ✅ Very fast (~0.1s per object)
- ✅ No GPU required
- ✅ Good for testing pipeline logic
- ❌ Generic template descriptions
- ❌ Limited semantic value

### FastVLM Setup

The model is automatically downloaded from HuggingFace on first use:

```python
# Model is loaded automatically in Part 1
# Location: apple/FastVLM-0.5B from HuggingFace Hub
# Size: ~2GB download
# Cache: ~/.cache/huggingface/
```

**First run** will download the model (2GB):
```
Loading FastVLM model (apple/FastVLM-0.5B)...
Downloading model files...  [This may take a few minutes]
FastVLM model loaded successfully
```

**Subsequent runs** use the cached model:
```
Loading FastVLM model (apple/FastVLM-0.5B)...
FastVLM model loaded successfully  [~5-10 seconds]
```

### Device Selection

FastVLM automatically selects the best available device:

- **CUDA (NVIDIA GPU)**: Fastest, recommended for large videos
- **MPS (Apple Silicon)**: Fast, great for M1/M2/M3 Macs
- **CPU**: Slowest, works everywhere but not recommended for production

```python
# Auto-detect (recommended)
run_integrated_pipeline(video_path, use_fastvlm=True)

# Force specific device (advanced)
from production.part1_perception_engine import Config
Config.FASTVLM_DEVICE = "cuda"  # or "mps" or "cpu"
```

## Configuration Presets

### Part 1 Configurations

| Config | Description | FastVLM | Objects/Min | Best For |
|--------|-------------|---------|-------------|----------|
| `fast` | Quick processing | Recommended | ~100 | Testing |
| `balanced` | Default quality | Recommended | ~50 | Production |
| `accurate` | Maximum detail | **Required** | ~20 | Analysis |

### Part 2 Configurations

| Config | Description | Entity Precision | State Sensitivity |
|--------|-------------|------------------|-------------------|
| `fast` | Quick graph building | Medium | Low |
| `balanced` | Default quality | High | Medium |
| `accurate` | Maximum insight | Very High | High |
| `sensitive` | Catch subtle changes | High | Very High |

### Recommended Combinations

**For Testing (Fast iteration)**:
```python
run_integrated_pipeline(
    video_path,
    use_fastvlm=False,  # Placeholder
    part1_config="fast",
    part2_config="fast"
)
# Expected time: 30-60s for 1min video
```

**For Production (Balanced quality/speed)**:
```python
run_integrated_pipeline(
    video_path,
    use_fastvlm=True,  # Real FastVLM
    part1_config="balanced",
    part2_config="balanced"
)
# Expected time: 5-10min for 1min video (GPU)
```

**For Research (Maximum quality)**:
```python
run_integrated_pipeline(
    video_path,
    use_fastvlm=True,
    part1_config="accurate",
    part2_config="accurate"
)
# Expected time: 15-30min for 1min video (GPU)
```

## Output Files

### Perception Log (Part 1)
**Location**: `data/testing/perception_log_{video}_{timestamp}.json`

**Structure**:
```json
[
  {
    "frame_number": 42,
    "timestamp": 1.4,
    "object_class": "person",
    "confidence": 0.92,
    "bbox": [100, 200, 150, 300],
    "visual_embedding": [512-dim vector],
    "rich_description": "A person wearing blue shirt, walking forward..."
  },
  ...
]
```

### Neo4j Knowledge Graph (Part 2)
**Access**: Neo4j Browser at http://localhost:7474

**Sample Queries**:
```cypher
// View all entities
MATCH (e:Entity) RETURN e LIMIT 25

// Find state changes
MATCH (e:Entity)-[:PARTICIPATED_IN]->(ev:Event {type: 'state_change'})
RETURN e.label, ev.description

// Entity timeline
MATCH (e:Entity {id: 'entity_cluster_0001'})-[:PARTICIPATED_IN]->(ev:Event)
RETURN ev.timestamp, ev.description
ORDER BY ev.timestamp
```

## Performance Characteristics

### Part 1 Processing Speed

**With Placeholder** (use_fastvlm=False):
- ~100-200 objects/min (single-threaded)
- ~500-1000 objects/min (with 4 workers)

**With Real FastVLM** (use_fastvlm=True):
- **GPU (CUDA)**: ~20-30 objects/min
- **Apple Silicon (MPS)**: ~15-25 objects/min
- **CPU**: ~5-10 objects/min (not recommended)

### Part 2 Processing Speed

- Entity tracking: ~1000 objects/min
- State detection: ~500 objects/min
- Event composition (with LLM): ~30-60 windows/min
- Neo4j ingestion: ~1000 nodes/min

### Total Pipeline Time Estimates

For a **60-second video** at 30fps (~50 objects after filtering):

| Configuration | FastVLM | Part 1 | Part 2 | Total |
|---------------|---------|--------|--------|-------|
| Fast + Placeholder | No | 30s | 20s | **~1min** |
| Balanced + FastVLM (GPU) | Yes | 5min | 30s | **~6min** |
| Accurate + FastVLM (GPU) | Yes | 15min | 1min | **~16min** |

## Troubleshooting

### Issue: "FastVLM model not loading"

**Symptoms**: Falls back to placeholder despite `use_fastvlm=True`

**Solutions**:
```bash
# Check transformers installation
pip install --upgrade transformers torch

# Test FastVLM directly
python production/fastvlm_wrapper.py path/to/test/image.jpg

# Check for CUDA/MPS availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

### Issue: "Out of memory (OOM) with FastVLM"

**Solutions**:
1. Use CPU instead of GPU (slower but more memory):
```python
from production.part1_perception_engine import Config
Config.FASTVLM_DEVICE = "cpu"
```

2. Reduce batch size or use fewer workers:
```python
Config.NUM_DESCRIPTION_WORKERS = 1  # Reduce parallelism
```

3. Process shorter video segments

### Issue: "Neo4j connection failed"

**Solutions**:
```bash
# Check if Neo4j is running
curl http://localhost:7474

# Test connection
python -c "from neo4j import GraphDatabase; GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password')).verify_connectivity(); print('OK')"

# Restart Neo4j
docker restart neo4j  # If using Docker
```

### Issue: "Ollama not available"

**Note**: Ollama is optional. The system will fall back to template-based Cypher generation.

**To enable Ollama** (better quality):
```bash
# Install
brew install ollama

# Start server
ollama serve

# Pull model
ollama pull llama3

# Verify
curl http://localhost:11434/api/tags
```

### Issue: "YOLO model download failing"

**Solutions**:
```bash
# Pre-download YOLO model
python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"

# Or download manually from Ultralytics GitHub
```

## Integration API

### Python API

```python
from production.integrated_pipeline import run_integrated_pipeline

# Full pipeline
results = run_integrated_pipeline(
    video_path="video.mp4",
    output_dir="data/results",
    use_fastvlm=True,
    part1_config="balanced",
    part2_config="balanced",
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    skip_part1=False,
    skip_part2=False
)

# Check results
if results['success']:
    print(f"✓ Pipeline completed successfully")
    print(f"  Part 1: {results['part1']['num_objects']} objects")
    print(f"  Part 2: {results['part2']['num_entities']} entities")
else:
    print(f"✗ Pipeline failed: {results['errors']}")
```

### Run Individual Parts

```python
# Part 1 only
from production.part1_perception_engine import run_perception_engine
perception_log = run_perception_engine("video.mp4")

# Part 2 only (requires existing perception log)
from production.part2_semantic_uplift import run_semantic_uplift
import json

with open('data/testing/perception_log.json') as f:
    perception_log = json.load(f)

results = run_semantic_uplift(perception_log)
```

## Next Steps

After running the integrated pipeline:

1. **Explore the Knowledge Graph**
   - Open Neo4j Browser: http://localhost:7474
   - Try sample queries
   - Visualize relationships

2. **Analyze Results**
   ```python
   from production.test_part2 import visualize_graph_stats, query_sample_data
   
   visualize_graph_stats()
   query_sample_data()
   ```

3. **Prepare for Part 3**
   - The knowledge graph is ready for querying
   - Part 3 will add Q&A agents
   - Evaluation metrics (EC-15, LOT-Q)

## Architecture Details

### Two-Tier Processing (Part 1)

- **Tier 1 (Fast)**: Scene detection + Object detection + Embeddings
- **Tier 2 (Slow)**: Async FastVLM description generation
- Workers process descriptions in parallel while video continues

### Five-Stage Uplift (Part 2)

1. **Entity Tracking**: Cluster visual embeddings → permanent IDs
2. **State Detection**: Compare descriptions → identify changes
3. **Temporal Windows**: Group events into time periods
4. **Event Composition**: LLM generates structured Cypher
5. **Graph Ingestion**: Batch insert into Neo4j

## Performance Optimization Tips

### For Faster Processing

1. **Use placeholder descriptions** for testing:
   ```python
   use_fastvlm=False  # ~10x faster Part 1
   ```

2. **Reduce video resolution** in Part 1 config:
   ```python
   Config.SCENE_DETECTION_SIZE = (320, 240)  # Smaller
   ```

3. **Increase worker count** (if you have cores):
   ```python
   Config.NUM_DESCRIPTION_WORKERS = 8  # More parallelism
   ```

4. **Disable Ollama** in Part 2:
   ```python
   Config.USE_OLLAMA = False  # Use template Cypher
   ```

### For Higher Quality

1. **Use accurate configs**:
   ```python
   part1_config="accurate"
   part2_config="accurate"
   ```

2. **Enable FastVLM with GPU**:
   ```python
   use_fastvlm=True
   Config.FASTVLM_DEVICE = "cuda"  # or "mps"
   ```

3. **Lower state change threshold** (more sensitive):
   ```python
   Config.STATE_CHANGE_THRESHOLD = 0.75  # Default: 0.85
   ```

4. **Enable Ollama** for better events:
   ```bash
   ollama serve
   ollama pull llama3
   ```

## License

Part of the Orion Research project. See LICENSE for details.

## Contributors

- Riddhiman Rana
- Aryav Semwal
- Yogesh Atluru
- Jason Zhang

---

**Last Updated**: January 2025
