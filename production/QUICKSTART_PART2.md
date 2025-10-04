# Quick Start: Part 2 - Semantic Uplift Engine

## 🚀 30-Second Start

```bash
# 1. Install dependencies
pip install hdbscan sentence-transformers neo4j

# 2. Start Neo4j (choose one method)
# Docker:
docker run --name neo4j -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
# OR Desktop: Download from https://neo4j.com/download/

# 3. (Optional) Start Ollama
ollama serve
ollama pull llama3

# 4. Run with Part 1 output
python production/test_part2.py --use-part1-output
```

## 📋 Prerequisites

### Required
- ✅ Python 3.10+
- ✅ Part 1 perception log (JSON file)
- ✅ Neo4j database running
- ✅ Dependencies: `hdbscan`, `sentence-transformers`, `neo4j`

### Optional (but recommended)
- 🔄 Ollama with llama3 model (for better event composition)

## 🔧 Setup

### Step 1: Install Python Packages

```bash
pip install hdbscan==0.8.39
pip install sentence-transformers==3.3.1
pip install neo4j==5.26.0
```

### Step 2: Setup Neo4j

**Option A: Neo4j Desktop** (Easiest)
1. Download from https://neo4j.com/download/
2. Install and launch
3. Create new project → Create database
4. Set password to `password`
5. Click "Start"
6. Access at http://localhost:7474

**Option B: Docker** (Fast)
```bash
docker run \
    --name neo4j \
    -p7474:7474 -p7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    -v $HOME/neo4j/data:/data \
    neo4j:latest
```

**Option C: Neo4j Aura** (Cloud)
1. Create free account at https://neo4j.com/cloud/aura/
2. Create instance
3. Save connection string and password
4. Update config:
```python
from production.part2_config import use_neo4j_aura
use_neo4j_aura("neo4j+s://xxxxx.databases.neo4j.io", "your-password")
```

### Step 3: Install Ollama (Optional)

```bash
# macOS
brew install ollama

# Start server
ollama serve

# Pull model
ollama pull llama3
```

## 💡 Basic Usage

### Example 1: From Part 1 Output

```python
import json
from production.part2_semantic_uplift import run_semantic_uplift

# Load perception log from Part 1
with open('data/testing/perception_log.json', 'r') as f:
    perception_log = json.load(f)

# Run semantic uplift
results = run_semantic_uplift(perception_log)

# Check results
print(f"✓ Tracked {results['num_entities']} entities")
print(f"✓ Detected {results['num_state_changes']} state changes")
print(f"✓ Created {results['graph_stats']['entity_nodes']} graph nodes")
```

### Example 2: With Custom Config

```python
from production.part2_config import apply_config, ACCURATE_CONFIG
from production.part2_semantic_uplift import run_semantic_uplift

# Use accurate configuration
apply_config(ACCURATE_CONFIG)

# Process
results = run_semantic_uplift(perception_log)
```

### Example 3: Using the Test Script

```bash
# Basic test with Part 1 output
python production/test_part2.py --use-part1-output

# With custom perception log
python production/test_part2.py --perception-log path/to/log.json

# With custom Neo4j
python production/test_part2.py \
    --use-part1-output \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password mypassword
```

## 📊 Verifying Results

### In Neo4j Browser (http://localhost:7474)

```cypher
// Count entities
MATCH (e:Entity) RETURN count(e)

// View entity distribution
MATCH (e:Entity)
RETURN e.label, count(e) as count
ORDER BY count DESC

// Find state changes
MATCH (e:Entity)-[:PARTICIPATED_IN]->(ev:Event {type: 'state_change'})
RETURN e.label, ev.description
LIMIT 10
```

### Using Python

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)

with driver.session() as session:
    # Count entities
    result = session.run("MATCH (e:Entity) RETURN count(e) as count")
    print(f"Entities: {result.single()['count']}")
    
    # Get entity labels
    result = session.run("""
        MATCH (e:Entity)
        RETURN e.label, count(e) as count
        ORDER BY count DESC
    """)
    for record in result:
        print(f"{record['e.label']}: {record['count']}")

driver.close()
```

## ⚙️ Configuration Presets

### Quick Config Switch

```python
from production.part2_config import apply_config, print_current_config
from production.part2_config import (
    FAST_CONFIG,          # Quick processing
    BALANCED_CONFIG,      # Default
    ACCURATE_CONFIG,      # Maximum detail
)

# Apply preset
apply_config(ACCURATE_CONFIG)

# Verify
print_current_config()
```

### Preset Comparison

| Config | Speed | Entity Precision | State Sensitivity | Best For |
|--------|-------|------------------|-------------------|----------|
| FAST | ⚡⚡⚡ | 🎯 | 🔍 | Quick tests |
| BALANCED | ⚡⚡ | 🎯🎯 | 🔍🔍 | Production |
| ACCURATE | ⚡ | 🎯🎯🎯 | 🔍🔍🔍 | Analysis |

## 🔧 Common Configurations

### Catch More State Changes

```python
from production.part2_config import create_custom_config, apply_config

config = create_custom_config(
    STATE_CHANGE_THRESHOLD=0.80,  # Lower = more sensitive (default: 0.85)
    MIN_EVENTS_PER_WINDOW=1       # Detect single changes
)
apply_config(config)
```

### Stricter Entity Clustering

```python
config = create_custom_config(
    MIN_CLUSTER_SIZE=5,             # Require 5+ appearances
    CLUSTER_SELECTION_EPSILON=0.10  # Tighter clustering
)
apply_config(config)
```

### Faster Processing (No LLM)

```python
config = create_custom_config(
    USE_OLLAMA=False  # Use template-based Cypher
)
apply_config(config)
```

## 🐛 Troubleshooting

### "hdbscan not available"
```bash
pip install hdbscan==0.8.39

# If build fails on macOS:
conda install -c conda-forge hdbscan
```

### "Could not connect to Neo4j"
```bash
# Check if running
curl http://localhost:7474

# Restart Docker
docker restart neo4j

# Check credentials
python -c "from neo4j import GraphDatabase; GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password')).verify_connectivity()"
```

### "Could not connect to Ollama"
```bash
# Start Ollama server
ollama serve

# Verify model
ollama list

# Pull if missing
ollama pull llama3

# OR disable Ollama
python -c "from production.part2_config import create_custom_config, apply_config; apply_config(create_custom_config(USE_OLLAMA=False))"
```

### "No state changes detected"
This means objects are too similar or threshold is too high:
```python
# Lower threshold
config = create_custom_config(STATE_CHANGE_THRESHOLD=0.75)
apply_config(config)
```

### "Too many unique entities"
HDBSCAN marked everything as noise:
```python
# Looser clustering
config = create_custom_config(
    MIN_CLUSTER_SIZE=2,
    CLUSTER_SELECTION_EPSILON=0.20
)
apply_config(config)
```

## 📈 Expected Performance

### Typical Processing Times

For a 60-second video with ~100 perception objects:

- **Entity Tracking**: 2-5 seconds
- **State Detection**: 3-8 seconds
- **Event Composition**: 
  - With Ollama: 10-30 seconds
  - Without Ollama: 1-2 seconds
- **Neo4j Ingestion**: 1-3 seconds
- **Total**: 15-45 seconds

### Scaling Guidelines

| Video Length | Objects | Processing Time | Memory |
|--------------|---------|-----------------|--------|
| 1 min | 100 | 30s | 2 GB |
| 5 min | 500 | 2 min | 2.5 GB |
| 30 min | 3000 | 10 min | 4 GB |

## 📁 Output Files

After processing, you'll have:

1. **Neo4j Knowledge Graph**: Persistent database at `bolt://localhost:7687`
2. **Console Logs**: Processing statistics and warnings
3. **Results Dictionary**: Returned by `run_semantic_uplift()`

## 🔄 Integration Workflow

### Full Pipeline (Part 1 → Part 2)

```python
from production.part1_perception_engine import run_perception_engine
from production.part2_semantic_uplift import run_semantic_uplift

# Step 1: Extract perception data
print("🎥 Processing video...")
perception_log = run_perception_engine("video.mp4")

# Step 2: Build knowledge graph
print("🧠 Building knowledge graph...")
results = run_semantic_uplift(perception_log)

print(f"✅ Complete! {results['num_entities']} entities in graph")
```

### Querying Results (Part 2 → Part 3 Ready)

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

# Ready for Part 3 query agent
# (to be implemented)
```

## 📚 Next Steps

1. **Query the graph** in Neo4j Browser
2. **Experiment with configs** to optimize for your use case
3. **Wait for Part 3** to enable natural language querying
4. **Integrate with real FastVLM** for better descriptions

## 🎯 Success Checklist

- [ ] Neo4j running and accessible
- [ ] Perception log from Part 1 available
- [ ] All dependencies installed
- [ ] Test script runs without errors
- [ ] Graph contains entities in Neo4j Browser
- [ ] (Optional) Ollama running and reachable

## 💬 Support

Questions? Check:
- Full documentation: `README_PART2.md`
- Part 1 docs: `README_PART1.md`
- Configuration details: `part2_config.py`
- Test examples: `test_part2.py`

Happy graph building! 🎉
