# Orion

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)
[![Ollama](https://img.shields.io/badge/ollama-local%20LLM-blue)](https://ollama.com)
[![Neo4j](https://img.shields.io/badge/neo4j-optional-blue)](https://neo4j.com)

Orion turns videos into a simple knowledge graph you can query in plain English. It detects objects, describes scenes, tracks changes over time, and stores events in Neo4j. A local LLM (via Ollama) answers questions grounded by this graph.

## Requirements
 
- macOS or Linux
- Python 3.10+
- Ollama (local LLM)
- Neo4j (optional; for graph storage/visualization)

## Install

```bash
# Clone
git clone https://github.com/riddhimanrana/orion-research
cd orion-research

# Environment (example with conda)
conda create -n orion python=3.10 -y
conda activate orion

# Install
pip install -e .
```

## Initialize (one-time)

```bash
# Download models and prepare folders
orion init

# Start Ollama (in another terminal)
ollama serve
```

This will:

- Download YOLO11m (detection) and FastVLM-0.5B (descriptions)
- Pull Ollama models: gemma3:4b (Q&A/event composition) and embeddinggemma (embeddings)
- Create data/ and models/ directories

## Neo4j (optional)

You can run Neo4j locally or use Neo4j Aura. Orion works without Neo4j, but using it lets you visualize and query the event graph.

Option A — Docker:

```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

Option B — Neo4j Desktop:

- Download: <https://neo4j.com/download/>
- Create a local DBMS and start it (note Bolt port, username, password)

Environment variables (used by Orion):

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
```

Quick check:

- Browser: <http://localhost:7474> (login and run: MATCH (n) RETURN count(n))
- Aura: set NEO4J_URI to your neo4j+s:// URI and use your Aura credentials

## Use

```bash
# Analyze a video
orion analyze path/to/video.mp4

# Analyze and then start interactive Q&A
orion analyze path/to/video.mp4 -i

# Q&A only (after data is ingested)
orion qa
```

Helpful:

```bash
orion models    # Show model information
orion modes     # Show processing modes
orion --help    # CLI help
```

## Configuration (optional)

- Perception: production/perception_config.py (thresholds, sampling, model names)
- Uplift/graph: production/semantic_uplift.py (clustering, window size, LLM)

Defaults work out of the box. Event composition uses Ollama with gemma3:4b and validates Cypher before execution; invalid queries fall back to safe, idempotent ones. Neo4j is optional.

## Troubleshooting

- Ollama not running → run "ollama serve" in another terminal
- Downloads slow/failing → re-run "orion init" and check network
- Neo4j not required → you can skip it; if used, ensure it’s reachable
- Ensure enough disk space (several GB for models)

## License

MIT — see LICENSE
