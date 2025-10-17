# Orion

[![CI](https://github.com/riddhimanrana/orion-research/actions/workflows/ci.yml/badge.svg)](https://github.com/riddhimanrana/orion-research/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)
[![Ollama](https://img.shields.io/badge/ollama-local%20LLM-blue)](https://ollama.com)
[![Neo4j](https://img.shields.io/badge/neo4j-optional-blue)](https://neo4j.com)

Orion turns videos into a rich knowledge graph you can query in plain English. It detects objects with **YOLO11x**, generates embeddings with **CLIP**, describes scenes with **FastVLM**, tracks entities across time, and builds an intelligent graph with scene understanding, spatial relationships, and causal reasoning.

**Enhanced Knowledge Graph Features:**
- 🏠 **Scene Classification** - Automatically detects room types (office, kitchen, bedroom, etc.)
- 🔗 **Spatial Relationships** - Tracks which objects are near, above, below, or contain each other
- 🧠 **Contextual Embeddings** - Combines visual + spatial + scene context for better understanding
- ⚡ **Causal Reasoning** - Infers potential cause-effect relationships between state changes
- 💬 **Intelligent QA** - Context-aware question answering with multi-modal retrieval
- 📊 **Scene Transitions** - Temporal graph showing how scenes flow and change

The system uses a unified **ModelManager** for efficient resource sharing and **OrionConfig** for centralized configuration, achieving 21x efficiency improvement over naive approaches.

## Requirements

- macOS or Linux (Apple Silicon or CUDA recommended for speed)
- Python 3.10+
- Ollama (local LLM)
- Neo4j (optional; for graph storage/visualization)

Runtime note: Orion ships a single PyTorch backend that auto-picks CPU, Apple MPS, or CUDA; there is no separate MLX build to maintain across platforms.

## Quick Start (one command)

```bash
curl -sSfL https://raw.githubusercontent.com/riddhimanrana/orion-research/refs/heads/main/scripts/bootstrap.sh | bash
```

You can override pieces of the bootstrap with environment variables:

- `ORION_TARGET_DIR` – where to clone (default: `orion-research`)
- `ORION_PYTHON` – interpreter to use (default: `python3`)
- `ORION_VENV` – name of the created virtualenv (default: `.orion-venv`)

After it finishes:

```bash
source orion-research/.orion-venv/bin/activate
python -m orion.cli --help
pip install -e .[dev]         # optional: install test/lint tooling
pytest tests/test_quickstart.py  # smoke check shared with CI
```

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
orion init           # or: python -m orion.cli init

# Start Ollama (in another terminal)
ollama serve
```

This collects everything Orion needs under `models/`:

- YOLO11m detector and FastVLM-0.5B describer (Torch backend)
- Suggested Ollama models: `gemma3:4b` for Q&A and `embeddinggemma` for embeddings
- Shared caches for Hugging Face, Ultralytics, and Torch assets

## Neo4j

You can run Neo4j locally or use Neo4j Aura. Orion works without Neo4j, but using it lets you visualize and query the event graph which is sort of the main point of the project.

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

Orion stores its connection details in `~/.orion/config.json`. You can set them once via the CLI.

## Use

```bash
# Analyze a video
orion analyze path/to/video.mp4

# Analyze and then start interactive Q&A
orion analyze path/to/video.mp4 -i

# Q&A only (after data is ingested)
orion qa

# Inspect configuration defaults
orion config show

# Update settings (examples)
orion config set neo4j.uri bolt://localhost:7687
orion config set neo4j.password changeme
orion config set qa.model mistral:7b
orion config set embedding.backend sentence-transformer
orion config set embedding.model all-MiniLM-L6-v2
```

On-demand overrides are also available:

```bash
orion analyze video.mp4 --neo4j-uri neo4j://other-host:7687 --qa-model llama3.2:3b
```

Helpful commands:

```bash
orion models    # Show model information
orion modes     # Show processing modes
orion init      # Re-sync model assets (or: python -m orion.cli init)
orion --help    # CLI help and subcommand docs
python -m orion.cli --help   # Direct module invocation if the entry point is absent
```

## Scene & Location Graph

- `Scene` nodes capture each analyzed frame with its dominant objects, descriptive text, and a vector embedding for similarity search.
- `Location` nodes cluster scenes by their shared objects (e.g., “desk, monitor, keyboard”), enabling questions like “find rooms similar to the office scene.”
- Relationships include `APPEARS_IN` (entities ↔ scenes with timestamps), `IN_LOCATION` (scene → location grouping), `TRANSITIONS_TO` (ordered timeline with gaps), and `SIMILAR_TO` (bi-directional cosine similarity scores between scenes).

These structures are populated automatically during `orion analyze`, so downstream queries gain immediate context about spaces, transitions, and object co-occurrence.

## Configuration (optional)

The CLI settings live in `~/.orion/config.json` and can be managed with `orion config`:

```bash
orion config show            # View current values (passwords are masked)
orion config set runtime torch
orion config set embedding.backend sentence-transformer
orion config reset           # Restore defaults
```

Perception and uplift presets remain in the repository if you wish to tweak advanced behaviour:

- Perception presets: `production/perception_config.py`
- Semantic uplift: `production/semantic_uplift.py`

Defaults work out of the box. Event composition uses Ollama with `gemma3:4b` (or your configured model) and validates Cypher before execution; invalid queries fall back to safe, idempotent ones. Neo4j is optional.

## Continuous Integration

GitHub Actions (`.github/workflows/ci.yml`) installs Orion with developer dependencies, runs formatting and lint checks, and executes `pytest tests/test_quickstart.py` to confirm the CLI boots. The status badge above reflects the latest run on `main`.

## Troubleshooting

- Ollama not running → run "ollama serve" in another terminal
- Downloads slow/failing → re-run `orion init` (or `python -m orion.cli init`) and check network
- Neo4j not required → you can skip it; if used, ensure it’s reachable
- Ensure enough disk space (several GB for models)
- Ollama issues → verify the model name you configured exists locally (`ollama list`)

## Research & Evaluation

Orion implements a novel **two-stage causal inference** approach for building knowledge graphs from videos:

1. **Mathematical Causal Influence Score (CIS)**: Filters agent-patient pairs using spatial proximity, directed motion, temporal decay, and visual similarity
2. **LLM Verification**: Only high-scoring pairs are passed to Gemma 3 4B(or gpt-oss-20b but you choose based on the models available to be run in your environments as you need) for semantic labeling
  
### Evaluation Framework

Compare CIS+LLM vs. heuristic baseline:

```bash
# Single video evaluation
python scripts/run_evaluation.py --video path/to/video.mp4

# VSGR benchmark evaluation
python scripts/run_evaluation.py --mode benchmark --benchmark vsgr --dataset-path /path/to/vsgr
```

See [EVALUATION.md](EVALUATION.md) for detailed documentation on:
- Causal Inference Score (CIS) calculation
- Heuristic baseline implementation
- VSGR benchmark support
- Metrics and comparison tools

See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for technical architecture details.

## License

MIT — see LICENSE
