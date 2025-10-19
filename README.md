# Orion Video Analysis Pipeline

A high-performance video understanding system that builds rich knowledge graphs from video content.

**Status:** Production Ready | **Version:** 1.0 | **Last Updated:** October 2025

## Quick Start

```bash
# One-time setup
orion init

# Process a video
orion process video.mp4

# Query the knowledge graph  
orion query "What objects appeared in the kitchen?"
```

## Features

- **Object Detection:** YOLO11x for 80+ object classes
- **Rich Descriptions:** FastVLM for detailed object descriptions
- **Spatial Understanding:** Automatic spatial zone detection
- **Scene Recognition:** Kitchen, bedroom, office, living room detection
- **Object Tracking:** HDBSCAN clustering with state change detection
- **Causal Inference:** Mathematical scoring + LLM verification
- **Knowledge Graph:** Neo4j graph with temporal, spatial, and causal relationships
- **Q&A System:** Natural language queries over the knowledge graph

## Installation

### Prerequisites

- **macOS or Linux**
- **Python 3.10+**
- **8GB+ RAM** (16GB recommended)
- **Optional:** NVIDIA GPU (CUDA) or Apple Silicon (M1/M2/M3+)

### Step 1: Install System Dependencies

#### macOS (Homebrew)

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install wget (minimal dependency, used for model downloads)
brew install wget
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y wget python3.10 python3.10-venv
```

### Step 2: Set Up Conda Environment

```bash
# Create and activate Conda environment
conda create -n orion python=3.10
conda activate orion
```

### Step 3: Install Orion

```bash
# Clone the repository
git clone https://github.com/riddhimanrana/orion-research.git
cd orion-research

# Install Orion in development mode
pip install -e .
```

### Step 4: Set Up Docker + Neo4j

Neo4j must run in Docker for consistent, secure setup.

```bash
# Install Docker Desktop
# macOS: https://docs.docker.com/desktop/install/mac-install/
# Linux: https://docs.docker.com/engine/install/

# Start Neo4j container with secure password
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-secure-password-here \
  neo4j:latest
```

**Important:** Replace `your-secure-password-here` with a strong password. You'll set this via environment variable in the next step.

### Step 5: Set Up Environment Variables

Create `~/.orion/.env` (or set these in your shell profile):

```bash
# Neo4j configuration
export ORION_NEO4J_URI="neo4j://127.0.0.1:7687"
export ORION_NEO4J_USER="neo4j"
export ORION_NEO4J_PASSWORD="your-secure-password-here"

# Ollama configuration
export ORION_OLLAMA_URL="http://localhost:11434"
export ORION_OLLAMA_QA_MODEL="gemma3:4b"

# Runtime backend (auto-detected on Apple Silicon)
export ORION_RUNTIME_BACKEND="auto"
```

Source these in your shell:

```bash
source ~/.orion/.env
```

Or add to `~/.zshrc` (macOS) / `~/.bashrc` (Linux):

```bash
echo 'source ~/.orion/.env' >> ~/.zshrc
source ~/.zshrc
```

### Step 6: Set Up Ollama

Ollama provides the Q&A and embedding models.

```bash
# Install Ollama
# macOS/Linux: https://ollama.com/download

# Start Ollama service
ollama serve

# In a new terminal, download required models
ollama pull gemma3:4b
ollama pull openai/clip-vit-base-patch32
```

Keep Ollama running in the background while using Orion.

### Step 7: Initialize Orion

```bash
# Run one-time initialization
orion init
```

This will:

- Detect your hardware (Apple Silicon, NVIDIA GPU, etc.)
- Download required models
- Set up the Neo4j knowledge graph schema
- Test all connections
- Save configuration

## Usage

### Process a Video

```bash
# Basic usage
orion process video.mp4

# With options
orion process video.mp4 \
  --output-dir ./results \
  --config balanced \
  --verbose

# Configuration presets: fast, balanced, accurate
orion process video.mp4 --config accurate  # Best accuracy, slowest
orion process video.mp4 --config fast      # Fastest, lower accuracy
```

### Query the Knowledge Graph

```bash
# Interactive Q&A
orion query "What objects appeared in the scene?"
orion query "Where was the person located?"
orion query "What state changes occurred?"

# With options
orion query "What happened?" --verbose
```

### View Configuration

```bash
# Display current configuration
orion config show

# Set configuration values
orion config set neo4j.uri "neo4j://my-server:7687"

# Reset to defaults
orion config reset
```

## Hardware Notes

### Apple Silicon (M1/M2/M3+)

Orion automatically uses the MLX backend for optimal performance:

- Automatic GPU acceleration via Metal Performance Shaders
- Lower memory footprint than NVIDIA/CUDA
- Runs fast on MacBook Air, Mac Studio, etc.

```bash
# Verify MLX backend is detected
orion init  # Will show "Auto-detected MLX backend"
```

### NVIDIA GPU (CUDA)

If you have an NVIDIA GPU, Orion automatically uses CUDA:

```bash
# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

### CPU-Only

Orion runs on CPU, but slower. Recommended for testing only.

## Project Structure

```text
orion-research/
├── orion/
│   ├── config.py                 # Centralized configuration
│   ├── config_manager.py         # Config loading and management
│   ├── cli.py                    # Command-line interface
│   ├── perception_engine.py      # YOLO + FastVLM
│   ├── contextual_engine.py      # Spatial + scene analysis
│   ├── semantic_uplift.py        # Tracking + graph building
│   ├── knowledge_graph.py        # Neo4j schema and ingestion
│   ├── neo4j_manager.py          # Neo4j utilities
│   ├── embedding_model.py        # CLIP embeddings
│   ├── video_qa/                 # Q&A system
│   └── run_pipeline.py           # Main pipeline orchestration
├── scripts/
│   ├── init.py                   # Setup script (use `orion init` instead)
│   ├── test_optimizations.py     # Performance tests
│   └── ...
├── docs/
│   ├── SYSTEM_ARCHITECTURE.md    # Detailed system design
│   └── EVALUATION_README.md      # Evaluation framework
└── README.md                      # This file
```

## Troubleshooting

### Neo4j Connection Failed

```bash
# Check Neo4j is running
docker ps | grep neo4j

# Restart Neo4j
docker restart neo4j

# Check logs
docker logs neo4j

# Verify credentials in ~/.orion/.env
cat ~/.orion/.env
```

### Ollama Connection Failed

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve

# Verify models are downloaded
ollama list
```

### GPU Not Detected

```bash
# Check PyTorch GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check MLX (Apple Silicon)
python -c "import mlx.core as mx; print(f'MLX available: True')" 2>/dev/null || echo "MLX not available"

# Force CPU-only mode
export ORION_RUNTIME_BACKEND="torch"
```

## Performance

**Processing 1-minute video:**

- Total time: ~110 seconds
- Perception: 60s (YOLO + FastVLM)
- Contextual: 25s (spatial zones + corrections)
- Semantic: 25s (tracking + graph building)

**Key Optimizations:**

- Batch LLM processing (15x fewer API calls)
- Smart filtering (skip 70% of obvious cases)
- 90%+ spatial zone accuracy
- Evidence-based scene inference
- Efficient Neo4j operations

## Architecture

```text
Video Input
    ↓
Perception Engine (YOLO11x + FastVLM)
    ↓
Contextual Analysis (spatial zones + scene)
    ↓
Semantic Uplift (tracking + state changes)
    ↓
Neo4j Knowledge Graph (entities + relationships)
    ↓
Q&A System (Gemma3 via Ollama)
```

## Documentation

- **[SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md)** - Detailed architecture
- **[EVALUATION_README.md](docs/EVALUATION_README.md)** - Testing framework

## Testing

```bash
# Run test suite
python3 scripts/test_optimizations.py

# Run specific tests
python3 scripts/test_complete_pipeline.py
python3 scripts/test_contextual_understanding.py
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ORION_NEO4J_URI` | `neo4j://127.0.0.1:7687` | Neo4j connection URI |
| `ORION_NEO4J_USER` | `neo4j` | Neo4j username |
| `ORION_NEO4J_PASSWORD` | *(required)* | Neo4j password (set via env var for security) |
| `ORION_OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `ORION_OLLAMA_QA_MODEL` | `gemma3:4b` | Model for Q&A |
| `ORION_RUNTIME_BACKEND` | `auto` | Compute backend (auto, torch, mlx) |
| `ORION_CONFIG_DIR` | `~/.orion` | Configuration directory |
| `ORION_CONFIG_PATH` | `~/.orion/config.json` | Configuration file path |

## Support

- **Issues:** [GitHub Issues](https://github.com/riddhimanrana/orion-research/issues)
- **Documentation:** See `/docs` folder

---

**Built with:** YOLO11 • CLIP • FastVLM • Neo4j • Ollama
