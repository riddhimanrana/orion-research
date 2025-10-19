# Orion: Intelligent Video Understanding Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Production](https://img.shields.io/badge/Status-Production-brightgreen)](https://github.com/riddhimanrana/orion-research)

A comprehensive video understanding system that analyzes video content using object detection, entity tracking, and knowledge graphs. Ask questions about your videos and get intelligent answers.

## What It Does

Video → Intelligence

1. **Detect Objects** - YOLO11x identifies 80+ object classes
2. **Understand Visuals** - CLIP + FastVLM generate descriptions
3. **Track Entities** - Hungarian algorithm tracks objects across frames
4. **Build Knowledge Graph** - Neo4j stores relationships (temporal, spatial, causal)
5. **Answer Questions** - Query videos in natural language via Ollama LLM

## Quick Start

```bash
# One-time setup (prompts for configuration)
orion init

# Analyze a video
orion analyze video.mp4

# Ask questions
orion qa
```

## Pre-Installation Requirements

Before starting, install these tools:

1. **Conda** - For Python environment management
   - [Download Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) (recommended, minimal)
   - Or [Anaconda](https://www.anaconda.com/download) (full version)
   - Verify: `conda --version`

2. **Git** - To clone the repository
   - macOS: `brew install git`
   - Linux: `sudo apt-get install git`
   - Windows: [Download Git](https://git-scm.com/)
   - Verify: `git --version`

3. **Docker** (optional but recommended)
   - Makes Neo4j/Ollama setup automatic
   - [Install Docker Desktop](https://www.docker.com/products/docker-desktop)

4. **System Space**
   - 20GB+ free disk space (for ML models)
   - 8GB+ RAM (16GB recommended)

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/riddhimanrana/orion-research.git
cd orion-research
```

### 2. Create Conda Environment

```bash
conda create -n orion python=3.10
conda activate orion
```

### 3. Install Orion

```bash
pip install -e .
```

### 4. Initialize (First Time Only)

```bash
orion init
```

This will:

- **Detect your hardware** (Apple Silicon / NVIDIA GPU / CPU)
- **Prompt for configuration** (Neo4j, Ollama settings)
- **Download models** (~5-15 minutes)
- **Test connections**

If you don't have Docker, `orion init` will show commands to start Neo4j and Ollama manually.

### 5. Verify Setup

```bash
orion status
```

Should show all services connected ✓

## Usage

### Analyze Videos

```bash
# Standard analysis
orion analyze video.mp4

# Quick mode (faster, less accurate)
orion analyze video.mp4 --fast

# High quality (slower, more accurate)
orion analyze video.mp4 --accurate
```

### Ask Questions

```bash
orion qa

# Examples:
# > What objects appeared in the video?
# > Where was the person located?
# > What events happened?
```

### Check Status

```bash
orion status
```

### Configure Settings

```bash
# View configuration
orion config show

# Update a setting
orion config set neo4j.uri bolt://my-server:7687

# Reset to defaults
orion config reset
```

## Hardware Support

- **Apple Silicon (M1/M2/M3+)** - Auto-detected, uses MLX backend (recommended)
- **NVIDIA GPU (CUDA)** - Auto-detected, uses CUDA backend  
- **CPU** - Works but slow (~30-60s per frame)

Orion automatically selects the best option during `orion init`.

## Troubleshooting

### Neo4j Connection Failed

```bash
# Check if running
docker ps | grep neo4j

# Restart it
docker restart orion-neo4j

# Check logs
docker logs orion-neo4j
```

### Ollama Not Responding

```bash
# Check if running
curl http://localhost:11434/api/tags

# Restart it
docker restart orion-ollama
```

### Models Not Downloading

```bash
# Check internet, then re-run
orion init
```

### Reset Everything

```bash
orion config reset
orion init
```

## Documentation

- **[TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md)** - System design
- **[KNOWLEDGE_GRAPH_SCHEMA.md](KNOWLEDGE_GRAPH_SCHEMA.md)** - Database schema

## Performance

**Processing 1-minute video:**

- Total time: ~110 seconds
- Perception: 60s (detection + descriptions)
- Tracking & Graph: 50s (entity tracking + relationships)

**Optimizations:**

- 15x fewer LLM calls via batching
- 90%+ accuracy on spatial relationships
- Efficient Neo4j operations

## Supported Platforms

- macOS (Intel & Apple Silicon)
- Linux (Ubuntu/Debian)
- Windows (WSL2)

## License

MIT - see [LICENSE](LICENSE) file

## Citation

If you use Orion in research:

```bibtex
@software{orion2025,
  title={Orion: Intelligent Video Understanding Pipeline},
  author={Rana, Riddhiman},
  year={2025}
}
```

---

**Ready to analyze videos?**

```bash
conda activate orion
orion analyze your-video.mp4
```
