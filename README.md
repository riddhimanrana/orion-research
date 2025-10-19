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

**REQUIRED** - Install these first (in order):

### 1. Conda - Python Environment Manager

Conda isolates Orion from your system Python. Install one:

- **[Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)** (5 min, ~500MB) - Recommended, lightweight
- **[Anaconda](https://www.anaconda.com/download)** (10 min, ~3GB) - Full version with extra tools

After installation, verify:

```bash
conda --version
```
### 2. Docker - Service Containers (Optional but Strongly Recommended)

Automatically starts Neo4j and Ollama. Without Docker, you'll need to start them manually.

- [Install Docker Desktop](https://www.docker.com/products/docker-desktop)
- Supports: macOS, Linux, Windows
- Verify: `docker --version`

**No Docker?** → `orion init` will show you how to install and run Neo4j and Ollama manually (takes ~10 min more).

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/riddhimanrana/orion-research
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

This runs a 3-step initialization:

#### Step 1: Pre-flight Check

- Verifies Neo4j and Ollama are running
- If missing: Shows you exactly how to install and start them
- Suggests Docker (easiest) or manual installation
- Won't proceed until services are ready

#### Step 2: Download Models

- Detects your hardware (Apple Silicon / NVIDIA GPU / CPU)
- Downloads vision models (YOLO11x, FastVLM, CLIP)
- Prepares runtime backend (~5-15 minutes depending on internet)

#### Step 3: Interactive Setup

- Prompts for Neo4j credentials
- Configures Ollama URL
- Tests all connections
- Saves configuration to `~/.orion/config.json`

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