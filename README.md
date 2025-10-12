# Orion - Video Analysis Pipeline

Transform videos into queryable knowledge graphs with AI-powered analysis.

```
Video → Object Detection → Descriptions → Knowledge Graph → Q&A
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/riddhimanrana/orion-research
cd orion-research

# Install conda (if needed)
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all

# Create environment
conda create --name orion python=3.10
conda activate orion
pip install -e .


# Setup Ollama
brew install ollama
ollama pull gemma3:4b
ollama pull embeddinggemma

# Install Neo4j (optional - for knowledge graph storage)
# Download from: https://neo4j.com/download/
# Or use Docker: docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/orion123 neo4j
```

### 2. Basic Usage

```bash
# Analyze a video
./orion analyze video.mp4

# With interactive Q&A
./orion analyze video.mp4 -i

# Fast mode (for long videos)
./orion analyze video.mp4 --fast

# Accurate mode (for short clips)
./orion analyze video.mp4 --accurate

# Q&A only (after processing)
./orion qa
```

## 📚 What It Does

Orion uses **5 AI models** to understand your video:

| Model | Job | Example |
|-------|-----|---------|
| **YOLO11m** | Detect objects | Bounding boxes, class labels |
| **FastVLM-0.5B** | Describe scenes | "A red car parked near a building" |
| **ResNet50** | Track objects | Visual similarity scores |
| **EmbeddingGemma** | Understand text | Semantic similarity vectors |
| **Gemma3:4b** | Answer questions | "The lights turned on at 10 seconds" |

### The Pipeline

```
┌─────────────────────────────────────┐
│  Part 1: Visual Perception          │
│  - Detect objects (YOLO)            │
│  - Generate descriptions (FastVLM)  │
│  - Create embeddings (ResNet50)     │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Part 2: Knowledge Graph            │
│  - Track entities                   │
│  - Detect state changes             │
│  - Find relationships               │
│  - Store in Neo4j                   │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Part 3: Interactive Q&A            │
│  - Ask questions in natural language│
│  - Get intelligent answers          │
└─────────────────────────────────────┘
```

## Command Reference

### Analyze Command

```bash
# Basic
./orion analyze video.mp4

# Options
./orion analyze video.mp4 \
  --fast                    # Fast processing mode
  --accurate                # Accurate mode (slower)
  -i, --interactive         # Start Q&A after processing
  --skip-perception         # Skip Part 1 (use existing data)
  --skip-graph              # Skip Part 2 (only detect objects)
  --keep-db                 # Keep existing Neo4j data
  -o OUTPUT                 # Output directory
  --neo4j-password PWD      # Neo4j password (default: orion123)
  -v, --verbose             # Debug logging
```

### Q&A Command

```bash
# Start Q&A session
./orion qa

# Use different model
./orion qa --model llama3.2:3b

# Custom Neo4j password
./orion qa --neo4j-password mypassword
```

### Info Commands

```bash
./orion models              # Show model information
./orion modes               # Show processing modes
```

## 🎛️ Processing Modes

| Mode | FPS | Descriptions | Best For |
|------|-----|--------------|----------|
| **Fast** | 3 | Every 10th object | Long videos, testing |
| **Balanced** | 5 | Every 5th object | General use ⭐ |
| **Accurate** | 10 | Every 2nd object | Short clips, detail |

## 📊 Example Output

```bash
$ ./orion analyze gaming.mp4 -i

   ___  ____  ____  ___  _   _ 
  / _ \|  _ \|  _ \|_ _|/ \ | |
 | | | | |_) | |_) || |/ _ \| |
 | |_| |  _ <|  _ < | / ___ \ |
  \___/|_| \_\_| \_\___/   \_\_|
  
Video → Knowledge Graph → Q&A

Analyzing: gaming.mp4
Mode: balanced

✓ Neo4j cleared

═════════════════════════════════════════
PART 1: VISUAL PERCEPTION ENGINE
═════════════════════════════════════════

┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Metric           ┃ Value         ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Objects Detected │ 436           │
│ Processing Time  │ 180.3s        │
└──────────────────┴───────────────┘

═════════════════════════════════════════
PART 2: SEMANTIC KNOWLEDGE GRAPH
═════════════════════════════════════════

┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Metric           ┃ Value         ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Entities Tracked │ 12            │
│ State Changes    │ 28            │
│ Cypher Queries   │ 60            │
└──────────────────┴───────────────┘

✓ Pipeline complete!

Starting Q&A mode...

You: What happened in the video?
Orion: The video shows a person gaming. A black gaming keyboard with 
red LED lights is visible, along with a mouse and monitor. The keyboard 
lights turned on around 10 seconds into the video, and the person began 
typing shortly after.

You: exit
```

## 🛠️ Configuration

### Neo4j Setup

1. **Neo4j Desktop**: Download from https://neo4j.com/download/
2. **Docker**:
   ```bash
   docker run -d \
     --name neo4j \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/orion123 \
     neo4j:latest
   ```
3. **Access**: http://localhost:7474
4. **Default credentials**: `neo4j / orion123`

### Model Customization

Edit configuration files in `production/`:
- `perception_config.py` - Part 1 settings (FPS, workers, thresholds)
- `semantic_config.py` - Part 2 settings (clustering, state detection)
- `query_config.py` - Part 3 settings (Q&A models)

## 📁 Project Structure

```
orion-research/
├── orion                    # CLI entry point ⭐
├── production/              # Production code
│   ├── perception_engine.py # Part 1: Visual perception
│   ├── semantic_uplift.py   # Part 2: Knowledge graph
│   ├── video_qa.py          # Part 3: Q&A system
│   ├── neo4j_manager.py     # Database utilities
│   ├── embedding_model.py   # Unified embeddings
│   ├── fastvlm_mlx_wrapper.py # FastVLM interface
│   └── *_config.py          # Configuration files
├── data/                    # Output directory
├── models/                  # Model weights
└── README.md               # This file
```

## 🧠 Model Details

### Visual Models

- **YOLO11m** (20MB): Real-time object detection, 80 COCO classes
- **FastVLM-0.5B** (600MB): Vision-language model optimized for Apple MLX
- **ResNet50** (100MB): Visual embeddings for object tracking

### Language Models

- **EmbeddingGemma** (622MB): 768-dim semantic embeddings, 100+ languages
- **Gemma3:4b** (3.3GB): Conversational Q&A, runs via Ollama

**Total**: ~5.0GB of models, all running locally!

## 🔧 Troubleshooting

### "No module named 'timm'"
```bash
pip install timm transformers
```

### "Ollama not found"
```bash
brew install ollama
ollama pull gemma2:2b
ollama pull embeddinggemma
```

### "Neo4j connection failed"
- Verify Neo4j is running: http://localhost:7474
- Check password (default: `orion123`)
- Or skip Neo4j: `./orion analyze video.mp4 --skip-graph`

### "FastVLM not generating descriptions"
- Ensure models are in `models/fastvlm-0.5b-captions/`
- Check spawn mode is enabled (automatic on macOS)

## 💡 Tips

- **Long videos**: Use `--fast` mode
- **Short clips**: Use `--accurate` mode  
- **Testing**: Skip perception with `--skip-perception` to reuse data
- **Storage**: Clear Neo4j between runs (default) or use `--keep-db`
- **Debugging**: Add `-v` flag for detailed logs

## 📖 How It Works

### Simple Explanation

1. **YOLO** finds objects in each frame
2. **FastVLM** describes interesting objects
3. **ResNet50** creates visual fingerprints for tracking
4. **EmbeddingGemma** understands text meaning
5. Objects are clustered into **entities**
6. **State changes** are detected by comparing descriptions
7. **Relationships** between entities are inferred
8. Everything is stored in a **Neo4j graph database**
9. **Gemma3** answers your questions using the graph

### Why Multiple Models?

Each model has one specialized job:
- YOLO is fast at finding but can't describe
- FastVLM describes well but too slow for every frame
- ResNet50 is fast for visual comparison but doesn't understand text
- EmbeddingGemma understands meaning but can't see images
- Gemma3 answers questions but needs data from others

Together they create a complete understanding!

## 🚀 Advanced Usage

### Python API

```python
from production.run_pipeline import run_pipeline

results = run_pipeline(
    video_path='video.mp4',
    output_dir='data/output',
    part1_config='balanced',
    part2_config='balanced',
    neo4j_password='orion123'
)

print(f"Detected {results['part1']['num_objects']} objects")
print(f"Found {results['part2']['num_entities']} entities")
```

### Custom Configurations

```python
# In perception_config.py
CUSTOM_CONFIG = {
    'TARGET_FPS': 7,
    'NUM_WORKERS': 4,
    'DESCRIPTION_FREQUENCY': 3,
    # ... more settings
}
```

## 📝 License

See LICENSE file for details.

## 🙏 Acknowledgements

- **FastVLM**: Apple MLX team
- **YOLO**: Ultralytics
- **Neo4j**: Graph database platform
- **Ollama**: Local LLM runtime
- **Google**: EmbeddingGemma and Gemma3 models

---

**Made with ❤️ by the Orion team**

For issues: Check logs in `data/testing/pipeline.log`
