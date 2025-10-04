# Orion Research Documentation

All documentation for the "From Moments to Memory" pipeline.

## 📁 Directory Restructure (October 2025)

The codebase was restructured to remove "part1/2/3" prefixes from production code files. These parts were implementation phases, not final system components.

### File Renaming Map

**Main Code Files:**
- `part1_perception_engine.py` → `perception_engine.py`
- `part2_semantic_uplift.py` → `semantic_uplift.py`
- `part3_query_evaluation.py` → `query_evaluation.py`
- `part3_agents.py` → `agents.py`

**Configuration Files:**
- `part1_config.py` → `perception_config.py`
- `part2_config.py` → `semantic_config.py`
- `part3_config.py` → `query_config.py`

**Test Files:**
- `test_part1.py` → `test_perception.py`
- `test_part2.py` → `test_semantic.py`
- `test_part3.py` → `test_query.py`

**Documentation:**
All `.md` files moved from `production/` to `production/docs/`

### Import Changes

All imports changed from:
```python
from production.part1_perception_engine import run_perception_engine
```

To:
```python
from perception_engine import run_perception_engine
```

This works because scripts set up the path correctly:
```python
sys.path.insert(0, str(Path(__file__).parent))
```

## 📚 Documentation Index

### Getting Started
- [Integrated Pipeline Guide](README_INTEGRATED_PIPELINE.md) - Complete pipeline overview
- [Quickstart - Perception](QUICKSTART_PART1.md) - Video perception quickstart
- [Quickstart - Semantic](QUICKSTART_PART2.md) - Knowledge graph quickstart
- [Quickstart - Query](QUICKSTART_PART3.md) - Query engine quickstart

### System Components

**Perception Engine (formerly Part 1)**
- [README_PART1.md](README_PART1.md) - Detailed documentation
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Implementation details

**Semantic Uplift (formerly Part 2)**
- [README_PART2.md](README_PART2.md) - Detailed documentation
- [IMPLEMENTATION_SUMMARY_PART2.md](IMPLEMENTATION_SUMMARY_PART2.md) - Implementation details

**Query & Evaluation (formerly Part 3)**
- [README_PART3.md](README_PART3.md) - Detailed documentation
- [PART3_COMPLETE.md](PART3_COMPLETE.md) - Complete implementation guide
- [PART3_INDEX.md](PART3_INDEX.md) - Navigation guide

### Model Documentation
- [MODELS_GUIDE.md](MODELS_GUIDE.md) - Complete model configuration guide
- [FASTVLM_MODEL_GUIDE.md](FASTVLM_MODEL_GUIDE.md) - FastVLM architecture guide
- [PART1_MODEL_UPDATE.md](PART1_MODEL_UPDATE.md) - YOLO11m + FastVLM update
- [PART1_UPDATE_COMPLETE.md](PART1_UPDATE_COMPLETE.md) - Update summary

### Integration
- [INTEGRATION_COMPLETE.md](INTEGRATION_COMPLETE.md) - Full integration guide

## 🎯 Quick Reference

### Running the Pipeline

```bash
# Full pipeline (perception + knowledge graph)
python production/integrated_pipeline.py ./examples/video1.mp4

# Just perception
python production/perception_engine.py --video ./examples/video1.mp4

# Test individual components
python production/test_perception.py --video ./examples/video1.mp4
python production/test_semantic.py --perception-log output/perception_log.json
python production/test_query.py
```

### Configuration

```python
# Perception engine
from perception_config import apply_config, FAST_CONFIG, BALANCED_CONFIG
apply_config(BALANCED_CONFIG)

# Semantic uplift
from semantic_config import apply_config, ACCURATE_CONFIG
apply_config(ACCURATE_CONFIG)

# Query engine
from query_config import apply_config, HIGH_QUALITY_CONFIG
apply_config(HIGH_QUALITY_CONFIG)
```

## 🏗️ Architecture

```
Video Input
    ↓
[Perception Engine]  ← YOLO11m, FastVLM, FastViT, ResNet50
    ↓
Object Detections + Rich Descriptions
    ↓
[Semantic Uplift]  ← HDBSCAN, Sentence Transformers
    ↓
Knowledge Graph (Neo4j)
    ↓
[Query Engine]  ← Gemini, Graph Queries
    ↓
Answers
```

## 📊 Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Object Detection | YOLO11m (20.1M params) | Detect objects in frames |
| Scene Understanding | FastViT-T8 | Select representative frames |
| Object Embeddings | ResNet50 | Track objects across frames |
| Description Generation | FastVLM-0.5B (custom fine-tuned) | Rich text descriptions |
| Entity Clustering | HDBSCAN | Group similar objects |
| Semantic Embeddings | all-MiniLM-L6-v2 | Event/activity similarity |
| Visual Q&A | Gemini 2.0 Flash | Answer questions |

## 🔧 Dependencies

See `requirements.txt` in the project root for complete list.

Key dependencies:
- PyTorch 2.8.0
- Ultralytics 8.3.0 (YOLO11m)
- Transformers 4.56.2 (FastVLM)
- Neo4j 5.x
- HDBSCAN
- Sentence Transformers

## 📝 Notes

- All documentation references to "Part 1/2/3" remain for historical context
- The part numbers refer to implementation phases, not final system architecture
- Code files now use descriptive names matching their function
