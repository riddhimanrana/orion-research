# Copilot Instructions for Orion Research Repository

## Project Overview

**Orion** is a semantic video understanding pipeline that transforms raw egocentric/third-person video into causally-aware knowledge graphs. It bridges low-level perception (YOLO, CLIP) with high-level reasoning (LLM-based event composition).

**Tech Stack:** Python 3.10+, PyTorch, MLX, Ultralytics YOLO11, OpenAI CLIP, Neo4j, FastAPI, Rich CLI

---

## Architecture: Three-Phase Pipeline

### Phase 1: Perception (`orion/perception/`)
- **Detection**: YOLO11x object detection (80 COCO classes)
- **Embedding**: CLIP ViT-L/14 visual embeddings (512-2048 dim)
- **Clustering**: HDBSCAN entity tracking across frames
- **Description**: FastVLM/LLaVA async description generation
- **Entry Point**: `PerceptionEngine.process_video()` → `PerceptionResult`

### Phase 2: Semantic Uplift (`orion/semantic/`)
- **Entity Tracking**: Consolidate observations into semantic entities
- **State Changes**: Detect appearance/location changes via embedding similarity
- **Causal Inference**: Score causal relationships using CIS formula (learned weights)
- **Event Composition**: LLM-generated natural language events
- **Entry Point**: `SemanticEngine.process()` → `SemanticResult`

### Phase 3: Knowledge Graph (`orion/graph/`)
- **Nodes**: Entity, Scene, Event, Zone, Location
- **Relationships**: CAUSES, APPEARS_IN, NEAR, PART_OF, etc.
- **Storage**: Neo4j with vector indices (HNSW)
- **Entry Point**: `GraphBuilder.build()` → Neo4j database

**Full Pipeline**: `VideoPipeline.process_video()` orchestrates all phases.

---

## Critical Code Patterns

### 1. Configuration System (Three-Tier)
```python
# User Settings (credentials, preferences) → ~/.orion/config.json
from orion.settings import OrionSettings
settings = OrionSettings.load()  # Auto-creates defaults

# Runtime Config (detection params, thresholds)
from orion.perception.config import get_balanced_config  # or fast/accurate
config = get_balanced_config()

# Never hardcode credentials - use environment variables:
# ORION_NEO4J_PASSWORD, ORION_OLLAMA_URL
```

**Config Presets**: Always use `get_fast_config()`, `get_balanced_config()`, or `get_accurate_config()` rather than manually constructing configs. These are tested and validated.

### 2. Type System (Pydantic-like Dataclasses)
```python
# Core types in orion/{perception,semantic}/types.py
from orion.perception import Observation, PerceptionEntity, PerceptionResult
from orion.semantic import StateChange, CausalLink, Event, SemanticResult

# All results use dataclasses with type hints - ALWAYS preserve this pattern
# Bad: result = {"entities": [...]}
# Good: result = PerceptionResult(entities=[...], total_frames=100)
```

### 3. Logging (Rich Console)
```python
import logging
logger = logging.getLogger(__name__)

# Structured logging for pipeline stages:
logger.info("="*80)
logger.info("PERCEPTION ENGINE - PHASE 1")
logger.info("="*80)
logger.info(f"  Total detections: {count}")

# Rich console for CLI output - see orion/cli.py for patterns
```

### 4. Error Handling (Graceful Degradation)
```python
# Preferred pattern: fallback to simpler methods
try:
    descriptions = vlm_describer.generate(entities)
except VLMError:
    logger.warning("VLM unavailable, using YOLO classes only")
    descriptions = [e.object_class for e in entities]

# Neo4j connections should be optional:
if not self.neo4j_manager:
    logger.warning("Neo4j unavailable, skipping graph ingestion")
    return {"status": "skipped"}
```

### 5. CIS (Causal Influence Score) Formula
```python
# Located in orion/semantic/causal_scorer.py
# CIS = w_temporal·f_temporal + w_spatial·f_spatial + w_motion·f_motion + w_semantic·f_semantic
# Weights learned via Bayesian optimization (see docs/CIS_TRAINING_GUIDE.md)

# Load optimized weights from hpo_results/cis_weights.json
# Default weights: [0.4, 0.3, 0.2, 0.1] if file missing
```

---

## Developer Workflows

### Running the Pipeline
```bash
# Full pipeline (CLI)
orion analyze data/examples/video.mp4 --mode balanced

# Phase-by-phase inspection
orion analyze video.mp4 --inspect perception  # Stop after Phase 1
orion analyze video.mp4 --inspect semantic    # Stop after Phase 2

# Programmatic usage
from orion import VideoPipeline, PipelineConfig
pipeline = VideoPipeline(PipelineConfig())
results = pipeline.process_video("video.mp4", scene_id="test")
```

### Testing Strategy
```bash
# Unit tests: Component-level
pytest tests/unit/test_causal_inference.py

# Integration tests: End-to-end
pytest tests/test_quickstart.py

# Contextual tests (CIS validation)
python -m pytest tests/ -k "cis" --skip-correction

# Run tasks defined in .vscode/tasks.json
# Task: "Run contextual tests" → Uses scripts/test_contextual_understanding.py
```

**Test Patterns**:
- Mock YOLO/CLIP for fast tests: `@patch('orion.perception.observer.YOLO')`
- Use `data/examples/video.mp4` for integration tests
- CIS tests compare against `data/benchmarks/ground_truth/` annotations

### Debugging
```bash
# Debug image crops (visual inspection)
python scripts/debug_image_crops.py --video video.mp4

# Causal diagnostics (detailed CIS breakdown)
python scripts/run_causal_diagnostics.py --video video.mp4

# Neo4j query browser
# Open http://localhost:7474 after running pipeline
```

---

## Project-Specific Conventions

### 1. Class Correction (Semantic Validation)
**File**: `orion/perception/class_correction.py`  
**Problem**: YOLO misclassifies objects (e.g., "tire" → "suitcase")  
**Solution**: Semantic validation with part-of detection

```python
# Pattern: Validate corrections before applying
if corrector.should_correct(description, yolo_class):
    new_class = corrector.semantic_class_match(description)
    if corrector.validate_correction(new_class, description):
        return new_class  # Apply correction
return yolo_class  # Keep original
```

**Key Insight**: Part-of relationships ("car tire" contains "car") are NOT corrections. Use `is_part_of_relationship()` to filter.

### 2. Async Perception (Producer-Consumer)
**File**: `orion/perception/engine.py`  
**Fast Loop (30-60 FPS)**: Detection + Embedding  
**Slow Loop (1-5 FPS)**: VLM Description (async queue)

```python
# Describe ONCE per entity (not every detection)
# This is why PerceptionResult has:
# - raw_observations: List[Observation]  # All detections
# - entities: List[PerceptionEntity]     # Unique entities with descriptions
```

### 3. Neo4j Schema Conventions
```cypher
// Node labels: Entity, Scene, Event, Zone, Location
// Relationship types: APPEARS_IN, CAUSES, NEAR, PART_OF, etc.

// Always create constraints before ingesting:
CREATE CONSTRAINT entity_id_unique FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// Use parameterized queries (never string formatting):
session.run("MATCH (e:Entity {id: $id}) RETURN e", id=entity_id)
```

### 4. MLX vs PyTorch Backend
```python
# Backend auto-selected based on platform (orion/backends/)
# MLX: Apple Silicon (M1/M2/M3) - faster inference
# PyTorch: CUDA/CPU - broader compatibility

# User never specifies backend directly - use config presets
# orion.settings.OrionSettings.runtime_backend = "auto"  # Default
```

### 5. Documentation Style
- **Architecture**: See `docs/SYSTEM_ARCHITECTURE_2025.md` for comprehensive overview
- **Training**: See `docs/CIS_TRAINING_GUIDE.md` for CIS optimization
- **Code Comments**: Use docstrings with Args/Returns/Raises sections
- **Top-Level Rule**: Do NOT create README files unless explicitly requested

---

## Common Pitfalls

1. **Don't hardcode COCO class names** - Use `ObjectClass` enum or dynamic mapping
2. **Don't modify PerceptionResult after creation** - Immutable dataclasses
3. **Don't skip config validation** - Always call `config.validate()` before use
4. **Don't use `python -c` for snippets** - Use `mcp_pylance_mcp_s_pylanceRunCodeSnippet` tool
5. **Don't create tasks.json from scratch** - Use `create_and_run_task` tool

---

## Key Files Reference

| Component | Entry Point | Config |
|-----------|-------------|--------|
| Perception | `orion/perception/engine.py` | `orion/perception/config.py` |
| Semantic | `orion/semantic/engine.py` | `orion/semantic/config.py` |
| Graph | `orion/graph/builder.py` | `orion/graph/config.py` |
| Pipeline | `orion/pipeline.py` | `orion/pipeline.py` (PipelineConfig) |
| CLI | `orion/cli.py` | `orion/settings.py` (OrionSettings) |
| Types | `orion/{perception,semantic}/types.py` | - |

---

## When Contributing Code

1. **Add type hints**: All functions must have parameter and return types
2. **Use dataclasses**: For structured data (not dicts)
3. **Log pipeline stages**: Use `logger.info("="*80)` for phase boundaries
4. **Handle missing models**: Graceful degradation with warnings
5. **Test with presets**: Validate against fast/balanced/accurate configs
6. **Document CIS changes**: Update `docs/CIS_TRAINING_GUIDE.md` if modifying causal scoring

---

## External Dependencies

- **VSGR Benchmark**: Video Scene Graph dataset (aspirational - not yet integrated)
- **TAO-Amodal**: Object tracking annotations (used for CIS training)
- **Ollama**: Local LLM inference (Gemma, LLaMA) - must be running
- **Neo4j**: Graph database - docker run recommended

---

**Last Updated**: October 25, 2025  
**Authors**: Riddhiman Rana, Aryav Semwal, Yogesh Atluru