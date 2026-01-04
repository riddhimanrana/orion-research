# Orion Research: AI Coding Agent Instructions

## Project Overview

**Orion** is a memory-centric video understanding system for persistent object tracking, spatial reasoning, and long-term scene memory. Built for Apple Silicon (M-series) with end-to-end perception → memory → query pipeline.

### Architecture Layers (understand dependencies)

1. **Perception Engine** (`orion/perception/engine.py`): Frame-level detection, embedding, tracking
   - Detectors: YOLO11, GroundingDINO, Grounded SAM2
   - Embedders: CLIP, DINO, DINOv3 (for Re-ID)
   - Trackers: EnhancedTracker with adaptive Re-ID thresholds
   - Depth: DepthAnythingV3 for spatial mapping

2. **Scene Graph** (`orion/graph/`): Temporal object relationships
   - Per-frame: `SGNode` (objects) + `SGEdge` (relations: on, near, held_by)
   - Temporal: `VideoSceneGraph` aggregates frame graphs
   - Data flow: perception observations → scene graph → Memgraph export

3. **Memory/Query** (Phase 3-4 roadmap):
   - Re-ID Index: DINOv3 embeddings + cosine similarity
   - Vector Search: semantic object queries
   - Memgraph: persistent graph backend (`orion/graph/memgraph_backend.py`)

## Critical Data Structures

**Perception Flow:**
```
Frame → Observation (detections) 
  → PerceptionEntity (tracked object)
    → PerceptionResult (per-frame output with tracks, scene_graph, entities)
```

**Key Types:**
- `PerceptionConfig`: Detection, embedding, depth, tracking settings (presets: fast/balanced/accurate)
- `DetectionConfig`: Backend (yolo/groundingdino/grounded_sam2), thresholds, model size
- `EmbeddingConfig`: Backend (clip/dino/dinov3), dimension, model name
- `SceneGraph`: nodes (SGNode: id, label, bbox, attributes) + edges (SGEdge: subject→predicate→object)

See [orion/perception/types.py](orion/perception/types.py) and [orion/graph/types.py](orion/graph/types.py).

## Episode & Results Format

**Input:** `data/examples/episodes/<episode_id>/meta.json` + `video.mp4`
**Output:** `results/<episode_id>/tracks.jsonl` + `scene_graph.jsonl` + `memory.json` (Phase 3+)

Example:
```bash
python -c "from orion.config import list_episodes; print(list_episodes())"
python -m orion.cli.run_showcase --episode test_demo --video data/examples/test.mp4
```

Results are JSONL (1 object per line) for streaming. Load with `json.loads(line)` in loops.

## Developer Workflows

### Running Perception
```bash
# End-to-end pipeline (detection → embedding → tracking → scene graph)
python -m orion.cli.run_showcase --episode test_demo --video data/examples/test.mp4

# Reuse tracks, rebuild only scene graph & overlay
python -m orion.cli.run_showcase --episode test_demo --skip-phase1

# Export to Memgraph (requires docker run -p 7687:7687 memgraph/memgraph)
python -m orion.cli.run_showcase --episode test_demo --memgraph --memgraph-host 127.0.0.1

# Quality sweep (perception → graph → optional Gemini validation)
python -m orion.cli.run_quality_sweep --episode test_demo

# Standalone overlay regeneration
python scripts/render_video_overlay.py --video data/examples/test.mp4 --results results/test_demo
```

### Testing & Validation
- `scripts/run_dataset.py`: Batch process videos (json or ActionGenome format)
- `scripts/eval_perception_run.py`: Per-detection metrics (conf, bbox accuracy)
- `scripts/eval_reid.py`: Re-ID consistency across tracks
- `scripts/analyze_scene_graph.py`: Graph structure analysis (nodes, edges, attributes)

### Configuration Patterns
- **Presets**: PerceptionConfig(mode="fast"|"balanced"|"accurate") sets all thresholds automatically
- **Override**: Create PerceptionConfig(mode="balanced"), then `.detection.model = "yolo11x"`
- **Validation**: All `@dataclass` configs have `__post_init__` validation with clear error messages
- **Device**: Auto-detects MPS (Apple Silicon), CUDA, else CPU. Override with `detection.device = "mps"`

## Code Patterns & Conventions

### Detection & Embedding Backend Abstraction
```python
# DetectionConfig.backend ∈ {yolo, groundingdino, grounded_sam2}
# EmbeddingConfig.backend ∈ {clip, dino, dinov3}
# Instantiation happens in PerceptionEngine.__init__() via factory pattern
# Each backend has validate() method called in config __post_init__
```

### Re-ID Adaptive Thresholds
- `EnhancedTracker` uses class-specific `reid_thresholds.py` (tuned empirically)
- Threshold is multiplied by `motion_factor` (faster objects → stricter matching)
- `max_gallery_size` limits appearance history per track (default: 30 frames)
- See `orion/perception/reid_manager.py` for similarity calculation

### Scene Graph Generation
```python
# Per-frame: PerceptionEngine.process_frame() calls:
# 1. Detect: get observations (detections + embeddings)
# 2. Track: match to existing tracks (PerceptionEntity)
# 3. Build SG: call build_research_scene_graph(tracks, depth, spatial_zones)
# 4. Serialize: scene_graph.to_dict() → JSON line in tracks.jsonl
```

### Spatial Relationships (Phase 3 readiness)
- Zones: Define with `shapely` polygons in config (e.g., "kitchen", "living_room")
- Relations: Compute with depth + bbox overlaps + heuristics
- Confidence: Each edge has `confidence` (0-1) reflecting spatial certainty

## Integration Points & External Dependencies

**Model Weights:**
- YOLO: `models/_torch/yolo11*.pt` (auto-download from Ultralytics)
- CLIP/DINO: HuggingFace `transformers` (auto-cached)
- DepthAnythingV3: `models/_torch/depth_anything_v3.pth`
- SAM2: `orion/models/external/segment-anything-2/checkpoints/sam2_hiera_small.pt`

**Backends:**
- Memgraph (Docker): For Phase 4 persistent graph (optional)
- Neo4j (deprecated, legacy CLI refs only)
- OLLAMA: For local LLM reasoning (optional, used in query stage)

**Video I/O:**
- OpenCV: Frame reading + overlay rendering
- MediaPipe: Face/pose detection (optional, separate pipeline)

## Common Pitfalls & Fixes

1. **Missing model weights**: Scripts auto-download, but slow first run. Check `models/` directory.
2. **Device mismatch**: Tensors must match detector device. Use `to(device)` in perception modules.
3. **JSON JSONL confusion**: Results are JSONL (one JSON object per line), not single array.
4. **Threshold tuning**: Don't hardcode values in code. Always use `PerceptionConfig` + `reid_thresholds.py`.
5. **Scene graph edges missing**: Ensure `build_research_scene_graph()` is called with valid tracks + depth. Check spatial zone config.
6. **Memgraph ingest fails**: Verify `orion/graph/memgraph_backend.py` matches schema (Entity, Frame, NEAR edge, etc.).

## Getting Started on New Features

1. **Add detection backend**: Extend `DetectionConfig.backend` enum, implement wrapper in `perception/engine.py`
2. **Add embedding model**: Extend `EmbeddingConfig.backend`, add to `embedder.py` factory
3. **Improve Re-ID**: Tune `reid_thresholds.py` per object class, test with `eval_reid.py`
4. **Add spatial relations**: Extend predicate types in `SGEdge.predicate`, implement in `build_research_scene_graph()`
5. **Phase 4 memory**: Use `MemgraphBackend.add_entity_observation()` in `PerceptionEngine.process_frame()`

## References

- **Detailed schemas**: [docs/episodes.md](docs/episodes.md), [docs/results_schema.md](docs/results_schema.md)
- **Phase status**: [docs/PHASE_4_PLAN.md](docs/PHASE_4_PLAN.md)
- **Config reference**: [orion/perception/config.py](orion/perception/config.py) (150+ lines, all documented)
- **CLI entry**: [orion/cli/main.py](orion/cli/main.py)
- **Core engine**: [orion/perception/engine.py](orion/perception/engine.py) (~1400 lines, main logic)

**Apple Silicon note:** Project built & tested on M-series. MPS (Metal Performance Shaders) is default device. No special setup needed beyond standard PyTorch with MPS support.
