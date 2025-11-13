### Copilot Instructions for Orion Research Repository

**Top-Level Rule:** Do not create or suggest markdown files unless explicitly requested.

---

## Architecture Overview

**Orion** is a multi-phase egocentric video understanding system that processes video through perception → semantic → knowledge graph stages.

### Pipeline Phases (5-Phase Architecture)
1. **Phase 1: Perception** (`orion/perception/`) - Object detection (YOLO11), depth estimation (MiDaS/DepthAnything), CLIP embeddings, 3D backprojection
2. **Phase 2: Semantic** (`orion/semantic/`) - Entity tracking, state change detection, temporal windowing, causal inference, event composition  
3. **Phase 3: SLAM** (`orion/slam/`) - Visual odometry, camera pose estimation, loop closure, pose graph optimization
4. **Phase 4: Tracking** - Temporal deduplication across frames (reduces ~130 detections → 6 entities)
5. **Phase 5: Re-ID** - Semantic deduplication using CLIP embeddings with cross-view merging

### Core Entry Points
- **`orion/perception/engine.py::PerceptionEngine`** - Orchestrates Phase 1 (detection → embedding → tracking → description)
- **`orion/semantic/engine.py::SemanticEngine`** - Orchestrates Phase 2 (entity consolidation → state changes → events → causality)
- **`orion/slam/slam_engine.py::SLAMEngine`** - Feature-based visual odometry with ICP fallback for low-texture scenes
- **`orion/pipeline.py::VideoPipeline`** - Full end-to-end orchestrator (perception → semantic → graph ingestion)
- **`orion/managers/model_manager.py::ModelManager`** - Singleton for lazy-loading YOLO, CLIP, FastVLM (shared across modules)

### Data Flow Pattern
```
Video Frame → PerceptionEngine → PerceptionResult (observations, entities)
            ↓
    SemanticEngine → SemanticResult (entities, state_changes, events, causal_links)
            ↓
    GraphBuilder → Neo4j/Memgraph (knowledge graph storage)
```

**Key Types**: `orion/perception/types.py` defines canonical contracts (`Observation`, `PerceptionEntity`, `PerceptionResult`). `orion/semantic/types.py` defines `SemanticEntity`, `StateChange`, `Event`, `SemanticResult`.

---

## Critical Development Workflows

### Running Tests
- **Unit tests**: `pytest tests/` or `make test` (use `--device cpu|mps|cuda` for hardware selection)
- **Pipeline tests**: `python scripts/test_pipeline.py` (validates 5-phase reduction chain)
- **Component tests**: `python scripts/test_components_simple.py` (YOLO, Depth, CLIP verification)
- **Integration**: `pytest tests/integration/` (full perception + semantic flow)

### Building & Installation
- **Dev install**: `make install-dev` (includes pytest, black, mypy, profiling tools)
- **Research deps**: `make install-research` (adds Jupyter, matplotlib, benchmarking tools)
- **Production**: `make install` (minimal dependencies only)

### CLI Usage (`python -m orion` or `orion`)
- **Analyze video**: `orion analyze --video <path> --output-dir results/`
- **With Q&A**: `orion analyze-with-qa --video <path> --questions "What did I hold?"`
- **Run unified pipeline**: `orion run --video <path> --rerun` (launches 3D Rerun visualizer)
- **Config management**: `orion config show|set|reset` (stored in `~/.orion/config.json`)

### Performance Profiling
- **Fast mode**: YOLO11n, skip 40 frames, ~8-10 FPS overall (2-3GB memory)
- **Accurate mode**: YOLO11m, skip 10 frames, scene graphs enabled, ~1-2 FPS (4-5GB memory)
- **Benchmark**: `make benchmark-ego4d` or `make profile` (outputs to `results/`)

---

## Project-Specific Conventions

### Code Organization
- **Engines over functions**: Use `PerceptionEngine`, `SemanticEngine`, `SLAMEngine` pattern for stateful orchestration
- **Lazy loading**: Models loaded on-demand via `ModelManager.get_instance()` (avoids redundant GPU memory)
- **Config dataclasses**: All configs use `@dataclass` with `default_factory` (see `orion/perception/config.py`, `orion/semantic/config.py`)
- **Type safety**: Always use type hints (enforced by `mypy`). Import `TYPE_CHECKING` for circular dependency resolution.

### Backend Selection
- **Runtime**: `auto` (default, uses MPS on macOS, CUDA on Linux/Windows), `torch` (explicit PyTorch)
- **Embedding backend**: `auto`, `ollama`, `sentence-transformer` (configurable via `orion config set embedding.backend`)
- **YOLO models**: `yolo11n` (fastest), `yolo11s`, `yolo11m` (balanced), `yolo11x` (most accurate)

### Depth & SLAM Integration
- **Depth preprocessing**: Always use `preprocess_depth_map()` from `orion/slam/depth_utils.py` (bilateral filter + outlier removal)
- **Scale recovery**: Use `ScaleEstimator` from `orion/perception/scale_estimator.py` with object size priors (keyboards=350mm, laptops=400mm, etc.)
- **Loop closure**: `LoopClosureDetector` in `orion/slam/loop_closure.py` uses VLAD embeddings for scene recognition

### Testing Patterns
- **Mock entities**: Use `test_phase1_mock_entities.py` pattern for lightweight entity tests
- **Fixtures**: Define reusable test data in `tests/fixtures/` (e.g., sample frames, depth maps)
- **Device flags**: All tests accept `--device` for hardware-specific runs (default CPU for CI)

### Visualization
- **Rerun.io**: Use `UnifiedRerunVisualizer` from `orion/perception/rerun_visualizer.py` for 3D interactive visualizations
- **Matplotlib**: For static plots (2D bounding boxes, depth heatmaps) in research scripts

---

## Integration Points

### Knowledge Graph (Neo4j → Memgraph Migration)
- **Current**: `GraphBuilder` in `orion/graph/builder.py` is a stub (Neo4j deprecated, Memgraph integration pending)
- **Future**: Use Memgraph backend via `pymgclient` (install separately, not in default deps)
- **Export**: `export_memgraph_structure()` in pipelines generates graph JSON for manual import

### External Dependencies
- **Detectron2**: Optional (install via `bash scripts/install_detectron2_macos.sh`), used for hybrid detection fallback
- **OSNet Re-ID**: Optional (`pip install torchreid`), advanced appearance-based re-identification
- **Ollama**: Required for Q&A (`qa.model = "gemma3:4b"` in config), not managed by ModelManager

### MLX-VLM Submodule
- **Location**: `mlx-vlm/` (Apple MLX-optimized vision-language models)
- **Usage**: `ModelManager.fastvlm` wraps either MLX or PyTorch backend based on device
- **Installation**: Included in dependencies as `mlx-vlm @ file:./mlx-vlm`

---

## Common Pitfalls to Avoid

1. **Model Manager Singleton**: Never instantiate models directly—always use `ModelManager.get_instance()` to prevent duplicate GPU allocations
2. **Depth Units**: Raw depth from MiDaS is in meters (0-10 range). Convert to mm (`* 1000`) for `UnifiedFrame` and SLAM
3. **CLIP Embeddings**: Must be L2-normalized before similarity computation (`embedding /= np.linalg.norm(embedding)`)
4. **Frame Skipping**: Use `--skip N` in CLI or `target_fps` in configs—processing every frame kills performance
5. **Type Imports**: Use `from typing import TYPE_CHECKING` for engine imports in type signatures to avoid circular deps
6. **Config Persistence**: Changes via `orion config set` are persisted to `~/.orion/config.json` (base64-encoded passwords)

---

## Debugging & Research Tools

- **Debug scripts**: `scripts/debug_research_slam.py` for SLAM visualization with trajectory plots
- **Profiling**: `python scripts/profile_performance.py --device auto --output results/profile.json`
- **Ablation studies**: `make ablation` runs 2D vs 3D CIS, hands, occlusion studies (outputs to `results/ablation_results.json`)
- **WACV Demo**: `make demo` runs conference demo on sample video with HTML output

---

## Key Files Reference

- **`orion/settings.py`**: User config management with `OrionSettings` dataclass (Neo4j, runtime, QA model)
- **`orion/__main__.py`**: Module entry point (`python -m orion` delegates to `orion.cli.main`)
- **`Makefile`**: Centralized commands (test, lint, format, profile, benchmark)
- **`QUICKSTART.md`**: Performance benchmarks, model loading times, command cheat sheet
- **`pyproject.toml`**: Dependency specs (torch 2.6.0, transformers ≥4.47.1, ultralytics 8.3.217)