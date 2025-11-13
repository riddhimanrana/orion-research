# Orion Archive

**Date Archived**: November 12, 2025  
**Reason**: Codebase cleanup and consolidation

This directory contains legacy, experimental, and duplicate code that was archived during the major cleanup initiative. These files are preserved for reference but are no longer part of the active codebase.

---

## What's Archived

### Perception Modules (~20 files)

#### Trackers (`perception/trackers/`)
- `tracking.py` — EntityTracker3D (Phase 2, replaced by EnhancedTracker)
- `enhanced_tracker_adapter.py` — Adapter for old EntityTracker3D
- `tracker_reid.py` — MultiHypothesisTracker (experimental)
- `temporal_tracker.py` — TemporalTracker variant
- `object_tracker.py` — Simple 2D+3D tracker
- `pipeline_adapter.py` — Legacy pipeline adapters

**Why**: `EnhancedTracker` (StrongSORT-inspired) is now the single tracker implementation with 3D Kalman filtering, appearance-based Re-ID, and camera motion compensation.

#### Re-ID Modules (`perception/reid/`)
- `advanced_reid.py` — Advanced Re-ID with gallery
- `appearance_reid.py` — Appearance-based Re-ID
- `clip_reid.py` — CLIP-based Re-ID
- `fastvlm_reid.py` — FastVLM Re-ID
- `geometric_reid.py` — Geometric Re-ID
- `appearance_extractor.py` — Feature extraction
- `reid_matcher.py` — Re-ID matching logic

**Why**: `EnhancedTracker` has built-in appearance embeddings (CLIP/FastVLM) with EMA smoothing and gallery management. Future Re-ID improvements will use Faiss index for long-term memory.

#### Spatial/SLAM Fusion (`perception/spatial/`)
- `slam_fusion.py` — SLAM fusion experiments
- `reconstruction_3d.py` — 3D reconstruction
- `spatial_map_builder.py` — Spatial mapping
- `semantic_scale.py` — Semantic scale estimation
- `spatial_analyzer.py` — Spatial analysis utilities

**Why**: Spatial reasoning consolidated in `SLAMEngine` and `Perception3DEngine`. World coordinates handled by SLAM pose transforms.

#### Visualization (`perception/visualization/`)
- `visualization.py` — Old matplotlib/CV2 visualization

**Why**: Replaced by `rerun_visualizer.py` (Rerun.io SDK) for interactive 3D visualization.

#### Detection (`perception/detection/`)
- `advanced_detection.py` — Detectron2-based detection (experimental)

**Why**: Using YOLO11 in `observer.py` as primary detector.

#### Depth (`perception/depth/`)
- `depth_anything.py` — DepthAnything wrapper

**Why**: Core `depth.py` supports MiDaS and ZoeDepth. DepthAnything can be re-integrated if needed.

#### Other
- `unified_pipeline.py` — Old unified pipeline
- `corrector.py` — Correction logic

---

### Semantic Modules (~14 files)

#### Captioning (`semantic/captioning/`)
- `rich_captioning.py` — Rich caption generation
- `smart_caption_prioritizer.py` — Caption prioritization
- `strategic_captioner.py` — Strategic captioning
- `temporal_description_generator.py` — Temporal descriptions

**Why**: Core captioning now handled by `EntityDescriber` (FastVLM). These modules were experimental NLG variations.

#### Scene Understanding (`semantic/scene/`)
- `scene_assembler.py` — Scene assembly
- `scene_classifier.py` — Scene classification
- `scene_graph.py` — In-memory scene graphs
- `scene_understanding.py` — Scene understanding

**Why**: Scene graphs now stored in **Memgraph** (external graph database). Scene understanding consolidated in `SemanticEngine`.

#### Experimental (`semantic/experimental/`)
- `causal_scorer.py` — Causal scoring (duplicate)
- `cis_scorer_3d.py` — 3D CIS scoring
- `enhanced_spatial_reasoning.py` — Enhanced spatial reasoning
- `query_intelligence.py` — Query intelligence
- `spatial_nlg.py` — Spatial NLG
- `entity_tracker.py` — Entity tracking (duplicate)

**Why**: Core causal inference in `causal.py`, spatial relationships in `zone_manager.py` and `spatial_utils.py`. These were experimental extensions.

---

### SLAM Modules (~4 files)

#### Odometry (`slam/odometry/`)
- `depth_odometry.py` — Depth-only odometry
- `semantic_slam.py` — Semantic SLAM (experimental)

**Why**: Using `hybrid_odometry.py` for fused visual-depth odometry in `SLAMEngine`.

#### Projection (`slam/projection/`)
- `projection_3d.py` — 3D projection utilities
- `world_coordinate_tracker.py` — World coordinate tracking

**Why**: 3D backprojection handled by `camera_intrinsics.py` in perception. World coordinates from SLAM pose transforms.

---

## Active Codebase Structure

### Perception (15 files - kept)
- `engine.py` — PerceptionEngine orchestrator
- `config.py`, `types.py` — Configuration and types
- `observer.py` — YOLO detection
- `embedder.py` — CLIP embeddings
- `describer.py` — FastVLM descriptions
- `enhanced_tracker.py` — Unified tracker (3D+appearance)
- `tracker_base.py` — Tracker protocol
- `tracker.py` — EntityTracker (observation clustering)
- `perception_3d.py` — 3D perception (depth + hands + occlusion)
- `depth.py` — Depth estimation
- `camera_intrinsics.py` — 3D backprojection
- `scale_estimator.py` — Scale recovery
- `rerun_visualizer.py` — Rerun.io visualization
- `unified_frame.py` — UnifiedFrame dataclass
- `occlusion.py`, `hand_tracking.py` — Occlusion/hand detection

### Semantic (9 files - kept)
- `engine.py` — SemanticEngine orchestrator
- `config.py`, `types.py` — Configuration and types
- `state_detector.py` — State change detection
- `event_composer.py` — Event composition
- `causal.py` — Causal inference
- `temporal_windows.py` — Temporal windowing
- `zone_manager.py` — Zone management
- `spatial_utils.py` — Spatial relationships

### SLAM (7 files - kept)
- `slam_engine.py` — SLAMEngine (visual odometry + loop closure)
- `loop_closure.py` — Loop closure detection
- `pose_graph.py` — Pose graph optimization
- `depth_utils.py` — Depth preprocessing
- `hybrid_odometry.py` — Visual-depth fusion
- `depth_consistency.py` — Depth consistency checking
- `multi_frame_depth_fusion.py` — Multi-frame fusion

---

## Can I Use Archived Code?

**Yes!** Archived code is preserved for:
1. **Reference**: Understanding past approaches
2. **Recovery**: Re-integrating useful features
3. **Research**: Experimental modules for papers/ablations

To recover archived code:
```bash
# Copy back to active codebase
cp orion/_archive/perception/reid/clip_reid.py orion/perception/

# Or reference directly
from orion._archive.perception.reid.clip_reid import CLIPReID
```

**Note**: Archived code may have broken imports after the cleanup. Update imports if recovering.

---

## Migration Notes

### Tracker Migration
- **Old**: `EntityTracker3D`, `TemporalTracker`, `ObjectTracker`, etc.
- **New**: `EnhancedTracker` (single unified implementation)
- **Adapter**: See `enhanced_tracker_adapter.py` (archived) for migration examples

### Re-ID Migration
- **Old**: Separate `clip_reid.py`, `fastvlm_reid.py`, etc.
- **New**: Built into `EnhancedTracker` with CLIP/FastVLM embeddings
- **Future**: Faiss index for long-term memory (see `TRACKER_INTEGRATION.md`)

### Scene Graph Migration
- **Old**: In-memory `scene_graph.py`
- **New**: **Memgraph** external graph database
- **Schema**: See `CLEANUP_AUDIT.md` for Cypher schema examples

### Depth Migration
- **Old**: Multiple depth wrappers (`depth_anything.py`, etc.)
- **New**: Single `depth.py` with MiDaS/ZoeDepth support

---

## Statistics

- **Total files archived**: ~38 files
- **Codebase reduction**: ~50%
- **Perception**: 20 → 15 files (25% reduction)
- **Semantic**: 24 → 9 files (62% reduction)
- **SLAM**: 11 → 7 files (36% reduction)

---

## Questions?

See main documentation:
- `CLEANUP_AUDIT.md` — Full audit and cleanup plan
- `TRACKER_INTEGRATION.md` — Tracker integration guide
- `QUICKSTART.md` — Performance benchmarks and usage

**Last Updated**: November 12, 2025
