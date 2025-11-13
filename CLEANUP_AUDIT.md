# Orion Codebase Cleanup Audit

**Date**: November 12, 2025  
**Goal**: Archive legacy code, consolidate duplicates, establish clean foundation

---

## Perception Folder Analysis

### 🔴 DUPLICATE TRACKERS (Archive All Except `enhanced_tracker.py` + `tracker_base.py`)

**Keep**:
- ✅ `enhanced_tracker.py` — StrongSORT-inspired, 3D+appearance, integrated with engine
- ✅ `tracker_base.py` — Protocol interface (NEW, just created)

**Archive** (7 duplicate tracker implementations):
1. `tracking.py` — EntityTracker3D (old Phase 2, replaced by EnhancedTracker)
2. `enhanced_tracker_adapter.py` — Adapter for old EntityTracker3D
3. `tracker.py` — EntityTracker (basic clustering, keep for legacy entity grouping?)
4. `tracker_reid.py` — MultiHypothesisTracker (experimental)
5. `temporal_tracker.py` — TemporalTracker (another variant)
6. `object_tracker.py` — ObjectTracker (simple 2D+3D)
7. `pipeline_adapter.py` — Adapters for old pipelines

**Decision**: Keep `tracker.py` (EntityTracker) for now as it's used by `engine.py` for clustering observations into entities (different from frame-level tracking). Archive the rest.

---

### 🟡 DUPLICATE RE-ID MODULES (Archive All, Use EnhancedTracker's Built-in)

**Archive**:
1. `advanced_reid.py` — Advanced Re-ID with gallery
2. `appearance_reid.py` — Appearance-based Re-ID
3. `clip_reid.py` — CLIP-based Re-ID
4. `fastvlm_reid.py` — FastVLM Re-ID
5. `geometric_reid.py` — Geometric Re-ID
6. `appearance_extractor.py` — Feature extraction
7. `reid_matcher.py` — Re-ID matching logic

**Reason**: `EnhancedTracker` already has appearance embeddings + EMA + gallery. Consolidate later if needed.

---

### 🟡 DUPLICATE DEPTH MODULES (Keep `depth.py`, Archive Variants)

**Keep**:
- ✅ `depth.py` — Main DepthEstimator (MiDaS/ZoeDepth)

**Archive**:
1. `depth_anything.py` — DepthAnything wrapper (if not used)
2. `depth_anything_v2/` — Submodule (check if actively used)

**Action**: Check if DepthAnything is referenced, otherwise archive.

---

### 🟡 DUPLICATE DETECTION MODULES (Keep `observer.py`, Archive Advanced)

**Keep**:
- ✅ `observer.py` — FrameObserver with YOLO

**Archive**:
1. `advanced_detection.py` — Advanced detection (Detectron2? Experimental?)

---

### 🟡 SPATIAL/SLAM FUSION (Keep Core, Archive Experimental)

**Keep**:
- ✅ `perception_3d.py` — Perception3DEngine (depth + hands + 3D)
- ✅ `camera_intrinsics.py` — Backprojection utilities
- ✅ `scale_estimator.py` — Scale recovery

**Archive**:
1. `slam_fusion.py` — SLAM fusion experiments
2. `reconstruction_3d.py` — 3D reconstruction (duplicate with perception_3d?)
3. `spatial_map_builder.py` — Spatial mapping
4. `semantic_scale.py` — Semantic scale estimation

---

### 🟡 VISUALIZATION (Keep Rerun, Archive Old)

**Keep**:
- ✅ `rerun_visualizer.py` — UnifiedRerunVisualizer (current)

**Archive**:
1. `visualization.py` — Old visualization (matplotlib/CV2)

---

### 🟡 OTHER MODULES

**Keep**:
- ✅ `engine.py` — PerceptionEngine (core orchestrator)
- ✅ `config.py` — PerceptionConfig
- ✅ `embedder.py` — VisualEmbedder (CLIP)
- ✅ `describer.py` — EntityDescriber (FastVLM)
- ✅ `types.py` — Type definitions
- ✅ `unified_frame.py` — UnifiedFrame dataclass
- ✅ `occlusion.py` — OcclusionDetector
- ✅ `hand_tracking.py` — HandTracker (MediaPipe, disabled for now)

**Archive**:
1. `unified_pipeline.py` — Old unified pipeline?
2. `spatial_analyzer.py` — Spatial analysis
3. `corrector.py` — Correction logic?

---

## Semantic Folder Analysis

### 🟢 KEEP (Core Semantic Processing)

- ✅ `engine.py` — SemanticEngine (Phase 2 orchestrator)
- ✅ `config.py` — SemanticConfig
- ✅ `types.py` — Type definitions
- ✅ `state_detector.py` — StateChangeDetector
- ✅ `event_composer.py` — EventComposer
- ✅ `causal.py` — CausalInference
- ✅ `temporal_windows.py` — TemporalWindowManager
- ✅ `zone_manager.py` — ZoneManager
- ✅ `spatial_utils.py` — Spatial relationship utilities

### 🟡 ARCHIVE (Experimental/Duplicate)

1. `causal_scorer.py` — Duplicate causal logic?
2. `cis_scorer_3d.py` — CIS scoring (experimental)
3. `scene_assembler.py` — Scene assembly
4. `scene_classifier.py` — Scene classification
5. `scene_graph.py` — Scene graph (use Memgraph instead)
6. `scene_understanding.py` — Scene understanding
7. `rich_captioning.py` — Rich captions
8. `smart_caption_prioritizer.py` — Caption prioritization
9. `strategic_captioner.py` — Strategic captions
10. `temporal_description_generator.py` — Temporal descriptions
11. `enhanced_spatial_reasoning.py` — Enhanced spatial reasoning
12. `query_intelligence.py` — Query intelligence
13. `spatial_nlg.py` — Spatial NLG
14. `entity_tracker.py` — Entity tracker (duplicate?)

**Reason**: Many of these are experimental or duplicate functionality already in `engine.py`. Archive and consolidate later if needed.

---

## SLAM Folder Analysis

### 🟢 KEEP (Core SLAM)

- ✅ `slam_engine.py` — SLAMEngine (visual odometry + loop closure)
- ✅ `loop_closure.py` — LoopClosureDetector
- ✅ `pose_graph.py` — PoseGraphOptimizer
- ✅ `depth_utils.py` — Depth preprocessing utilities
- ✅ `hybrid_odometry.py` — Hybrid visual-depth odometry
- ✅ `depth_consistency.py` — Depth consistency checking
- ✅ `multi_frame_depth_fusion.py` — Multi-frame depth fusion

### 🟡 ARCHIVE (Experimental/Duplicate)

1. `depth_odometry.py` — Depth-only odometry (replaced by hybrid?)
2. `projection_3d.py` — 3D projection utilities (duplicate with perception?)
3. `semantic_slam.py` — Semantic SLAM (experimental)
4. `world_coordinate_tracker.py` — World coordinate tracking

---

## Graph Folder (Quick Look)

**Keep**:
- `builder.py` — GraphBuilder (stub, migrate to Memgraph)

**Archive**:
- Any Neo4j-specific code

---

## Summary Statistics

### Perception
- **Keep**: 15 files
- **Archive**: 20+ files (trackers, Re-ID, spatial, visualization)

### Semantic
- **Keep**: 9 files
- **Archive**: 14 files (captioning, scene understanding, experimental)

### SLAM
- **Keep**: 7 files
- **Archive**: 4 files (experimental odometry, projections)

**Total to Archive**: ~38 files  
**Reduction**: ~50% of codebase

---

## Proposed Archive Structure

```
orion/
  _archive/
    README.md                     # What's here and why
    perception/
      trackers/                   # All duplicate trackers
      reid/                       # All Re-ID modules
      spatial/                    # Spatial/SLAM fusion experiments
      visualization/              # Old visualization code
      detection/                  # Advanced detection
      depth/                      # Depth variants
    semantic/
      captioning/                 # Captioning modules
      scene/                      # Scene understanding
      experimental/               # Experimental features
    slam/
      odometry/                   # Experimental odometry
      projection/                 # 3D projection utilities
```

---

## Next Steps

1. ✅ Create `orion/_archive/` structure
2. Move files according to plan above
3. Update imports in remaining files
4. Remove Neo4j stubs in `graph/builder.py`
5. Consolidate settings in `orion/settings.py`
6. Add validation and tests
7. Update documentation

---

## Memory/Storage Architecture Proposal

### For Long-Term Object Memory & Re-ID

**Proposed Stack**:

1. **Short-Term (In-Memory)**:
   - `EnhancedTracker` appearance gallery (5-20 embeddings per track)
   - Track state (position, velocity, class beliefs)
   - Active for `max_age` frames (default 30 = ~7 seconds at 4 FPS)

2. **Medium-Term (Embedding Index)**:
   - **Faiss HNSW** index for fast nearest-neighbor search
   - Store embeddings + metadata (track_id, class, last_seen, bbox_3d)
   - Query when new unmatched detections appear
   - Persist to disk every N minutes (e.g., `~/.orion/embeddings.index`)

3. **Long-Term (Memgraph)**:
   - **Entities** as nodes: `(:Entity {id, class, first_seen, last_seen, confidence})`
   - **Tracks** as edges: `(:Track {track_id, start_frame, end_frame, avg_confidence})`
   - **Relationships**: `(:Entity)-[:SEEN_AT {timestamp, zone, bbox_3d}]->(:Frame)`
   - **Spatial**: `(:Entity)-[:ON|NEAR|HELD_BY]->(:Entity)`
   - **State changes**: `(:Entity)-[:STATE_CHANGE {from, to, timestamp}]->()`

**Why This Stack**:
- **Faiss**: 10-100x faster than brute-force for >1000 embeddings, supports GPU
- **Memgraph**: C++-based in-memory graph DB, 10x faster than Neo4j for real-time queries
- **Separation of concerns**: Embeddings for Re-ID, graph for relationships/queries

**Schema Example**:
```cypher
// Entity with tracking history
CREATE (e:Entity {
  id: 'entity_123',
  class: 'book',
  first_seen: timestamp(),
  last_seen: timestamp(),
  total_appearances: 15
})

// Spatial relationship
CREATE (e1:Entity {id: 'book_1'})-[:ON {
  confidence: 0.92,
  timestamp: timestamp(),
  zone: 'desk_center'
}]->(e2:Entity {id: 'desk_1'})

// State change
CREATE (e:Entity {id: 'cup_1'})-[:STATE_CHANGE {
  from: 'on_table',
  to: 'held',
  timestamp: timestamp(),
  causal_score: 0.85
}]->()

// Query: "Where was the book last seen?"
MATCH (e:Entity {class: 'book'})-[r:SEEN_AT]->(f:Frame)
RETURN e.id, r.zone, r.timestamp
ORDER BY r.timestamp DESC
LIMIT 1
```

**Decision**: Use **Faiss + Memgraph** for hybrid memory.

---

**Ready to proceed with archiving?** Say "yes" and I'll execute the move.
