# Phase 4 Plan: Long-term Memory & Spatial Reasoning

## Goals
1.  **Robust Re-ID:** Consistent object identity across long videos (30+ mins) and occlusions.
2.  **Spatial Memory:** Accurate spatial zones and relationships stored in a graph.
3.  **Vector Search:** Natural language querying of the long-term memory.
4.  **Memgraph Integration:** Real-time graph updates from the perception pipeline.

## 1. Memgraph Vector Indexing
**Status:** `MemgraphBackend` exists but lacks vector support.
**Action:**
-   Update `orion/graph/memgraph_backend.py`:
    -   Add `embedding` property to `Entity` nodes.
    -   Implement `create_vector_index()` using Memgraph's vector search capabilities (MAGE or native).
    -   Implement `search_similar_entities(embedding, limit)` for semantic search.

## 2. Perception Engine Integration
**Status:** `PerceptionEngine` does not currently use `MemgraphBackend`.
**Action:**
-   Update `orion/perception/engine.py`:
    -   Initialize `MemgraphBackend` if configured.
    -   Push `Entity` updates (bbox, class, embedding) to Memgraph on every frame/interval.
    -   Push `SpatialRelationship` updates to Memgraph.

## 3. Re-ID Improvements
**Status:** `EnhancedTracker` uses appearance but needs tuning for long-term consistency.
**Action:**
-   Tune `appearance_threshold` and `max_gallery_size`.
-   Implement "Split/Merge" logic in the graph:
    -   If a new track has high similarity to an old (lost) track, merge them in the graph.
    -   Use `MemgraphBackend.query_temporal_coexistence` to validate merges.

## 4. Evaluation Framework
**Status:** No automated Re-ID evaluation.
**Action:**
-   Create `scripts/evaluate_reid.py`:
    -   Input: Video + Ground Truth (if available) or manual annotation mode.
    -   Metrics: ID Switches, Track Length, Re-ID Accuracy (if GT exists).
    -   Run on `video.mp4`, `test`, and `room`.

## 5. Spatial Zones
**Status:** `Zone` nodes exist in Memgraph schema.
**Action:**
-   Ensure `PerceptionEngine` calculates which zone an entity is in (using `shapely` or simple bounds).
-   Pass `zone_id` to `MemgraphBackend.add_entity_observation`.

## Next Steps
1.  Update `MemgraphBackend` with vector support.
2.  Integrate `MemgraphBackend` into `PerceptionEngine`.
3.  Run `video.mp4` and verify data in Memgraph.
