# Orion Research: Issues & Optimization Plan

**Date:** October 25, 2025  
**Status:** Analysis Phase

---

## Issue 1: Motion Tracker Redundancy

### Problem
- `orion/motion_tracker.py` exists at top-level
- `orion/perception/tracker.py` exists separately  
- `orion/semantic/causal.py` imports from motion_tracker
- Tests expect both APIs

### Current State
```
orion/
├── motion_tracker.py          ← Lightweight utilities (MotionData, MotionTracker)
├── perception/
│   └── tracker.py             ← Entity clustering & tracking logic
└── semantic/
    └── causal.py              ← Uses motion_tracker
```

### Solution
**Move `motion_tracker.py` utilities into `perception/tracker.py`**
- Consolidate MotionData, MotionTracker, and helper functions
- Update semantic imports to reference `orion.perception.tracker`
- Keep clean API surface without top-level imports

---

## Issue 2: Graph Builder Duplication

### Problem
- `orion/semantic/graph_builder.py` (414 lines) - ingests semantic results into Neo4j
- `orion/graph/builder.py` (1179 lines) - comprehensive knowledge graph building
- Both build graphs, different purposes and interfaces
- Creates confusion about which to use

### Current State
```
Semantic Result → orion/semantic/graph_builder.py → Neo4j
                  (simple ingest)

Perception + Semantic → orion/graph/builder.py → Neo4j
                        (comprehensive with spatial reasoning)
```

### Root Cause
- `semantic/graph_builder.py` was created for Phase 2 semantic pipeline
- `graph/builder.py` is the original, more complete implementation
- Never consolidated after refactoring

### Solution
**Delete `orion/semantic/graph_builder.py`**
- `orion/graph/builder.py` is the authoritative implementation
- Update `SemanticEngine` to use `orion.graph.builder.GraphBuilder`
- Clean up imports in semantic/__init__.py

---

## Issue 3: Perception vs Semantic Engine Redundancy

### Analysis

#### Perception Engine (`orion/perception/engine.py`)
**Responsibilities:**
1. Frame observation (video → detections)
2. Visual embedding generation
3. Entity clustering & tracking
4. Entity description generation

**Key Output:** `PerceptionResult` with:
- `entities: List[PerceptionEntity]`
- `raw_observations: List[Observation]`
- Embeddings per entity

---

#### Semantic Engine (`orion/semantic/engine.py`)
**Responsibilities:**
1. Entity consolidation (re-tracking!)
2. Spatial zone detection (HDBSCAN clustering)
3. State change detection
4. Scene assembly
5. Causal reasoning
6. Event composition
7. Graph ingestion

**Key Input:** `PerceptionResult`

---

### Redundancy Points

#### 1. **Entity Tracking (MAJOR)**

**Perception:** `EntityTracker` clusters observations by spatial proximity + embedding similarity
```python
# perception/tracker.py
def cluster_observations(observations: List[Observation]) -> List[PerceptionEntity]:
    # Groups detections into entities over time using DBSCAN + embedding
```

**Semantic:** `SemanticEntityTracker` re-consolidates entities
```python
# semantic/entity_tracker.py
def consolidate_entities(perception_result: PerceptionResult) -> List[SemanticEntity]:
    # Re-groups perception entities (why?)
```

**Problem:** Why consolidate twice? Perception already groups objects.

---

#### 2. **Spatial Analysis**

**Perception:** `spatial_analyzer.py` does zone classification
```python
def calculate_spatial_zone(bbox) -> SpatialContext:
    # Returns: ceiling, wall_upper, wall_lower, floor, etc.
    # Hierarchical zones (vertical + horizontal)
```

**Semantic:** `spatial_utils.py` does HDBSCAN clustering for zones
```python
def cluster_entities_hdbscan(entities) -> List[SpatialZone]:
    # Returns: desk_area, bedroom_area, kitchen_area
    # Semantic zones (high-level scene areas)
```

**Distinction:**
- Perception: **Low-level frame zones** (ceiling/wall/floor)
- Semantic: **High-level scene areas** (room-level clustering)
- **NOT redundant** - they serve different purposes

---

#### 3. **Embedding & Description**

**Perception:** Generates both
```python
# perception/embedder.py
def embed_detections(detections) -> List[Detection]:
    # CLIP embeddings per object

# perception/describer.py  
def describe_entities(entities) -> List[PerceptionEntity]:
    # FastVLM descriptions
```

**Semantic:** Uses them, doesn't regenerate
```python
# semantic/engine.py
embeddings = {
    e.entity_id: e.average_embedding
    for e in entities
}
```

**Status:** ✓ No redundancy here

---

### Root Cause of Entity Re-consolidation

Looking at `semantic/entity_tracker.py`:
```python
def consolidate_entities(perception_result: PerceptionResult):
    # 1. Iterates over perception entities
    # 2. Merges nearby entities
    # 3. Returns "consolidated" entities
```

**Why?** Unclear. Perception already did this work.

**Hypothesis:** 
- Semantic pipeline needed different entity structure (SemanticEntity vs PerceptionEntity)
- Instead of converting, they re-consolidated
- This is architectural debt

---

## Optimization Plan

### Phase 1: Motion Tracker Consolidation
**Priority:** HIGH | **Effort:** 1 hour

1. Move `motion_tracker.py` contents → `perception/tracker.py`
2. Update imports in `semantic/causal.py`
3. Delete `orion/motion_tracker.py`
4. Update tests

---

### Phase 2: Graph Builder Consolidation  
**Priority:** HIGH | **Effort:** 30 min

1. Delete `orion/semantic/graph_builder.py`
2. Update `SemanticEngine` to import from `orion.graph.builder`
3. Check if APIs are compatible; adapt if needed
4. Update `semantic/__init__.py` imports

---

### Phase 3: Entity Tracking Refactor
**Priority:** MEDIUM | **Effort:** 2-3 hours

**Option A: Eliminate Redundancy (RECOMMENDED)**
1. Remove `SemanticEntityTracker.consolidate_entities()`
2. Convert `PerceptionEntity` → `SemanticEntity` directly
3. Semantic pipeline receives already-tracked entities
4. **Benefit:** Simpler, faster pipeline

**Option B: Unify Tracking Logic**
1. Keep both if semantic filtering is valuable
2. Document why re-consolidation happens
3. Make it explicit in naming/config

**Recommendation:** Option A - perception tracking is sufficient

---

### Phase 4: Verify Spatial Analysis Separation
**Priority:** LOW | **Effort:** 30 min

1. Confirm perception zones vs semantic zones are intentionally different
2. Document the distinction in architecture guide
3. Add examples showing both in action

---

## Testing Strategy for CIS & Temporal Components

### CIS (Causal Influence Score) Testing

#### 1. **Component Tests** (Unit Level)
Test each CIS component in isolation:

```python
# Test temporal decay
- test_temporal_proximity_immediate()      # t=0, should be ~1.0
- test_temporal_proximity_decay()          # exponential decay
- test_temporal_proximity_threshold()      # beyond decay horizon

# Test spatial proximity
- test_spatial_proximity_touching()        # distance=0, should be 1.0
- test_spatial_proximity_far()             # max_distance, should be 0.0
- test_spatial_proximity_quadratic()       # verify quadratic falloff

# Test motion alignment
- test_motion_alignment_perfect()          # velocity aligned to target
- test_motion_alignment_perpendicular()    # 90° angle
- test_motion_alignment_opposite()         # opposite direction

# Test semantic proximity
- test_semantic_proximity_identical()      # same embedding
- test_semantic_proximity_orthogonal()     # unrelated objects
- test_semantic_proximity_opposite()       # semantically opposite
```

#### 2. **Integration Tests**
Test full CIS calculation with realistic scenarios:

```python
# High-causality scenario
- agent close to patient
- agent moving toward patient
- very recent in time
- semantically related
→ Expected: CIS > 0.7

# Low-causality scenario  
- agent far from patient
- agent stationary
- temporal distance > threshold
- unrelated semantics
→ Expected: CIS < 0.3

# Edge cases
- agent == patient (same entity)
- missing motion data
- missing embeddings
- invalid timestamps
```

#### 3. **HPO Weight Validation**
Test that HPO-learned weights from `hpo_results/cis_weights.json` work well:

```python
# Current weights from your JSON:
{
  "temporal": 0.296,      # 30% - weighted toward space/motion
  "spatial": 0.436,       # 44% - HIGHEST weight (spatial matters most)
  "motion": 0.208,        # 21%
  "semantic": 0.060       # 6% - LOWEST (semantics less important)
  "threshold": 0.543      # 54% threshold
}

Tests:
- test_cis_with_learned_weights()
  → Score should predict causal pairs better than default weights
  
- test_weight_ablation()
  → Remove each weight, verify F1 degrades
  → spatial weight most critical (44%)
  → semantic weight least critical (6%)
```

---

### Temporal Component Testing

#### 1. **Temporal Window Creation**
```python
# State change grouping
- test_window_creation_sparse()       # Few changes, single window
- test_window_creation_clustered()    # Dense changes → multiple windows
- test_window_max_duration()          # Enforce max_duration_seconds
- test_window_max_gap()               # Enforce max_gap_between_changes

# Edge cases
- test_empty_state_changes()          # No changes
- test_single_state_change()          # One change
- test_changes_at_boundaries()        # Changes near window limits
```

#### 2. **Temporal Decay**
```python
# Verify decay formula: exp(-t / decay_constant)
- test_decay_at_zero()                # t=0, decay should be 1.0
- test_decay_at_constant()            # t=decay_seconds, should drop to ~0.37 (1/e)
- test_decay_shape()                  # Verify exponential shape
- test_decay_with_config_values()     # Use actual decay_seconds from config
```

#### 3. **Temporal Causality**
```python
# Events close in time should have high temporal score
- test_temporal_causality_immediate()
  → agent_change at t=1.0
  → patient_change at t=1.1
  → temporal_score should be high (close in time)

- test_temporal_causality_distant()
  → agent_change at t=1.0
  → patient_change at t=10.0 (beyond decay horizon)
  → temporal_score should be low
```

#### 4. **Multi-Window Causal Links**
```python
# Causal links across temporal windows
- test_causal_links_same_window()     # Both changes in same window
- test_causal_links_adjacent_windows() # Changes in adjacent windows
- test_causal_links_distant_windows()  # Changes far apart in time
```

---

## Performance Recommendations

### 1. **CIS Optimization**
- **Problem:** Spatial weight is dominant (44%) → computation-heavy
- **Solution:** Cache spatial distance calculations
- **Benefit:** 2-3x faster causal link computation

### 2. **Temporal Window Optimization**
- **Problem:** State change detection is quadratic (N² comparisons)
- **Solution:** 
  - Sort changes by timestamp first
  - Early exit when time gap exceeds threshold
  - Use binary search for window boundaries
- **Benefit:** O(N log N) instead of O(N²)

### 3. **Entity Consolidation**
- **Problem:** Re-tracking in semantic engine wastes computation
- **Solution:** Skip `SemanticEntityTracker`, convert directly
- **Benefit:** ~20% faster semantic pipeline

### 4. **Embedding Caching**
- **Problem:** CLIP embeddings recalculated for each state change
- **Solution:** Use perception engine's embeddings directly
- **Benefit:** Already done in causal scorer, verify no duplication

---

## Implementation Order

1. **Week 1:** Motion tracker consolidation (Issue #1)
2. **Week 1:** Graph builder consolidation (Issue #2)  
3. **Week 2:** Entity tracking refactor (Issue #3)
4. **Week 2-3:** CIS & Temporal tests (validation)
5. **Week 3:** Performance optimizations

---

## Current Issues Summary Table

| Issue | Location | Type | Impact | Status |
|-------|----------|------|--------|--------|
| Motion Tracker | `orion/motion_tracker.py` | Duplication | Low | TODO |
| Graph Builder | `orion/semantic/graph_builder.py` | Duplication | Medium | TODO |
| Entity Re-consolidation | `SemanticEntityTracker` | Inefficiency | Medium | TODO |
| Spatial Zones | Perception + Semantic | By Design | None | ✓ OK |
| CIS Testing | Tests missing | Coverage | Medium | TODO |
| Temporal Testing | Tests missing | Coverage | Medium | TODO |

