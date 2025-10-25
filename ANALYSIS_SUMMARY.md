# Summary: Architecture Analysis & Testing Guide

## Issues Found & Recommendations

### 1. **Motion Tracker Redundancy** 🔴
- **Location:** `orion/motion_tracker.py` (top-level, duplicates `perception/tracker.py`)
- **Solution:** Merge into `perception/tracker.py`, delete source
- **Effort:** 1 hour
- **Impact:** Cleaner imports

### 2. **Graph Builder Duplication** 🔴
- **Locations:** 
  - `orion/semantic/graph_builder.py` (414 lines, simple)
  - `orion/graph/builder.py` (1179 lines, comprehensive)
- **Solution:** Delete semantic version, use authoritative `orion/graph/builder.py`
- **Effort:** 30 minutes
- **Impact:** Single source of truth

### 3. **Entity Re-consolidation Inefficiency** 🟡
- **Location:** `orion/semantic/entity_tracker.py`
- **Problem:** Perception already clusters entities, semantic re-clusters them
- **Solution:** Direct conversion `PerceptionEntity` → `SemanticEntity`
- **Effort:** 2-3 hours
- **Impact:** ~20% faster semantic pipeline

### 4. **Spatial Analysis Separation** ✅
- **Status:** INTENTIONAL (not a bug)
- **Difference:**
  - Perception: Frame-level zones (ceiling/wall/floor)
  - Semantic: Scene-level zones (desk_area/kitchen_area)
- **Action:** Keep separate

---

## CIS Testing Guide

Your HPO-learned weights (from `hpo_results/cis_weights.json`):

```json
{
  "temporal": 0.296,      // 30%  - Time proximity
  "spatial": 0.436,       // 44%  - DOMINANT (space matters most)
  "motion": 0.208,        // 21%  - Motion alignment
  "semantic": 0.060,      // 6%   - Semantic similarity (least important)
  "threshold": 0.543      // Decision boundary
}
```

### What to Test

**1. Component Tests (Unit Level)**
```
✓ Temporal: exp(-t/4) decay curve
  - t=0 → 1.0
  - t=4 → 0.37
  - t>20 → near 0

✓ Spatial: (1 - d/600)² quadratic falloff
  - d=0 → 1.0
  - d=150 → 0.56
  - d=300 → 0.25
  - d=600 → 0.0

✓ Motion: Alignment to target
  - Perfect (0°) → high score
  - Perpendicular (90°) → low
  - Opposite (180°) → very low

✓ Semantic: Embedding similarity
  - Identical → 1.0
  - Related → 0.8
  - Unrelated → 0.5
  - Opposite → 0.0
```

**2. Integration Tests (Full CIS)**
```
✓ Perfect causal scenario
  - Hand touches door, moving toward it, same time
  - Expected CIS: > 0.85

✓ Non-causal scenario
  - Far objects, opposite motion, old event
  - Expected CIS: < 0.05

✓ Boundary scenario
  - Right at threshold (0.543)
  - Test stability, rounding
```

**3. Weight Validation**
```
✓ Verify spatial weight (0.436) is most critical
✓ Verify removing it causes biggest F1 drop
✓ Compare HPO weights vs defaults on your data
```

---

## Temporal Testing Guide

### What to Test

**1. Decay Formula: exp(-t / 4.0)**
```
t=0:   1.0000
t=1:   0.7788
t=4:   0.3679
t=10:  0.0821
t>20:  <0.01
```

**2. Window Creation**
```
✓ Single change → 1 window
✓ Changes within 1.5s gap → same window
✓ Changes > 1.5s apart → separate windows
✓ Max duration (5.0s) → split if exceeded
✓ Max changes (20) → overflow handling
```

**3. Causal Links Across Windows**
```
✓ Same window: both changes in one temporal window
✓ Adjacent: changes in nearby windows
✓ Distant: changes far apart in time
```

---

## Documents Created

1. **`ARCHITECTURE_SUMMARY.md`** - Full architectural analysis with diagrams
2. **`ISSUES_AND_PLAN.md`** - Detailed issue breakdown + optimization plan
3. **`CIS_TEMPORAL_TESTING_GUIDE.py`** - 40+ specific test cases with expected values
4. **`TEST_QUICK_REFERENCE.py`** - Simple test templates you can use

---

## Next Steps

### Phase 1: Quick Wins (Week 1)
1. Move motion_tracker → perception/tracker
2. Delete semantic/graph_builder.py
3. Consolidate imports

### Phase 2: Validation (Week 2)
1. Run CIS component tests
2. Run temporal window tests
3. Compare HPO vs default weights

### Phase 3: Optimization (Week 3)
1. Profile pipeline
2. Cache spatial distances
3. Parallelize window creation

---

## Key Metrics to Track

- **CIS per-link computation:** < 1ms
- **Temporal window creation:** < 10ms for 100 changes
- **F1 improvement from HPO weights:** > 5%
- **Pipeline throughput:** > 30 fps

---

## Questions You Can Answer Now

1. ✅ **What's the redundancy between perception & semantic?**
   - Entity re-consolidation (architectural debt from Phase 2)

2. ✅ **Why two graph builders?**
   - One simple (semantic ingest), one comprehensive (authoritative)

3. ✅ **Why motion tracker at top level?**
   - Architectural sloppiness from refactoring

4. ✅ **What to test for CIS?**
   - All 4 components + 3 integration scenarios
   - See `CIS_TEMPORAL_TESTING_GUIDE.py` for specifics

5. ✅ **What to test for temporal?**
   - Decay formula + window creation logic
   - See test guide for expected values

---

All analysis documents are ready for review. You now have:
- Clear architecture issues identified
- Specific action items with effort estimates
- Comprehensive testing guide with expected values
- Templates to use for writing tests
- Performance optimization roadmap
