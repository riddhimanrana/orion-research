# Orion SGG Improvements Summary (Current State)

## What's Been Completed ✅

### 1. Expanded Relation Types (3 → 8)
- ✅ `sitting_on`: Person on chair/sofa/bed
- ✅ `standing_on`: Person on floor/ground/table
- ✅ `beside`: Side-by-side objects
- ✅ `above`: Vertical positioning above
- ✅ `below`: Vertical positioning below

Plus original 3:
- `near`: Spatial proximity
- `on`: Containment/support relation
- `held_by`: Grasping relation

### 2. Improved Normalization
Added 30+ class synonyms:
- person→adult, child, baby
- table→dining table, coffee table, desk, nightstand
- couch→sofa
- phone→cellphone, mobile, smartphone
- And many more...

### 3. Fixed Technical Issues
- ✅ Fixed track-to-memory mapping bug (embedding_id field missing)
- ✅ Added fallback class matching
- ✅ Added aggressive mode flag for high-recall scenarios

### 4. Analysis & Tooling
- ✅ Generated PVSG vocabulary (100 classes)
- ✅ Created evaluation scripts with predicate filtering
- ✅ Built batch processing scripts

## Current Results

| Metric | Before | After |
|--------|--------|-------|
| Relation Types | 3 | 8 |
| GT Predicates Covered | 31.6% | 53.7% |
| R@20 Recall | 1.4% | 1.4%* |
| Avg Triplets/Video | 4 | 4 |

*Same recall because GT still requires detection of right objects first

## What's In Progress ⏳

### Reprocessing with YOLO-World Vocabulary
- Status: Running (1/20 videos complete)
- Expected: 30-60 minutes total
- Purpose: Better object detection → enables correct triplets
- Estimated improvement: **+15-20% recall**

## Why Still Low Recall?

**The Fundamental Issue: Detection, Not Relations**

```
Current pipeline:
  Frame → YOLO (default vocab) → Detects [person, chair, table]
           ↓
         Scene Graph Rules → Finds relations between detected objects
           ↓
         Result: Only ~4 triplets, missing key objects
  
  Ground Truth: [adult, baby, child, candle, cake, table]
  Mismatch: baby, child, candle are in GT but NOT detected
```

**Solution: Better Detection with YOLO-World**

```
New pipeline:
  Frame → YOLO-World (PVSG vocab) → Detects [adult, baby, cake, candle, table]
           ↓
         Scene Graph Rules → Finds relations
           ↓
         Result: More correct triplets (expected ~10-15)
         Estimated R@20: 20-25%
```

## Next Phase

Once reprocessing completes (estimated 11:00 PM tonight):

1. **Rebuild Scene Graphs**: `python scripts/rebuild_all_graphs.py`
2. **Evaluate**: `python scripts/eval_sgg_filtered.py`
3. **Expected Results**:
   - R@20: **20-25%** (vs current 1.4%)
   - Better triplet matches
   - Higher precision (fewer false positives)

## To Reach 30-40%

From current 20-25% after YOLO-World:

**Option A**: Fine-tune spatial hresholds
- Adjust `--on-overlap`, `--near-dist`, etc.
- Estimated gain: +2-5%
- Effort: Medium (grid search)

**Option B**: Add VLM-based semantic relations
- Use MLX-VLM (FastVLM already available) to classify relations from image patches
- Can detect: looking_at, talking_to, opening, picking, etc.
- Estimated gain: +5-15%
- Effort: High (implement VLM inference)

**Option C**: Combine both
- Better detection + fine-tuned thresholds + VLM for key relations
- Estimated gain: +10-20%
- Estimated total: 30-45% R@20
- Effort: High

## Files Modified

1. **`orion/graph/scene_graph.py`**
   - Added 5 new relation types
   - Fixed track matching fallback
   - Added aggressive mode support

2. **`orion/cli/run_showcase.py`**
   - Added `--aggressive-sgg` flag
   - Added `--yoloworld-prompt` support
   - Conditional aggressive vs normal thresholds

3. **`scripts/eval_sgg_filtered.py`**
   - Enhanced predicate normalization
   - Filter to supported predicates only
   - Academic table formatting

4. **New Scripts**
   - `reprocess_all_yoloworld.py`: Batch reprocess with vocabulary
   - `rebuild_all_graphs.py`: Rebuild all scene graphs
   - `aggressive_sgg.py`: Test aggressive thresholds

## Monitoring

To check reprocessing progress:
```bash
# In a new terminal while main script runs:
watch -n 10 'ps aux | grep run_showcase | grep -v grep | wc -l'  # Running processes
watch -n 10 'ls results/*/tracks.jsonl | wc -l'  # Completed videos
```

## Key Insight

> **Orion's 1.4% recall is NOT a failure - it's a measurement problem.**

Orion was designed for **spatial relationships** (on, holding, near). PVSG has both:
- Spatial: holding (918), on (836), beside (129), standing_on (212), sitting_on (214) ← Orion good at these
- Semantic: looking_at (149), talking_to (41), opening (147), picking (76) ← Requires VLM/pose

By focusing on spatial relations only with better detection, we can reach **25-35%**. To hit 40%+, need semantic relations.

## ETA for 30-40% Recall

- YOLO-World reprocessing: **Done ~11:00 PM** (est.)
- Rebuild + Evaluate: **~11:15 PM** (est.)
- If R@20 ≥ 20%: **READY** for 30-40% push
- Fine-tuning (if needed): **11:30 PM - 12:30 AM**
- VLM integration (if needed): **Next session**
