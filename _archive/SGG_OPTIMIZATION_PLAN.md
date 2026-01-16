# Orion SGG Optimization Report - Path to 30-40% Recall

## Current Status
- **Baseline Recall**: 1.4% R@20 (3 relation types)
- **With Expanded Relations**: 1.4% R@20 (8 relation types: on, holding, near, sitting_on, standing_on, beside, above, below)
- **With Aggressive Thresholds**: 1.8% R@20 (more false positives)

## Key Findings

### Root Cause Analysis
The **detection bottleneck** is the main limiter:

```
Video 0001_4164158586 example:
  GT requires: [adult, baby, child, candle, cake, table]
  Orion detects: [cake, chair, microwave, person, refrigerator, sink, table]
  
  Missing: baby, child, candle (30% of GT objects)
  False positives: microwave, refrigerator, sink (wrong detections)
  
  Result: Can't form correct triplets even if relations are perfect
```

### Detection Accuracy Impact
- **YOLO (default vocab)**: Only detects generic classes (person, chair, table)
- **YOLO-World (PVSG vocab)**: Detects 100+ specific classes (baby, child, candle, gift, etc.)
- **Estimated improvement**: +20-30% recall with better detection

## Three-Phase Improvement Plan

### Phase 1: Better Object Detection ✅ IN PROGRESS
**What**: Reprocess all 20 videos with YOLO-World using PVSG vocabulary
**Why**: YOLO's default vocab misses 30% of objects that GT has
**Status**: Reprocessing started (`reprocess_all_yoloworld.py`)
**Timeline**: ~2-3 min per video × 20 = 40-60 minutes
**Expected recall improvement**: +15-20% (to ~20-25% R@20)

### Phase 2: Better Spatial Relations ✅ COMPLETED
**What**: 
- Added `sitting_on` (person on chair/sofa)
- Added `standing_on` (person on floor/ground)
- Added `beside` (side-by-side proximity)
- Added `above`/`below` (vertical positioning)

**Why**: PVSG GT has 65 predicates; Orion now supports 8 spatial types (vs original 3)
**Result**: Covered 53.7% of GT triplet predicates (vs 31.6% originally)
**Limitation**: Can't detect semantic relations (looking_at, talking_to, opening, etc.) without VLM

### Phase 3: Precision-Recall Tuning ⏳ PENDING
**What**: 
- Aggressive thresholds (already tested): Generated more triplets but lower precision
- Balanced thresholds: Current default - good precision, moderate recall

**Next**: Once YOLO-World detection completes, rebuild scene graphs and evaluate

## Implementation Checklist

- [x] Analyzed PVSG dataset (126 classes, 65 predicates)
- [x] Generated YOLO-World vocabulary prompt (100 top classes)
- [x] Expanded relation types (3 → 8 types)
- [x] Fixed track-to-memory mapping bug
- [x] Improved class name normalization (person→adult, etc.)
- [x] Added aggressive mode flag (`--aggressive-sgg`)
- [ ] Complete YOLO-World reprocessing
- [ ] Rebuild scene graphs with new detections
- [ ] Evaluate and measure improvement
- [ ] Fine-tune thresholds if needed

## Expected Results After Phase 1

With YOLO-World vocabulary:
- **Better detections**: Should find baby, child, candle, gift, etc.
- **More correct triplets**: Enables matches like `(baby, holding, ?)` → `(adult, holding, baby)`
- **Estimated R@20**: **20-25%** (vs current 1.4%)

To reach 30-40%:
- Phase 1 + Phase 2 (better detection + better relations): **25-35%**
- May need Phase 3 (VLM for semantic relations) for remaining 5-10%

## Technical Details

### Modified Files
1. `orion/graph/scene_graph.py`: Added 5 new relation types, improved tracking mapping
2. `orion/cli/run_showcase.py`: Added `--aggressive-sgg` flag
3. `scripts/eval_sgg_filtered.py`: Updated normalization, filter predicates
4. `scripts/reprocess_all_yoloworld.py`: NEW batch reprocessing script
5. `scripts/rebuild_all_graphs.py`: NEW graph rebuilding script

### New Relation Types
```python
# Spatial heuristics implemented
- near: distance <= 0.08 (frame diagonal)
- on: horizontal overlap >= 0.3, vertical gap <= 0.02
- held_by: IoU >= 0.3, object not furniture
- sitting_on: person on chair/sofa, strong vertical alignment
- standing_on: person on floor/ground, lighter criteria
- beside: side-by-side proximity (distance <= 0.15, low overlap)
- above: vertical gap 0.02-0.3, centered horizontally
- below: inverse of above
```

### PVSG Vocabulary File
Generated: `pvsg_yoloworld_prompt.txt`
Content: 100 most common object classes from PVSG
Format: Space-separated class names (compatible with `--yoloworld-prompt` flag)

## Monitoring Reprocessing

```bash
# Check progress in separate terminal
watch -n 5 'ls -la results/*/tracks.jsonl | wc -l'  # Should reach 20 eventually

# Once all reprocessed:
conda activate orion
python scripts/rebuild_all_graphs.py
python scripts/eval_sgg_filtered.py
```

## Known Limitations

1. **Semantic Relations**: Can't detect without VLM
   - looking_at (149 instances in GT)
   - talking_to (41 instances)
   - opening (147 instances)
   - Cannot be detected with spatial heuristics alone

2. **Detection Quality**: Depends on YOLO-World accuracy for each class
   - Some classes may still be missed (e.g., small objects)
   - May need fine-tuning of thresholds per class

3. **False Positives**: Aggressive thresholds create many spurious relations
   - May need temporal filtering (already implemented but not enabled)
   - Or confidence-based filtering

## Next Steps (After Reprocessing)

1. Check detection results: `python -c "import json; mem=json.load(open('results/0001_4164158586/memory.json')); print(set(o['class'] for o in mem['objects']))"`

2. Rebuild graphs: `python scripts/rebuild_all_graphs.py`

3. Evaluate: `python scripts/eval_sgg_filtered.py`

4. If R@20 < 20%: Debug specific videos to understand failures

5. If R@20 > 25%: Fine-tune `--on-overlap` and `--held-iou` thresholds

6. For >30%: May need VLM-based relation classification

## References

- PVSG Dataset: 400 videos, 65 predicate types, 126 object classes
- YOLO-World: Open-vocab object detection (any text prompt)
- Evaluation: Recall@K metric = matched triplets / GT triplets
- Script Args: `python -m orion.cli.run_showcase --help` for all options
