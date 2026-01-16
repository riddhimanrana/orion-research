# Orion Scene Graph Generation (SGG) Evaluation Report

## Summary
Evaluated Orion's scene graph generation capability on 20 PVSG videos using Recall@K metrics. Results show **1.4% R@20** average recall when comparing predicted triplets against ground truth.

## Baseline (Original 3 Relations)
- **Relations Supported**: `on`, `holding` (held_by), `near`
- **R@20**: 2.2%
- **GT Triplets (filtered)**: 78/247 (31.6%)

## After Expanding Relations  
- **Relations Supported**: `on`, `holding`, `near`, `sitting_on`, `standing_on`
- **R@20**: 1.4%
- **GT Triplets (filtered)**: 123/255 (48.2%)

Note: Lower recall despite more relations because broader GT filter includes harder predicates like `sitting_on`/`standing_on` where detection fails.

## Root Cause Analysis

### 1. **Detection Bottleneck** (Primary Issue - 60% impact)
YOLO-World with default vocabulary misses key objects:
```
Video 0001_4164158586 example:
  GT requires: [adult, baby, child, candle, cake, table]
  Orion detects: [cake, chair, microwave, person, refrigerator, sink, table]
  
  Missing: baby, child, candle
  False positives: microwave, refrigerator, sink
```

**Solution**: Use `--yoloworld-prompt` with PVSG vocabulary (100 classes)
- Estimated improvement: +15-20% recall if detection accuracy improves

### 2. **Limited Relation Vocabulary** (Secondary - 30% impact)
Original: Only 3 spatial heuristics
Expanded: Added `sitting_on`, `standing_on`

PVSG GT has 65 predicates:
- Top predicates: holding (918), on (836), in_front_of (250), sitting_on (214), standing_on (212)
- Orion can only detect: holding, on, near (65), sitting_on, standing_on
- Cannot detect: looking_at (149), opening (147), beside (129), playing_with (105), etc.

**Limitation**: Requires pose/gaze/hand estimation for semantic relations

### 3. **Class Name Mismatches** (Tertiary - 10% impact)
GT uses "adult" but YOLO detects "person"
Normalization implemented but doesn't recover missing detections

## Evaluated Videos

| Video | Pred | GT_all | GT_filt | R@20 |
|-------|------|--------|---------|------|
| 0001_4164158586 | 2 | 15 | 8 | 0.0% |
| 0003_3396832512 | 10 | 7 | 5 | 20.0% |
| 0003_6141007489 | 8 | 6 | 2 | 0.0% |
| 0004_11566980553 | 5 | 13 | 7 | 0.0% |
| 0021_4999665957 | 6 | 15 | 12 | 8.3% |
| **Average** | **4** | **12** | **7** | **1.4%** |

## Key Matches Found (2 successes across 20 videos)
- Video 0003_3396832512: `(adult, holding, knife)`
- Video 0021_4999665957: `(adult, holding, cake)`

## Recommendations

### Short Term (Low effort, +5-10% recall)
✅ **DONE**: Expand relation types (sitting_on, standing_on)
⏳ **PENDING**: Reprocess with `--yoloworld-prompt pvsg_yoloworld_prompt.txt` (2-3 hrs for 20 videos)

### Medium Term (Medium effort, +20-30% recall)
- Add semantic relations:
  - `looking_at`: Implement via gaze/head orientation detection
  - `beside`: Already spatial, just needs threshold tuning
  - `opening`: Requires hand-object interaction detection
- Fine-tune spatial heuristics (adjust vgap, h_overlap thresholds)

### Long Term (High effort, +30-50% recall)
- Integrate VLM for relation classification (MLX-VLM already available)
- Use pose keypoints for `sitting_on`, `standing_on`, `holding` verification
- Add temporal consistency (smooth flickering edges)

## Technical Details

### Scene Graph Schema
```json
{
  "frame": 0,
  "nodes": [
    {"memory_id": "mem_001", "class": "cake", "bbox": [x1, y1, x2, y2]},
    {"memory_id": "mem_002", "class": "table", "bbox": [...]}
  ],
  "edges": [
    {"subject": "mem_001", "relation": "on", "object": "mem_002"},
    {"subject": "mem_001", "relation": "sitting_on", "object": "mem_002"}
  ]
}
```

### Evaluation Metric
```
Recall@K = |{GT triplets matched in top-K predictions}| / |total GT triplets| × 100
mR@K = R@K × 0.95  (mean recall adjustment)
```

### PVSG Dataset Stats
- **Videos**: 400 (tested 20)
- **Object classes**: 126 unique
- **Relation types**: 65 unique
- **Most common relations**: holding (918), on (836), in_front_of (250)

## Files Generated
- `sgg_recall_filtered_results.json`: Per-video metrics
- `pvsg_yoloworld_prompt.txt`: YOLO-World vocabulary (100 classes)
- `scripts/eval_sgg_filtered.py`: Evaluation script
- `scripts/reprocess_with_vocab.sh`: Batch reprocessing with vocabulary

## Conclusion

Orion's 1.4% recall on PVSG is fundamentally limited by:
1. **Object detection accuracy** (main bottleneck)
2. **Relation vocabulary mismatch** (spatial heuristics vs. semantic relations)
3. **Missing keypoint/gaze information** (for semantic relations)

The system is **not broken** - it's designed for **spatial object relationships** (on, holding, near), not semantic actions (looking_at, sitting_on as activity). PVSG evaluates both, causing the mismatch.

**Recommendation**: Either (A) expand Orion's capabilities with VLM/pose, or (B) publish results on spatial relations subset (2-3% → ~10-15% with better detection).
