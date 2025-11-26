# Gemini Validation Results - Scene Graph Relations

**Validated**: 2025-11-16 21:41:47  
**Total Samples**: 8 frames  
**Total Relations**: 10  

## Overall Results

✅ **CORRECT**: 8 (80.0%)  
❌ **INCORRECT**: 2 (20.0%)  
⚠️ **AMBIGUOUS**: 0 (0.0%)  

## Quality Distribution

- **HIGH**: 5 frames (62.5%)
- **MEDIUM**: 1 frame (12.5%)
- **LOW**: 2 frames (25.0%)

## Detailed Findings

### ✅ Correct Relations (8)

1. **Frame 665**: `near` (book ↔ bed) - "The book is near the bed"
2. **Frame 670**: `near` (book ↔ bed) - "The book is near the bed"  
3. **Frame 710**: `held_by` (book → person) - "The book is being held by a person"
4. **Frame 775**: `held_by` (book → person) - "The person is holding the book"
5. **Frame 1040**: `held_by` (book → person) - "The book is held by the person"
6. **Frame 1050**: `held_by` (book → person) - "The book is held by the person"

### ❌ Incorrect Relations (2)

1. **Frame 785**: `held_by` (book → mem_008)
   - **Issue**: Wrong person ID - book is held by mem_009, not mem_008
   - **Root cause**: Multiple people in frame, IoU overlap picked wrong person
   - **Fix**: Need person disambiguation logic or higher IoU threshold

2. **Frame 1905**: `on` (tv → keyboard)
   - **Issue**: TV is on a stand/table, not directly on keyboard
   - **Root cause**: Vertical alignment + horizontal overlap detected, but intermediate object (table) not considered
   - **Fix**: Need multi-hop spatial reasoning or stricter vertical gap threshold

## Key Insights

### Strengths
- ✅ `near` relations: 100% accurate (4/4)
- ✅ `held_by` with single person: 80% accurate (4/5)
- ✅ Class filtering worked: No "bed held_by person" false positives
- ✅ Debouncing prevented noise

### Weaknesses
- ❌ `held_by` with multiple people: Needs person ID disambiguation
- ❌ `on` detection: Doesn't handle intermediate surfaces
- ⚠️ Missing common relations: Gemini flagged missing "book on bed" in multiple frames

## Commonly Missing Relations

Identified by Gemini across samples:
- **"book on bed"** (frames 665, 670, 710) - Threshold too strict for `on` relation
- **"person on bed"** (frame 710) - Same issue
- **"book held_by correct person"** (frame 785) - Need multi-person handling

## Recommended Improvements

### 1. Relax `on` Relation Thresholds
Current: `on_h_overlap=0.3, on_vgap_norm=0.02`  
**Suggested**: `on_h_overlap=0.2, on_vgap_norm=0.05`

Rationale: Gemini flagged multiple missing "on" relations that should be detected

### 2. Add Person Disambiguation for `held_by`
When multiple persons present:
- Prefer person with highest IoU with object
- OR: Require object centroid inside person bbox (stricter)
- OR: Use hand keypoint detection if available

### 3. Add Intermediate Object Reasoning
For `on` relations:
- Check for support surfaces between subject and object
- If gap > threshold, look for intermediate objects
- Emit chain: A on B, B on C

### 4. Tune Thresholds Based on Validation

| Relation | Current | Suggested | Reason |
|----------|---------|-----------|--------|
| `near_dist_norm` | 0.08 | 0.08 | Working well ✅ |
| `on_h_overlap` | 0.3 | 0.2 | Too strict, missing valid cases |
| `on_vgap_norm` | 0.02 | 0.05 | Same as above |
| `held_by_iou` | 0.3 | 0.35 | Reduce multi-person confusion |

## Files Generated

- `results/video_validation/scene_graph.jsonl` - 197 frames, 11 edges
- `results/video_validation/graph_summary.json` - Aggregate stats
- `results/video_validation/gemini_feedback.json` - Full validation details
- `results/video_validation/graph_samples/` - 8 annotated frames

## Next Actions

1. ✅ Apply class filtering (DONE - removed 4 false positives)
2. ⏳ Tune `on` relation thresholds for better recall
3. ⏳ Add multi-person handling for `held_by`
4. ⏳ Consider spatial reasoning chains (A on B on C)
5. ⏳ Re-validate after changes

## Commands to Re-run with Tuned Params

```bash
# Relaxed on relation
python -m orion.cli.run_scene_graph --results results/video_validation \
  --on-overlap 0.2 --on-vgap 0.05

# Stricter held_by to reduce multi-person errors
python -m orion.cli.run_scene_graph --results results/video_validation \
  --held-iou 0.4

# Re-export and re-validate
rm -rf results/video_validation/graph_samples
python scripts/export_graph_samples.py --results results/video_validation \
  --video data/examples/video.mp4 --output results/video_validation/graph_samples
python scripts/gemini_validate_relations.py --samples results/video_validation/graph_samples \
  --output results/video_validation/gemini_feedback_v2.json
```

## Accuracy by Relation Type

| Relation | Correct | Incorrect | Accuracy |
|----------|---------|-----------|----------|
| `near` | 4 | 0 | 100% ✅ |
| `held_by` | 4 | 1 | 80% |
| `on` | 0 | 1 | 0% ❌ |

**Overall**: 80% accuracy is good for initial implementation with automated validation!
