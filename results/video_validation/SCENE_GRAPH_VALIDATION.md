# Scene Graph Validation Summary

## Generated Outputs

### Video Validation Episode
- **Total frames with detections**: 197
- **Frames with relations**: 12
- **Average nodes per frame**: 1.86
- **Average edges per frame**: 0.08

### Relation Counts
- `near`: 4 instances
- `held_by`: 6 instances  
- `on`: 5 instances

## Sample Validations

### Frame 710: Person Holding Objects
**Scene**: Person in frame with bed and book

**Detected Relations**:
1. `mem_010 (bed) --[held_by]--> mem_008 (person)`
   - IoU: 0.301
   - Normalized distance: 0.298
   - Horizontal overlap: 1.000
   - Vertical gap: 0.538

2. `mem_014 (book) --[held_by]--> mem_008 (person)`
   - IoU: 0.341
   - Normalized distance: 0.129
   - Horizontal overlap: 1.000
   - Vertical gap: 0.433

**Analysis**: 
- Book held_by person: **VALID** (IoU 0.34, close proximity)
- Bed held_by person: **QUESTIONABLE** - High overlap but bed is likely a support surface, not held

### Frame 1075: On Relation
**Detected Relations**:
1. `mem_011 (bed) --[on]--> mem_014 (book)`
   - IoU: 0.000 (no overlap)
   - Horizontal overlap: 0.869
   - Vertical gap: 0.002 (very small)

**Analysis**: 
- Bed on book: **LIKELY INVERTED** - Should be "book on bed" based on semantics
- Metrics look valid for an "on" relation, but direction may need review

### Frames 665, 670: Near Relations
**Detected Relations**: bed near book (bidirectional)
- IoU: ~0.08 (minimal overlap)
- Normalized distance: 0.047-0.067
- Both nodes present in same scene

**Analysis**: **VALID** - Objects in same space with proximity

## Tuning Recommendations

### Current Thresholds
```python
near_dist_norm = 0.08      # Normalized centroid distance
on_h_overlap = 0.3         # Horizontal overlap ratio
on_vgap_norm = 0.02        # Vertical gap normalized
held_by_iou = 0.3          # IoU threshold
iou_exclude = 0.1          # Exclude overlaps from 'near'
```

### Suggested Adjustments

1. **Add class constraints for `held_by`**:
   - Exclude large furniture (bed, couch, table) as subjects
   - Limit to portable objects (book, bottle, phone, etc.)

2. **Fix `on` relation directionality**:
   - Use semantic constraints: smaller/movable object should be subject
   - Or use Y-coordinate consistently: higher object "on" lower object

3. **Tighten `held_by` for person**:
   - Current IoU 0.3 catches large overlaps (person behind furniture)
   - Consider requiring subject centroid inside person bbox + minimum size constraint

4. **Add confidence scores**:
   - Weight by IoU for `held_by`
   - Weight by proximity for `near`
   - Weight by overlap + gap for `on`

## Next Steps for Validation

1. **Visual inspection**: Review exported JPGs in `results/video_validation/graph_samples/`
2. **Gemini verification**: Send sample frames + detected relations for ground truth validation
3. **Iterate thresholds**: Adjust based on false positives/negatives
4. **Add semantic rules**: Class-based relation constraints (e.g., "bed cannot be held_by person")

## Files to Review
- `results/video_validation/graph_samples/frame_000710.jpg` - held_by examples
- `results/video_validation/graph_samples/frame_001075.jpg` - on relation
- `results/video_validation/graph_samples/frame_000665.jpg` - near relation

## Commands to Re-run with Tuned Thresholds

```bash
# Stricter held_by (higher IoU)
python -m orion.cli.run_scene_graph --results results/video_validation --held-iou 0.5

# Tighter near (smaller distance)
python -m orion.cli.run_scene_graph --results results/video_validation --near-dist 0.05

# Require more overlap for on
python -m orion.cli.run_scene_graph --results results/video_validation --on-overlap 0.5 --on-vgap 0.01
```
