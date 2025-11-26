# Scene Graph Tuning Summary

## Overview
Iterative tuning of spatial relation heuristics with automated Gemini validation.

## Progression

### v2 (Initial Adaptive Heuristics)
**Configuration:**
- Adaptive "on" thresholds for small subjects
- `on_overlap=0.2`, `on_vgap=0.05`, `held_iou=0.4`
- Subject-overlap constraint active

**Results:**
- Total relations: 9
- CORRECT: 6 (66.7%)
- INCORRECT: 3 (33.3%)
- Missing: 4

**Issues:**
- chair→bed false positives (all 3 incorrect)
- held_by misassignment (wrong person)

---

### v3 (Tighter "on" Thresholds)
**Configuration:**
- Tightened global thresholds: `on_overlap=0.3`, `on_vgap=0.03`
- Subject-overlap requirement: `on_subj_overlap=0.6`
- Small-subject relaxation: `on_small_area=0.04`, `on_small_overlap=0.2`, `on_small_vgap=0.06`

**Results:**
- Total relations: 9
- CORRECT: 7 (77.8%)
- INCORRECT: 2 (22.2%)
- Missing: 8

**Improvements:**
- Reduced chair→bed false positives from 3 to 1
- Better precision overall

**Remaining Issues:**
- chair→bed: frame 595 still flagged (next to, not on)
- held_by misassignment: frame 785 wrong person

---

### v4 (Subject-Within + Unique Best-Person) ⭐ **FINAL**
**Configuration:**
- Added subject-within-object constraint: `on_subj_within=0.7` (70% of subject x-span must be inside object x-span)
- Unique best-person selection for `held_by`: picks person with highest IoU/centroid score per subject
- Maintained v3 thresholds

**Results:**
- Total relations: 9
- CORRECT: 7 (77.8%)
- INCORRECT: 2 (22.2%)
- Missing: 8

**Analysis:**
- Precision maintained at 77.8%
- False positives:
  - chair→bed (frame 595): persistent spatial ambiguity; may need manual class rules
  - held_by person ambiguity (frame 785): multiple people close to object; may need pose/hand detection
- Recall gaps (8 missing):
  - "on" for objects partially occluded or small (keyboard/tv on table, book on bed)
  - "near" for TV-keyboard pairs (threshold too conservative?)
  - held_by for correct person when best-score heuristic picks wrong one

---

## Recommendations for Next Phase

### High Priority
1. **Class-specific "on" rules:**
   - Disable chair→bed by adding explicit class-pair exclusions
   - Example: `if (subject_class == "chair" and object_class == "bed"): skip`

2. **Pose-aware held_by:**
   - Integrate hand keypoint detection (MediaPipe/OpenPose) to disambiguate multi-person holding
   - Fallback to current IoU/centroid for single-person frames

3. **Adaptive "near" threshold:**
   - Lower `near_dist_norm` to 0.06 for small objects (keyboard, mouse, remote)
   - Keep 0.08 for larger objects

### Medium Priority
4. **On-chain reasoning:**
   - Support transitive "on" (A on B on C) for stacked objects
   - Example: keyboard on table, table on floor → detect both

5. **Occlusion handling:**
   - Relax `on_vgap` for small subjects with high overlap (likely stacked but partially hidden)

### Low Priority (Ego4D dataset testing)
6. **Multi-video validation:**
   - Run on Ego4D EASG dataset to measure generalization
   - Track precision/recall across diverse scenes (kitchen, office, outdoor)

---

## Command for Current Best Configuration
```bash
python -m orion.cli.run_scene_graph \
  --results results/video_validation \
  --on-overlap 0.3 \
  --on-vgap 0.03 \
  --on-subj-overlap 0.6 \
  --on-subj-within 0.7 \
  --on-small-area 0.04 \
  --on-small-overlap 0.2 \
  --on-small-vgap 0.06
```

---

## Gemini Validation Details

### Correct Relations (7/9)
- frame 10: mouse on keyboard ✓
- frame 20: mouse on keyboard, mouse near keyboard ✓
- frame 40: mouse on keyboard ✓
- frame 50: tv on keyboard ✓
- frame 665: person (mem_008) held_by person (mem_009) ✓ *(Note: This may be a Gemini mislabel; "person held_by person" is semantically odd)*
- frame 710: person (mem_008) held_by person (mem_009) ✓ *(Same note)*

### Incorrect Relations (2/9)
- **frame 595:** chair→bed flagged as "on"
  - Reason: chair is next to/behind bed, not on it
  - Root cause: subject-within threshold still too permissive for furniture pairs
- **frame 785:** book held_by wrong person
  - Reason: mem_008 detected, but mem_009 is holding the book
  - Root cause: IoU/centroid heuristic picks closer person; need hand proximity

### Missing Relations (8 across 4 frames)
- **frame 20 (2 missing):**
  - keyboard on table
  - tv on table
- **frame 40 (3 missing):**
  - keyboard near tv
  - tv behind keyboard *(Note: "behind" not currently modeled; would need depth/occlusion)*
  - tv behind mouse
- **frame 710 (2 missing):**
  - person on bed
  - book on bed
- **frame 785 (1 missing):**
  - book held_by correct person (mem_009)

---

## Summary Statistics

| Version | Correct | Incorrect | Precision | Missing | Notes |
|---------|---------|-----------|-----------|---------|-------|
| v2      | 6       | 3         | 66.7%     | 4       | Initial adaptive |
| v3      | 7       | 2         | 77.8%     | 8       | Tighter thresholds |
| v4      | 7       | 2         | 77.8%     | 8       | + subject-within + unique held_by |

**Key Insight:** v3→v4 maintained precision while adding robustness (unique held_by selection prevents multi-person duplication). Recall remains limited by conservative thresholds and lack of occlusion/pose reasoning.

---

## Next Steps
1. Add class-pair exclusion rules for known false positives (chair→bed)
2. Test on Ego4D EASG dataset with current config
3. Implement pose-based held_by if multi-person misassignment persists
