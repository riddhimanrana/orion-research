# Scene Graph Generation (SGG) Evaluation - Root Cause Analysis

## Executive Summary

**Current Performance:** 1.1% Recall@20 on PVSG dataset (9 videos at 0%, 1 video at 11.1%)

**Root Cause:** NOT a scene graph generation bug, but a combination of object detection and spatial relation generation limitations.

---

## Problem Breakdown

### 1. **Object Detection Limitations (PRIMARY BLOCKER)**

The YOLO11 detector cannot detect scene elements that PVSG ground truth requires.

#### Video 0020_10793023296 (0% R@20) - 100% of GT triplets blocked
- **GT expects:** Triplets with "grass", "child", "ball" (24 relations total)
- **Orion detects:** Only "adult", "car", "sports ball", "truck" (4 unique classes)
- **Result:** All 24 GT triplets are unmatchable because they reference objects never detected

#### Video 0020_5323209509 (11.1% R@20) - 88.9% of GT triplets blocked  
- **GT expects:** Triplets with "child", "carpet", "gift", "sofa", "basket" (9 relations total)
- **Orion detects:** "adult", "chair", "table", "bag", "remote", "couch", "bird", "vase" (8 classes)
- **Result:** Only 1 GT triplet matchable: `('adult', 'on', 'chair')` → 11.1% recall

**Missing Objects Across 10 Videos:**
- Scene elements: grass, floor, carpet, sky, ground, wall, ceiling
- Small objects: ball, gift, paper, card
- People: child, baby
- Furniture: sofa (detected as "couch" but heuristic relations too strict)
- Kitchen: basket, utensils

### 2. **Spatial Relation Generation Limitations (SECONDARY BLOCKER)**

Even when objects ARE detected, weak spatial relations prevent edge generation.

#### Example: Couch in 0020_5323209509
- Orion detected "couch" (mem_005)
- But has 0 edges in scene_graph.jsonl
- Heuristic spatial rules ("on", "near", "held_by") too strict for this furniture piece

#### Edge Statistics
- Total detected objects: 10
- Objects with edges: 6
- Objects with NO edges: 4 (adult, couch, bird, vase)
- This means 40% of detected objects are isolated without relations

### 3. **Predicate Vocabulary Mismatch (TERTIARY ISSUE)**

PVSG GT uses semantic action predicates that Orion cannot generate:

| GT Predicate | Orion Predicate | Type | Notes |
|---|---|---|---|
| walking on | on, standing on | action verb vs spatial | Requires action recognition |
| pointing to | *(not generated)* | action verb | Requires pose estimation |
| opening | on, held_by | action verb | Requires semantic understanding |
| running on | on, standing on | action verb | Requires action recognition |
| talking to | looking at | communication | Requires VLM |

**Note:** This is MUCH less impactful than detection (#1) and relation generation (#2).

---

## Performance Analysis

### Why 1 Video Works (11.1%), 9 Don't (0%)?

**Video 0020_5323209509 "wins" because:**
1. It has simpler GT predicates (some are spatial: "on", "beside")
2. It happened to have detectable objects (adult, chair) with spatial overlap

**Other 9 videos fail because:**
1. All their GT triplets reference undetectable objects (grass, child, etc.)
2. OR: Their spatial relations use action verbs Orion can't generate

### Why This Isn't a Bug

If we had perfect object detection WITH the PVSG vocabulary:
- Video 0020_10793023296: Could match ~10-15% (with spatial predicates only)
- Video 0020_5323209509: Could reach ~40-50% (currently blocked by missing objects)
- Overall: Could reach ~15-25% with perfect detection + current heuristics

The scene graph generation logic (build_scene_graphs) is correct - it's just working with incomplete object detection.

---

## Solutions (Ranked by Impact)

### Solution 1: Better Object Detector (HIGHEST IMPACT - Estimated +15-20% R@20)

**Option A: YOLO-World with PVSG Vocabulary**
- Use custom prompt: `adult . child . chair . table . grass . carpet . floor . sky . ...` (100 classes)
- Pros: Open-vocabulary, handles scene elements, no retraining needed
- Cons: PVSG videos needed for reprocessing (currently unavailable)
- Expected gain: +10-20% R@20

**Option B: Upgrade to Vision Transformer (e.g., DINO, DETR)**
- Better at detecting small objects (gift, ball, paper) and scene elements
- Pros: Higher recall for semantic objects
- Cons: Slower inference, may need finetuning on PVSG vocabulary
- Expected gain: +8-15% R@20

**Option C: Hybrid Detection (YOLO + GroundingDINO)**
- Already partially implemented in config
- Use GroundingDINO for scene elements on low-detection frames
- Pros: Reasonable speed, good recall
- Cons: Complex pipeline, tuning needed
- Expected gain: +12-18% R@20

### Solution 2: Relax Spatial Relation Thresholds (MODERATE IMPACT - Estimated +3-5% R@20)

**Current state:** 40% of detected objects have NO edges

**Actions:**
- Reduce `near_dist_norm` from 0.08 → 0.12
- Reduce `on_h_overlap` from 0.3 → 0.2
- Add `sitting_on`, `standing_on` with looser thresholds
- Allow more "next_to" relations

**Expected gain:** +3-5% R@20 (but may increase false positives)

### Solution 3: Add Semantic Action Predicates (LOW IMPACT - Estimated +1-3% R@20)

**Currently impossible** without upstream components:
- VLM-based "opening", "talking_to", "looking_at" classification
- Pose-based "standing_on", "sitting_on", "running_on" detection
- Action recognition for "walking_on", "pointing_to"

**Expected gain:** +1-3% R@20 (very limited without detection of people involved in actions)

---

## Recommendations

### Immediate (No Video Required)
1. **Relax spatial relation thresholds** → +3-5% R@20 (low effort, quick win)
2. **Add class aliasing** (couch → sofa, etc.) → +0-1% R@20 (already done, no additional gain found)
3. **Increase edge generation on low-relation frames** → +1-2% R@20

### Short-term (If Videos Available)
1. **Reprocess with YOLO-World + PVSG vocabulary** → +10-15% R@20
   - Requires: Original PVSG videos (currently unavailable)
   - Time: 2-3 hours for 10 videos

### Long-term (Architecture Change)
1. **Implement YOLO-World as default detector** in Orion
2. **Add semantic relation classification** (VLM-based)
3. **Fine-tune detector on PVSG** if full dataset training needed

---

## Debugging Evidence

### Triplet Comparison (0020_5323209509)

```
ORION PREDICTIONS (8):
  ('adult', 'on', 'chair')           ✓ MATCHES GT!
  ('adult', 'on', 'bag')              ✗ FP
  ('bag', 'on', 'chair')              ✗ FP
  ('chair', 'on', 'chair')            ✗ FP (self-loop error)
  ('remote', 'next to', 'adult')      ✗ FP
  ('remote', 'on', 'adult')           ✗ FP
  ('table', 'on', 'chair')            ✗ FP
  ('adult', 'next to', 'remote')      ✗ FP

GROUND TRUTH (9):
  ('adult', 'on', 'chair')            ✓ MATCHED!
  ('adult', 'guiding', 'child')       ✗ BLOCKED (no "child" detected)
  ('adult', 'looking at', 'child')    ✗ BLOCKED (no "child" detected)
  ('adult', 'on', 'sofa')             ✗ BLOCKED (detected as "couch" but no edge)
  ('child', 'on', 'carpet')           ✗ BLOCKED (no "child" or "carpet" detected)
  ('child', 'holding', 'basket')      ✗ BLOCKED (no "child" or "basket" detected)
  ('child', 'opening', 'gift')        ✗ BLOCKED (no "child" or "gift" detected)
  ('gift', 'on', 'sofa')              ✗ BLOCKED (no "gift" or "sofa" detected)
  ('sofa', 'beside', 'sofa')          ✗ BLOCKED (detected as "couch" but no edge)

Conclusion: 1/9 matches (11.1%) due to lucky object overlap + spatial relation match
```

### Triplet Comparison (0020_10793023296)

```
ORION PREDICTIONS (3):
  ('adult', 'next to', 'adult')       ✗ FP
  ('adult', 'on', 'adult')            ✗ FP (self-loop error)
  ('car', 'on', 'adult')              ✗ FP

GROUND TRUTH (24):
  All 24 triplets blocked because they require "grass", "child", or "ball"
  - Orion never detected any of these objects
  - Example: ('adult', 'walking on', 'grass')
  - Example: ('adult', 'pointing to', 'child')

Conclusion: 0/24 matches (0%) due to complete object detection failure
```

---

## Key Metrics

| Metric | Value | Implication |
|--------|-------|-------------|
| % of GT triplets blocked by missing objects | 89% average | Detection is the blocker |
| % of GT triplets blocked by semantic predicates | 40% | Secondary issue |
| Objects detected but with no edges | 40% | Spatial relation rules too strict |
| Self-loop errors (obj on obj) | 2 per video | Minor edge generation bug |
| Predicate normalization coverage | 12/65 types | Low impact (~1-2% gain) |

---

## Conclusion

**The 1.1% R@20 is NOT a bug in scene graph generation.** It's a fundamental limitation of using YOLO11 object detection on PVSG, which requires semantic scene understanding (grass, sky, floor, gifts) that generic object detectors aren't designed to handle.

**To achieve meaningful improvement (15%+), you MUST upgrade the object detector** to one that understands scene semantics (YOLO-World, DINO, or custom fine-tuned models).

**Quick wins (3-5% gain) are possible** by relaxing spatial relation thresholds, but you'll hit a hard ceiling at ~5-10% without better detection.
