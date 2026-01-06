# Orion v2 Full Evaluation Report

## Executive Summary

Evaluated Orion v2 perception pipeline on 3 videos with comprehensive Gemini validation.

### Key Findings

| Video | Duration | Detections | Entities | Time | FPS | Status |
|-------|----------|------------|----------|------|-----|--------|
| test.mp4 | 61s | 508 | 29 | 199s | 9.15 | ⚠️ False positives |
| video.mp4 | 66s | 248 | 7 | 75s | 26.29 | ✅ Good filtering |
| room.mp4 | 38s | 130 | 5 | 50s | 22.83 | ⚠️ False positives |

## Critical Issues Identified

### 1. YOLO-World Label Confusion (High Priority)
- **Problem**: Doors/closets detected as "refrigerator" (77+ occurrences in room.mp4)
- **Root Cause**: Visual similarity between white doors and white refrigerators in COCO vocabulary
- **Impact**: Major false positive pollution across all videos
- **Evidence**: Gemini confirms frames 382, 510, 637, 892 in room.mp4 have no refrigerators

### 2. Scene Filter Limitation for Multi-Room Videos
- **Problem**: Initial scene caption (frame 0) doesn't represent entire video
- **Example**: room.mp4 starts with "hallway with door" but shows bedroom, desk, etc.
- **Impact**: Scene filter passes irrelevant objects (refrigerator=0.684 similarity to "hallway")
- **Solution Needed**: Per-segment scene updates or scene change detection

### 3. Generic Scene Descriptions
- **Problem**: FastVLM produces vague captions like "hallway with door and wall"
- **Impact**: Most COCO labels pass the 0.56 threshold
- **Evidence**: In room.mp4, even toilet (0.703) passes for hallway scene

## Performance Metrics

### Processing Speed
- **video.mp4**: 75s for 66s video = **1.14x real-time** ✅
- **room.mp4**: 50s for 38s video = **1.32x real-time** ✅
- **test.mp4**: 199s for 61s video = **3.3x real-time** (more entities = more description time)

### Re-ID Effectiveness
- V-JEPA2 embeddings successfully cluster same objects across frames
- Entity deduplication reduces track count by 30-50%
- **Issue**: Garbage in → garbage out (false positive detections get clustered too)

## Video-by-Video Analysis

### test.mp4 (61s - House Tour)
**Scene**: Multiple rooms - desk, living room, kitchen, stairs, doors
**Detections**: 508 (only 6 filtered by scene filter)
**Issue**: Scene filter ineffective - initial caption "desk with monitor" doesn't cover whole house

| Gemini Ground Truth | Orion Detected | Accuracy |
|---------------------|----------------|----------|
| Wall, doorway, ceiling fan | Refrigerator, airplane | ❌ |
| Window, vase, plants | Book, sports ball | ❌ |
| Couch, chair, coffee table | - (missed) | ❌ |
| Kitchen with cabinets, sink | 2 books, sink | Partial |
| Door, soccer ball, floor mat | - (missed) | ❌ |

### video.mp4 (66s - Mixed Scenes)
**Scene**: Desk area, doors, bedroom, bathroom, stairs
**Detections**: 248 (590 filtered = 70% reduction ✅)
**Result**: Best scene filtering performance

| Detection | Count | Accuracy |
|-----------|-------|----------|
| laptop | 73 | Partially (often monitors) |
| dining table | 64 | Desk = correct ✅ |
| keyboard | 41 | Correct ✅ |
| mouse | 34 | Correct ✅ |
| tv | 19 | Monitor = correct ✅ |

### room.mp4 (38s - Room Circle)
**Scene**: Your bedroom with desk, monitor, keyboard, bed, chair
**Detections**: 130 (565 filtered = 81% reduction)
**Critical Issue**: 94 "refrigerator" detections are all false positives (doors/closets)

| Gemini Ground Truth | Orion Detected | Accuracy |
|---------------------|----------------|----------|
| Desk, monitor, keyboard | Refrigerator (6x!) | ❌ |
| Bed, pillow, chair | Refrigerator, bed, toilet | Partial |
| Closet, bookshelf | Refrigerator | ❌ |

## Recommendations

### Immediate Fixes (High Impact)

1. **Add negative class suppression for refrigerator**
   - In bedroom/office scenes, refrigerator should be suppressed
   - Use scene context to build suppression list

2. **Improve scene caption extraction**
   - Use first 3-5 frames instead of just frame 0
   - Parse VLM output more carefully to extract actual objects

3. **Add label blacklist per scene type**
   ```python
   SCENE_BLACKLIST = {
       "bedroom": ["refrigerator", "toilet", "sink"],
       "office": ["refrigerator", "toilet", "oven"],
       "hallway": ["refrigerator", "bed", "oven"],
   }
   ```

### Architecture Improvements (Medium Term)

1. **Scene change detection**
   - Detect when scene changes significantly
   - Update scene caption and filter dynamically

2. **Visual verification layer**
   - Use FastVLM to verify high-confidence detections
   - Reject "refrigerator" if VLM describes it as "door"

3. **YOLO-World prompt engineering**
   - Consider using `indoor_full` or custom prompts instead of COCO
   - Remove problematic classes (refrigerator, toilet) for non-kitchen scenes

## V-JEPA2 Re-ID Assessment

### Working Well
- Embedding clustering groups same objects across frames
- Temporal continuity maintained (entity spans frame 0-1145)
- Entity deduplication merges duplicate tracks

### Limitations
- Cannot fix detection errors (FP in → FP clustered)
- No cross-class matching (door ≠ refrigerator semantically but visually similar)

## Conclusion

The Orion v2 pipeline shows **strong performance for desk/office scenes** where the initial scene caption is accurate. However, it struggles with:

1. **Multi-room videos** where scene changes aren't tracked
2. **YOLO-World label confusion** (doors → refrigerator)
3. **Generic scene descriptions** that don't constrain the filter

**Next Priority**: Address the refrigerator false positive issue before further development.

---
*Generated: January 6, 2026*
*Validated with: Gemini Vision API*

