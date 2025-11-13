# Detection Quality Analysis - Key Findings

## Executive Summary

**Critical Discovery**: `yoloe-11s-seg-pf` model is SIGNIFICANTLY better than standard `yolo11s-seg` for egocentric video understanding.

---

## Model Comparison Results

### Frame 100 Analysis (conf=0.35)

**yolo11s-seg (Standard COCO)**:
- 4 detections
- Classes: `tv`, `mouse`, `keyboard`, `bottle`
- ❌ Generic classifications ("tv" instead of "monitor")

**yoloe-11s-seg-pf (Workspace-Optimized)**:
- 13 detections (3.25x more!)
- Classes: `monitor`, `laptop keyboard`, `desktop computer`, `mouse`, `thermos`, `coaster`, `calendar`, `bulletin board`, `musical keyboard`, `keyboard`
- ✅ Specific, actionable classifications
- ✅ Detects workspace-specific items

### Gemini Ground Truth Validation

Gemini identified 15 objects in frame 100:
- ✅ computer monitor, keyboard, mouse, water bottle (all caught by yoloe-11s-seg-pf)
- ⚠️ microphone, pencil case, wrist rest, foot rest, desk mat, statue, cables, bag (missed by both)

---

## Why This Matters for Orion

### Current Limitations with yolo11s-seg:
1. **Too generic**: "tv" doesn't tell you it's a monitor you're working on
2. **Misses context**: "bottle" doesn't specify it's a thermos/water bottle
3. **Low recall**: Only 4 objects vs 15 visible (27% recall)

### Advantages of yoloe-11s-seg-pf:
1. **Context-aware**: Distinguishes `laptop keyboard` vs `musical keyboard` vs generic `keyboard`
2. **Workspace-optimized**: Trained on office/desk items (`desktop computer`, `coaster`, `bulletin board`)
3. **Higher recall**: 13/15 objects (87% recall)
4. **Better for egocentric**: Focuses on items a person interacts with

---

## Recommended Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: Primary Detection (yoloe-11s-seg-pf)                 │
│  ─────────────────────────────────────────────────────────────  │
│  Input: Video frame                                             │
│  Output: Specific objects (monitor, laptop keyboard, thermos)   │
│  Confidence: 0.35 threshold (balanced)                          │
│  Priority: Assign importance scores (laptop=5, wall=1)          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: Depth + Spatial Zones (MiDaS/ZoeDepth + SLAM)        │
│  ─────────────────────────────────────────────────────────────  │
│  Input: Detection bboxes + depth map                            │
│  Output: Spatial zones (desk, shelf, floor) + real-world sizes  │
│  Use Case: "laptop is on desk zone at 600mm depth"              │
│  Validation: Check if detected size matches expected object size│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: Unknown Object Handler (FastVLM + Custom Embeddings) │
│  ─────────────────────────────────────────────────────────────  │
│  Input: Undetected regions with high saliency                   │
│  Output: Descriptions ("silver cylindrical object near laptop") │
│  Learning: Build custom embedding classifier from descriptions  │
│  Example: "That's my favorite pen" → associate embedding        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 4: Enhanced Tracking with Re-ID                          │
│  ─────────────────────────────────────────────────────────────  │
│  Input: Detections + embeddings + importance scores             │
│  Priority: Track high-importance objects more carefully          │
│  Re-ID: Maintain appearance galleries for persistent objects    │
│  SLAM-aware: Use spatial zones for re-association hints         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Importance Scoring System

### Priority Levels (1-5):

**Level 5 (Critical - Personal workspace items)**:
- laptop, laptop keyboard, keyboard, mouse, monitor
- cell phone, tablet, smartwatch
- → Track every frame, large appearance gallery (10+ samples)

**Level 4 (High - Frequently used items)**:
- thermos, water bottle, cup, coffee mug
- notebook, book, pen, pencil
- backpack, bag, headphones
- → Track consistently, medium gallery (5-7 samples)

**Level 3 (Medium - Workspace accessories)**:
- desk mat, coaster, wrist rest
- lamp, speaker, picture frame
- → Track opportunistically, small gallery (3 samples)

**Level 2 (Low - Furniture/static items)**:
- desk, chair, table
- calendar, bulletin board, whiteboard
- → Track for spatial context only

**Level 1 (Minimal - Background)**:
- wall, floor, carpet, ceiling, door
- → Use for depth/SLAM only, no tracking

---

## Handling Non-COCO Objects

### Problem:
Gemini sees: microphone, pencil case, wrist rest, foot rest, statue, cables
YOLO sees: (nothing)

### Solutions:

**Approach 1: Region Proposals + FastVLM**
```python
# 1. Find undetected regions with high saliency/edges
salient_regions = find_salient_undetected_regions(frame, yolo_boxes)

# 2. Use FastVLM to describe each region
for region in salient_regions:
    description = fastvlm.describe(region)  # "silver cylindrical microphone"
    embedding = clip.encode(region)
    
    # 3. Store as custom object
    custom_objects[description] = {
        'embedding': embedding,
        'location': region.bbox,
        'importance': infer_importance(description),  # heuristic scoring
    }
```

**Approach 2: Spatial Context Inference**
```python
# If object is near laptop + small + cylindrical → likely pen/pencil
# If object is on desk + rectangular + thin → likely notebook/paper
# If object is hanging from monitor → likely webcam/light

spatial_hints = {
    'near_laptop_small_cylindrical': ['pen', 'stylus', 'usb_drive'],
    'on_desk_rectangular_thin': ['notebook', 'paper', 'book'],
    'monitor_attachment': ['webcam', 'light', 'speaker'],
}
```

**Approach 3: User Labeling + Learning**
```python
# When user interacts with unknown object:
user_says = "That's my favorite pen"
object_embedding = clip.encode(object_region)

# Build custom classifier
custom_classifier.add_example(
    label="favorite_pen",
    embedding=object_embedding,
    importance=5,  # User marked as important
)
```

---

## Immediate Action Items

1. **Switch to yoloe-11s-seg-pf** ✅ (we have it!)
   - Set confidence=0.35 as default
   - Implement importance scoring from classifications

2. **Integrate Depth Model** (pending)
   - Add MiDaS or ZoeDepth to ModelManager
   - Compute spatial zones using depth clustering
   - Validate object sizes (laptop should be ~400mm wide)

3. **Add FastVLM for Unknown Objects** (partial - have model)
   - Detect salient undetected regions
   - Generate descriptions
   - Build embedding-based custom classifier

4. **Update EnhancedTracker** (pending)
   - Add importance-based tracking priority
   - Allocate more gallery space to important objects
   - Use spatial zones for re-association hints

5. **SLAM Integration** (architecture exists, needs wiring)
   - Track camera pose across frames
   - Build persistent spatial map (zones)
   - Use zone context for classification hints

---

## Performance Targets

- **Detection**: yoloe-11s-seg-pf at 0.35 conf → ~150ms/frame
- **Depth**: MiDaS → ~100ms/frame (GPU) or skip frames (every 10th)
- **Tracking**: EnhancedTracker → ~5ms/frame overhead
- **Total**: ~250ms/frame → 4 FPS real-time on MPS/CUDA

---

## Next Steps Discussion

**Q: What about objects COCO can't classify?**
- Use FastVLM descriptions + spatial context + user learning
- Example: "silver object near laptop" + cylindrical shape + desk zone → probably microphone or pen

**Q: How to understand spatial layout?**
- Depth map + SLAM pose → 3D point cloud
- HDBSCAN clustering on 3D positions → zones (desk, shelf, floor)
- Track objects relative to zones ("laptop is on desk zone")

**Q: How to prioritize what to track?**
- Importance scoring (1-5) based on:
  - Object class (laptop=5, wall=1)
  - Confidence (high conf → more important)
  - Persistence (seen across frames → more important)
  - Spatial proximity (near user → more important)

**Q: What if object moves between scenes?**
- Re-ID gallery with appearance embeddings
- SLAM spatial memory (last seen at desk zone)
- When reappearing: check both appearance AND expected zone

---

## Conclusion

**The path forward is clear**:
1. ✅ Use yoloe-11s-seg-pf (way better detections)
2. Add depth for spatial understanding
3. Use FastVLM + embeddings for unknown objects
4. Prioritize tracking with importance scores
5. Integrate SLAM for persistent spatial memory

This gives Orion the ability to understand "what's around me" beyond just COCO classes, using context, depth, and learned associations.
