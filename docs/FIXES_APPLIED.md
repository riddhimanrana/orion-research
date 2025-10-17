# Critical Fixes Applied - Addressing Hallucinations

## What Was Wrong

### 1. **Biased Prompts → Hallucinations**
```python
# OLD (WRONG):
prompt = f"Describe this {class_name} in detail..."
# Problem: If YOLO says "bottle", FastVLM is primed to see a bottle
```

**Result:** Chain of errors
- YOLO misclassifies pixels → FastVLM forced to describe the wrong thing
- Your results: "bird in flight" (no bird), "wine bottle" (not wine), "TV" (it's a monitor)

### 2. **Over-Sensitive State Detection**
- Threshold: 0.90 cosine similarity
- **Too sensitive!** Triggers on: camera shake, angle changes, lighting, minor occlusion
- Result: 23 "state changes" for a static bottle

### 3. **No Quality Filtering**
- Described everything, even low-confidence junk detections
- Wasted FastVLM compute on false positives

## Fixes Applied

### Fix #1: Unbiased Open-Ended Descriptions ✅

**New approach:**
```python
# DON'T mention YOLO class - let FastVLM decide what it sees
prompt = """What do you see in this image? Provide a detailed description.

Focus on:
- What type of object this is
- Its appearance, color, and shape
- Any distinguishing features
"""
```

**Verification added:**
- After description, check if FastVLM agrees with YOLO
- If disagreement + low confidence → flag as "potential misclassification"
- Warnings logged for human review

**Example flow:**
```
YOLO: "bird" (conf=0.35)
FastVLM: "I see a yellow triangular shape..."
System: ⚠️  "Potential misclassification: YOLO said 'bird', but FastVLM sees..."
```

### Fix #2: Much Less Sensitive State Detection ✅

**Changed threshold:**
```python
# OLD:
STATE_CHANGE_THRESHOLD = 0.90  # Way too sensitive

# NEW:
STATE_CHANGE_THRESHOLD = 0.75  # Only major visual changes
```

**What this means:**
- **0.90**: Detects tiny changes (lighting, slight movement) = noisy
- **0.75**: Only detects significant changes (different pose, occlusion, major movement)

**Expected improvement:**
- Bottle: 23 changes → 2-3 changes (only when picked up/put down)
- Keyboard: 10 changes → 0-1 changes (should be mostly stable)

### Fix #3: Confidence-Based Filtering ✅

**New thresholds:**
```python
HIGH_CONFIDENCE_THRESHOLD = 0.6  # Trust YOLO above this
LOW_CONFIDENCE_THRESHOLD = 0.4   # Skip below this (false positives)
```

**Logic:**
```python
if confidence < 0.4:
    # Likely false positive - don't waste FastVLM on it
    description = "Low confidence detection - likely false positive"
    skip_fastvlm_call()

elif confidence < 0.6:
    # Medium confidence - verify with FastVLM
    description = open_ended_prompt()  # Let FastVLM decide
    
else:
    # High confidence - trust YOLO
    description = open_ended_prompt()  # Still unbiased, but YOLO probably right
```

**Benefits:**
- Saves FastVLM calls on junk detections
- Focuses compute on reliable observations
- Should filter out the "bird", "suitcase", etc.

## Expected Improvements

### Accuracy
| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Hallucinations | ~50% (bird, wine, etc.) | <10% |
| YOLO errors caught | 0% | ~70-80% |
| False positives | High | Much lower |

### Efficiency
| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Total time | 4 min for 1 min video | 2.5-3 min |
| FastVLM calls | 49 (all entities) | ~35-40 (filtered) |
| State changes | 247 total | ~50-80 total |

### Quality
- **Before**: "Wine bottle" (wrong), "TV" (it's a monitor), "bird in flight" (???)
- **After**: "Blue cylindrical object" (accurate), "Monitor/screen" (correct), warnings about birds

## What You'll See

### 1. New Log Messages
```
⚠️  Potential misclassification: YOLO said 'bird' (conf=0.35), 
    but FastVLM sees: a yellow triangular shape on dark surface...

✓ Described 35 entities
  Skipped 14 low-confidence entities (< 0.4)
```

### 2. Better Descriptions
```json
{
  "id": "entity_0008",
  "class": "bottle",
  "description": "A blue cylindrical container, approximately 8 inches tall, 
                  with a white cap. The surface is smooth and appears to be 
                  plastic or glass. Located on a desk surface.",
  "yolo_confidence": 0.62,
  "potential_misclassification": false
}
```

### 3. Fewer Bogus State Changes
```json
{
  "state_changes": 2,  // Was 23!
  "changes": [
    {
      "from_frame": 588,
      "to_frame": 756,
      "similarity": 0.72,
      "note": "Significant visual change - object moved or camera angle changed"
    }
  ]
}
```

## Testing the Fixes

Run the same video:
```bash
python scripts/test_tracking.py data/examples/video1.mp4
```

### What to Check

1. **Look for warning messages:**
   - "Potential misclassification" warnings
   - Should catch the "bird", "suitcase", etc.

2. **Count low-confidence skips:**
   - Should see "Skipped X low-confidence entities"
   - These were probably false positives

3. **Check state changes:**
   - Keyboard: should be 0-2 (was 10)
   - Bottle: should be 2-5 (was 23)
   - Person: should be much lower (was 247!)

4. **Read descriptions:**
   - Should be more objective
   - No more "wine bottle" if it's not wine
   - Better object identification

## Next Steps (If Still Issues)

### If still seeing hallucinations:
1. **Try multi-crop context** (wide + tight crops)
2. **Add scene understanding** (describe overall scene first)
3. **Switch to EmbeddingGemma** (better semantic embeddings)

### If still too many state changes:
1. **Lower threshold further** (try 0.70)
2. **Add temporal filtering** (require 3+ frames persistence)
3. **Semantic verification** (use FastVLM to verify changes)

### If still too slow:
1. **Increase LOW_CONFIDENCE_THRESHOLD** to 0.5 (skip more)
2. **Reduce TARGET_FPS** to 3.0 (fewer frames)
3. **Batch FastVLM calls** (describe multiple at once)

## Files Modified

- `src/orion/tracking_engine.py`:
  - `Config.STATE_CHANGE_THRESHOLD`: 0.90 → 0.75
  - `Config.HIGH/LOW_CONFIDENCE_THRESHOLD`: Added
  - `_generate_description()`: Completely rewritten (unbiased prompts)
  - `describe_entities()`: Added confidence filtering

## Performance Impact

### Before
```
Phase 1: Observations - 72s
Phase 2: Clustering - 1s  
Phase 3: Descriptions - 167s (49 entities × 3.4s)
Phase 4: State Detection - <1s
Total: ~240s (4 minutes)
```

### After (Expected)
```
Phase 1: Observations - 72s (same)
Phase 2: Clustering - 1s (same)
Phase 3: Descriptions - ~120s (35 entities × 3.4s) ← 28% faster
Phase 4: State Detection - <1s (same)
Total: ~193s (3.2 minutes) ← 20% faster overall
```

**Plus** much better accuracy! 🎯

## Summary

We've addressed all three core issues:
1. ✅ **Biased prompts** → Unbiased open-ended descriptions + verification
2. ✅ **Over-sensitive states** → Much higher threshold (0.90 → 0.75)
3. ✅ **No filtering** → Skip low-confidence detections

This should dramatically reduce hallucinations while making the system faster and more accurate.

Test it now and let me know the results! 🚀
