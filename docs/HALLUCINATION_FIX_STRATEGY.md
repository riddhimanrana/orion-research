# Fixing Hallucinations & Improving Accuracy

## Problem Analysis

Your tracking results show several critical issues:

### 1. YOLO Misclassification → FastVLM Hallucination Pipeline
```
YOLO: "bottle" (wrong!) → FastVLM: "wine bottle" (hallucination)
YOLO: "tv" (it's a monitor) → FastVLM: "TV on desk" (reinforces error)
YOLO: "bird" (???) → FastVLM: "bird in flight" (pure hallucination)
```

**Root cause:** FastVLM receives:
- **Tight crop** (no context)
- **Biased prompt** based on YOLO class: `"Describe this {class_name}"`
- Only visual features within bounding box

### 2. Meaningless "State Changes"
- 23 changes for a bottle = camera movement, lighting, occlusion, blur
- These are **perceptual changes**, not **semantic changes**
- Current threshold (0.90 cosine similarity) is too sensitive
- No distinction between "object moved" vs "object changed state"

### 3. No Semantic Understanding
- ResNet50 extracts visual features, not semantic meaning
- Clustering groups "things that look similar" not "same object instances"
- No verification that YOLO classifications are correct

## Solution Strategy

### Phase 1: Fix the Description Generation (IMMEDIATE)

**Problem:** Biased prompts cause hallucinations

**Current code:**
```python
prompt = f"Describe this {class_name} in detail..."
description = fastvlm.generate(crop, prompt)
```

**Solution A: Open-Ended Description**
```python
# Let FastVLM decide what it is, don't bias it
prompt = "What is this object? Describe what you see in detail, including its appearance, color, shape, and any distinguishing features."
description = fastvlm.generate(crop, prompt)
```

**Solution B: Classification Verification**
```python
# Two-stage: verify first, then describe
verify_prompt = f"Is this a {class_name}? Answer yes or no, then briefly explain what you see."
verification = fastvlm.generate(crop, verify_prompt, max_tokens=50)

if "yes" in verification.lower()[:10]:
    # YOLO was right
    detail_prompt = f"Describe this {class_name} in detail..."
else:
    # YOLO was wrong - open-ended description
    detail_prompt = "What is this object? Describe it in detail..."

description = fastvlm.generate(crop, detail_prompt)
```

**Solution C: Multi-Crop Context (BEST)**
```python
# Provide wider context
tight_crop = get_crop(bbox)  # What we have now
wide_crop = get_crop(expand_bbox(bbox, factor=1.5))  # 50% bigger

prompt = f"""
You are viewing two images of the same detection:
1. A tight crop focusing on the object
2. A wider view showing surrounding context

The object detector classified this as: {class_name}

Please:
1. Verify if this classification is correct
2. If incorrect, identify what this actually is
3. Describe the object in detail

Consider the surrounding context to understand what you're looking at.
"""

description = fastvlm.generate([tight_crop, wide_crop], prompt)
```

### Phase 2: Better State Change Detection (MEDIUM PRIORITY)

**Problem:** Too sensitive, detects perceptual not semantic changes

**Current approach:**
```python
if cosine_similarity < 0.90:
    # State change!
```

**Solution A: Higher Threshold + Temporal Filtering**
```python
class Config:
    STATE_CHANGE_THRESHOLD = 0.80  # Much less sensitive (was 0.90)
    MIN_STATE_DURATION = 3  # Must persist for 3+ frames
    
# Only flag state changes that persist
def detect_stable_state_changes(entity):
    changes = []
    current_state_start = 0
    
    for i in range(len(entity.observations) - 1):
        similarity = cosine_similarity(obs[i], obs[i+1])
        
        if similarity < 0.80:  # Significant visual difference
            # Check if change persists
            if i - current_state_start >= 3:
                # This is a real state change, not camera jitter
                changes.append({
                    'from_frame': current_state_start,
                    'to_frame': i,
                    'type': 'persistent_visual_change'
                })
            current_state_start = i + 1
    
    return changes
```

**Solution B: Use FastVLM for Semantic State Verification (Selective)**
```python
# Only verify significant, persistent changes
def verify_semantic_state_change(entity, change_idx):
    """
    Only called for high-magnitude, persistent changes
    Uses FastVLM to check if the change is semantically meaningful
    """
    before_obs = entity.observations[change_idx]
    after_obs = entity.observations[change_idx + 1]
    
    before_crop = get_crop(before_obs)
    after_crop = get_crop(after_obs)
    
    prompt = f"""
    Compare these two images of the same {entity.class_name}:
    
    Image 1 (before): [first image]
    Image 2 (after): [second image]
    
    Has this object undergone a meaningful STATE CHANGE?
    Examples of state changes:
    - Person: sitting → standing, eyes open → closed
    - Laptop: closed → open
    - Door: open → closed
    - Bottle: full → empty
    
    Examples of NON-state changes (ignore these):
    - Camera angle changed
    - Lighting changed
    - Object moved position
    - Temporary occlusion
    
    Answer: Yes or No, then explain briefly.
    """
    
    response = fastvlm.generate([before_crop, after_crop], prompt)
    
    return {
        'is_semantic_change': 'yes' in response.lower()[:10],
        'explanation': response
    }

# Only verify changes with high magnitude + duration
for change in entity.state_changes:
    if change['change_magnitude'] > 0.25 and change['duration'] > 2.0:
        verification = verify_semantic_state_change(entity, change['index'])
        change['semantic_verification'] = verification
```

### Phase 3: Replace ResNet50 with EmbeddingGemma (HIGH IMPACT)

**Problem:** ResNet50 only understands visual similarity, not semantic identity

**Solution: Use Google's EmbeddingGemma**
- Multimodal embeddings (vision + language)
- Better semantic understanding
- Could help distinguish "monitor" vs "TV"

**Implementation:**
```python
# src/orion/embedding_model.py (NEW)
from transformers import AutoModel, AutoProcessor
import torch

class EmbeddingGemmaWrapper:
    def __init__(self):
        self.model = AutoModel.from_pretrained(
            "google/embedding-gemma-2b",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            "google/embedding-gemma-2b"
        )
    
    def get_embedding(self, image, text_context=None):
        """
        Get multimodal embedding
        
        Args:
            image: PIL Image
            text_context: Optional text (e.g., YOLO class) to condition on
        
        Returns:
            Normalized embedding vector
        """
        if text_context:
            # Multimodal: image + text
            inputs = self.processor(
                images=image,
                text=text_context,
                return_tensors="pt"
            )
        else:
            # Vision only
            inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)
        
        # Normalize
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy()[0]
```

**Usage:**
```python
# In observation collection
embedding = embedding_model.get_embedding(
    image=crop,
    text_context=f"A {class_name} in a video"  # Semantic hint
)
```

**Benefits:**
- Better clustering (semantic not just visual)
- Could catch YOLO errors (image doesn't match class name → different embedding)
- More robust to lighting/angle changes

### Phase 4: Confidence-Based Verification (EFFICIENCY)

**Problem:** We're describing everything, even low-confidence detections

**Solution: Only describe high-confidence detections, verify low-confidence**
```python
class Config:
    HIGH_CONFIDENCE_THRESHOLD = 0.7  # Trust YOLO
    LOW_CONFIDENCE_THRESHOLD = 0.4   # Verify with FastVLM
    
def smart_description_pipeline(entity):
    best_obs = select_best_observation(entity)
    
    if best_obs.confidence > Config.HIGH_CONFIDENCE_THRESHOLD:
        # Trust YOLO, use standard description
        prompt = f"Describe this {entity.class_name} in detail..."
        description = fastvlm.generate(crop, prompt)
    
    elif best_obs.confidence > Config.LOW_CONFIDENCE_THRESHOLD:
        # YOLO uncertain - verify first
        verify_prompt = f"Is this a {entity.class_name}? What do you see?"
        description = fastvlm.generate(crop, verify_prompt)
        
        # Parse response to potentially update class_name
        if "not a" in description.lower() or "actually" in description.lower():
            entity.class_name = "unknown"
            entity.needs_reclassification = True
    
    else:
        # Very low confidence - skip or mark for review
        description = f"Low confidence detection (conf={best_obs.confidence:.2f})"
        entity.low_confidence = True
    
    return description
```

### Phase 5: Add Contextual Understanding (ADVANCED)

**Problem:** Crops lose spatial context

**Solution A: Include Spatial Context in Prompt**
```python
def generate_contextual_description(entity, best_obs):
    # Calculate spatial position
    cx, cy = best_obs.get_center()
    h, w = video_height, video_width
    
    # Determine location
    if cy < h/3:
        position = "top"
    elif cy < 2*h/3:
        position = "middle"
    else:
        position = "bottom"
    
    if cx < w/3:
        position += " left"
    elif cx < 2*w/3:
        position += " center"
    else:
        position += " right"
    
    # Get nearby objects
    nearby = find_nearby_objects(best_obs, all_observations)
    
    prompt = f"""
    Describe the {entity.class_name} you see in the image.
    
    Context:
    - Location: {position} of frame
    - Nearby objects: {', '.join([o.class_name for o in nearby])}
    - Size: {best_obs.get_bbox_area()*100:.1f}% of frame
    
    Focus on what this object actually is and its distinctive features.
    """
    
    return fastvlm.generate(crop, prompt)
```

**Solution B: Scene Understanding First**
```python
def hierarchical_understanding(video_path):
    # Step 1: Understand the overall scene
    key_frame = extract_middle_frame(video_path)
    scene_prompt = """
    Describe this scene in detail:
    - What type of location is this?
    - What are the main objects/areas?
    - What activity seems to be happening?
    """
    scene_description = fastvlm.generate(key_frame, scene_prompt)
    
    # Step 2: Detect objects with scene context
    # Now we know "this is a desk workspace with computer equipment"
    # So when YOLO says "bird", we can flag it as unlikely
    
    # Step 3: Describe objects with scene context
    for entity in entities:
        prompt = f"""
        Scene context: {scene_description}
        
        Describe this {entity.class_name} within this scene.
        Does this object make sense in this context?
        """
```

## Recommended Implementation Order

### Week 1: Quick Wins
1. ✅ **Open-ended prompts** (remove bias)
2. ✅ **Confidence filtering** (don't describe junk)
3. ✅ **Raise state change threshold** (0.80 instead of 0.90)

### Week 2: Quality Improvements
4. ✅ **Two-stage verification** (check YOLO classifications)
5. ✅ **Multi-crop context** (wide + tight crops)
6. ✅ **Temporal filtering** (persistent state changes only)

### Week 3: Architecture Upgrade
7. ✅ **Switch to EmbeddingGemma** (better embeddings)
8. ✅ **Semantic state verification** (selective FastVLM verification)
9. ✅ **Spatial context** (location + nearby objects)

### Week 4: Advanced
10. ✅ **Scene understanding** (hierarchical approach)
11. ✅ **Confidence-based routing** (smart description pipeline)
12. ✅ **Hallucination detection** (cross-validate with scene context)

## Expected Improvements

### Accuracy
- **Before**: 50% hallucinations (bird, suitcase, wine, etc.)
- **After**: <10% errors (mostly edge cases)

### Efficiency
- **Before**: 4 minutes for 1 minute video
- **After**: 2-3 minutes (by skipping low-confidence detections)

### State Changes
- **Before**: 23 meaningless changes for a bottle
- **After**: 0-2 semantic state changes (if any)

## Testing Strategy

1. **Run with open-ended prompts first** (easiest change)
2. **Compare results** to current output
3. **Measure hallucination rate** (manual review of 50 descriptions)
4. **Iterate** on prompt engineering

Would you like me to implement any of these solutions?
