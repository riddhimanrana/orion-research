"""
LABEL-AGNOSTIC OBJECT MEMORY SYSTEM

Problem: YOLO labels are unreliable and contaminate downstream systems
Solution: Track objects by visual appearance + spatial position, not class names

Architecture:
1. Detection → bbox + mask + CLIP embedding (NO class name used for identity)
2. Re-ID → Match by CLIP embedding similarity (appearance-based)
3. Spatial Memory → Track by position + size (independent of label)
4. Semantic Layer → FastVLM descriptions + user corrections (ground truth)
5. Knowledge Graph → Store descriptions, not YOLO guesses
"""

## PROPOSED ARCHITECTURE

### Phase 1: Detection (What We Can Trust)
```python
class Detection:
    bbox: [x1, y1, x2, y2]           # ✓ Reliable spatial data
    mask: np.ndarray                  # ✓ Reliable segmentation
    clip_embedding: np.ndarray        # ✓ Reliable visual signature (512-dim)
    confidence: float                 # ✓ Reliable detection score
    
    # Spatial features (derived from bbox + depth if available)
    area: float                       # ✓ Reliable
    center: (x, y)                    # ✓ Reliable
    depth: Optional[float]            # ✓ Reliable (if depth model used)
    
    # WEAK HINTS (don't trust for identity)
    yolo_class_hint: str              # ⚠️  Use only for initial grouping
    yolo_confidence: float            # ⚠️  Treat as "objectness" not class certainty
```

### Phase 2: Tracking (Appearance + Spatial)
```python
class TrackedObject:
    track_id: int                     # Unique ID (never reused)
    
    # Visual identity (what makes this object unique)
    appearance_gallery: Deque[np.ndarray]  # CLIP embeddings over time (EMA)
    canonical_embedding: np.ndarray        # Average CLIP embedding
    
    # Spatial memory (where has this object been)
    position_history: List[(x, y, z, timestamp)]
    last_seen_position: (x, y, z)
    spatial_zone: Optional[str]       # "desk", "shelf", "floor" (from SLAM)
    
    # Semantic understanding (built over time, not from YOLO)
    fastvlm_descriptions: List[str]   # Descriptions from visual language model
    user_label: Optional[str]         # Ground truth from user ("my backpack")
    inferred_category: Optional[str]  # High-confidence category (laptop, bottle, etc)
    
    # Lifecycle
    first_seen: int                   # Frame number
    last_seen: int
    times_reidentified: int
    confidence_score: float           # Track quality (how stable is Re-ID)
    
    # DO NOT STORE: yolo_class_name (unreliable, will pollute data)
```

### Phase 3: Re-Identification (Embedding-Based)
```python
def match_detection_to_tracks(
    detection: Detection,
    active_tracks: List[TrackedObject]
) -> Optional[int]:
    """
    Match detection to existing track using ONLY reliable data
    
    NO CLASS NAME MATCHING - only visual + spatial
    """
    
    candidates = []
    
    for track in active_tracks:
        # Appearance similarity (CLIP embedding cosine similarity)
        appearance_sim = cosine_similarity(
            detection.clip_embedding,
            track.canonical_embedding
        )
        
        # Spatial proximity (is detection near where track was last seen?)
        spatial_dist = euclidean_distance(
            detection.center,
            track.last_seen_position
        )
        
        # Size similarity (objects don't drastically change size)
        size_ratio = detection.area / track.last_seen_area
        size_sim = 1.0 - abs(1.0 - size_ratio)  # 1.0 = same size
        
        # Combined score (NO CLASS NAME!)
        score = (
            0.6 * appearance_sim +      # Visual similarity most important
            0.3 * (1 / (1 + spatial_dist)) +  # Spatial consistency
            0.1 * size_sim              # Size consistency
        )
        
        candidates.append((track.track_id, score))
    
    # Return best match if above threshold
    if candidates:
        best_id, best_score = max(candidates, key=lambda x: x[1])
        if best_score > 0.5:  # Threshold for Re-ID
            return best_id
    
    return None  # New track
```

### Phase 4: Semantic Understanding (Separate from Tracking)
```python
def build_semantic_understanding(track: TrackedObject):
    """
    Build accurate semantic understanding AFTER tracking is stable
    
    Only run this when:
    - Track has been seen multiple times (confident it's real object)
    - Scene is stable (not too much motion blur)
    - Object is large enough (good crop quality)
    """
    
    # Get best frame (highest quality crop)
    best_crop = get_best_crop(track)
    
    # Use FastVLM for accurate description
    description = fastvlm.describe(
        best_crop,
        prompt="Describe this object in detail. What is it? What color? What material?"
    )
    track.fastvlm_descriptions.append(description)
    
    # Parse description to extract category (optional)
    # e.g., "A black leather backpack with..." → category = "backpack"
    category = extract_category(description)
    
    # Only set inferred_category if highly confident
    if confidence_in_description(description) > 0.8:
        track.inferred_category = category
    
    # NEVER use yolo_class_hint for semantic identity
```

### Phase 5: Knowledge Graph Storage
```python
# WRONG (old way - pollutes graph with bad labels):
graph.add_entity(
    id=track.track_id,
    type=detection.yolo_class_hint,  # ❌ "diaper bag" contaminates data
    embedding=track.canonical_embedding
)

# RIGHT (new way - accurate semantic data):
graph.add_entity(
    id=track.track_id,
    
    # Visual identity (reliable)
    embedding=track.canonical_embedding,
    appearance_hash=hash(track.canonical_embedding),
    
    # Spatial identity (reliable)
    typical_location=track.spatial_zone,
    position_history=track.position_history,
    
    # Semantic identity (accurate or unknown)
    description=track.fastvlm_descriptions[-1] if track.fastvlm_descriptions else "unknown object",
    user_label=track.user_label,  # Ground truth if available
    category=track.inferred_category,  # Only if confident
    
    # Metadata
    confidence=track.confidence_score,
    times_observed=track.times_reidentified
)
```

---

## CONCRETE IMPROVEMENTS

### 1. Don't Trust YOLO Labels for Identity
```python
# BAD (current):
if detection.class_name == "keyboard" and track.class_name == "keyboard":
    match_score += 0.3  # ❌ Both could be wrong!

# GOOD (proposed):
# Just use CLIP embeddings - if they look similar, they probably are similar
appearance_sim = cosine_similarity(detection.embedding, track.embedding)
```

### 2. Use FastVLM for Accurate Descriptions (Not Real-Time)
```python
# Run FastVLM only when:
# - Track is stable (seen 5+ times)
# - Scene is clear (low motion blur)
# - Object is important (on desk, held by user)

if track.times_seen > 5 and track.fastvlm_descriptions == []:
    description = fastvlm.describe(track.best_crop)
    # Example: "A black leather backpack with gray straps and a laptop compartment"
    # Much more accurate than YOLO's "diaper bag"
```

### 3. Enable User Corrections
```python
# User says: "That's my backpack"
track.user_label = "backpack"
track.confidence_score = 1.0  # Ground truth

# Update CLIP classifier with this example
custom_classifier.add_positive_example(
    label="backpack",
    embedding=track.canonical_embedding
)
```

### 4. Spatial Context for Disambiguation
```python
# "keyboard on desk" vs "keyboard on floor" (musical instrument)
# Use SLAM zones to infer likely categories

if track.spatial_zone == "desk" and "keyboard" in track.yolo_class_hint:
    likely_category = "computer_keyboard"
elif track.spatial_zone == "floor" and "keyboard" in track.yolo_class_hint:
    likely_category = "musical_keyboard"
```

---

## MIGRATION PATH

### Step 1: Update EnhancedTracker
- Remove class_name from track identity
- Use ONLY appearance (CLIP) + spatial (position) for Re-ID
- Store yolo_class_hint but don't use for matching

### Step 2: Add FastVLM Semantic Layer
- Queue tracks for FastVLM description
- Run in background when scene is stable
- Store descriptions in track metadata

### Step 3: Enable User Corrections
- UI for "correct this label"
- Store ground truth in track.user_label
- Use for training custom classifier

### Step 4: Update Graph Storage
- Store descriptions not class names
- Store embeddings as primary identity
- Query by "objects similar to this" not "all keyboards"

---

## EXPECTED IMPROVEMENTS

**Accuracy:**
- Re-ID: 95%+ (currently ~80% due to class confusion)
- Semantic: High confidence only, or "unknown" (better than wrong)
- SLAM: Spatial memory independent of labels (more robust)

**Data Quality:**
- Knowledge graph: Accurate descriptions, not polluted guesses
- No "diaper bag" when it's a backpack
- Can handle objects YOLO doesn't know (FastVLM describes them)

**Flexibility:**
- Works for custom objects (user's specific backpack, unique laptop)
- Learns from user corrections
- Adapts to new object types without retraining YOLO
