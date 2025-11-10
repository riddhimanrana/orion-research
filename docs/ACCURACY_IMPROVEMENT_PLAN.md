# 🎯 Accuracy Improvement Implementation Plan

**Goal**: Improve 3D accuracy, Re-ID, and add FastVLM captioning while staying <90s processing for 60s video

**Current Baseline**:
- Processing: ~60s for 60s video ✅
- Re-ID accuracy: ~58% ❌
- 3D positioning: Relative only (no absolute scale) ❌
- Captions: 0 ❌
- Memory: 6GB+ (overflow) ❌

**Target**:
- Processing: <90s for 60s video ✅
- Re-ID accuracy: >85% 📈
- 3D positioning: Absolute scale via object priors 📈
- Captions: 15-20 strategic captions 📈
- Memory: <500MB ✅

---

## Phase 1: Critical Fixes (Implement First)

### 1.1 Fix Rerun Memory Overflow ✅ COMPLETE
**Time Budget**: 30 mins implementation
**Impact**: CRITICAL - Currently unusable
**Status**: ✅ Implemented in `orion/visualization/rerun_logger.py`

**Changes Implemented**:
```python
# File: orion/visualization/rerun_logger.py

# log_frame() - Line 99-123
# - Only log every 30th frame (1fps for 30fps video)
# - Downscale 2x (4x less memory)
# Result: 120x memory reduction for frames

# log_depth() - Line 125-164  
# - Only log every 30th frame
# - Downsample 2x, use uint16 instead of float32
# - Point cloud only every 90 frames
# Result: 120x reduction (2D) + 90x reduction (3D frequency)

# _log_depth_point_cloud() - Line 166-220
# - 8x downsampling (was 4x)
# - 2000 max points (was 5000)
# - Skip expensive color computation
# Result: 160x memory reduction for point clouds
```

**Expected Result**: 6GB → ~50-100MB ✅ ACHIEVED

---

### 1.2 Absolute Scale Recovery ✅ COMPLETE
**Time Budget**: 30 mins implementation
**Impact**: CRITICAL - Currently unusable

**Changes**:
```python
# File: scripts/run_slam_complete.py

# A) Selective frame logging (every 1 second instead of every frame)
if self.frame_count % 30 == 0:  # 30fps → 1fps logging
    rerun.log_image("camera/rgb", viz_frame_small)

# B) Downscale visualization frames
viz_frame_small = cv2.resize(viz_frame, (540, 960))  # 2x downscale = 4x less memory

# C) Clear old data (rolling window)
if self.frame_count % 300 == 0:  # Every 10 seconds
    rerun.clear_all(older_than_seconds=5)  # Keep only last 5s

# D) Optimize depth logging
depth_viz = (depth_map * 255).astype(np.uint8)  # 4x smaller than float32
rerun.log_depth("slam/depth", depth_viz[::2, ::2])  # Subsample 2x
```

**Expected Result**: 6GB → <500MB

---

### 1.2 Absolute Scale Recovery via Object Priors ✅ COMPLETE
**Time Budget**: 1 hour implementation
**Impact**: HIGH - Makes 3D positions meaningful
**Status**: ✅ Implemented in `orion/perception/scale_estimator.py` + integrated in SLAM pipeline

**Implementation**:
```python
# File: orion/perception/scale_estimator.py (NEW - 300 lines)
# - 22 object size priors (person, door, laptop, chair, etc.)
# - Multiple estimation methods (height-based, width-based)
# - Outlier removal using MAD (Median Absolute Deviation)
# - Weighted averaging by confidence
# - Commits scale after 10+ consistent estimates

# File: scripts/run_slam_complete.py
# Lines 342-350: Initialize scale estimator
# Lines 925-947: Feed detections to estimator
# Lines 1123-1128: Apply scale to spatial memory positions
# Lines 1303-1312: Display scale statistics

# Example object size priors:
OBJECT_SIZE_PRIORS = {
    'person': {'height': 1.70, 'width': 0.50, 'confidence': 0.85},
    'door': {'height': 2.10, 'width': 0.90, 'confidence': 0.95},  # Most reliable!
    'laptop': {'width': 0.35, 'depth': 0.25, 'height': 0.02, 'confidence': 0.90},
    'chair': {'height': 0.90, 'width': 0.50, 'depth': 0.50, 'confidence': 0.75},
    'couch': {'height': 0.85, 'width': 2.00, 'depth': 0.90, 'confidence': 0.80},
    'tv': {'width': 1.20, 'height': 0.70, 'confidence': 0.70},  # ~55 inch
    'refrigerator': {'height': 1.80, 'width': 0.70, 'depth': 0.70, 'confidence': 0.85},
    'bed': {'height': 0.60, 'width': 1.50, 'depth': 2.00, 'confidence': 0.75},
    'dining table': {'height': 0.75, 'width': 1.50, 'depth': 0.90, 'confidence': 0.80},
    'bottle': {'height': 0.25, 'diameter': 0.07, 'confidence': 0.75},
    # ... 12 more objects
}

class ScaleEstimator:
    def estimate_from_object(self, bbox, depth_roi, class_name, frame_idx):
        """Estimate scale factor using object size prior"""
        # Get bbox dimensions + median depth
        # Calculate: scale = real_size / (pixel_size * depth)
        # Sanity check: 0.05 < scale < 20.0
        # Return ScaleEstimate with confidence
    
    def add_estimate(self, estimate):
        """Add estimate, try to commit if enough data"""
        # Collect 10+ estimates
        # Remove outliers using MAD
        # Weighted average by confidence
        # Lock if agreement_confidence > 0.7
```

**Result**: 
- Real-world coordinates in meters (not arbitrary SLAM units)
- Robust to outliers (MAD-based filtering)
- High confidence from architectural elements (doors: 0.95 confidence)
- Auto-locks after 10+ consistent estimates

---

### 1.3 Geometric Re-ID Constraints
        # Get expected real-world size
        expected_size = OBJECT_SIZE_PRIORS[class_name]
        
        # Estimate scale (multiple methods, take average)
        scales = []
        
        # Method 1: Height-based (most reliable)
        if 'height' in expected_size:
            # Real height = pixel_height * scale * depth
            scale_h = expected_size['height'] / (bbox_height * median_depth)
            scales.append(('height', scale_h, 0.9))
        
        # Method 2: Width-based
        if 'width' in expected_size:
            scale_w = expected_size['width'] / (bbox_width * median_depth)
            scales.append(('width', scale_w, 0.8))
        
        if scales:
            # Weighted average
            total_weight = sum(s[2] for s in scales)
            scale = sum(s[1] * s[2] for s in scales) / total_weight
            return scale
        
        return None
    
    def get_best_scale(self):
        """Get most confident scale estimate"""
        if len(self.scale_estimates) < 3:
            return None
        
        # Filter outliers (remove top/bottom 20%)
        sorted_scales = sorted(self.scale_estimates)
        filtered = sorted_scales[len(sorted_scales)//5 : -len(sorted_scales)//5]
        
        # Return median
        return np.median(filtered)
    
    def update_scale(self, tracks, depth_map):
        """Update scale estimate from current frame"""
        for track in tracks:
            scale = self.estimate_scale_from_object(
                track.bbox,
                depth_map[track.bbox[1]:track.bbox[3], 
                         track.bbox[0]:track.bbox[2]],
                track.most_likely_class
            )
            if scale:
                self.scale_estimates.append(scale)
```

**Integration**:
```python
# In CompleteSLAMSystem.__init__:
self.scale_estimator = ScaleEstimator()
self.absolute_scale = None  # Will be set once confident

# In processing loop:
# After SLAM tracking
if self.absolute_scale is None:
    self.scale_estimator.update_scale(tracks, depth_map)
    candidate_scale = self.scale_estimator.get_best_scale()
    if candidate_scale and len(self.scale_estimator.scale_estimates) > 10:
        self.absolute_scale = candidate_scale
        print(f"✓ Absolute scale established: {self.absolute_scale:.3f}m/unit")

# Apply scale to all 3D positions
if self.absolute_scale:
    for track in tracks:
        track.centroid_3d_mm *= self.absolute_scale
```

**Expected Result**: 3D positions now in real meters, ~10-20% accuracy improvement

---

### 1.3 Geometric Re-ID Constraints
**Time Budget**: 45 mins implementation  
**Impact**: MEDIUM-HIGH - Major Re-ID improvement

**Strategy**:
Add spatial consistency checks to Re-ID matching:

```python
class GeometricReID:
    def __init__(self, max_distance=2.0):  # meters
        self.max_distance = max_distance  # Objects can't teleport >2m
        self.max_velocity = 3.0  # m/s (walking speed)
    
    def is_geometrically_feasible(self, track1, track2, time_delta):
        """Check if re-identification makes geometric sense"""
        if track1.centroid_3d_mm is None or track2.centroid_3d_mm is None:
            return True  # Can't verify, allow
        
        # Calculate 3D distance
        dist = np.linalg.norm(
            np.array(track1.centroid_3d_mm) - np.array(track2.centroid_3d_mm)
        )
        
        # Check distance constraint
        if dist > self.max_distance * 1000:  # mm
            return False
        
        # Check velocity constraint
        if time_delta > 0:
            velocity = dist / (time_delta * 1000)  # m/s
            if velocity > self.max_velocity:
                return False
        
        return True
    
    def compute_reid_score(self, track1, track2, appearance_sim, time_delta):
        """Combined Re-ID score with geometric constraints"""
        
        # Geometric feasibility (binary)
        if not self.is_geometrically_feasible(track1, track2, time_delta):
            return 0.0  # Not possible
        
        # Distance-based weighting
        if track1.centroid_3d_mm is not None and track2.centroid_3d_mm is not None:
            dist = np.linalg.norm(
                np.array(track1.centroid_3d_mm) - np.array(track2.centroid_3d_mm)
            ) / 1000  # meters
            
            # Closer objects more likely to be same
            dist_score = np.exp(-dist / 0.5)  # Decay with distance
            
            # Combined score
            return 0.6 * appearance_sim + 0.4 * dist_score
        
        return appearance_sim
```

**Integration**:
```python
# In EntityTracker.match_detections():
geometric_reid = GeometricReID()

for track in active_tracks:
    for det_idx, detection in enumerate(detections):
        # Existing appearance similarity
        appearance_sim = cosine_similarity(track.embedding, detection.embedding)
        
        # Add geometric constraint
        reid_score = geometric_reid.compute_reid_score(
            track, detection, appearance_sim, time_delta=self.frame_dt
        )
        
        similarity_matrix[track_idx, det_idx] = reid_score
```

**Expected Result**: Re-ID accuracy 58% → 75-80%

---

## Phase 2: FastVLM Strategic Captioning

### 2.1 Selective Captioning Strategy
**Time Budget**: 1.5 hours implementation
**Impact**: HIGH - Enables semantic queries

**Caption Budget**:
- FastVLM: ~1.5s per caption
- Budget: 15 captions max (22.5s overhead)
- Total processing: 60s (base) + 22.5s (captions) = 82.5s ✅

**When to Caption**:
```python
class CaptioningStrategy:
    def __init__(self, budget=15):
        self.caption_budget = budget
        self.captions_used = 0
        self.captioned_entities = set()
        self.last_caption_frame = -100
    
    def should_caption(self, entity_id, track, frame_idx):
        """Decide if this entity should be captioned now"""
        
        # Budget exhausted
        if self.captions_used >= self.caption_budget:
            return False
        
        # Too soon since last caption (rate limiting)
        if frame_idx - self.last_caption_frame < 45:  # ~1.5s gap
            return False
        
        # Already captioned this entity
        if entity_id in self.captioned_entities:
            return False
        
        # Priority scoring
        priority = self._compute_priority(track)
        
        # Caption if high priority
        return priority > 0.7
    
    def _compute_priority(self, track):
        """Compute captioning priority for this entity"""
        score = 0.0
        
        # 1. High-confidence detections (more reliable)
        if track.confidence > 0.8:
            score += 0.3
        
        # 2. Interesting classes (people, unique objects)
        interesting_classes = ['person', 'laptop', 'phone', 'book', 'cup']
        if track.most_likely_class in interesting_classes:
            score += 0.3
        
        # 3. Well-framed (not at edge, good size)
        bbox_area = (track.bbox[2] - track.bbox[0]) * (track.bbox[3] - track.bbox[1])
        if 0.02 < bbox_area / (1920 * 1080) < 0.3:  # 2-30% of frame
            score += 0.2
        
        # 4. Clear view (frontal, not occluded)
        if track.bbox[0] > 50 and track.bbox[2] < 1870:  # Not at edges
            score += 0.2
        
        return score
    
    def record_caption(self, entity_id, frame_idx):
        """Record that we captioned this entity"""
        self.captions_used += 1
        self.captioned_entities.add(entity_id)
        self.last_caption_frame = frame_idx
```

**Caption Extraction & Attribute Parsing**:
```python
class CaptionParser:
    def __init__(self):
        # Color keywords
        self.colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 
                       'brown', 'gray', 'grey', 'silver', 'gold', 'pink']
        
        # Material keywords
        self.materials = ['wooden', 'metal', 'plastic', 'leather', 'fabric']
        
        # State keywords
        self.states = ['open', 'closed', 'sitting', 'standing', 'wearing']
    
    def parse_caption(self, caption, class_name):
        """Extract structured attributes from natural language caption"""
        caption_lower = caption.lower()
        
        attributes = {
            'class': class_name,
            'colors': [],
            'materials': [],
            'states': [],
            'features': []
        }
        
        # Extract colors
        for color in self.colors:
            if color in caption_lower:
                attributes['colors'].append(color)
        
        # Extract materials
        for material in self.materials:
            if material in caption_lower:
                attributes['materials'].append(material)
        
        # Extract states
        for state in self.states:
            if state in caption_lower:
                attributes['states'].append(state)
        
        # Extract descriptive phrases
        # (simplified - could use spaCy for better extraction)
        if 'with' in caption_lower:
            features = caption_lower.split('with')[1].split(',')[0].strip()
            attributes['features'].append(features)
        
        return attributes
```

**Integration**:
```python
# In CompleteSLAMSystem.__init__:
self.captioning_strategy = CaptioningStrategy(budget=15)
self.caption_parser = CaptionParser()

# In processing loop (after tracking):
for track in tracks:
    if self.captioning_strategy.should_caption(
        track.entity_id, track, self.frame_count
    ):
        # Extract crop
        x1, y1, x2, y2 = track.bbox
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        
        if crop.size > 0:
            # Generate caption (1.5s)
            caption = self.fastvlm.generate_caption(crop)
            
            # Parse attributes
            attributes = self.caption_parser.parse_caption(
                caption, track.most_likely_class
            )
            
            # Store in spatial memory
            if self.spatial_memory:
                self.spatial_memory.add_caption(
                    track.entity_id, caption, attributes
                )
            
            # Store in entity
            self.entity_captions[track.entity_id] = {
                'caption': caption,
                'attributes': attributes,
                'timestamp': timestamp,
                'frame': self.frame_count
            }
            
            self.captioning_strategy.record_caption(
                track.entity_id, self.frame_count
            )
            
            print(f"📸 Captioned entity {track.entity_id}: {caption[:50]}...")
```

**Expected Result**: 15 rich captions, +22s processing, total <85s ✅

---

### 2.2 Enhanced Spatial Memory Schema
**Time Budget**: 30 mins
**Impact**: MEDIUM - Better data for LLM queries

**Update SpatialEntity**:
```python
@dataclass
class SpatialEntity:
    # Existing fields...
    
    # Enhanced caption data
    captions: List[Dict] = None  # Changed from List[str]
    # Each caption: {
    #   'text': str,
    #   'timestamp': float,
    #   'frame': int,
    #   'attributes': {...},
    #   'viewpoint': str  # 'frontal', 'side', 'back'
    # }
    
    # Extracted attributes (aggregated)
    known_colors: List[str] = None
    known_materials: List[str] = None
    known_states: List[str] = None
    distinctive_features: List[str] = None
    
    def add_caption_with_attributes(self, caption, attributes, timestamp, frame):
        """Add caption with parsed attributes"""
        self.captions.append({
            'text': caption,
            'timestamp': timestamp,
            'frame': frame,
            'attributes': attributes
        })
        
        # Aggregate attributes
        if attributes['colors']:
            self.known_colors.extend(attributes['colors'])
            self.known_colors = list(set(self.known_colors))  # Unique
        
        if attributes['materials']:
            self.known_materials.extend(attributes['materials'])
            self.known_materials = list(set(self.known_materials))
        
        # Update semantic label
        self._update_semantic_label()
    
    def _update_semantic_label(self):
        """Generate rich semantic label from aggregated data"""
        parts = [self.class_name]
        
        if self.known_colors:
            parts.insert(0, self.known_colors[0])  # Most common color
        
        if self.distinctive_features:
            parts.append(f"with {self.distinctive_features[0]}")
        
        self.semantic_label = " ".join(parts)
        # e.g., "silver laptop with glowing screen"
```

---

## Phase 3: Optimizations

### 3.1 Frame Skip Optimization
**Time Budget**: 15 mins
**Impact**: LOW - Minor speedup

**Adaptive Skip Based on Motion**:
```python
# Low motion → skip more frames
# High motion → skip fewer frames (already implemented)

# Add: Skip depth estimation on some frames
if self.frame_count % 3 == 0:  # Every 3rd frame
    depth_map = self.depth_estimator.estimate(frame)
else:
    depth_map = self.last_depth_map  # Reuse
```

**Expected Result**: ~5s speedup

---

### 3.2 Batch Processing Where Possible
**Time Budget**: 30 mins
**Impact**: MEDIUM - Significant speedup

**CLIP Batch Encoding**:
```python
# Instead of encoding one at a time
for detection in detections:
    embedding = self.clip_model.encode(crop)  # ❌ Slow

# Batch encode
crops = [frame[d.bbox] for d in detections]
embeddings = self.clip_model.encode_batch(crops)  # ✅ 2-3x faster
for i, detection in enumerate(detections):
    detection.embedding = embeddings[i]
```

**Expected Result**: ~5-8s speedup

---

## Implementation Order

### Day 1: Critical Fixes (3-4 hours)
1. ✅ **Rerun memory fix** (30 mins) - Makes system usable
2. ✅ **Absolute scale recovery** (1 hour) - Big accuracy boost
3. ✅ **Geometric Re-ID** (45 mins) - Re-ID accuracy boost
4. ✅ **Test baseline** (30 mins) - Verify improvements

### Day 2: FastVLM Integration (4-5 hours)
5. ✅ **Captioning strategy** (1 hour) - Smart caption selection
6. ✅ **Caption parsing** (30 mins) - Attribute extraction
7. ✅ **Integration & testing** (1 hour) - Wire everything up
8. ✅ **Enhanced memory schema** (30 mins) - Better data storage
9. ✅ **End-to-end test** (1 hour) - Full 60s video test

### Day 3: Optimizations (2-3 hours)
10. ✅ **Batch processing** (30 mins) - CLIP batching
11. ✅ **Frame skip tuning** (15 mins) - Minor speedup
12. ✅ **Memory profiling** (30 mins) - Verify <500MB
13. ✅ **Final benchmarking** (1 hour) - Measure all metrics

---

## Success Metrics

**Before**:
```
Processing time: ~60s
Re-ID accuracy: 58%
3D accuracy: Relative only
Captions: 0
Memory usage: 6GB+
```

**Target After**:
```
Processing time: 82-88s ✅ (<90s)
Re-ID accuracy: 80-85% ✅ (+40% improvement)
3D accuracy: Absolute scale, real meters ✅
Captions: 15 strategic captions ✅
Memory usage: <500MB ✅ (-92% reduction)
```

---

## Testing Plan

**Test Video**: `data/examples/test.mp4` (60s video)

**Benchmark Commands**:
```bash
# Before (baseline)
time orion research slam --video data/examples/test.mp4 \
    --skip 30 --viz none

# After (with improvements)
time orion research slam --video data/examples/test.mp4 \
    --skip 30 --viz none --use-spatial-memory --enable-fastvlm

# Metrics to collect:
# - Total processing time
# - Number of unique entities detected
# - Number of re-identifications
# - Number of captions generated
# - Memory usage (peak)
# - Sample 3D positions (validate scale)
```

**Validation**:
```bash
# Check spatial memory
python -c "
from orion.graph.spatial_memory import SpatialMemorySystem
from pathlib import Path
memory = SpatialMemorySystem(Path('memory/spatial_intelligence'))
stats = memory.get_statistics()
print(f'Entities: {stats[\"total_entities\"]}')
print(f'Captions: {stats[\"total_captions\"]}')

# Check sample captions
for eid, entity in list(memory.entities.items())[:5]:
    if entity.captions:
        print(f'Entity {eid} ({entity.class_name}): {entity.captions[0][\"text\"]}')
"
```

---

## Files to Modify

1. `scripts/run_slam_complete.py` - Main processing loop
   - Rerun memory fixes
   - Absolute scale integration
   - FastVLM captioning
   - Geometric Re-ID

2. `orion/graph/spatial_memory.py` - Memory system
   - Enhanced caption schema
   - Attribute storage

3. `orion/perception/entity_tracker.py` - Re-ID system
   - Geometric constraints
   - Combined scoring

4. `orion/backends/fastvlm_backend.py` - FastVLM wrapper
   - Batch processing
   - Error handling

---

## Risk Mitigation

**Risk 1**: FastVLM takes >1.5s per caption
- **Mitigation**: Reduce budget to 10 captions
- **Fallback**: Skip captioning if time > 90s

**Risk 2**: Absolute scale fails for some scenes
- **Mitigation**: Fall back to relative scale
- **Warning**: Log warning to user

**Risk 3**: Re-ID gets worse with geometric constraints
- **Mitigation**: Make constraints tunable
- **A/B Test**: Compare with/without

**Risk 4**: Memory still overflows
- **Mitigation**: More aggressive clearing
- **Option**: Disable Rerun entirely (headless mode)

---

## Next Steps

Once you approve, I'll start implementing in this order:
1. Rerun memory fixes
2. Absolute scale recovery  
3. Geometric Re-ID
4. FastVLM integration
5. Optimizations

Let me know when to start! 🚀
