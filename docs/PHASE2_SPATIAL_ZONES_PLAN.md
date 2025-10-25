# Phase 2 Implementation Plan: Spatial Zone Detection

## 🎯 Objectives

Fix the issue where `num_spatial_zones: 0` by implementing:
1. **HDBSCAN-based spatial clustering** of entities
2. **Zone labeling** (e.g., "desk area", "bedroom area")
3. **Spatial relationship detection** (near, on, inside)
4. **Zone-aware entity enrichment**

---

## 📊 Current Problem

**From test results:**
```json
{
  "semantic": {
    "num_spatial_zones": 0,  // ❌ NOT WORKING
    "num_entities": 11,
    "state_changes": 0,
    "events": 0
  }
}
```

**Root Cause:**
- `SemanticEngine.process()` doesn't call spatial zone detection
- `_detect_spatial_zones()` method doesn't exist or isn't implemented
- No HDBSCAN clustering happening

---

## 🏗️ Architecture Design

### **Data Flow:**
```
PerceptionResult (entities with bboxes, timestamps)
    ↓
SemanticEngine.process()
    ↓
_detect_spatial_zones(entities)
    ↓
    1. Extract spatial features
    2. HDBSCAN clustering
    3. Label zones
    4. Compute zone relationships
    ↓
SpatialZone objects + enriched entities
```

### **Spatial Zone Structure:**
```python
@dataclass
class SpatialZone:
    zone_id: str
    label: str  # "desk_area", "bedroom_area", etc.
    entity_ids: List[str]
    centroid: Tuple[float, float]  # (x, y) in normalized coords
    bounding_box: BoundingBox
    confidence: float
    
    # Relationships
    adjacent_zones: List[str]
    contained_zones: List[str]  # Hierarchical zones
```

---

## 🔧 Implementation Steps

### **Step 1: Create Spatial Utilities Module**

**File:** `orion/semantic/spatial_utils.py`

**Functions:**
```python
def extract_spatial_features(entities: List[Entity]) -> np.ndarray:
    """
    Extract features for HDBSCAN clustering.
    
    Features (per entity):
    - Centroid X (normalized)
    - Centroid Y (normalized)
    - Temporal co-occurrence score
    - Bbox area (normalized)
    
    Returns: (N, 4) array
    """
    pass

def cluster_entities_hdbscan(
    features: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: int = 2,
) -> np.ndarray:
    """
    Cluster entities using HDBSCAN.
    
    Returns: cluster labels (-1 = noise)
    """
    from hdbscan import HDBSCAN
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean'
    )
    return clusterer.fit_predict(features)

def label_zone(
    entity_classes: List[str],
    centroid: Tuple[float, float],
) -> str:
    """
    Infer zone label from entity classes.
    
    Examples:
    - ["keyboard", "mouse", "monitor"] → "desk_area"
    - ["bed", "pillow"] → "bedroom_area"
    - ["refrigerator", "oven"] → "kitchen_area"
    """
    pass

def compute_zone_relationships(
    zones: List[SpatialZone],
) -> Dict[str, List[str]]:
    """
    Compute adjacency and containment relationships.
    
    Returns: {zone_id: [adjacent_zone_ids]}
    """
    pass
```

---

### **Step 2: Implement in SemanticEngine**

**File:** `orion/semantic/engine.py`

**Add method:**
```python
def _detect_spatial_zones(
    self,
    entities: List[ConsolidatedEntity],
) -> List[SpatialZone]:
    """
    Detect spatial zones using HDBSCAN clustering.
    
    Args:
        entities: Consolidated entities with spatial info
        
    Returns:
        List of detected spatial zones
    """
    if len(entities) < 3:
        logger.info("Not enough entities for spatial clustering")
        return []
    
    # Extract spatial features
    features = extract_spatial_features(entities)
    
    # Cluster with HDBSCAN
    labels = cluster_entities_hdbscan(
        features,
        min_cluster_size=self.config.spatial_min_cluster_size,
    )
    
    # Create zones
    zones = []
    unique_labels = set(labels) - {-1}  # Exclude noise
    
    for label_id in unique_labels:
        zone_entities = [e for e, l in zip(entities, labels) if l == label_id]
        
        # Compute zone properties
        zone_centroid = compute_centroid(zone_entities)
        zone_bbox = compute_bounding_box(zone_entities)
        zone_label = label_zone(
            [e.object_class for e in zone_entities],
            zone_centroid
        )
        
        zone = SpatialZone(
            zone_id=f"zone_{label_id}",
            label=zone_label,
            entity_ids=[e.entity_id for e in zone_entities],
            centroid=zone_centroid,
            bounding_box=zone_bbox,
            confidence=0.8,  # Could be based on cluster stability
        )
        zones.append(zone)
    
    logger.info(f"Detected {len(zones)} spatial zones")
    return zones
```

**Update `process()` method:**
```python
def process(self, perception_result: PerceptionResult) -> SemanticResult:
    # ... existing consolidation ...
    entities = self._consolidate_entities(perception_result.entities)
    
    # NEW: Detect spatial zones
    spatial_zones = self._detect_spatial_zones(entities)
    
    # ... rest of processing ...
    
    return SemanticResult(
        entities=entities,
        spatial_zones=spatial_zones,  # NEW
        state_changes=state_changes,
        # ...
    )
```

---

### **Step 3: Add Configuration**

**File:** `orion/semantic/config.py`

**Add to SemanticConfig:**
```python
@dataclass
class SemanticConfig:
    # ... existing fields ...
    
    # Spatial zone detection
    enable_spatial_zones: bool = True
    spatial_min_cluster_size: int = 3
    spatial_min_samples: int = 2
    spatial_feature_weights: Dict[str, float] = field(default_factory=lambda: {
        'position': 0.6,
        'temporal': 0.2,
        'size': 0.2,
    })
```

---

### **Step 4: Zone Labeling Logic**

**Pattern-based labeling:**
```python
ZONE_PATTERNS = {
    'desk_area': ['keyboard', 'mouse', 'monitor', 'tv', 'laptop'],
    'bedroom_area': ['bed', 'pillow'],
    'kitchen_area': ['refrigerator', 'oven', 'microwave', 'sink'],
    'living_area': ['couch', 'tv', 'remote'],
    'workspace': ['chair', 'desk', 'laptop'],
}

def label_zone(entity_classes: List[str], centroid: Tuple[float, float]) -> str:
    # Count matches for each pattern
    scores = {}
    for zone_label, patterns in ZONE_PATTERNS.items():
        score = sum(1 for cls in entity_classes if cls in patterns)
        scores[zone_label] = score
    
    # Return best match or generic
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    else:
        # Use position heuristic
        x, y = centroid
        if y < 0.3:  # Top of frame
            return 'upper_area'
        elif y > 0.7:  # Bottom
            return 'lower_area'
        elif x < 0.3:  # Left
            return 'left_area'
        elif x > 0.7:  # Right
            return 'right_area'
        else:
            return 'central_area'
```

---

### **Step 5: Entity Enrichment**

**Add zone info to entities:**
```python
class ConsolidatedEntity:
    # ... existing fields ...
    spatial_zone_id: Optional[str] = None
    spatial_zone_label: Optional[str] = None
    
    # Relationships within zone
    nearby_entities: List[str] = field(default_factory=list)
```

**In `_detect_spatial_zones()`:**
```python
# After creating zones, enrich entities
for zone in zones:
    for entity_id in zone.entity_ids:
        entity = next(e for e in entities if e.entity_id == entity_id)
        entity.spatial_zone_id = zone.zone_id
        entity.spatial_zone_label = zone.label
        entity.nearby_entities = [
            eid for eid in zone.entity_ids if eid != entity_id
        ]
```

---

## 📊 Expected Results

### **Before Phase 2:**
```json
{
  "semantic": {
    "num_spatial_zones": 0,
    "entities": [
      {"entity_id": "entity_1", "class": "tv"},
      {"entity_id": "entity_2", "class": "keyboard"},
      {"entity_id": "entity_5", "class": "mouse"}
    ]
  }
}
```

### **After Phase 2:**
```json
{
  "semantic": {
    "num_spatial_zones": 2,
    "spatial_zones": [
      {
        "zone_id": "zone_0",
        "label": "desk_area",
        "entity_ids": ["entity_1", "entity_2", "entity_5"],
        "centroid": [0.45, 0.35],
        "confidence": 0.85
      },
      {
        "zone_id": "zone_1",
        "label": "bedroom_area",
        "entity_ids": ["entity_9", "entity_10"],
        "centroid": [0.72, 0.68],
        "confidence": 0.78
      }
    ],
    "entities": [
      {
        "entity_id": "entity_1",
        "class": "tv",
        "spatial_zone_id": "zone_0",
        "spatial_zone_label": "desk_area",
        "nearby_entities": ["entity_2", "entity_5"]
      }
    ]
  }
}
```

---

## 🧪 Testing Strategy

### **Test 1: Basic Clustering**
```python
# Entities: keyboard, mouse, tv (close together) + bed (far away)
# Expected: 2 zones
```

### **Test 2: Zone Labeling**
```python
# Zone with [keyboard, mouse, monitor] → "desk_area"
# Zone with [bed, pillow] → "bedroom_area"
```

### **Test 3: Noise Handling**
```python
# Single isolated entity → Not assigned to any zone (noise)
```

### **Test 4: Hierarchical Zones**
```python
# Large zone contains smaller sub-zones
# E.g., "room" contains "desk_area" and "bed_area"
```

---

## 🔧 Dependencies

**Install HDBSCAN:**
```bash
conda activate orion
pip install hdbscan scikit-learn
```

**Import structure:**
```python
from hdbscan import HDBSCAN
import numpy as np
from sklearn.preprocessing import StandardScaler
```

---

## 🚀 Implementation Order

1. ✅ Create `spatial_utils.py` with feature extraction
2. ✅ Implement HDBSCAN clustering function
3. ✅ Add zone labeling logic
4. ✅ Update `SemanticEngine._detect_spatial_zones()`
5. ✅ Update `SemanticEngine.process()` to call it
6. ✅ Add configuration options
7. ✅ Enrich entity objects with zone info
8. ✅ Update SemanticResult to include zones
9. ✅ Create test script
10. ✅ Validate on full video

---

## 📝 Code Locations

**New Files:**
- `orion/semantic/spatial_utils.py` (NEW)
- `orion/semantic/types.py` (add SpatialZone dataclass)
- `test_phase2_spatial_zones.py` (NEW)

**Modified Files:**
- `orion/semantic/engine.py` (add _detect_spatial_zones)
- `orion/semantic/config.py` (add spatial config)
- `orion/semantic/types.py` (add zone fields to entities)

---

## ⚠️ Edge Cases

1. **Too few entities:** < 3 entities → no clustering possible
2. **Noise entities:** Isolated entities don't belong to any zone
3. **Overlapping zones:** Rare, but HDBSCAN might create overlaps
4. **Temporal spread:** Entities appearing at different times
   - Solution: Use temporal co-occurrence in features
5. **Scale variance:** TV (large) vs phone (small)
   - Solution: Normalize bbox sizes

---

## 🎯 Success Criteria

- [ ] `num_spatial_zones` > 0 for test video
- [ ] Desk entities (keyboard, mouse, TV) in same zone
- [ ] Bed entities in separate zone
- [ ] Zone labels make semantic sense
- [ ] Processing time < 500ms for spatial stage
- [ ] Noise entities (< 3 in cluster) handled gracefully

---

## 🔮 Future Enhancements (Phase 2.5)

1. **Hierarchical zones** (room → desk area → monitor)
2. **Temporal zone stability** (zones persist across time)
3. **Zone transitions** (entity moves from zone A to B)
4. **3D spatial reasoning** (above, below relationships)
5. **Semantic zone refinement** (use descriptions to refine labels)

---

**Status:** 📋 Plan Complete - Ready for Implementation  
**Next:** Start coding `spatial_utils.py`
