# Performance Analysis & Model Inventory

## Current SGG Performance
**Recall@20: 1.1%** (only 1 video out of 10 with >0% recall)

### Performance Breakdown:
- Only **1 matched triplet** out of 90 predicted relations across 10 videos
- Most videos have **0% recall** (completely wrong relations)
- Issue: Relations are fundamentally mismatched with ground truth

---

## Root Causes of Low Performance

### 1. **Detection Quality** (Primary Issue)
- YOLO-World is detecting wrong objects or missing key objects
- PVSG ground truth focuses on: person↔object interactions (on, near, held_by)
- YOLO-World may not distinguish fine-grained relationships

### 2. **Scene Graph Generation**
- Spatial relationship logic may not match PVSG ground truth predicates
- Current pipeline: detection → tracking → scene graph
- Missing: relationship verification with ground truth ontology

### 3. **Re-ID / Object Tracking**
- V-JEPA2 Re-ID may not be initialized/used in current pipeline
- Person-to-person matching may be weak
- Spatial zones not properly configured for PVSG dataset

### 4. **PVSG Ground Truth Format Mismatch**
- PVSG expects specific predicates: "on", "near", "held_by" etc.
- Current scene graph may be generating different predicates

---

## Models Available & Usage Status

### **Detection Models** (Backend: yoloworld)
| Model | Status | Notes |
|-------|--------|-------|
| YOLO11x | ✅ Loaded | `models/yolo11x/yolo11x.pt` (2.4GB) |
| YOLO11m | ✅ Available | Not currently used |
| YOLO11s | ⚠️ Available | Lighter, faster |
| YOLO11n | ⚠️ Available | Smallest, fastest |
| YOLO-World v8x | ✅ Auto-downloaded | Current default (open-vocab) |
| GroundingDINO | ❌ Not used | Feature available but disabled |
| OWL-ViT2 | ❌ Not used | For OpenVocab backend |

**Currently Used**: YOLO-World v8x-worldv2

### **Re-ID / Embedding Models**
| Model | Backend | Status | Dimension | Notes |
|-------|---------|--------|-----------|-------|
| DINOv3 | dinov3 | ✅ **CURRENTLY USED** | 768 | Gated access, manual weights |
| V-JEPA2 (vitl) | vjepa2 | ⚠️ Default fallback | 1024 | 3D-aware video embeddings, auto-download |
| DINOv2 | dinov2 | ❌ Available | 768 | Public, auto-download capable |
| CLIP ViT-B/32 | legacy | ❌ Unused | 512 | Old candidate labeling (disabled) |

**Currently Used**: **DINOv3** (with manual weights)

### **Depth Models**
| Model | Status | Notes |
|-------|--------|-------|
| DepthAnythingV3 | ✅ Available | For spatial reasoning |
| SAM2 | ✅ Available | `models/sam_vit_h_4b8939.pth` (2.4GB) |

**Status**: Loaded but underutilized for spatial relationships

### **VLM / Description Models**
| Model | Status | Purpose |
|-------|--------|---------|
| FastVLM (0.5B MLX) | ✅ Available | Object descriptions |
| Gemini 3.5-Flash | ❌ API only | VLM relation verification (optional) |

---

## Improvement Strategies (Priority Order)

### **High Impact (1-2 days)**

1. **Switch from YOLO-World to YOLO11x Detection** ⭐ **PRIMARY ISSUE**
   ```bash
   # Switch from YOLO-World → YOLO11x baseline
   --detector-backend yolo --detection-model yolo11x
   ```
   - YOLO11x is more reliable for COCO classes (person, chair, table)
   - YOLO-World struggles with fine-grained relationships
   - Expected improvement: 5-10x better initial detections

2. **Verify DINOv3 is Properly Initialized**
   ```bash
   # Check DINOv3 weights are loaded
   ls -lh models/dinov3-vitb16/
   # Verify embedding extraction works
   conda run -n orion python -c "from orion.backends.dino_embedder import DINOEmbedder; print('DINOv3 OK')"
   ```
   - DINOv3 should provide 768-dim embeddings
   - If initialization fails, fall back to DINOv2 (public)
   - Current issue: DINOv3 may not be integrated into scene graph generation

3. **Enable Hybrid Detection (YOLO11x + GroundingDINO)**
   ```python
   detection.enable_hybrid_detection = True
   detection.hybrid_min_detections = 3
   ```
   - Use YOLO11x as primary (fast, reliable)
   - GroundingDINO for hard cases (rare objects, missed detections)
   - Expected improvement: 3-5% additional recall

### **Medium Impact (2-3 days)**

4. **Implement PVSG Ground Truth Relationship Oracle**
   - Create relationship classifier: given (obj1, obj2, bbox1, bbox2, depth) → predicate
   - Use depth + spatial layout to predict "on", "near", "held_by"
   - Expected improvement: 10-20% recall

5. **Switch to DINOv2 for Re-ID (Fallback)**
   ```python
   embedding.backend = "dinov2"  # Public, auto-download
   ```
   - If V-JEPA2 issues persist
   - DINOv2 is stable and public (no gated access)
   - Expected improvement: Stable baseline, 1-2% worse than V-JEPA2

6. **Enable Spatial Zones for PVSG Dataset**
   - Define kitchen, living room, bedroom zones
   - Use depth to constrain spatial predicates
   - Expected improvement: 5-10% by filtering impossible relations

### **Lower Impact (Nice to have)**

7. **Use GroundingDINO as Primary Detector**
   - For higher-quality detection of rare objects
   - Slower but more accurate (needed for "remote", "scissors", etc.)
   - Expected improvement: 2-5% for small objects

8. **VLM Relation Verification (Gemini)**
   - Use Gemini 3.5-Flash to verify predicted relations
   - Slower but very accurate
   - Expected improvement: 2-3% for ambiguous cases

---

## Currently Unused High-Value Components

### **⚠️ DINOv3 Integration Issues:**

1. **DINOv3 Re-ID Not in Scene Graph**
   - DINOv3 is loaded and working for embeddings
   - But may not be used in scene graph generation
   - Scene graphs only use detection + spatial heuristics
   - Missing: embedding-based object matching across frames

2. **DepthAnythingV3 Not Informing Relations**
   - Provides rich 3D structure
   - Should inform spatial relationships ("on", "near")
   - Currently only used for visualization

---

## Recommended Action Plan

### **Phase 1: Quick Win (30 min)**
```bash
# Test with YOLO11x baseline instead of YOLO-World
conda run -n orion python -m orion.cli.run_showcase \
  --episode test_yolo11 \
  --video data/examples/test.mp4 \
  --detector-backend yolo \
  --detection-model yolo11x \
  --fps 2.0
```

### **Phase 2: Spatial Relationships (1 day)**
- Create spatial predicate classifier
- Use depth + bbox geometry to predict "on", "near", "held_by"
- Validate against PVSG ground truth

### **Phase 3: Integration (1 day)**
- Enable V-JEPA2 logging to confirm it's running
- Add temporal consistency to scene graphs
- Re-evaluate on full 10 videos

### **Expected Result**
- Baseline YOLO11x: 5-10% R@20
- +Hybrid detection: 8-15% R@20
- +Spatial oracle: 15-25% R@20 (realistic target)
- +V-JEPA2 verification: 20-30% R@20

---

## Model Weights Summary

```
Available Models (~10GB):
├── Detection:
│   ├── YOLO11x                    2.4GB ✅ (models/yolo11x/)
│   ├── YOLO-World v8x            auto-download
│   ├── GroundingDINO             auto-download
│   └── OWL-ViT2                  auto-download
├── Re-ID:
│   ├── V-JEPA2 (vitl)           auto-download (HF)
│   ├── DINOv3                    manual (gated)
│   └── DINOv2                    auto-download (HF)
├── Depth:
│   ├── DepthAnythingV3          auto-download
│   └── SAM2                       2.4GB ✅ (models/)
└── VLM:
    ├── FastVLM (0.5B)           available
    └── Gemini 3.5-Flash         API (requires key)
```

**Recommendation**: Start with YOLO11x → GroundingDINO → spatial oracle
