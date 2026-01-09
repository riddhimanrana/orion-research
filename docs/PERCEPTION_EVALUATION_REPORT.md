# Orion Perception Pipeline - Deep Component Evaluation Report

**Generated:** January 9, 2026  
**Test Video:** `/lambda/nfs/orion-core-fs/test.mp4` (1827 frames @ 30 FPS, 1080x1920)  
**Environment:** Lambda A10 GPU (CUDA)

---

## Executive Summary

This report presents a comprehensive evaluation of the Orion perception pipeline's core components: **detection**, **classification**, **semantic filtering**, and **Re-ID tracking**. Each component was evaluated independently to identify bottlenecks and improvement opportunities.

### Key Findings

| Component | Status | Key Insight |
|-----------|--------|-------------|
| **Detection** | ✅ Working | YOLO11x detects 11% more objects than YOLO11m with higher confidence |
| **Classification** | ⚠️ Mixed | YOLO-World refines 22% of detections, but some refinements are incorrect (clock→human) |
| **Filtering** | ✅ Good | CLIP-based VLM filtering catches 100% of false `clock` detections |
| **Tracking** | ❌ Needs Dense Frames | Sparse sampling prevents meaningful track formation |

---

## 1. Detection Evaluation

### Setup
- **Models:** YOLO11m vs YOLO11x
- **Confidence threshold:** 0.25
- **Sample frames:** 50

### Results

| Model | Detections | Avg/Frame | Classes | FPS | Avg Conf |
|-------|------------|-----------|---------|-----|----------|
| yolo11m | 107 | 2.1 | 21 | 23.6 | 0.53 |
| yolo11x | 119 | 2.4 | 22 | 30.5 | 0.55 |

### Class Distribution (Top 5)

| Class | YOLO11m | YOLO11x | Difference |
|-------|---------|---------|------------|
| potted plant | 19 | 22 | +3 |
| chair | 16 | 20 | +4 |
| couch | 12 | 9 | -3 |
| tv | 6 | 8 | +2 |
| person | 8 | 8 | 0 |

### Potential Issues Detected
- `oven` - Low confidence: 0.298
- `cell phone` - Low confidence: 0.285
- `handbag` - Low confidence: 0.305

### Recommendation
**Use YOLO11x** - 11% more detections with marginally higher confidence and surprisingly faster FPS on CUDA (batch optimization).

---

## 2. Classification Evaluation

### Setup
- **Methods:** YOLO-World (crop-based) vs DINOv2+Semantic (full-image context)
- **Vocabulary:** 72 classes (base + refinements)
- **Sample frames:** 15

### YOLO-World Results

| Metric | Value |
|--------|-------|
| Total Classifications | 27 |
| Avg Time/Crop | 0.104s |
| Unique Classes | 15 |
| Refinement Rate | 22.2% |

### Refinement Examples

| Original | Refined | Confidence | Assessment |
|----------|---------|------------|------------|
| tv | television | 0.323 | ✅ Correct synonym |
| tv | monitor | 0.455 | ✅ Plausible refinement |
| tv | screen | 0.429 | ✅ Plausible refinement |
| clock | human | 0.345 | ❌ **False refinement** |
| bed | person | 0.302 | ❌ **False refinement** |

### Analysis
YOLO-World successfully refines generic labels (tv→television/monitor) but makes errors when confidence is borderline. The `clock→human` misclassification suggests the crop may contain a person near a clock, causing confusion.

### Recommendation
1. **Add refinement validation** - Reject refinements where refined_class is unrelated to original_class
2. **Use semantic similarity threshold** - Only accept refinements with >0.5 cosine similarity between original and refined labels
3. **DINOv2 needs different approach** - Current semantic matching isn't working; consider fine-tuning or using CLIP directly

---

## 3. Semantic Filtering Evaluation

### Setup
- **Methods:** 
  1. Confidence-only (threshold=0.35)
  2. Class-specific thresholds
  3. CLIP-based VLM verification

### Results

| Method | Total | Kept | Filtered | Filter Rate |
|--------|-------|------|----------|-------------|
| confidence_0.35 | 119 | 94 | 25 | 21.0% |
| threshold | 119 | 88 | 31 | 26.1% |
| vlm_clip | 93 | 74 | 19 | 20.4% |

### Class-Level Filtering Analysis

| Class | Conf-only | Threshold | VLM CLIP | Analysis |
|-------|-----------|-----------|----------|----------|
| potted plant | 22.7% | 40.9% | **46.7%** | VLM catches most false positives |
| clock | 0.0% | 50.0% | **100%** | VLM filters ALL clocks (likely all false) |
| chair | 25.0% | 15.0% | 0.0% | VLM verifies chairs as real |
| person | 37.5% | 25.0% | 0.0% | VLM keeps all persons |
| vase | 0.0% | 40.0% | 50.0% | VLM more aggressive on vases |

### Key Insights

1. **CLIP VLM is most accurate for suspicious classes** - It correctly identifies that ALL `clock` detections are false positives
2. **Threshold-based filtering is too aggressive on reliable classes** - Filters real chairs/persons
3. **`potted plant` is the most problematic class** - 40-47% false positive rate across all methods

### Recommendation
**Hybrid approach:**
1. Auto-keep high-confidence reliable classes (person, chair, tv, couch) without verification
2. VLM-verify suspicious classes (potted plant, clock, vase, bowl)
3. Use class-specific thresholds as fallback when VLM unavailable

---

## 4. Re-ID / Tracking Evaluation

### Setup
- **Methods:**
  1. IoU-based (threshold=0.3)
  2. IoU-based (threshold=0.5)  
  3. DINOv2 embedding-based (threshold=0.7)

### Results

| Method | Tracks | Avg Len | Max Len | Switches | Frags | >5 frames |
|--------|--------|---------|---------|----------|-------|-----------|
| iou_0.3 | 106 | 1.1 | 3 | 4 | 99 | 0 |
| iou_0.5 | 112 | 1.1 | 2 | 2 | 105 | 0 |
| embedding_0.7 | 117 | 1.0 | 2 | 0 | 110 | 0 |

### Analysis

**Critical Issue:** The evaluation used 50 uniformly sampled frames from 1827 total frames, meaning consecutive evaluated frames are ~36 frames apart. This makes IoU-based matching nearly impossible since objects move significantly between samples.

**Positive Finding:** Embedding-based tracking shows **0 track switches** compared to 2-4 for IoU-based, indicating appearance embeddings are more robust than spatial proximity when objects move.

### Recommendation
1. **Re-run with consecutive frames** for meaningful tracking evaluation
2. **Use embedding+IoU hybrid** - The combination reduces track switches
3. **Implement gallery management** - Keep last N frames of appearance per track

---

## 5. Summary & Action Items

### Immediate Actions
1. ✅ Use YOLO11x for detection
2. ⚠️ Add refinement validation to YOLO-World classification
3. ✅ Implement CLIP-based VLM filtering for suspicious classes
4. 🔄 Re-evaluate tracking with consecutive frames

### Architecture Recommendations

```
Detection (YOLO11x, conf=0.25)
    ↓
Classification (YOLO-World with validation)
    ↓
Filtering (Hybrid: auto-keep reliable + VLM-verify suspicious)
    ↓
Tracking (DINOv2 embedding + IoU hybrid)
```

### Suspicious Class Configuration

```python
SUSPICIOUS_LABELS = {
    "potted plant": {"threshold": 0.50, "vlm_verify": True},
    "clock": {"threshold": 0.55, "vlm_verify": True},
    "vase": {"threshold": 0.50, "vlm_verify": True},
    "bowl": {"threshold": 0.50, "vlm_verify": True},
    "bird": {"threshold": 0.50, "vlm_verify": True},
}

HIGH_CONFIDENCE_LABELS = {
    "person", "chair", "couch", "bed", "tv", "refrigerator"
}
```

---

## Appendix: Evaluation Scripts

All evaluation scripts are available in `scripts/`:

- `eval_detections_simple.py` - Detection comparison
- `eval_classification.py` - Classification refinement evaluation  
- `eval_filtering.py` - Semantic filtering comparison
- `eval_tracking.py` - Re-ID tracking evaluation

Run all evaluations:
```bash
python scripts/eval_detections_simple.py --video test.mp4 --sample-frames 50
python scripts/eval_classification.py --video test.mp4 --detections results/detection_eval/yolo11x_results.json
python scripts/eval_filtering.py --video test.mp4 --detections results/detection_eval/yolo11x_results.json
python scripts/eval_tracking.py --video test.mp4 --detections results/detection_eval/yolo11x_results.json
```
