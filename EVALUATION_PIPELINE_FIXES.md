# Orion Evaluation Pipeline - Fixes & Implementation

## Summary of Changes

### ✅ FIXED: Script 3 - Run Orion Pipeline on Action Genome

**Problem:** Original script tried to feed single JPEG frames to `run_pipeline()`, which expects video files. Result was zero actual testing of Orion's scene graph generation.

**Solution:** Complete rewrite of `scripts/3_run_orion_ag_eval.py`

#### Key Improvements:

1. **Proper Video Handling**
   - Added `find_video_for_clip()` - searches for video files by clip ID
   - Added `create_video_from_frames()` - generates MP4 from frame sequences using ffmpeg
   - Graceful fallback chain: existing video → create from frames → skip

2. **Full Pipeline Integration**
   - Now correctly passes VIDEO files to `run_pipeline()`
   - Uses both Part 1 (perception) and Part 2 (semantic uplift)
   - Extracts actual scene graph output (entities, relationships, events, causal links)

3. **Robust Error Handling**
   - Tracks failed clips separately
   - Logs informative warnings/errors
   - Returns statistics on success/failure rates

4. **Output Format**
   - Predictions now include:
     - `entities`: Detected objects with classes, bboxes, temporal info
     - `relationships`: Spatial/semantic relationships between entities
     - `events`: Detected actions with temporal bounds
     - `causal_links`: Inferred causal relationships

---

### ✅ NEW: Recall@K and Mean Rank (MR) Metrics

**Added to:** `orion/evaluation/recall_at_k.py`

#### Metrics Implemented:

1. **R@K (Recall at K)**
   - R@10, R@20, R@50
   - Measures: % of ground truth relationships recalled in top-K predictions
   - Formula: `recalled_relationships[K] / total_gt_relationships * 100`

2. **mR (Mean Recall)**
   - Average recall across all relationship categories
   - Per-category recall is averaged

3. **MR (Mean Rank)**
   - Average rank position of recalled items
   - Lower is better (closer to 1 = predictions appear early)
   - Useful for ranking quality assessment

#### Key Features:
- Computes IoU-based matching for bounding boxes (configurable threshold)
- Handles per-category breakdown
- Compatible with HyperGLM evaluation protocol
- Detailed per-category statistics

---

### ✅ ENHANCED: Script 4 - Evaluation with All Metrics

**File:** `scripts/4_evaluate_ag_predictions.py`

#### Now Computes:

**Standard Metrics:**
- Edge precision/recall/F1
- Event precision/recall/F1
- Causal link precision/recall/F1
- Entity Jaccard similarity

**Ranking Metrics (NEW):**
- Recall@10, Recall@20, Recall@50
- Mean Recall (mR)
- Mean Rank (MR)
- Per-category breakdown

#### Output Format:
```json
{
  "aggregated": {
    "edges": {"precision": 0.XX, "recall": 0.XX, "f1": 0.XX},
    "events": {"precision": 0.XX, "recall": 0.XX, "f1": 0.XX},
    "causal": {"precision": 0.XX, "recall": 0.XX, "f1": 0.XX},
    "entities": {"jaccard_similarity": 0.XX}
  },
  "recall_at_k": {
    "R@10": 45.23,
    "R@20": 62.15,
    "R@50": 78.32,
    "mR": 61.90,
    "MR": 25.5,
    "per_category": {...}
  }
}
```

---

## How to Use

### Step 1: Prepare Data
```bash
python scripts/1_prepare_ag_data.py
```
- Extracts 50 Action Genome clips
- Creates ground truth graphs from annotations
- Outputs: `data/ag_50/ground_truth_graphs.json`

### Step 2: Run Orion Pipeline
```bash
python scripts/3_run_orion_ag_eval.py
```
- Processes 50 video clips through full Orion pipeline
- Handles video creation from frame sequences if needed
- Outputs: `data/ag_50/results/predictions.json`

**Requirements:**
- Action Genome dataset: `dataset/ag/videos/` and `dataset/ag/frames/`
- ffmpeg installed (for frame → video conversion if needed)
- Neo4j running (for Part 2 semantic uplift)

### Step 3: Evaluate Results
```bash
python scripts/4_evaluate_ag_predictions.py
```
- Compares predictions against ground truth
- Computes all metrics (standard + Recall@K + MR)
- Outputs: `data/ag_50/results/metrics.json`

---

## Architecture Changes

### Data Flow (Corrected)

```
Raw Videos (MP4/AVI)
       ↓
Frame Extraction (if needed)
       ↓
run_pipeline() [VIDEO FILE]
       ↓
  Part 1: Perception Engine
  ├─ Object Detection (YOLO)
  ├─ Tracking (DeepSORT/ByteTrack)
  └─ Bounding Boxes + Class Labels
       ↓
  Part 2: Semantic Uplift
  ├─ Scene Graph Generation
  ├─ Relationship Extraction
  ├─ Event Detection
  └─ Causal Reasoning
       ↓
Knowledge Graph (Neo4j)
       ↓
Predictions JSON
       ↓
Evaluation Metrics
  ├─ Precision/Recall/F1
  ├─ Recall@K (R@10, R@20, R@50)
  ├─ Mean Recall (mR)
  └─ Mean Rank (MR)
```

### Important Notes

1. **Video Files Required**
   - Script 3 now correctly processes VIDEO FILES, not single frames
   - Supports multiple formats: MP4, AVI, MOV, MKV
   - If only frames available, creates temporary MP4 via ffmpeg

2. **Metric Meanings (for Paper Comparison)**
   - **R@K**: What % of relationships are found in top-K predictions?
   - **mR**: Per-category average - how good are we on average?
   - **MR**: Ranking metric - how early do good predictions appear?

3. **Evaluation vs. Ground Truth**
   - Uses IoU ≥ 0.5 for bounding box matching
   - Matches predicates exactly
   - Per-category statistics available for detailed analysis

---

## File Modifications Summary

### Modified Files:
1. **scripts/3_run_orion_ag_eval.py** - Complete rewrite (113 lines)
2. **scripts/4_evaluate_ag_predictions.py** - Major enhancement (232 lines)
3. **orion/evaluation/recall_at_k.py** - Added MR computation (10 lines)

### Backward Compatibility:
- All changes are additive
- Existing metrics unchanged
- RecallAtK class has new `MR` field in output

---

## Next Steps

1. **Download Action Genome Dataset**
   ```bash
   # Download Charades videos (480p) from https://prior.allenai.org/projects/charades
   # Extract to: dataset/ag/videos/
   
   # Download annotations from Google Drive
   # Extract to: dataset/ag/annotations/
   ```

2. **Run Scripts in Order**
   ```bash
   python scripts/1_prepare_ag_data.py
   python scripts/3_run_orion_ag_eval.py      # Takes 30-60 mins for 50 clips
   python scripts/4_evaluate_ag_predictions.py
   ```

3. **Analyze Results**
   - Check `data/ag_50/results/metrics.json` for detailed scores
   - Compare R@K with paper baselines
   - Use per-category breakdown to identify weak areas

---

## Troubleshooting

### Script 3 Issues:

1. **"Video not found for clip"**
   - Ensure videos are in `dataset/ag/videos/`
   - Check clip naming matches video filenames
   - Verify ffmpeg is installed: `which ffmpeg`

2. **"Pipeline failed"**
   - Check Neo4j is running: `neo4j status`
   - Verify frames exist in `dataset/ag/frames/`
   - Check disk space for temporary videos

3. **"Frame extraction failed"**
   - Install ffmpeg: `brew install ffmpeg` (macOS)
   - Verify video file format

### Script 4 Issues:

1. **"Predictions not found"**
   - Ensure Script 3 completed successfully
   - Check `data/ag_50/results/predictions.json` exists

2. **Metrics showing all zeros**
   - Ground truth format mismatch
   - Predictions empty or malformed
   - Check first prediction manually: `head -50 data/ag_50/results/predictions.json`

---

## Validation Checklist

- [x] Script 3 passes video files to run_pipeline() (not single frames)
- [x] Recall@K metrics implemented with IoU matching
- [x] Mean Rank (MR) metric added
- [x] Script 4 includes all metrics in output
- [x] Per-category statistics available
- [x] Error handling and logging improved
- [x] Syntax validated (no Python errors)
- [x] Backward compatible with existing evaluation code

