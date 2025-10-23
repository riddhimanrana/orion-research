# Quick Start: Orion Evaluation Pipeline

## What Was Fixed

✅ **Script 3** now correctly runs Orion's full pipeline on VIDEO FILES (not frames)
✅ **Metrics** include Recall@K and Mean Rank for paper comparison  
✅ **Script 4** computes all metrics and saves to JSON

---

## TL;DR - What to Run

### Prerequisite: Get Action Genome Dataset

```bash
# Download Charades 480p videos
# From: https://prior.allenai.org/projects/charades
# Save to: dataset/ag/videos/

# Download annotations
# From: https://drive.google.com/drive/folders/1LGGPK_QgGbh9gH9SDFv_9LIhBliZbZys
# Save to: dataset/ag/annotations/
```

### Run the Pipeline (3 Commands)

```bash
# Step 1: Prepare 50 test clips and ground truth
python scripts/1_prepare_ag_data.py

# Step 2: Run Orion on all 50 clips (creates scene graphs)
# WARNING: Takes 30-60 minutes depending on hardware
python scripts/3_run_orion_ag_eval.py

# Step 3: Evaluate and compute metrics
python scripts/4_evaluate_ag_predictions.py
```

---

## What Each Script Does

### 1️⃣ `scripts/1_prepare_ag_data.py`
- Loads Action Genome annotations
- Extracts first 50 video clips
- Creates ground truth scene graphs
- **Output:** `data/ag_50/ground_truth_graphs.json`

### 2️⃣ `scripts/3_run_orion_ag_eval.py` ⭐ FIXED
- Takes VIDEO files from `dataset/ag/videos/`
- Runs Orion's full pipeline on each video
- Part 1: Perception (object detection + tracking)
- Part 2: Semantic Uplift (scene graph generation)
- **Output:** `data/ag_50/results/predictions.json`
- **Key Fix:** Now passes actual videos, not single frames

### 3️⃣ `scripts/4_evaluate_ag_predictions.py` ⭐ ENHANCED
- Compares predictions vs ground truth
- **Computes:**
  - Precision/Recall/F1 for edges, events, causal links
  - **R@10, R@20, R@50** (Recall@K ranking metrics)
  - **mR** (Mean Recall per category)
  - **MR** (Mean Rank - average position of good predictions)
  - Entity Jaccard similarity
- **Output:** `data/ag_50/results/metrics.json`

---

## Understanding the Metrics

### Standard Metrics (Already existed)
- **Precision/Recall/F1**: How accurate are entities, relationships, events?
- **Jaccard**: How much entity overlap with ground truth?

### NEW Ranking Metrics (For Paper Comparison)

| Metric | Meaning | Better if | Range |
|--------|---------|-----------|-------|
| **R@10** | % of GT relationships found in top-10 predictions | Higher | 0-100% |
| **R@20** | % of GT relationships found in top-20 predictions | Higher | 0-100% |
| **R@50** | % of GT relationships found in top-50 predictions | Higher | 0-100% |
| **mR** | Average recall across all relationship types | Higher | 0-100% |
| **MR** | Average rank position of correct predictions | Lower | 1-50 |

**Example Interpretation:**
- R@10 = 45% means we find 45% of relationships in top-10 predictions
- MR = 25 means our good predictions appear at position 25 on average
- Higher R@K + Lower MR = Better ranking quality

---

## Expected Output

After running all 3 scripts, you'll have:

```
data/ag_50/
├── ground_truth_graphs.json      (Ground truth scene graphs)
└── results/
    ├── predictions.json           (Orion's predictions)
    ├── metrics.json               (All evaluation metrics)
    └── intermediate/              (Per-clip intermediate files)
```

### Example metrics.json structure:
```json
{
  "aggregated": {
    "edges": {
      "precision": 0.45,
      "recall": 0.52,
      "f1": 0.48
    },
    "recall_at_k": {
      "R@10": 35.67,
      "R@20": 52.14,
      "R@50": 71.32,
      "mR": 53.38,
      "MR": 24.5
    }
  }
}
```

---

## Troubleshooting

### Script 3 Fails with "Video not found"
- Check videos are in `dataset/ag/videos/`
- Verify filenames match clip IDs
- Have ffmpeg installed: `which ffmpeg`

### Script 4 Shows all zeros
- Ensure Script 3 completed (check `predictions.json` exists)
- Check first few predictions are non-empty
- Verify ground truth format is correct

### Performance is slow
- Part 1 (perception) takes 5-10 min/video
- Part 2 (semantic uplift) takes 3-5 min/video
- Total: ~15 min/video = ~750 mins for 50 clips (~12.5 hours)
- Try with faster config: edit script to use 'fast' instead of 'balanced'

---

## Key Improvements Made

### Before (Broken):
- Script 3 fed single JPEG frames to run_pipeline()
- Pipeline treated frame as full video, gave fake/empty results
- Metrics always showed 1.0 F1 (copying ground truth)
- No ranking metrics for paper comparison

### After (Fixed):
- ✅ Script 3 handles full video files properly
- ✅ Creates temporary MP4 from frames if needed
- ✅ Full Orion pipeline runs (perception + semantic uplift)
- ✅ Real scene graph predictions vs ground truth
- ✅ Recall@K metrics for paper baseline comparison
- ✅ Per-category performance breakdown
- ✅ Better error handling and logging

---

## Paper Comparison

To compare against HyperGLM or other baselines:
1. Check `MR` (Mean Rank) - lower is better
2. Check `R@50` - higher is better for final ranking metric
3. Check per-category breakdown for domain-specific performance

Use these metrics in your paper evaluation section.

