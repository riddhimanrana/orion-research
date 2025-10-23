# ORION Evaluation Setup - Ready to Go!

## ✅ Setup Complete

All scripts are created and tested. Here's exactly what to run:

---

## 🚀 Quick Commands

### 1. Identify 50 TAO videos
```bash
python3 scripts/1_download_50_tao_videos.py
```
**Output:** `data/tao_validation_50/video_list.txt`

### 2. Prepare ground truth
```bash
python3 scripts/2_prepare_ground_truth.py
```
**Output:** `data/tao_validation_50/ground_truth.json`

### 3. Run Orion pipeline
```bash
python3 scripts/3_run_orion_eval.py
```
**Manual step:** Run Orion on videos, save predictions to:
```
data/tao_validation_50/results/predictions.json
```

### 4. Evaluate results
```bash
python3 scripts/4_evaluate_predictions.py
```
**Output:** `data/tao_validation_50/results/metrics.json`

---

## 📊 What Gets Measured

| Metric | What It Means |
|--------|--------------|
| **Recall@10** | How many GT objects are in top-10 predictions |
| **Recall@20** | How many GT objects are in top-20 predictions |
| **Recall@50** | How many GT objects are in top-50 predictions |
| **mRecall** | Average of above three |
| **BBox IoU@0.5** | Spatial accuracy of detected boxes |

---

## 📁 Files Created

```
scripts/
├── 1_download_50_tao_videos.py      ← Identify 50 videos
├── 2_prepare_ground_truth.py        ← Create ground truth JSON
├── 3_run_orion_eval.py              ← Instructions for Orion
└── 4_evaluate_predictions.py        ← Compare & score
```

---

## 🎯 Data Flow

```
aspire_2_test.json (ground truth)
        ↓
1_download_50_tao_videos.py
   → video_list.txt
        ↓
2_prepare_ground_truth.py
   → ground_truth.json
        ↓
[Run Orion manually]
   → predictions.json
        ↓
4_evaluate_predictions.py
   → metrics.json (Recall@K, IoU, mRecall)
```

---

## 📝 Predictions Format

When running Orion, save output as `data/tao_validation_50/results/predictions.json`:

```json
[
  {
    "image_id": 1,
    "bbox": [100, 50, 200, 150],
    "score": 0.95,
    "category_id": 1
  },
  {
    "image_id": 1,
    "bbox": [300, 200, 100, 80],
    "score": 0.87,
    "category_id": 2
  },
  ...
]
```

**Fields:**
- `image_id`: matches ground truth image ID
- `bbox`: [x, y, width, height]
- `score`: confidence (0-1)
- `category_id`: object class

---

## ⚡ Next Steps

1. ✅ Run step 1-2 to prep data
2. 🎬 Set up Orion to process the 50 videos
3. 💾 Save predictions.json in correct location
4. 📊 Run step 4 to get final metrics

---

## 🔍 Cleanup Notes

- All old scripts deleted (only kept essential 4)
- Using `aspire_2_test.json` as ground truth source
- No LLM/relationship evaluation (TAO has no relationship labels)
- Focus: **Object Detection** metrics only

**Ready to evaluate!** 🎉
