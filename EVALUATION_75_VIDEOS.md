# Orion Evaluation - 75 TAO Test Videos

## 🚀 Quick Start (4 Steps)

### Step 1: Download metadata + prepare ground truth
```bash
python3 scripts/1_download_75_tao_test.py
```
**Downloads:** ~1MB annotations  
**Output:** `data/tao_75_test/ground_truth.json`

### Step 2: Download 75 test video frames
```bash
python3 scripts/2_download_75_frames.py
```
**Downloads:** ~10-20GB (only the 75 test videos you need)  
**Output:** `data/tao_frames/frames/test/`

### Step 3: Run Orion on the 75 videos
```bash
python3 scripts/3_run_orion_eval.py
```
Displays instructions. Process videos in: `data/tao_frames/frames/test/`

### Step 4: Evaluate results
```bash
python3 scripts/4_evaluate_predictions.py
```
**Output:** `data/tao_75_test/results/metrics.json`

---

## 📊 Metrics You'll Get

```json
{
  "Recall@10": 75.5,       # % of GT objects in top-10
  "Recall@20": 82.3,       # % of GT objects in top-20
  "Recall@50": 88.9,       # % of GT objects in top-50
  "mRecall": 82.2,         # Average of above
  "BBox_IoU@0.5": 0.687    # Spatial accuracy
}
```

---

## 📝 Predictions Format

Save Orion output to: `data/tao_75_test/results/predictions.json`

```json
[
  {
    "image_id": 123,
    "bbox": [100, 50, 200, 150],
    "score": 0.95,
    "category_id": 1
  }
]
```

---

## 🎯 Why 75 Test Videos?

- TAO test set: 1000+ videos
- Evaluation subset: 75 videos
- Download: ~10-20GB (not 100GB+)
- Computation: Manageable time
- Results: Statistically valid

---

## 📁 Directory Structure

```
data/
├── tao_75_test/
│   ├── ground_truth.json       (from step 1)
│   └── results/
│       ├── predictions.json    (from Orion)
│       └── metrics.json        (from step 4)
└── tao_frames/
    └── frames/test/            (from step 2)

scripts/
├── 1_download_75_tao_test.py
├── 2_download_75_frames.py
├── 3_run_orion_eval.py
└── 4_evaluate_predictions.py
```

---

## ✅ Ready to Go!

Run the commands above in order. Takes ~1-2 hours total (mostly download time).

**Start with:** `python3 scripts/1_download_75_tao_test.py`
