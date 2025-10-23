# ORION Evaluation Pipeline - 50 TAO Videos

## Quick Start

Run these commands in order:

### Step 1: Extract 50 videos from TAO validation set
```bash
python scripts/1_download_50_tao_videos.py
```
**Output:** `data/tao_validation_50/video_list.txt` (list of 50 videos)

### Step 2: Prepare ground truth annotations
```bash
python scripts/2_prepare_ground_truth.py
```
**Output:** `data/tao_validation_50/ground_truth.json` (SGG format)

### Step 3: Run Orion pipeline
```bash
python scripts/3_run_orion_eval.py
```
This will display instructions for running Orion.

**Manual step:** Run Orion on the 50 videos and save predictions to:
```
data/tao_validation_50/results/predictions.json
```

### Step 4: Evaluate predictions
```bash
python scripts/4_evaluate_predictions.py
```
**Output:** Metrics in `data/tao_validation_50/results/metrics.json`

---

## Predictions Format

Save Orion predictions as JSON array to `data/tao_validation_50/results/predictions.json`:

```json
[
  {
    "image_id": 123,
    "bbox": [x, y, width, height],
    "score": 0.95,
    "category_id": 1
  },
  ...
]
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Recall@10** | % of ground truth objects in top-10 predictions |
| **Recall@20** | % of ground truth objects in top-20 predictions |
| **Recall@50** | % of ground truth objects in top-50 predictions |
| **mRecall** | Average of Recall@10, @20, @50 |
| **BBox IoU@0.5** | Average intersection-over-union for matched boxes |

---

## Directory Structure

```
data/tao_validation_50/
├── video_list.txt              # List of 50 videos
├── ground_truth.json           # Ground truth annotations (SGG format)
└── results/
    ├── predictions.json        # Orion predictions (manually created)
    └── metrics.json            # Evaluation results
```

---

## Notes

- TAO dataset has **no relationship annotations** (unlike VSGR)
- Evaluation focuses on **object detection** (bboxes only)
- Use frame_adapter to load TAO frame folders as video

---

## Full Pipeline

```
1_download_50_tao_videos.py
        ↓
2_prepare_ground_truth.py
        ↓
[Run Orion manually]
        ↓
4_evaluate_predictions.py
        ↓
metrics.json (Recall@K, BBox IoU, mRecall)
```
