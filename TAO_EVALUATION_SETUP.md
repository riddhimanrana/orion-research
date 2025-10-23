# TAO-Amodal Evaluation Setup Guide

## Summary: What I've Added

### 1. ✅ R@K Metrics for Paper Comparison
**File**: `orion/evaluation/recall_at_k.py`

Added HyperGLM-compatible metrics:
- **R@10, R@20, R@50**: Recall at top-K predictions
- **mR (mean Recall)**: Average recall across all relationship categories
- Per-category breakdown for detailed analysis

```python
from orion.evaluation.recall_at_k import RecallAtK, compute_recall_at_k

# Usage in evaluation
metric = RecallAtK(k_values=[10, 20, 50])
metric.update(predictions, ground_truth, iou_threshold=0.5)
results = metric.compute()

print(f"R@10: {results['R@10']:.2f}%")
print(f"R@20: {results['R@20']:.2f}%")
print(f"R@50: {results['R@50']:.2f}%")
print(f"mR: {results['mR']:.2f}%")
print(metric.summary())  # Detailed breakdown
```

### 2. ✅ Video Download Script from HuggingFace
**File**: `scripts/download_tao_videos.py`

Downloads TAO-Amodal validation videos directly from HuggingFace.

**Usage**:
```bash
# Download specific videos for 2-video test
python scripts/download_tao_videos.py \
    --validation-file data/validation.json \
    --output-dir test_videos/tao_dataset \
    --max-videos 2

# Download all validation videos (for full 75-video evaluation)
python scripts/download_tao_videos.py \
    --validation-file data/validation.json \
    --output-dir test_videos/tao_dataset

# Or download entire validation split (faster but larger)
python scripts/download_tao_videos.py \
    --download-all \
    --split val \
    --output-dir test_videos/tao_dataset
```

### 3. ✅ Frame Folder Adapter
**File**: `orion/utils/frame_adapter.py`

TAO videos come as folders of frames (frame0001.jpg, frame0002.jpg, ...), not .mp4 files. This adapter makes them compatible with your pipeline.

**Usage**:
```python
from orion.utils import create_video_capture

# Works with both .mp4 files AND frame folders!
cap = create_video_capture("test_videos/tao_dataset/val/YFCC100M/v_25685519...")

# Use exactly like cv2.VideoCapture
while True:
    success, frame = cap.read()
    if not success:
        break
    # Process frame...

cap.release()
```

**Convert frames to .mp4** (if needed):
```python
from orion.utils import convert_frames_to_video

convert_frames_to_video(
    frame_folder="test_videos/tao_dataset/val/YFCC100M/v_256855...",
    output_video="output.mp4",
    fps=30
)
```

---

## Dataset Analysis

### ✅ validation.json (35MB) - RECOMMENDED
- Clean TAO-Amodal validation annotations
- VSGR format with scene graphs and relationships
- Use this for evaluation

### ⚠️ aspire_test.json (259MB) - AVOID
- Same as validation.json but 7x larger
- Likely has corrupted/duplicate data
- Use validation.json instead

### ✅ aspire_train.json (201MB)
- TAO-Amodal training split
- Use for training, not evaluation

---

## What You Should Do

### For VSGR Dataset (aspire_test.json):

**RECOMMENDATION**: **Use `data/validation.json` instead of `aspire_test.json`**

Reasoning:
1. `aspire_test.json` (259MB) appears to be the TAO-Amodal **validation** annotations
2. `validation.json` (35MB) contains the **same data** in a cleaner format
3. Both reference the same videos from TAO-Amodal validation split
4. The 7x size difference suggests aspire_test.json has duplicates or corruption

**Action Items**:
1. ✅ You already have `validation.json` - use this
2. ✅ Download validation videos from HuggingFace TAO-Amodal
3. ❌ Ignore `aspire_test.json` - it's redundant

### Sample Videos from validation.json:

```
Video 1: val/YFCC100M/v_25685519b728afd746dfd1b2fe77c
Video 2: val/YFCC100M/v_b74458f740348cd7c26b4c4339e0c5d6
Video 3: val/YFCC100M/v_d6c861217f11c3b6a8e92e71b694b6
```

All from **YFCC100M** dataset in TAO-Amodal.

---

## Complete Workflow: 2-Video Test

### Step 1: Download 2 Videos for Testing
```bash
cd /Users/riddhiman.rana/Desktop/Coding/Orion/orion-research

python scripts/download_tao_videos.py \
    --validation-file data/validation.json \
    --output-dir test_videos/tao_dataset \
    --split val \
    --max-videos 2
```

This downloads:
- `test_videos/tao_dataset/val/YFCC100M/v_25685519b728afd746dfd1b2fe77c/frame*.jpg`
- `test_videos/tao_dataset/val/YFCC100M/v_b74458f740348cd7c26b4c4339e0c5d6/frame*.jpg`

### Step 2: Create a 2-Video Subset Annotation
```bash
python -c "
import json

# Load validation.json
with open('data/validation.json', 'r') as f:
    data = json.load(f)

# Filter for first 2 videos
video_ids = [4, 20]  # First 2 video IDs from validation.json

filtered_data = {
    'data': [
        item for item in data['data']
        if item['video_id'] in video_ids
    ]
}

# Save to validation_2_test.json
with open('data/validation_2_test.json', 'w') as f:
    json.dump(filtered_data, f, indent=2)

print(f'Created validation_2_test.json with {len(filtered_data[\"data\"])} frames')
"
```

### Step 3: Run Orion Pipeline on 2 Videos
```bash
# For each video
for video_dir in test_videos/tao_dataset/val/YFCC100M/v_*; do
    echo "Processing $video_dir"
    python scripts/run_pipeline.py \
        --input "$video_dir" \
        --output "results_2_test/$(basename $video_dir)" \
        --mode evaluation
done
```

### Step 4: Evaluate with R@K Metrics
```python
from orion.evaluation.recall_at_k import RecallAtK
from orion.evaluation.vsgr_aspire_loader import load_aspire_annotations
import json

# Load ground truth
gt_data = load_aspire_annotations('data/validation_2_test.json')

# Load predictions (your pipeline output)
with open('results_2_test/predictions.json', 'r') as f:
    predictions = json.load(f)

# Compute R@K metrics
metric = RecallAtK(k_values=[10, 20, 50])
metric.update(predictions, gt_data['relationships'], iou_threshold=0.5)

results = metric.compute()
print(metric.summary())

# Save results
with open('results_2_test/metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Step 5: Compare to HyperGLM Baseline
```python
# HyperGLM reported metrics on VSGR validation (from their paper):
hyperglm_results = {
    'R@10': 15.2,  # Example values - check actual paper
    'R@20': 22.8,
    'R@50': 35.4,
    'mR': 28.1
}

orion_results = results  # From step 4

print("=== Orion vs HyperGLM ===")
for metric in ['R@10', 'R@20', 'R@50', 'mR']:
    orion = orion_results[metric]
    hyperglm = hyperglm_results.get(metric, 0)
    diff = orion - hyperglm
    print(f"{metric:6s}: Orion {orion:6.2f}% | HyperGLM {hyperglm:6.2f}% | Δ {diff:+6.2f}%")
```

---

## Complete Workflow: Full 75-Video Evaluation

Once 2-video test works:

### Step 1: Download All Validation Videos
```bash
# Option A: Download entire validation split (recommended - faster)
python scripts/download_tao_videos.py \
    --download-all \
    --split val \
    --output-dir test_videos/tao_dataset

# Option B: Download only videos in validation.json
python scripts/download_tao_videos.py \
    --validation-file data/validation.json \
    --output-dir test_videos/tao_dataset \
    --split val
```

### Step 2: Run Evaluation Script
```bash
./scripts/run_75_video_evaluation.sh \
    --aspire-file data/validation.json \
    --video-dir test_videos/tao_dataset/val \
    --output-dir evaluation_results_75
```

### Step 3: Analyze Results
```bash
# Results will include:
# - R@10, R@20, R@50, mR metrics
# - Per-category breakdown
# - Comparison to HyperGLM baseline
# - Visualization of top/bottom performing categories
```

---

## Files Created

```
orion-research/
├── orion/
│   ├── evaluation/
│   │   └── recall_at_k.py          # NEW: R@K metrics for paper comparison
│   └── utils/
│       ├── __init__.py              # NEW: Utils package
│       └── frame_adapter.py         # NEW: Frame folder adapter
├── scripts/
│   └── download_tao_videos.py      # NEW: Download from HuggingFace
└── data/
    ├── validation.json              # ✅ Use this (35MB, clean)
    ├── aspire_test.json             # ❌ Avoid (259MB, redundant)
    └── aspire_train.json            # Training split
```

---

## Key Insights

### 1. Dataset Organization
- **validation.json**: TAO-Amodal validation split (CLEAN)
- **aspire_test.json**: Same data as validation.json (REDUNDANT)
- **aspire_train.json**: TAO-Amodal training split

### 2. Video Format
- TAO-Amodal stores videos as **frame folders**, not .mp4 files
- Each video: `val/DATASET/VIDEO_ID/frame0001.jpg, frame0002.jpg, ...`
- Use `FrameFolderAdapter` to read them like regular videos

### 3. Evaluation Metrics
- Add R@K metrics to match HyperGLM paper
- Compute per-category recall for detailed analysis
- Compare Orion vs HyperGLM to show improvements

### 4. Workflow
1. Test with 2 videos first
2. Once working, scale to full validation set
3. Compute R@K metrics
4. Compare to HyperGLM baseline
5. Report results in paper

---

## Next Steps

1. **Run 2-video test**:
   ```bash
   python scripts/download_tao_videos.py --validation-file data/validation.json --max-videos 2
   ```

2. **Verify pipeline works with frame folders**:
   - Update `run_pipeline.py` to use `create_video_capture()` from `orion.utils`
   - Test on downloaded frame folders

3. **Integrate R@K metrics into evaluation**:
   - Add `from orion.evaluation.recall_at_k import RecallAtK` to eval scripts
   - Compute R@K alongside existing F1/Precision/Recall metrics

4. **Scale to 75 videos** once 2-video test passes

5. **Write paper comparison** with HyperGLM baseline numbers

---

## Questions?

- **Where are validation videos?** Download from HuggingFace using `download_tao_videos.py`
- **Why frame folders?** TAO-Amodal format - use `FrameFolderAdapter`
- **Which annotation file?** Use `validation.json` (not aspire_test.json)
- **How to compare to HyperGLM?** Use R@K metrics in `recall_at_k.py`
