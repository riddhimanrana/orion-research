# VSGR Annotation Format Examples

## What You'll Get from HyperGLM Authors

When they send you the VSGR annotations, they'll look like this:

### Train.json Structure

```json
{
  "video_1/ArgoVerse_001": {
    "video_id": "ArgoVerse_001",
    "num_frames": 150,
    "temporal_order": [0, 1, 2, ..., 149],
    
    // Scene graphs (per frame)
    "scene_graphs": {
      "0": [
        {"subject": 5, "relation": "holding", "object": 12},
        {"subject": 5, "relation": "near", "object": 8},
        {"subject": 12, "relation": "on", "object": 15}
      ],
      "1": [
        {"subject": 5, "relation": "holding", "object": 12},
        {"subject": 12, "relation": "on", "object": 15}
      ]
    },
    
    // Causal relationships (between events/frames)
    "causal_links": [
      {
        "source_frame": 0,
        "target_frame": 5,
        "confidence": 0.85,
        "causal_type": "direct",
        "description": "Person picks up cup -> cup moves"
      },
      {
        "source_frame": 5,
        "target_frame": 10,
        "confidence": 0.72,
        "causal_type": "temporal",
        "description": "Cup moves -> cup placed on table"
      }
    ],
    
    // Objects with metadata
    "objects": [
      {
        "id": 5,
        "class_name": "person",
        "track_id": 0,
        "appearances": [0, 1, 2, ..., 149]
      },
      {
        "id": 12,
        "class_name": "cup",
        "track_id": 1,
        "appearances": [0, 1, 2, ..., 50]
      },
      {
        "id": 8,
        "class_name": "table",
        "track_id": 2,
        "appearances": [0, 1, 2, ..., 149]
      }
    ]
  },
  
  "video_2/BDD_002": {
    // ... same structure
  }
}
```

### Combining with TAO-Amodal

TAO-Amodal provides bounding boxes for same videos:

```json
// From TAO-Amodal amodal_annotations/test.json
{
  "ArgoVerse_001": {
    "annotations": [
      {
        "id": 1,
        "image_id": 0,
        "track_id": 0,
        "bbox": [100, 200, 150, 300],           // (x, y, w, h)
        "amodal_bbox": [95, 195, 160, 310],     // includes occluded parts
        "category_id": 1,                        // LVIS class ID
        "visibility": 0.95,                      // how much is visible (0-1)
        "area": 45000.0,
        "category_name": "person"
      },
      {
        "id": 2,
        "image_id": 0,
        "track_id": 1,
        "bbox": [300, 150, 80, 100],
        "amodal_bbox": [300, 150, 80, 100],
        "category_id": 50,
        "visibility": 1.0,
        "area": 8000.0,
        "category_name": "cup"
      }
    ]
  }
}
```

---

## What Your Evaluation Script Needs to Do

### 1. Match VSGR Scene Graphs to TAO-Amodal Bounding Boxes

```python
# For each frame in video:
frame_idx = 0

# Get VSGR scene graph triplets
sg_triplets = vsgr_ann[video_id]["scene_graphs"][str(frame_idx)]
# Returns: [{"subject": 5, "relation": "holding", "object": 12}, ...]

# Get TAO-Amodal bounding boxes for same frame
tao_bboxes = tao_ann[video_id]["annotations"]
# Filter for objects that appear in frame_idx
frame_bboxes = [bbox for bbox in tao_bboxes if frame_idx in bbox["appearances"]]

# Match: VSGR object IDs (5, 12, 8) -> TAO track IDs
# Your job: detect those objects with YOLO, compare to ground truth
```

### 2. Evaluate Predictions

**Your YOLO predictions for frame 0:**
```python
predictions = {
    'detections': [
        {'bbox': [102, 205, 148, 298], 'class': 'person', 'conf': 0.92},
        {'bbox': [298, 148, 82, 102], 'class': 'cup', 'conf': 0.88},
        {'bbox': [200, 180, 100, 120], 'class': 'spoon', 'conf': 0.65}  # false positive
    ]
}

ground_truth = {
    'detections': [
        {'bbox': [100, 200, 150, 300], 'class': 'person'},
        {'bbox': [300, 150, 80, 100], 'class': 'cup'}
    ]
}

# Match using IoU, compute metrics
# F1 = 2 * (P * R) / (P + R)
# Where P = 2/3 (2 correct, 1 false positive)
#       R = 2/2 (2 found, 2 ground truth)
```

### 3. Evaluate Scene Graphs

**Your predictions (from Orion):**
```python
predicted_triplets = [
    {'subject': 5, 'relation': 'holding', 'object': 12},
    {'subject': 12, 'relation': 'on', 'object': 8}
]

ground_truth_triplets = [
    {'subject': 5, 'relation': 'holding', 'object': 12},
    {'subject': 5, 'relation': 'near', 'object': 8},
    {'subject': 12, 'relation': 'on', 'object': 15}
]

# Compute Recall@K:
# Top-1: 1/3 correct (1st triplet matches)
# Recall@1 = 1/3 = 0.33
# 
# Top-2: 2/3 correct (1st and 2nd match)
# Recall@2 = 2/3 = 0.67
```

### 4. Evaluate Causality

**Your causal predictions:**
```python
predicted_causal = [
    {
        "source_frame": 0,
        "target_frame": 5,
        "entities": [5, 12],  # Person and cup involved
        "confidence": 0.88
    }
]

ground_truth_causal = [
    {
        "source_frame": 0,
        "target_frame": 5,
        "confidence": 0.85,
        "description": "Person picks up cup -> cup moves"
    }
]

# Evaluate:
# - Temporal accuracy: Is source before target? (0 < 5: YES)
# - Entity consistency: Do same entities appear in both frames?
# - Causal F1: Is causal link correct?
```

---

## Dataset Size Reference

| Metric | Value |
|--------|-------|
| Total Videos | ~5,000 |
| Total Frames | 1.9M |
| Scene Graph Triplets | ~500K |
| Causal Links | ~50K |
| Objects per Frame (avg) | 3-5 |
| Scene Graph Types | ~100 relationships |

---

## Downloading Strategy

```bash
# Option 1: Full download (500GB+)
git clone https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal data/tao-amodal
cd data/tao-amodal
python unzip_video.py

# Option 2: Subset download (start here!)
# Only download ArgoVerse subset (~50GB)
git clone https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal data/tao-amodal
cd data/tao-amodal/frames/train
# Delete BDD, Charades, HACS, LaSOT, YFCC100M folders
# Keep only ArgoVerse/

# Option 3: Stream from HuggingFace (slow but works)
from datasets import load_dataset
ds = load_dataset("chengyenhsieh/TAO-Amodal", split="train")
```

**My Recommendation:** Start with **ArgoVerse only** (~50GB, 50K videos) to test your pipeline. Once working, scale to full dataset.

---

## Quick Test to Verify Setup

```python
from orion.evaluation.datasets import VSGRDataset

# Try loading
dataset = VSGRDataset(
    root_dir='data/vsgr',
    tao_dir='data/tao-amodal',
    split='test'
)

sample = dataset[0]
print(f"✓ Video ID: {sample['video_id']}")
print(f"✓ Frames: {len(sample['frames'])}")
print(f"✓ Objects: {len(sample['objects'])}")
print(f"✓ Scene graphs: {len(sample['scene_graphs'])}")
print(f"✓ Causal links: {len(sample['causal_links'])}")

# If all print successfully, your setup is correct!
```

