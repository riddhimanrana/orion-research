# VSGR Dataset Setup & Evaluation - Step by Step

## What's Happening

**VSGR = ASPIRE annotations ON TOP OF TAO-Amodal videos**

So:
1. TAO-Amodal = base videos + amodal bounding boxes
2. ASPIRE = adds scene graphs, causal relationships, relationships on TAO-Amodal
3. VSGR = HyperGLM's published benchmark with 1.9M frames

---

## Step 1: Download TAO-Amodal (Base Videos)

```bash
# Install git-lfs first (if not already)
brew install git-lfs  # macOS
# OR
sudo apt-get install git-lfs  # Linux

# Clone the TAO-Amodal dataset
cd ~/Desktop/Coding/Orion
git lfs install
git clone git@hf.co:datasets/chengyenhsieh/TAO-Amodal data/tao-amodal

# Alternative if SSH fails (use HTTPS)
git clone https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal data/tao-amodal

# Unzip videos
cd data/tao-amodal
python unzip_video.py  # Modify dataset_root in the script to point here
```

**Expected structure:**
```
data/tao-amodal/
├── frames/
│   └── train/
│       ├── ArgoVerse/
│       ├── BDD/
│       ├── Charades/
│       ├── HACS/
│       ├── LaSOT/
│       └── YFCC100M/
├── amodal_annotations/
│   ├── train.json
│   ├── validation.json
│   └── test.json
└── annotations/  (TAO original)
```

**Size warning:** ~500GB+ for full dataset
**Workaround:** Download subset (e.g., ArgoVerse only) to test

---

## Step 2: Get VSGR Annotations (Scene Graphs + Causal)

Option A: **Contact HyperGLM authors directly**
```
Email: thuann@uark.edu
Subject: "Request VSGR dataset and annotations"
They'll provide the annotation JSONs with scene graphs + causal links
```

Option B: **Check their GitHub**
```
https://github.com/uark-cviu/HyperGLM
Look for "Download VSGR" or "Dataset" section
```

**Expected files:**
```
data/vsgr/
├── annotations/
│   ├── train.json      # Scene graphs + causal for training videos
│   ├── validation.json
│   └── test.json
└── (videos symlink to TAO-Amodal frames)
```

---

## Step 3: Create VSGR Data Loader

File: `orion/evaluation/datasets/vsgr_loader.py`

```python
import json
import cv2
from pathlib import Path
from typing import List, Dict
import numpy as np

class VSGRDataset:
    def __init__(self, root_dir: str, tao_dir: str, split: str = 'test'):
        self.root_dir = Path(root_dir)
        self.tao_dir = Path(tao_dir)
        self.split = split
        
        # Load VSGR annotations (scene graphs + causal)
        ann_path = self.root_dir / 'annotations' / f'{split}.json'
        with open(ann_path) as f:
            self.vsgr_ann = json.load(f)
        
        # Load TAO-Amodal annotations (bboxes)
        tao_ann_path = self.tao_dir / 'amodal_annotations' / f'{split}.json'
        with open(tao_ann_path) as f:
            self.tao_ann = json.load(f)
        
        self.videos = list(self.vsgr_ann.keys())
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx: int) -> Dict:
        video_id = self.videos[idx]
        vsgr_data = self.vsgr_ann[video_id]
        tao_data = self.tao_ann[video_id]
        
        # Get video frames from TAO-Amodal
        video_path = self._find_video_path(video_id)
        frames = self._load_video_frames(video_path)
        
        return {
            'video_id': video_id,
            'frames': frames,
            'objects': tao_data['annotations'],  # bbox + class
            'scene_graphs': vsgr_data['scene_graphs'],  # (subj, rel, obj) triplets
            'causal_links': vsgr_data.get('causal_links', []),  # causality
            'temporal_order': vsgr_data['temporal_order']
        }
    
    def _find_video_path(self, video_id: str) -> Path:
        # Map video_id to TAO-Amodal video path
        source, vid_name = video_id.split('/')
        return self.tao_dir / 'frames' / 'train' / source / vid_name / '*.jpg'
    
    def _load_video_frames(self, pattern: str) -> List[np.ndarray]:
        from glob import glob
        frames = []
        for img_path in sorted(glob(str(pattern))):
            img = cv2.imread(img_path)
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return frames
```

---

## Step 4: Run YOLO Baseline

File: `scripts/evaluation/run_yolo_baseline.py`

```python
from ultralytics import YOLO
from orion.evaluation.metrics import ClassificationEvaluator
from orion.evaluation.datasets import VSGRDataset
import json

# Setup
dataset = VSGRDataset(
    root_dir='data/vsgr',
    tao_dir='data/tao-amodal',
    split='test'
)

model = YOLO('yolo11x.pt')
evaluator = ClassificationEvaluator(class_names=COCO_CLASSES)

# Run on first 100 videos (sanity check)
results = []
for idx in range(min(100, len(dataset))):
    sample = dataset[idx]
    frames = sample['frames']
    ground_truth_objects = sample['objects']
    
    # Detect
    predictions = []
    for frame in frames:
        yolo_results = model(frame)
        predictions.append({
            'boxes': yolo_results.boxes,
            'classes': yolo_results.boxes.cls,
            'confs': yolo_results.boxes.conf
        })
    
    results.append({
        'video_id': sample['video_id'],
        'predictions': predictions,
        'ground_truth': ground_truth_objects
    })

# Evaluate
metrics = evaluator.evaluate(results, ground_truth_objects)
print(f"YOLO Baseline F1: {metrics['f1']:.3f}")

# Save
with open('evaluation_results/yolo_baseline.json', 'w') as f:
    json.dump(metrics, f)
```

**Run:**
```bash
python scripts/evaluation/run_yolo_baseline.py --num_videos 100  # sanity check
python scripts/evaluation/run_yolo_baseline.py --num_videos all   # full run
```

---

## Step 5: Run Orion Full Evaluation

File: `scripts/evaluation/run_orion_evaluation.py`

```python
from orion.pipeline import OrionPipeline
from orion.config import get_balanced_config
from orion.evaluation.datasets import VSGRDataset
from orion.evaluation.metrics import (
    ClassificationEvaluator,
    SceneGraphEvaluator,
    CausalReasoningEvaluator
)
import json

# Setup
dataset = VSGRDataset(
    root_dir='data/vsgr',
    tao_dir='data/tao-amodal',
    split='test'
)

config = get_balanced_config()
pipeline = OrionPipeline(config)

# Run Orion on all test videos
orion_predictions = []
for idx, sample in enumerate(dataset):
    print(f"Processing {idx+1}/{len(dataset)}")
    
    # Get video
    video_path = sample['video_path']
    
    # Run Orion pipeline
    result = pipeline.process_video(video_path)
    
    # Extract predictions
    orion_predictions.append({
        'video_id': sample['video_id'],
        'entities': result.entities,
        'relationships': result.scene_graph.relationships,
        'events': result.events,
        'causal_links': result.causal_edges,
        'corrections': result.perception_log.corrections
    })

# Evaluate all metrics
evaluators = {
    'classification': ClassificationEvaluator(class_names=COCO_CLASSES),
    'scene_graph': SceneGraphEvaluator(),
    'causal': CausalReasoningEvaluator()
}

results = {}
for name, evaluator in evaluators.items():
    results[name] = evaluator.evaluate(orion_predictions, dataset)

# Print results
print(f"\n=== Orion Results ===")
print(f"Classification F1: {results['classification']['f1']:.3f}")
print(f"Scene Graph R@50: {results['scene_graph']['recall_at_50']:.3f}")
print(f"Causal Reasoning F1: {results['causal']['causal_f1']:.3f}")

# Save
with open('evaluation_results/orion_full.json', 'w') as f:
    json.dump(results, f)
```

**Run:**
```bash
python scripts/evaluation/run_orion_evaluation.py --config balanced
```

---

## Step 6: Run Ablations (5 Configurations)

File: `scripts/evaluation/run_ablations.py`

```python
from orion.pipeline import OrionPipeline
from orion.config import get_balanced_config
from orion.evaluation.datasets import VSGRDataset
from orion.evaluation.metrics import ClassificationEvaluator

configs = {
    'yolo_only': {
        'use_clip': False,
        'use_vlm': False,
        'use_semantic_validation': False
    },
    'yolo_clip': {
        'use_clip': True,
        'use_vlm': False,
        'use_semantic_validation': False
    },
    'yolo_clip_vlm': {
        'use_clip': True,
        'use_vlm': True,
        'use_semantic_validation': False
    },
    'full_orion': {
        'use_clip': True,
        'use_vlm': True,
        'use_semantic_validation': True
    }
}

dataset = VSGRDataset('data/vsgr', 'data/tao-amodal', 'test')
ablation_results = {}

for config_name, overrides in configs.items():
    print(f"\n=== {config_name} ===")
    
    config = get_balanced_config()
    config.update(overrides)
    
    pipeline = OrionPipeline(config)
    predictions = []
    
    for sample in dataset:
        result = pipeline.process_video(sample['video_path'])
        predictions.append(result)
    
    evaluator = ClassificationEvaluator(class_names=COCO_CLASSES)
    metrics = evaluator.evaluate(predictions, dataset)
    
    ablation_results[config_name] = metrics
    print(f"F1: {metrics['f1']:.3f}")

# Save
with open('evaluation_results/ablation_results.json', 'w') as f:
    json.dump(ablation_results, f)
```

**Run:**
```bash
python scripts/evaluation/run_ablations.py
```

---

## Step 7: Generate Tables & Figures

File: `scripts/evaluation/generate_tables.py`

```python
import json
import pandas as pd

# Load all results
with open('evaluation_results/yolo_baseline.json') as f:
    yolo = json.load(f)
with open('evaluation_results/orion_full.json') as f:
    orion = json.load(f)

# Table 1: Classification
table1 = pd.DataFrame({
    'Method': ['YOLO Baseline', 'Orion'],
    'Precision': [yolo['precision'], orion['classification']['precision']],
    'Recall': [yolo['recall'], orion['classification']['recall']],
    'F1': [yolo['f1'], orion['classification']['f1']],
    'mAP': [yolo['mAP'], orion['classification']['mAP']]
})
print("=== Table 1: Classification ===")
print(table1.to_string(index=False))

# Save LaTeX
table1.to_latex('evaluation_results/tables/table1_classification.tex', index=False)
```

**Run:**
```bash
python scripts/evaluation/generate_tables.py
```

---

## Step 8: Statistical Testing

File: `scripts/evaluation/statistical_analysis.py`

```python
from scipy import stats
import numpy as np

# Load results
yolo_f1_scores = ...  # Per-video F1 scores
orion_f1_scores = ...

# Paired t-test
t_stat, p_value = stats.ttest_rel(orion_f1_scores, yolo_f1_scores)
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.6f}")  # Should be < 0.05

# Bootstrap confidence interval
from scipy.stats import bootstrap
def ci_func(x, axis):
    return np.mean(x, axis=axis)

rng = np.random.default_rng()
res = bootstrap((orion_f1_scores,), ci_func, rng=rng)
print(f"95% CI: [{res.confidence_interval.low:.3f}, {res.confidence_interval.high:.3f}]")
```

---

## Quick Timeline

```
Day 1: Download TAO-Amodal + VSGR annotations
Day 2: Implement data loader (test on 10 videos first)
Day 3: Run YOLO baseline (100 videos)
Day 4: Run full Orion (all test videos)
Day 5: Run ablations (5 configs)
Day 6: Generate tables, statistical testing
Day 7: Write results section
```

---

## Commands Cheat Sheet

```bash
# Setup
mkdir -p evaluation_results/{tables,figures}
pip install scikit-learn scipy matplotlib seaborn pandas

# Download
git clone https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal data/tao-amodal
# Get VSGR annotations from HyperGLM authors

# Run
python scripts/evaluation/run_yolo_baseline.py --num_videos 100
python scripts/evaluation/run_orion_evaluation.py
python scripts/evaluation/run_ablations.py
python scripts/evaluation/generate_tables.py
```

