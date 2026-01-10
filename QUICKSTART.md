# 🚀 Quick Start: PVSG + ActionGenome Evaluation

## ⚡ 5-Minute Setup

### 1. Install Dependencies (1 minute)
```bash
pip3 install google-generativeai pillow
```

### 2. Get Gemini API Key (2 minutes)
```bash
# Visit: https://aistudio.google.com/apikey
# Copy your API key and run:
export GOOGLE_API_KEY="your_api_key_here"

# Verify:
python3 -c "import google.generativeai; print('✓ Ready')"
```

### 3. Validate Setup (1 minute)
```bash
python3 scripts/setup_evaluation_datasets.py --check-datasets
```

### 4. Test Evaluator (1 minute)
```bash
python3 -c "
import sys
sys.path.insert(0, '/Users/yogeshatluru/orion-research')
from orion.evaluation.pvsg_evaluator import PVSGEvaluator

evaluator = PVSGEvaluator()
print(f'✅ PVSG evaluator ready with {len(evaluator.videos)} videos')
"
```

## 📖 Basic Usage

### Generate Scene Graphs from Video

```python
from orion.evaluation.scene_graph_pipeline import create_pipeline, PipelineMode

# Create pipeline (choose one)
pipeline = create_pipeline(PipelineMode.PAPER)       # Strong results for paper
# pipeline = create_pipeline(PipelineMode.LIGHTWEIGHT)  # Fast for deployment

# Process video
video_sgs = pipeline.process_video(
    "path/to/video.mp4",
    sample_rate=1,          # Process every frame
    context="kitchen"       # Optional: scene context
)

print(f"Processed {len(video_sgs.frame_graphs)} frames")
```

### Evaluate on PVSG Dataset

```python
# Evaluate against ground truth
metrics = pipeline.evaluate_on_pvsg(video_sgs)

print(f"Recall@1:  {metrics['recall@1']:.3f}")
print(f"Recall@5:  {metrics['recall@5']:.3f}")
print(f"Recall@10: {metrics['recall@10']:.3f}")
print(f"Precision: {metrics['mean_precision']:.3f}")
print(f"F1-Score:  {metrics['mean_f1']:.3f}")
```

### Predict Future Scene Graphs (ActionGenome SGA)

```python
# Anticipate future scene graphs
# Given: first 50% of video frames
# Predict: remaining 50% of frames

sga_metrics = pipeline.evaluate_on_actiongenome(
    video_sgs,
    prune_ratio=0.5  # Use first 50%, predict rest
)

print(f"Success Rate: {sga_metrics['anticipation_success_rate']:.3f}")
print(f"Recall@10:   {sga_metrics['mean_recall_at_10']:.3f}")
```

## 🎯 Two Release Versions

### Paper Version (For Publication)
```python
from orion.evaluation.scene_graph_pipeline import create_pipeline, PipelineMode

pipeline = create_pipeline(PipelineMode.PAPER)
# Uses:
#   - Detection: DINOv3 (open-vocab, strong)
#   - VLM: Gemini 3.5-Flash (GPT-4o quality)
# Result: Better Recall@K scores for paper
```

### Lightweight Version (For Deployment)
```python
from orion.evaluation.scene_graph_pipeline import create_pipeline, PipelineMode

pipeline = create_pipeline(PipelineMode.LIGHTWEIGHT)
# Uses:
#   - Detection: YOLO-World
#   - VLM: FastVLM
# Result: 3-5x faster, reasonable quality
```

## 📊 Metrics Explained

### Recall@K (HyperGLM Standard)
- **What**: % of ground truth relationships in top-K predictions
- **Formula**: (matching relationships) / (total GT relationships)
- **Example**: If GT has 10 relationships and 8 are in top-10 predictions → Recall@10 = 0.8
- **Range**: 0.0 - 1.0 (higher is better)

### Precision
- **What**: % of predictions that are correct
- **Formula**: (correct predictions) / (total predictions)
- **Range**: 0.0 - 1.0 (higher is better)

### F1-Score
- **What**: Harmonic mean of precision and recall
- **Formula**: 2 × (precision × recall) / (precision + recall)
- **Why**: Balances precision and recall

### Anticipation Success Rate (SGA)
- **What**: % of frames where future prediction is good (Recall@10 >= 50%)
- **Range**: 0.0 - 1.0 (higher is better)
- **Why**: Tests true scene understanding, not just detection

## 🔍 File Locations

```
orion-research/
├── orion/
│   ├── evaluation/
│   │   ├── pvsg_evaluator.py        # PVSG scene graph eval
│   │   ├── sga_evaluator.py         # ActionGenome SGA eval
│   │   └── scene_graph_pipeline.py  # End-to-end pipeline
│   └── backends/
│       └── gemini_vlm.py            # Gemini + FastVLM backends
│
├── scripts/
│   └── setup_evaluation_datasets.py  # Setup & testing
│
├── datasets/
│   └── PVSG/                         # ✓ Already downloaded
│       ├── pvsg.json                 # 3.9 MB annotations
│       ├── Ego4D/
│       ├── EpicKitchen/
│       └── VidOR/
│
└── docs/
    └── EVALUATION_IMPLEMENTATION.md  # Detailed guide
```

## 🐛 Troubleshooting

### "GOOGLE_API_KEY not set"
```bash
# Solution:
export GOOGLE_API_KEY="your_key_from_aistudio.google.com"

# Verify:
echo $GOOGLE_API_KEY  # Should show your key
```

### "google.generativeai not installed"
```bash
pip3 install google-generativeai pillow
```

### "PVSG dataset not found"
```bash
# PVSG is already downloaded at:
/Users/yogeshatluru/orion-research/datasets/PVSG

# If not found, check:
ls datasets/PVSG/pvsg.json
```

### "ActionGenome dataset not found"
```bash
# This is expected - needs to be downloaded
# Instructions at: https://github.com/jingkang50/OpenPVSG

mkdir -p datasets/ActionGenome
# Follow download instructions...
```

## 📈 Typical Workflow

```
1. Process Video
   video.mp4 → scene_graph_pipeline.py → frame_scene_graphs

2. Evaluate on PVSG
   frame_scene_graphs + ground_truth.json → Recall@K scores

3. Evaluate on ActionGenome
   frame_scene_graphs (first 50%) → predict (last 50%) → success rate

4. Compare with HyperGLM
   Your Recall@K vs HyperGLM Recall@K → Results!
```

## 🎯 Next Steps

1. **Download ActionGenome** (optional for SGA evaluation)
   ```bash
   mkdir -p datasets/ActionGenome
   # Follow: https://github.com/jingkang50/OpenPVSG
   ```

2. **Run Baseline**
   ```bash
   # Test on a sample video
   python3 -c "
   from orion.evaluation.scene_graph_pipeline import create_pipeline, PipelineMode
   pipeline = create_pipeline(PipelineMode.PAPER)
   # Process and evaluate...
   "
   ```

3. **Benchmark Against HyperGLM**
   - Run on test set
   - Compare Recall@K scores
   - Document results

4. **Deploy Lightweight Version**
   - Switch to PipelineMode.LIGHTWEIGHT
   - Test inference speed
   - Deploy to Lambda/servers

## 💡 Pro Tips

1. **Start with lightweight version** to test pipeline, then switch to paper for results
2. **Use sample_rate > 1** for faster testing (process every Nth frame)
3. **Set context** if known (e.g., "kitchen", "office") for better VLM understanding
4. **Recall@10 is most important** - standard metric, matches HyperGLM
5. **Batch evaluation** to process multiple videos efficiently

## 📚 Full Documentation

- Architecture & Design: `docs/EVALUATION_IMPLEMENTATION.md`
- Complete Summary: `EVALUATION_COMPLETE.md`
- Detection Improvements: `docs/DETECTION_IMPROVEMENTS_SUMMARY.md`

## ✅ Verification Checklist

- [ ] GOOGLE_API_KEY exported
- [ ] google-generativeai installed
- [ ] PVSG dataset accessible
- [ ] Setup script runs without errors
- [ ] Pipeline imports work
- [ ] First video processes successfully
- [ ] PVSG evaluation completes
- [ ] Ready to benchmark!

---

**Questions?** Check the detailed docs or run:
```bash
python3 scripts/setup_evaluation_datasets.py --all
```
