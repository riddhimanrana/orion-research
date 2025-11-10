# Orion - Video Understanding & Causal Inference

Video analysis system with learnable causality detection.

## Structure

```
orion-research/
├── orion/                    # Core package
│   ├── hpo/                  # CIS hyperparameter optimization
│   │   ├── cis_optimizer.py  # Bayesian optimization
│   │   └── tao_data_loader.py # TAO-Amodal loader
│   └── evaluation/           # VSGR benchmarks
├── data/
│   ├── benchmarks/ground_truth/  # VSGR causal labels
│   ├── examples/video.mp4    # Test video
│   ├── aspire_train.json     # TAO-Amodal annotations
│   └── aspire_test.json      # TAO-Amodal test set
├── scripts/
│   └── train_cis.py          # Train CIS weights
├── hpo_results/              # Trained weights
└── docs/
    └── CIS_TRAINING_GUIDE.md # Complete training guide
```

## Quick Start

### 1. Analyze Video
```bash
orion analyze data/examples/video.mp4
```

### 2. Research Mode (SLAM, 3D Visualization)

Run complete SLAM pipeline with 3D visualization:

```bash
# With Rerun 3D visualization (recommended)
orion research slam --video data/examples/video_short.mp4 --viz rerun

# With OpenCV visualization
orion research slam --video data/examples/video_short.mp4 --viz opencv

# Additional options
orion research slam --video X \
  --viz rerun \
  --max-frames 100 \
  --skip 2 \
  --zone-mode dense
```

**Research Mode Features**:
- Complete SLAM with loop closure detection
- Entity tracking with 3D Re-ID
- Spatial zone detection and classification
- Interactive 3D visualization with Rerun
- Trajectory tracking and velocity visualization
- Camera pose estimation

See **[`RERUN_QUICK_GUIDE.md`](RERUN_QUICK_GUIDE.md)** for visualization details.

### 3. Train CIS Weights (Optional)

Make causality detection more accurate using VSGR ground truth:

```bash
# Quick test (2 videos, 10 trials, ~30 seconds)
python scripts/train_cis.py --trials 10 --max-videos 2

# Full training (all videos, 200 trials, ~30 minutes)
python scripts/train_cis.py --trials 200

# Results auto-saved to: hpo_results/cis_weights.json
# Orion will automatically use these weights!
```

See **[`docs/CIS_TRAINING_GUIDE.md`](docs/CIS_TRAINING_GUIDE.md)** for complete documentation.

## Requirements

```bash
# Core installation
pip install -e .

# For research mode (SLAM + 3D visualization)
pip install -e .[research]

# For CIS training
pip install optuna
```

## Key Components

- **CIS Optimizer**: `orion/hpo/cis_optimizer.py` - Bayesian optimization
- **Causal Inference**: `orion/causal_inference.py` - CIS formula
- **VSGR Loader**: `orion/evaluation/benchmarks/vsgr_aspire_loader.py`
- **Training Guide**: `CIS_TRAINING_GUIDE.md` - Complete walkthrough

