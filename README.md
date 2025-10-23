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

### 2. Train CIS Weights (Optional)

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
pip install -e .
pip install optuna  # For CIS training
```

## Key Components

- **CIS Optimizer**: `orion/hpo/cis_optimizer.py` - Bayesian optimization
- **Causal Inference**: `orion/causal_inference.py` - CIS formula
- **VSGR Loader**: `orion/evaluation/benchmarks/vsgr_aspire_loader.py`
- **Training Guide**: `CIS_TRAINING_GUIDE.md` - Complete walkthrough

