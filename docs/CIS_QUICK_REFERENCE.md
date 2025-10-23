# CIS Training Quick Reference

## One-Command Training

```bash
# Full training (recommended)
python scripts/train_cis.py --trials 100

# Quick test
python scripts/train_cis.py --trials 10 --max-videos 2
```

## What It Does

1. Loads TAO-Amodal bounding box annotations
2. Loads VSGR ground truth causal labels  
3. Converts to Orion data structures (no videos needed!)
4. Runs Bayesian optimization to learn best CIS weights
5. Saves results to `hpo_results/cis_weights.json`
6. Orion automatically uses these weights

## Output Metrics

- **F1 Score**: Overall performance (target: >0.6)
- **Precision**: How many detections were correct
- **Recall**: How many true causalities were found
- **Weights**: Learned importance of temporal/spatial/motion/semantic
- **Threshold**: Minimum score to consider causal

## File Locations

```
data/
├── aspire_train.json          # TAO-Amodal annotations
└── benchmarks/ground_truth/
    └── vsgr_aspire_train_sample.json  # VSGR labels

hpo_results/
└── cis_weights.json           # Trained weights (auto-loaded)
```

## Full Documentation

See **`docs/CIS_TRAINING_GUIDE.md`** for:
- Detailed explanation of CIS formula
- Understanding training data
- Advanced usage and API
- Troubleshooting
- Scientific justification

## Integration

```python
# Orion automatically loads trained weights
from orion.causal_inference import CausalConfig

# Manual loading
config = CausalConfig.from_hpo_result('hpo_results/cis_weights.json')
```

## Validation

```bash
# Test the pipeline
python -c "from orion.hpo import load_tao_training_data; \
           agents, sc, gt = load_tao_training_data( \
               'data/aspire_train.json', \
               'data/benchmarks/ground_truth/vsgr_aspire_train_sample.json', \
               max_videos=1); \
           print(f'{len(agents)} agents, {len(sc)} state changes, {len(gt)} GT pairs')"
```

## Common Issues

**"No agent candidates found"**
→ Check data paths are correct

**"optuna not available"**
→ Run: `pip install optuna`

**Training too slow**
→ Use `--max-videos 5 --trials 20` for testing

**Low F1 score**
→ Need more training data or better annotations
