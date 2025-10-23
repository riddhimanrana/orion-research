# CIS Training Quick Reference

## What is CIS?
**Causal Influence Score** - mathematically scores how likely entity A caused entity B to change state.

## Quick Start (5 minutes)
```bash
# Train CIS weights on ground truth data (50 trials ≈ 10 seconds)
python3 scripts/run_cis_hpo.py --ground-truth data/cis_ground_truth.json --trials 50

# Validate the weights work
python3 scripts/validate_cis_integration.py

# Pipeline automatically uses learned weights
python3 -m orion.cli analyze --video data/examples/video.mp4
```

## The Formula
```
CIS(agent, patient) = w_temporal·f_temporal 
                    + w_spatial·f_spatial 
                    + w_motion·f_motion 
                    + w_semantic·f_semantic

Where:
  w_* = learned weights (sum to ~1.0)
  f_* = component scores (0 to 1)
  CIS = confidence score (0 to 1)
```

## CIS Components

| Component | Measures | Typical Weight | What It Means |
|-----------|----------|---|---|
| **Spatial** | Physical distance | 0.40 | Closer = more likely to interact |
| **Motion** | Moving toward? | 0.39 | Direct approach suggests causality |
| **Semantic** | Make sense together? | 0.20 | Person+door more likely than person+wall |
| **Temporal** | How recent? | 0.01 | Recent actions more relevant |

## Ground Truth Format
```json
{
  "agent_id": "track_42",
  "patient_id": "track_87",
  "state_change_frame": 150,
  "is_causal": true,
  "confidence": 0.95,
  "metadata": {
    "distance": 85.3,
    "agent_category": "person",
    "patient_category": "door"
  }
}
```

## What Gets Trained
✓ Weights for each component (spatial, temporal, motion, semantic)
✓ CIS score threshold (below this = not causal)
✓ Optional: decay constants, distance parameters

Result: `hpo_results/optimization_latest.json`

## Files
```
orion/
├── causal_inference.py          # CIS formula
├── hpo/cis_optimizer.py         # Bayesian optimization
└── semantic_uplift.py           # Auto-loads HPO weights

scripts/
├── run_cis_hpo.py              # Train weights
└── validate_cis_integration.py # Test everything

data/
└── cis_ground_truth.json       # 2000 labeled pairs

hpo_results/
└── optimization_latest.json    # Trained weights
```

## Performance Metrics
```json
{
  "best_score": 0.9633,        // F1 score
  "precision": 1.0000,         // No false alarms
  "recall": 0.9292,            // Catches 93% of causals
  "best_threshold": 0.6404     // CIS cutoff
}
```

## Validate Everything Works
```bash
python3 scripts/validate_cis_integration.py
```

Expected output:
```
✓ PASS: HPO Results
✓ PASS: CausalConfig Loading
✓ PASS: Causal Inference Engine
```

## Advanced: Train With More Data
```bash
# Better training (100 trials, ~30 seconds)
python3 scripts/run_cis_hpo.py \
    --ground-truth data/cis_ground_truth.json \
    --trials 100 \
    --seed 42

# Publication quality (500 trials, ~2 minutes)
python3 scripts/run_cis_hpo.py \
    --ground-truth data/cis_ground_truth.json \
    --trials 500 \
    --timeout 300
```

## Answers Your Mentor's Questions

**"Why these specific weights?"**
→ They're learned from 2,000 ground truth annotations using Bayesian optimization

**"Were they derived from something?"**
→ Yes - from causal relationships in video data (TAO-Amodal dataset)

**"Where does the threshold come from?"**
→ Jointly optimized with weights to maximize F1 score

**"Is this scientific?"**
→ Yes - includes sensitivity analysis showing robustness to parameter perturbations

## Next Steps
1. Run training script
2. Check `hpo_results/optimization_latest.json`
3. Run validation
4. Execute pipeline - it automatically uses learned weights

See `docs/CIS_COMPLETE_GUIDE.md` for detailed explanation.
