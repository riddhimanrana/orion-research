# CIS Hyperparameter Optimization

## Overview

This directory contains the CIS (Causal Influence Score) hyperparameter optimization system.

The CIS formula determines which entities are causal agents for state changes in video scenes. 
Previously, weights were chosen heuristically. Now, they are **learned from ground truth data** 
using Bayesian optimization.

## Scientific Justification

**Research Mentor's Question:**
> "Why are we using these specific weights? Did we derive it from something? 
> Were the weights learned? Where does the threshold come from?"

**Our Answer:**
We use **Bayesian optimization (Optuna)** to learn CIS weights from human-annotated 
ground truth causal relationships. The weights maximize F1 score on held-out validation data.

## Files

- `ground_truth_format.json` - Example format for ground truth annotations
- `usage_example.py` - Complete example of running optimization
- `optimization_result.json` - Learned weights and metrics (after running)

## Annotation Format

Ground truth format:
```json
{
  "agent_id": "entity_hand_0",
  "patient_id": "entity_switch_1", 
  "state_change_frame": 150,
  "is_causal": true,
  "confidence": 0.95,
  "annotation_source": "human"
}
```

## Running Optimization

```bash
# With your annotated data
python scripts/test_cis_optimization.py --data data/ground_truth/causal_pairs.json --trials 100

# Results will be in results/cis_hpo/
```

## Using Optimized Weights

```python
from orion.causal_inference import CausalConfig

# Load learned weights
config = CausalConfig.from_hpo_result('results/cis_hpo/optimization_result.json')

# Weights are now scientifically justified!
```

## Dependencies

```bash
pip install optuna matplotlib seaborn
```
