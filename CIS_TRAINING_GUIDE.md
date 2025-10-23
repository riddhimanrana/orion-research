# CIS Hyperparameter Optimization - Complete Guide

## What This Does

Trains the Causal Inference Score (CIS) weights using VSGR ground truth data, then integrates them into Orion so your `orion analyze` becomes more accurate at detecting causality.

## Quick Start (3 Steps)

### Step 1: Ground Truth (Already Done ✓)

Your ground truth is ready:
```bash
data/benchmarks/ground_truth/vsgr_aspire_train_full.json  # 2,578 labeled causal pairs
```

### Step 2: Run Orion on Your Video

Generate perception data from your example video:

```bash
orion analyze data/examples/video.mp4
```

This creates: `data/testing/pipeline_results_video_*.json`

### Step 3: Extract CIS Training Data

Convert the perception log to CIS format:

```bash
python scripts/extract_cis_data.py \
  data/testing/pipeline_results_video_*.json \
  --output data/hpo/extracted_data.pkl
```

### Step 4: Run CIS Optimization

```bash
python scripts/run_cis_hpo.py \
  --extracted-data data/hpo/extracted_data.pkl \
  --ground-truth data/benchmarks/ground_truth/vsgr_aspire_train_full.json \
  --trials 200 \
  --output hpo_results/cis_optimized
```

**What happens:**
1. Loads 2,578 labeled causal pairs from VSGR
2. Extracts agent candidates and state changes from your video
3. Uses Bayesian optimization (Optuna TPE) to learn optimal CIS weights
4. Saves best weights to `hpo_results/cis_optimized/best_weights.json`

**Expected output:**
```
Best F1: 0.XXX
Best Weights:
  w_temporal_proximity: X.XX
  w_spatial_proximity: X.XX  
  w_motion_correlation: X.XX
  w_class_compatibility: X.XX
  w_state_alignment: X.XX
```

## Integration into Orion

Once optimization completes, update the CIS weights in the code:

### Option A: Automatic (Recommended)

```bash
# Load and apply best weights
python << 'EOF'
import json
from pathlib import Path

# Load optimized weights
with open('hpo_results/cis_optimized/best_weights.json') as f:
    weights = json.load(f)['weights']

# Update causal_inference.py
causal_path = Path('orion/causal_inference.py')
content = causal_path.read_text()

# Find and replace default weights
for key, value in weights.items():
    # Look for patterns like: w_temporal_proximity: float = 0.3
    import re
    pattern = f"{key}: float = [0-9.]+"
    replacement = f"{key}: float = {value:.3f}"
    content = re.sub(pattern, replacement, content)

causal_path.write_text(content)
print("✓ CIS weights updated in orion/causal_inference.py")
EOF
```

### Option B: Manual

Edit `orion/causal_inference.py` around line 40:

```python
@dataclass
class CISWeights:
    """Learned CIS formula weights from VSGR optimization"""
    w_temporal_proximity: float = 0.XXX  # Update from best_weights.json
    w_spatial_proximity: float = 0.XXX   # Update from best_weights.json
    w_motion_correlation: float = 0.XXX  # Update from best_weights.json
    w_class_compatibility: float = 0.XXX # Update from best_weights.json
    w_state_alignment: float = 0.XXX     # Update from best_weights.json
```

## Verify Improvements

Run Orion again with the new weights:

```bash
orion analyze data/examples/video.mp4
```

Check the output:
- **Causal Links** count should increase if more causality exists
- **CIS Scores** should be better calibrated (check logs)
- **Knowledge Graph** should have more accurate causal relationships

Compare before/after in the Neo4j browser:
```cypher
// View causal relationships
MATCH (a:Entity)-[r:CAUSES]->(b:Entity)
RETURN a.name, r.cis_score, b.name
ORDER BY r.cis_score DESC
LIMIT 20
```

## Advanced: Multiple Videos

If you have multiple videos, extract data from all of them:

```bash
# 1. Run Orion on each video
for video in data/examples/*.mp4; do
    orion analyze "$video"
done

# 2. Combine all perception logs
mkdir -p data/hpo/perception_logs
cp data/testing/pipeline_results_*.json data/hpo/perception_logs/

# 3. Run HPO with all data
python scripts/run_cis_hpo.py \
  --perception-logs-dir data/hpo/perception_logs \
  --ground-truth data/benchmarks/ground_truth/vsgr_aspire_train_full.json \
  --trials 300
```

## Understanding the Results

### Output Files

```
hpo_results/cis_optimized/
├── best_weights.json          # Optimized CIS weights
├── optimization_history.json  # All trials and scores
├── study.db                   # Optuna database
└── plots/                     # Visualization (if generated)
    ├── param_importances.png
    └── optimization_history.png
```

### Interpreting Weights

**High weight** = more important for causality detection

- `w_temporal_proximity` → Events close in time
- `w_spatial_proximity` → Objects close in space  
- `w_motion_correlation` → Similar movement patterns
- `w_class_compatibility` → Semantically related (person→door)
- `w_state_alignment` → State change timing matches

### Metrics

- **Precision**: Of predicted causal links, how many are correct?
- **Recall**: Of actual causal links, how many did we find?
- **F1**: Harmonic mean of precision and recall (optimization target)

## Troubleshooting

### "No agent candidates found"

Your video needs state changes to detect causality. Make sure:
- Video has moving objects (tracked entities)
- Objects interact (proximity, state changes)
- Entities are detected and tracked properly

Run with verbose logging:
```bash
ORION_LOG_LEVEL=DEBUG orion analyze data/examples/video.mp4
```

### "Ground truth video_id not found"

The ground truth uses VSGR video IDs. If optimizing on your own videos:
1. Create custom ground truth annotations
2. Or use VSGR videos (requires download from HuggingFace)

### Low F1 Score

- Increase `--trials` (200 → 500)
- Check ground truth quality
- Verify perception log has rich data (many entities, state changes)

## Full Workflow Example

```bash
# 1. Start fresh
rm -rf data/testing/pipeline_results_*.json
rm -rf hpo_results/cis_optimized
rm -rf data/hpo/extracted_data.pkl

# 2. Generate perception data
orion analyze data/examples/video.mp4

# 3. Extract CIS training data
python scripts/extract_cis_data.py \
  data/testing/pipeline_results_video_*.json \
  --output data/hpo/extracted_data.pkl

# 4. Run optimization
python scripts/run_cis_hpo.py \
  --extracted-data data/hpo/extracted_data.pkl \
  --ground-truth data/benchmarks/ground_truth/vsgr_aspire_train_full.json \
  --trials 200 \
  --output hpo_results/cis_optimized

# 5. Check results
cat hpo_results/cis_optimized/best_weights.json

# 6. Update Orion code (see Integration section above)

# 7. Test improvements
orion analyze data/examples/video.mp4

# 8. View in Neo4j
# Open http://localhost:7474
# Run: MATCH (a)-[r:CAUSES]->(b) RETURN a, r, b
```

## Next Steps

After successful CIS optimization:

1. **Validate on test set**: Run on VSGR test split to measure generalization
2. **Add more features**: Incorporate CLIP embeddings, semantic similarity
3. **Tune other components**: HDBSCAN clustering, entity description prompts
4. **Benchmark**: Compare against Action Genome, VSGR baselines

## Files Reference

**Core CIS Code:**
- `orion/causal_inference.py` - CIS formula and weights
- `orion/hpo/cis_optimizer.py` - Bayesian optimization system

**Scripts:**
- `scripts/run_cis_hpo.py` - Main optimization script
- `scripts/extract_cis_data.py` - Convert perception logs to CIS format
- `scripts/extract_vsgr_ground_truth.py` - Parse VSGR annotations

**Data:**
- `data/benchmarks/ground_truth/` - Labeled causal pairs
- `data/aspire_train.json` - VSGR annotations (500 videos)
- `data/examples/video.mp4` - Test video

**Results:**
- `hpo_results/` - Optimization outputs
- `data/testing/` - Pipeline results from Orion

---

**Status**: Ready to optimize! Ground truth loaded with 2,578 causal pairs.

Run Steps 2-4 above to train your CIS weights.
