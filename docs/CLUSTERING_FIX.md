# Clustering Issue: Only 2 Entities Found

## Problem

Your video has 436 observations but HDBSCAN is finding only 2 entities. This means almost everything is being merged into 2 giant clusters.

## Root Cause

The epsilon value of **0.15** was tuned for semantic_uplift.py which used **512-dim OSNet embeddings**. 

You're now using **2048-dim ResNet50 embeddings**, which have a different scale:

- **512-dim embeddings**: euclidean distances typically 0.2-0.6
- **2048-dim embeddings**: euclidean distances typically 0.8-1.2

Your actual data shows:
- Mean euclidean distance: **1.07**
- Current epsilon: **0.15**  
- Result: Epsilon is **7x smaller** than mean distance → only 2 clusters

## Solution

Adjust epsilon to match the scale of 2048-dim embeddings:

```python
# In tracking_engine.py Config class:
CLUSTER_SELECTION_EPSILON = 0.5  # For 2048-dim embeddings
```

### Why 0.5?

- Your embeddings have mean euclidean distance ~1.07
- To merge objects that are within ~50% of mean similarity: 1.07 * 0.5 ≈ 0.5
- This should give you 20-40 entities (good range)

### Testing Different Values

Try these progressively:

**Conservative** (more entities, ~40-60):
```python
CLUSTER_SELECTION_EPSILON = 0.3
```

**Balanced** (good middle ground, ~20-40):
```python
CLUSTER_SELECTION_EPSILON = 0.5
```

**Aggressive** (fewer entities, ~10-20):
```python
CLUSTER_SELECTION_EPSILON = 0.7
```

## Expected Results

With epsilon=0.5, you should see:
```
Clustering results:
  Unique entities (clusters): 15-25
  Singleton objects (noise): 5-10
  Total unique objects: 20-35
```

This would give you **436 observations → 25 entities** ≈ **17x efficiency** ✓

## Quick Fix

Edit `src/orion/tracking_engine.py` line ~84:

```python
CLUSTER_SELECTION_EPSILON = 0.5  # Tuned for 2048-dim ResNet50 embeddings
```

Then run:
```bash
python scripts/test_tracking.py data/examples/video1.mp4
```
