# Clustering Debug Guide

## Problem

HDBSCAN is finding 436 entities from 436 observations (1.0x ratio), meaning it's treating every detection as a unique object. This defeats the whole purpose of entity tracking.

## Root Cause

Your embeddings show:
```
Sample cosine distances:
  min: 0.0195  (very similar objects)
  max: 0.7519  (very different objects)
  mean: 0.5692 (moderately different on average)
```

These cosine distances convert to euclidean distances (for normalized vectors):
```
euclidean_dist = sqrt(2 * cosine_dist)

  min: 0.0195 → 0.20
  max: 0.7519 → 1.23
  mean: 0.5692 → 1.07
```

## HDBSCAN Parameters

### `MIN_CLUSTER_SIZE`
- Minimum number of observations needed to form a cluster
- Lower = more aggressive merging
- **Current: 2**
- **Recommended: 2-3** (already optimal)

### `CLUSTER_SELECTION_EPSILON`
- Maximum distance to merge clusters
- Objects within this distance get merged into same entity
- **Current: 1.2** (corresponds to ~0.36 cosine distance)
- **Problem**: Too low for your data (mean dist is 1.07)

## Solution

The mean euclidean distance in your data is **1.07**, but epsilon is only **1.2**. This means HDBSCAN will only merge very similar objects (within 12% above mean).

### Try these values progressively:

#### Conservative (expect 50-100 entities):
```python
MIN_CLUSTER_SIZE = 3
CLUSTER_SELECTION_EPSILON = 1.0
```

#### Balanced (expect 30-60 entities):
```python
MIN_CLUSTER_SIZE = 2
CLUSTER_SELECTION_EPSILON = 1.3
```

#### Aggressive (expect 15-30 entities):
```python
MIN_CLUSTER_SIZE = 2
CLUSTER_SELECTION_EPSILON = 1.5
```

## How to Tune

### Method 1: Edit Config directly

Edit `src/orion/tracking_engine.py` line ~81:

```python
class Config:
    # ...
    MIN_CLUSTER_SIZE = 2
    CLUSTER_SELECTION_EPSILON = 1.3  # <-- Change this
```

### Method 2: Use environment variable (future enhancement)

```bash
export ORION_CLUSTER_EPSILON=1.3
python scripts/test_tracking.py data/examples/video1.mp4
```

## Expected Results

With epsilon=1.3, you should see:
```
Clustering results:
  Unique entities (clusters): 15-20
  Singleton objects (noise): 5-10
  Total unique objects: 20-30
```

This would give you:
- **436 observations → 25 entities**
- **Efficiency ratio: ~17x** ✓
- Much better than current 1.0x!

## Diagnostic Output

The tracking engine now logs helpful diagnostics:

```
Embedding norms - min: 1.0000, max: 1.0000, mean: 1.0000
  ✓ All embeddings are normalized correctly

Sample cosine distances - min: 0.0195, max: 0.7519, mean: 0.5692
  → Use this to estimate epsilon
  → epsilon should be around sqrt(2 * desired_cosine_threshold)
  → For cosine threshold 0.4, use epsilon = sqrt(2 * 0.4) ≈ 0.9
  → For cosine threshold 0.5, use epsilon = sqrt(2 * 0.5) ≈ 1.0
  → For cosine threshold 0.6, use epsilon = sqrt(2 * 0.6) ≈ 1.1
```

## Quick Formula

To merge objects with cosine distance < X:
```
CLUSTER_SELECTION_EPSILON = sqrt(2 * X)

X=0.3 → epsilon=0.77
X=0.4 → epsilon=0.89
X=0.5 → epsilon=1.00
X=0.6 → epsilon=1.10
X=0.7 → epsilon=1.18
```

## Video-Specific Tuning

Different videos need different thresholds:

### Static camera, few objects (office desk)
- Objects appear very similar across frames
- Use lower epsilon (0.9-1.1)
- Expect fewer entities

### Dynamic camera, many objects (street scene)
- Objects appear from different angles
- Use higher epsilon (1.2-1.5)
- Expect more entities

### Your video (appears to be desktop/office):
- Keyboard, mouse, laptop, monitor are static
- Should get 1 entity per physical object
- **Recommended: epsilon=1.2-1.4**

## Next Steps

1. Edit Config in tracking_engine.py
2. Set `CLUSTER_SELECTION_EPSILON = 1.3`
3. Run: `python scripts/test_tracking.py data/examples/video1.mp4`
4. Check output:
   ```
   Unique entities (clusters): XX
   Singleton objects (noise): YY
   Total unique objects: ZZ
   ```
5. If still too many entities (>100), increase epsilon to 1.5
6. If too few entities (<10), decrease epsilon to 1.1

## Understanding the Output

```
Clustering results:
  Unique entities (clusters): 18    ← Objects seen 2+ times
  Singleton objects (noise): 7      ← Objects seen only once
  Total unique objects: 25          ← Total entities to describe
```

Goal: 20-50 total entities for a typical video.
