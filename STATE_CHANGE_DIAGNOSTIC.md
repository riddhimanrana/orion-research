# State Change Detection Diagnostic

The system is not detecting state changes because:

## Root Cause
Entities only have **one description** (from initial perception), so there's nothing to compare.

The state change detection logic compares entity descriptions over time:
```python
# Need at least 2 descriptions to detect changes
if len(entity.descriptions) < 2:
    return changes
```

Currently: Each entity has exactly 1 description → No state changes detected ✗

## Why This Happens
1. Perception engine generates descriptions for entities at detection time
2. Entity tracker consolidates entities and stores their descriptions
3. State detector tries to find differences in descriptions over time
4. But there's no mechanism to **update descriptions** as the video progresses

## Solution: Add Temporal Description Updates

The semantic pipeline needs to:
1. Sample entities at multiple time points throughout the video
2. Generate fresh descriptions at each time point
3. Compare descriptions to detect changes

### Implementation Plan

**Option 1: Incremental Description Updates (Recommended)**
- Add description generator that samples every N seconds
- Use CLIP to describe entity state at sample points
- Feed descriptions through state change detector

**Option 2: Frame-by-Frame Descriptions**
- Generate description for each frame (expensive)
- Too many state changes, needs filtering

**Option 3: Heuristic Temporal Sampling**
- Sample at keyframes (e.g., when entity significantly changes position)
- Uses motion vectors to trigger sampling

### Quick Fix: Lower the Threshold (Temporary)
Change `embedding_similarity_threshold` from 0.85 to 0.5 in config.

**But this won't work** because there's still only 1 description per entity to compare.

## Next Steps
1. Add description update mechanism
2. Generate descriptions at multiple timestamps
3. Re-run state change detection
4. Validate on test video
