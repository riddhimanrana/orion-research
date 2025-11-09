# Zone Detection Fix

## Problem
The current zone manager creates way too many zones (18 zones for 1 room) because:
1. **Clustering every observation**: Each frame's detections are treated as separate points
2. **No entity aggregation**: Same entity at slightly different positions creates multiple "zones"
3. **Time-based splitting**: Temporal weight causes zones to split unnecessarily

## Root Cause
```python
# Current (WRONG): Cluster all observations
for obs in observation_buffer:  # 641 observations
    cluster(obs.centroid_3d_mm)  # Each observation is a point
# Result: 18 clusters (over-segmentation)
```

**Issue**: If we have 4 entities tracked over 50 frames, we get 4×50=200 observations. HDBSCAN sees 200 points and creates many clusters.

## Correct Approach

### For ONE ROOM detection:
```python
# Aggregate observations by entity first
entity_positions = {}
for obs in observation_buffer:
    if obs.entity_id not in entity_positions:
        entity_positions[obs.entity_id] = []
    entity_positions[obs.entity_id].append(obs.centroid_3d_mm)

# Compute mean position per entity
entity_centroids = []
for entity_id, positions in entity_positions.items():
    mean_pos = np.mean(positions, axis=0)
    entity_centroids.append(mean_pos)

# Cluster the aggregated entities
# If all entities in one room → should produce 1 cluster
```

### Algorithm
1. **Aggregate by entity**: Compute mean 3D position per unique entity
2. **Cluster entities**: Use spatial clustering on entity centroids (not raw observations)
3. **Large merge distance**: Use 10m+ for indoor (one room should be one zone)
4. **Ignore temporal**: Time should NOT affect spatial zones

### Parameters
- `min_cluster_size`: 3-5 entities (not 30 observations)
- `merge_distance_mm`: 10000mm (10 meters) for indoor room-scale
- `temporal_weight`: 0.0 (completely ignore time for spatial zones)
- Only cluster when `len(unique_entities) >= 3`

## Implementation Plan
1. Add `_aggregate_observations_by_entity()` method
2. Cluster on entity centroids, not raw observations
3. Use DBSCAN with eps=10m for room-scale
4. Return 1 zone for typical bedroom/office scene

## Expected Result
- **Before**: 18 zones for 1 room (4 entities, 641 observations)
- **After**: 1 zone for 1 room (4 entities, 4 centroids)
