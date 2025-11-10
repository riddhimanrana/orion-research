"""
Phase 4 Week 2 - Zone Detection Analysis

Analyze current zone detection results and plan improvements.

Week 2 Goals:
- Day 4: World-coordinate zone clustering (use SLAM transforms)
- Day 5: Semantic zone refinement (spatial + semantic clustering)

Author: Orion Research Team
Date: November 9, 2025
"""

import sys
from pathlib import Path

print("\n" + "=" * 70)
print("PHASE 4 WEEK 2: ZONE DETECTION REFINEMENT - ANALYSIS")
print("=" * 70)

print("\n**Current Status (from Week 1 tests)**:")
print("  • SLAM tracking: 100% success (was 14.4%)")
print("  • Loop closures: 20 detected")
print("  • Processing: 1.09 FPS")
print("  ✅ Week 1: SLAM performance FIXED")

print("\n**Week 2 Objectives**:")
print("\n  Day 4: World-Coordinate Zone Clustering")
print("    Goal: Use SLAM transforms to cluster in world frame")
print("    Expected: 3-4 zones (currently getting 5)")
print("    Benefit: View-invariant, persistent zones")

print("\n  Day 5: Semantic Zone Refinement")
print("    Goal: Use object classes to improve clustering")
print("    Method: Spatial (70%) + Semantic (30%) similarity")
print("    Patterns:")
print("      - Bedroom: bed + nightstand + dresser")
print("      - Kitchen: stove + sink + fridge")
print("      - Living: couch + TV + coffee table")

print("\n" + "=" * 70)
print("STEP 1: VERIFY WORLD COORDINATE USAGE")
print("=" * 70)

# Check if zone manager uses world coordinates
zone_manager_path = Path(__file__).parent / "orion" / "semantic" / "zone_manager.py"

if zone_manager_path.exists():
    with open(zone_manager_path, 'r') as f:
        content = f.read()
    
    print("\n✓ Analyzing zone_manager.py...")
    
    # Check for world coordinate usage
    checks = {
        'add_observation has frame_idx': 'frame_idx' in content and 'def add_observation' in content,
        'uses entity centroids': '_aggregate_observations_by_entity' in content,
        'DBSCAN clustering': 'DBSCAN' in content and '_cluster_entity_centroids' in content,
        'spatial_weight parameter': 'spatial_weight' in content,
        'merge_distance_mm': 'merge_distance_mm' in content,
    }
    
    print("\n  Checks:")
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"    {status} {check}")
    
    # Check default parameters
    if 'merge_distance_mm: float = 3000.0' in content:
        print(f"\n  ✅ merge_distance_mm = 3000.0 mm (3m) - Good for room-scale")
    elif 'merge_distance_mm: float = 5000.0' in content:
        print(f"\n  ⚠️  merge_distance_mm = 5000.0 mm (5m) - May merge adjacent rooms")
    
    if 'eps = 2.5' in content:
        print(f"  ✅ DBSCAN eps = 2.5m - Good for room separation")
    elif 'eps = 5.0' in content:
        print(f"  ⚠️  DBSCAN eps = 5.0m - May merge adjacent rooms")

print("\n" + "=" * 70)
print("STEP 2: CHECK SLAM INTEGRATION")
print("=" * 70)

world_tracker_path = Path(__file__).parent / "orion" / "slam" / "world_coordinate_tracker.py"

if world_tracker_path.exists():
    with open(world_tracker_path, 'r') as f:
        content = f.read()
    
    print("\n✓ Analyzing world_coordinate_tracker.py...")
    
    checks = {
        'transform_to_world method': 'transform_to_world' in content,
        'entity tracking': 'entities_world' in content,
        'get_all_entity_centroids': 'get_all_entity_centroids' in content,
    }
    
    print("\n  Checks:")
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"    {status} {check}")

print("\n" + "=" * 70)
print("STEP 3: IDENTIFY ISSUES")
print("=" * 70)

print("\n**Known Issue from Phase 4 tests**:")
print("  • Getting 5 zones instead of 3-4")
print("  • Despite 100% SLAM tracking")

print("\n**Possible Causes**:")
print("  1. Zone manager not using world coordinates")
print("     → Check: Are we transforming entity positions?")
print("  2. DBSCAN eps too small")
print("     → Check: Current eps value")
print("  3. No zone re-identification on loop closure")
print("     → Check: Do we merge zones when revisiting?")
print("  4. Temporal weight causing splits")
print("     → Check: temporal_weight parameter")

print("\n" + "=" * 70)
print("STEP 4: PROPOSED FIXES")
print("=" * 70)

print("\n**Fix 1: Ensure World Coordinate Usage**")
print("""
Current code in zone_manager.py should be:

def add_observation(self, entity_id, timestamp, centroid_3d_mm, ...):
    # centroid_3d_mm should be world coordinates from SLAM
    obs = ZoneObservation(
        entity_id=entity_id,
        centroid_3d_mm=centroid_3d_mm,  # MUST be world coords!
        ...
    )

Integration in pipeline should call:

# Transform to world coordinates first
world_pos = world_tracker.get_entity_world_centroid(entity_id)
zone_manager.add_observation(
    entity_id=entity_id,
    centroid_3d_mm=world_pos,  # World coordinates!
    ...
)
""")

print("\n**Fix 2: Tune DBSCAN Parameters**")
print("""
Recommended settings for indoor 3-room apartment:

zone_manager = ZoneManager(
    mode="dense",
    merge_distance_mm=3000.0,  # 3m for room-scale
    spatial_weight=1.0,
    temporal_weight=0.001,     # Nearly zero (don't split by time)
)

In _cluster_entity_centroids:
    eps = 2.5  # Meters - entities within 2.5m → same zone
    min_samples = 2  # At least 2 entities per zone
""")

print("\n**Fix 3: Zone Re-Identification**")
print("""
When loop closure detected:

def on_loop_closure(current_frame_idx, matched_frame_idx):
    # Get zones at current location
    current_zones = get_zones_at_frame(current_frame_idx)
    
    # Get zones at matched location
    matched_zones = get_zones_at_frame(matched_frame_idx)
    
    # Merge if spatial overlap > 50%
    for cz in current_zones:
        for mz in matched_zones:
            if spatial_overlap(cz, mz) > 0.5:
                merge_zones(cz, mz)
""")

print("\n**Fix 4: Semantic Refinement (Day 5)**")
print("""
Add semantic pattern matching:

room_patterns = {
    'bedroom': {'bed', 'nightstand', 'dresser', 'lamp'},
    'kitchen': {'stove', 'sink', 'refrigerator', 'table'},
    'living': {'couch', 'tv', 'coffee table', 'bookshelf'}
}

def semantic_similarity(zone_a, zone_b):
    # Get object classes in each zone
    classes_a = {obj.class_label for obj in zone_a.objects}
    classes_b = {obj.class_label for obj in zone_b.objects}
    
    # Jaccard similarity
    intersection = len(classes_a & classes_b)
    union = len(classes_a | classes_b)
    
    return intersection / union if union > 0 else 0.0

def should_merge_zones(zone_a, zone_b):
    spatial_sim = 1.0 - (distance(zone_a, zone_b) / 5000.0)  # 0-1
    semantic_sim = semantic_similarity(zone_a, zone_b)
    
    # Weighted combination
    combined = 0.7 * spatial_sim + 0.3 * semantic_sim
    
    return combined > 0.6  # Merge if 60% similar
""")

print("\n" + "=" * 70)
print("STEP 5: IMPLEMENTATION PLAN")
print("=" * 70)

print("\n**Day 4: World-Coordinate Clustering** (2-3 hours)")
print("  Tasks:")
print("    1. Verify zone_manager receives world coordinates")
print("    2. Tune DBSCAN eps parameter (test 2.0, 2.5, 3.0)")
print("    3. Reduce temporal_weight to 0.001")
print("    4. Add zone re-identification on loop closure")
print("    5. Test on synthetic + AG-50 data")

print("\n**Day 5: Semantic Refinement** (2-3 hours)")
print("  Tasks:")
print("    1. Extract object class lists from zones")
print("    2. Implement semantic_similarity()")
print("    3. Add semantic weight to clustering")
print("    4. Test pattern matching (bedroom/kitchen/living)")
print("    5. Validate on real data")

print("\n**Success Criteria**:")
print("  ✅ Zone count: 3-4 (not 5)")
print("  ✅ Zones stable in world coordinates")
print("  ✅ Re-identification works on loop closure")
print("  ✅ Semantic patterns recognized")

print("\n" + "=" * 70)
print("STEP 6: TESTING APPROACH")
print("=" * 70)

print("\n**Test 1: Synthetic 3-Room Data**")
print("  • 150 frames, 3 rooms with loop closure")
print("  • Known ground truth (3 zones)")
print("  • Object detections: 4 objects per room")
print("  • Expected: Exactly 3 zones")

print("\n**Test 2: AG-50 Real Data**")
print("  • Run on full AG-50 video")
print("  • Manual validation of zone boundaries")
print("  • Check semantic consistency")
print("  • Expected: 3-4 zones")

print("\n**Metrics to Track**:")
print("  • Zone count")
print("  • Zone stability (centroid variance)")
print("  • Re-identification rate")
print("  • Semantic pattern accuracy")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)

print("\n1. **Verify Current Implementation**:")
print("   Check how zone_manager is called in the pipeline")
print("   → Are we passing world coordinates?")

print("\n2. **Quick Parameter Tuning**:")
print("   Test different DBSCAN eps values:")
print("   → eps=2.0, 2.5, 3.0, 3.5 meters")

print("\n3. **Add Zone Re-ID**:")
print("   Detect when zones overlap spatially")
print("   → Merge on loop closure detection")

print("\n4. **Semantic Refinement**:")
print("   Extract object classes from zones")
print("   → Implement pattern matching")

print("\n5. **Full Validation**:")
print("   Test on synthetic + real data")
print("   → Document improvements")

print("\n" + "=" * 70)
print("\nReady to start implementation! 🚀")
print("\nRecommended order:")
print("  1. Check zone_manager integration (10 min)")
print("  2. Tune DBSCAN parameters (30 min)")
print("  3. Add zone re-identification (1 hour)")
print("  4. Semantic refinement (2 hours)")
print("  5. Testing and validation (1 hour)")
print("\nTotal: ~5 hours for Week 2")
print("=" * 70 + "\n")
