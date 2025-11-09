Short answer (one-liner)
Spatial zoning = convert per-frame 3D observations into a persistent, hierarchical partitioning of the environment (rooms → subzones → points of interest) by combining geometric signals (depth + camera pose), visual semantics (segmentation / CLIP embeddings), temporal continuity, and optional geo (GPS). Use clustering (DBSCAN/HDBSCAN), plane/room detection, and visual place recognition to identify and maintain zones across sessions.

Goals / contract
Input: per-frame PerceptionResult (depth_map, camera_pose, detected entities with 3D centroids, semantic labels, timestamps)
Output: Zone index:
Zone objects: {zone_id, type (room/subzone/outdoor_zone), canonical_centroid_3d, bounding_volume, label, confidence, first_seen, last_seen, members (entity IDs/timestamps)}
Hierarchical zone graph: adjacency (connected rooms), zone metadata (scene_type, embedding)
Error modes: moving camera, low depth quality, large open spaces, repeated near-identical rooms
Success criteria: stable zone IDs for same room (~>95% within-session), correct zone labeling (kitchen/bedroom) >80 on reasonable datasets
Pipeline (high-level)
Collect per-frame 3D observations: backprojected points, entity centroids (mm world coords), floor plane estimate, semantic segmentation masks.
Maintain a short-term voxel/occupancy accumulator and an entity centroid stream buffer (sliding window).
Run room candidate discovery:
Indoor: plane detection (RANSAC) + bounding extents on floor-aligned clusters → candidate rooms/subzones.
Outdoor: temporal + GPS clustering (if available) + visual embedding clustering for places.
Cluster entity centroids (space+time+embedding) with density clustering (HDBSCAN/DBSCAN) to form zones. Merge clusters using visual place recognition and scene classification.
Build zone graph and attach entities/events to zones; store canonical embeddings and representative frames.
Persist zones in the in-memory index (and optional graph DB) with TTL or permanence rules (dense indoor persistent, outdoor sliding window).
Algorithms and key components
1) Inputs you must maintain per frame
depth_map (H×W) + confidence_map
camera_pose (4×4 transform or position/quat) — if not available, estimate relative pose via visual odometry / frame-to-frame flow / SLAM
entity detections: {entity_id_temp, class, bbox_2d, centroid_3d_mm, embedding, confidence}
semantic map (optional): image-level scene label, panoptic segmentation (floor/wall/ceiling/object)
GPS (optional)
2) Low-level geometry steps
Backproject depth → point cloud in camera frame, then to world frame by camera_pose.
Estimate floor plane (RANSAC plane fit on near-floor normals / long flat region) to normalize Z and split vertical vs horizontal extents.
Compute axis-aligned bounding box of accumulated points in world frame, optionally fit oriented bounding box.
3) Clustering (zone discovery)
Two modes: indoor (dense) and outdoor (sparse).

Indoor clustering (dense, preferred in indoor scenes):

Accumulate entity centroids over a sliding buffer (e.g., last N seconds or M frames).
Use HDBSCAN (preferred for variable density) on features:
spatial: (x, y) in floor plane coordinates (use mm/units)
temporal: scaled time (t / τ_time) to prefer temporally close events when needed
semantic: low-dim embedding (CLIP or scene embedding) appended with small weight
HDBSCAN params: min_cluster_size=20 (or configurable), min_samples=5; distance metric: weighted Euclidean with spatial dominance.
Post-process clusters:
Remove tiny clusters or merge clusters whose centroids are within merge_distance (e.g., 2–4m) AND have high embedding similarity (>0.85)
Fit 2D polygon/convex hull on member points → zone footprint
Assign zone type using majority scene label or CLIP label of representative frames (e.g., kitchen, office)
Outdoor clustering (sparse, geo-aware):

Use GPS clustering (Haversine + DBSCAN with eps ~5–10m) if GPS available.
If no GPS use visual place embeddings (NetVLAD/DELG) or CLIP embeddings and DBSCAN/HDBSCAN on embeddings + coarse spatial coordinates.
Sliding-window horizon: keep zones only for last T minutes (configurable).
4) Room segmentation via floor/wall detection
Use semantic segmentation or plane detection:
If large floor plane plus separators (walls) are detected, group points by wall direction to infer separate rooms.
Combine with scene classification to label (kitchen/bedroom).
Use RGB panoptic segmentation (Detectron2, HRNet) if available: connected floor regions bounded by walls => rooms.
5) Temporal continuity & merging across sessions
For same-session: match new clusters to existing zones by centroid distance + embedding sim + topological connectivity (doorway detection from geometry).
Across sessions (different days): require stronger evidence — combine visual place recognition (NetVLAD or CLIP) with coarse location/GPS and structural similarity (room sizes/relative positions). Use re-identification score >0.85 to merge, otherwise create new zone.
6) Zone graph & adjacency
Create nodes for zones, edges if they are adjacent (common boundary) or frequent transitions observed between them.
Edge weights = transition frequency (useful for path queries, “where do I usually go after kitchen?”)
Data structures / types (Python sketch)
Zone:
zone_id: str
type: Enum {INDOOR_ROOM, SUBZONE, OUTDOOR_ZONE, UNKNOWN}
centroid_3d_mm: np.array([x,y,z])
footprint_2d: polygon (convex hull)
bounding_volume: (min,max)
representative_frames: list[(frame_idx, timestamp, img_path)]
embedding: np.array (512-d)
label: str (kitchen/bedroom/living/outdoor)
first_seen, last_seen
persistence_score: float
members: list of (entity_id, timestamps)
ZoneIndex:
zones: dict zone_id → Zone
spatial_index: R-tree or KD-tree to query zones by point
adjacency_graph: dict zone_id → list[(neighbor_id, weight)]
methods: add_zone(), merge_zones(), query_point(world_point), attach_entity(entity_id, point, timestamp)
API / Module suggestions
orion/backend/semantic/zone_manager.py
class ZoneManager:
add_observation(frame_id, timestamp, camera_pose, entities, depth_map, scene_label)
periodic_update() # runs clustering and zone maintenance
query_zone_by_point(world_point) -> zone_id / None
query_zone_history(zone_id) -> events, entities
merge_zones(zone_id_a, zone_id_b)
export_zones(format='json' / 'graphdb')
CLI flags:
--zone-mode {dense, sparse}
--zone-min-cluster-size
--zone-merge-distance-mm
--zone-persistence (indoor=permanent/outdoor=sliding_window_seconds=120)
Config defaults (suggested)
indoor:
clustering: HDBSCAN min_cluster_size=12, min_samples=3
spatial_weight = 1.0, embedding_weight = 0.2, temporal_weight = 0.1
merge_distance_mm = 2500 (2.5 m)
outdoor:
dbscan eps = 10 m (if GPS), min_samples = 3
sliding_window = 300 seconds (5 min)
plane detection:
ransac_threshold_mm = 40 mm
min_floor_ratio = 0.15 (must see enough floor)
Examples (how query/usage looks)
Query which zone the entity X is in at time t:
ZoneManager.query_zone_by_point(entity.centroid_3d_mm)
Ask “Where did I place the keys?”:
find entity class==key, collect its zone assignments across timeline, rank by count → return zone.label and timestamps.
Visualize zones:
export zone footprint to geojson-like structure and overlay on HTML viewer.
Edge cases & mitigations
moving camera without reliable pose → use relative clustering in camera coordinates with temporal smoothing; switch to “appearance-only” clustering (embeddings + bbox) with larger margin
low depth quality → reduce weight of spatial feature, rely more on embeddings and temporal proximity
open-plan spaces (kitchen+living) → detect subzones by semantic labels (kitchen island area vs lounge) and treat both as subzones inside a larger zone
repeated identical rooms (hotel corridors, identical office rooms) → require stronger visual place recognition (NetVLAD + structural cues) and time-based heuristics to avoid false merges
camera mounted on moving person — zone boundaries should be soft; use persistence and multiple visits before creating permanent zone
Evaluation & metrics
Zone purity (how many entity assignments belong to correct human-labeled zone)
Zone completeness (fraction of true zone area covered)
Re-identification rate for same-zone across sessions
Time-to-detect new zone (how fast we label a new room)
Failure modes counts (merged incorrectly, split incorrectly)
Use small labeled set (rooms with bounding polygons) and measure IoU of discovered footprint vs GT.
Implementation plan (concrete)
Create orion/backend/semantic/zone_manager.py and types file.
Implement per-frame ingestion: convert entity centroids to floor coordinates.
Implement short-term buffer and accumulation (circular buffer).
Implement HDBSCAN-based indoor clustering (use sklearn/hdbscan package).
Add merge/visual label assignment using CLIP or small scene classifier.
Integrate with FrameLogger and HTMLDebugger to visualize discovered zones (footprints and colors).
Add unit tests: synthetic scenes (two rooms separated by doorway) and verify clusters.
Add evaluation script under evaluation to compute zone metrics vs labeled data.
Quick pseudocode — core clustering loop
Short notes on cross-session persistence
Persist zone canonical data: footprint, representative embedding, label, adjacency
Use approximate nearest neighbor (FAISS) for embedding matching across sessions
Require thresholded match + structural check (size/footprint similarity) to merge
If you'd like, I can:

add a concrete zone_manager.py stub into orion/backend/semantic/ with the API and a small in-repo unit test (I can implement that next),
or draft the YAML config defaults and a small notebook to visualize how cluster parameters affect results on a sample video.
