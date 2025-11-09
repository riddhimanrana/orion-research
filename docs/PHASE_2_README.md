# PHASE 2: Tracking, Object Permanence, Occlusion Modeling

**Objective**: Build robust entity tracking in 3D space with Bayesian belief states. Handle occlusions, off-screen entities, and re-identification. Establish temporal coherence for downstream semantic analysis.

**Timeline**: Week 1–2

**Success Criteria**:
- ✅ 3D HDBSCAN clustering on 3D centroids + CLIP embeddings
- ✅ Bayesian re-identification with posterior beliefs (top-3 hypotheses per entity)
- ✅ Object permanence: entities remain tracked even when occluded/off-screen
- ✅ Re-appearance matching: disappeared entities correctly re-identified after occlu­sion
- ✅ Identity switch rate (ID-SW) < 5% on test video
- ✅ Complete entity trajectories with state history
- ✅ Latency: <200ms/frame (perception + tracking)

---

## Architecture Overview

```
PerceptionResult (from Phase 1)
├─ entities_frame_t: List[EntityState]  # 3D bboxes, visibility, embeddings
├─ hand_detections_frame_t: List[Hand]
└─ depth_map_frame_t: np.ndarray

    ↓
┌──────────────────────────────────────────┐
│  ENTITY TRACKING PIPELINE                │
├──────────────────────────────────────────┤
│                                          │
│  [Frame-to-Frame Matching]               │
│  ├─ Compute match scores: spatial +      │
│  │   embedding similarity                │
│  ├─ Hungarian algorithm (bipartite)      │
│  └─ Output: matched_pairs + unmatched    │
│                                          │
│  [Bayesian Belief Update]                │
│  ├─ For each entity: maintain posterior  │
│  │   P(class | observations)             │
│  ├─ Update with YOLO + VLM evidence      │
│  └─ Output: entity_id with confidence   │
│                                          │
│  [Disappearance Handling]                │
│  ├─ Track TTL (time-to-live) for        │
│  │   missing entities                    │
│  ├─ Retain in "disappeared" buffer      │
│  └─ Allow re-identification after gap   │
│                                          │
│  [Re-appearance Matching]                │
│  ├─ Match new observations to            │
│  │   disappeared entities                │
│  ├─ Use embedding + class + motion      │
│  └─ High threshold for re-id (>0.85)   │
│                                          │
└──────────────────────────────────────────┘

    ↓
TrackingResult
├─ active_entities: Dict[entity_id, Entity]
├─ entity_timelines: Dict[entity_id, List[EntityState]]
├─ trajectories: Dict[entity_id, Trajectory]
├─ tracking_metrics: Dict[str, float]
└─ disappeared_buffer: Dict[entity_id, (timestamp, state)]
```

---

## Key Components

### 1. Bayesian Entity State

```python
@dataclass
class BayesianEntityBelief:
    """
    Maintain belief state over entity class and identity
    """
    entity_id: str  # assigned ID (may change if merged/split)
    timestamp_updated: float
    
    # === Posterior over class labels ===
    class_posterior: Dict[str, float]  # {class_name: P(class | obs)}
    # Top-1 class
    primary_class: str
    primary_class_confidence: float
    
    # === Posterior over pose/state ===
    state_posterior: Optional[Dict[str, float]]  # e.g., {open: 0.7, closed: 0.3} for hands
    
    # === Observation history (circular buffer) ===
    recent_observations: List[EntityState]  # last 5-10 frames
    
    # === Motion model ===
    velocity_3d: Optional[Tuple[float, float, float]]  # mm/s, estimated from last 2 frames
    motion_covariance: Optional[np.ndarray]  # uncertainty in motion
    
    # === Re-id features ===
    embedding_mean: Optional[np.ndarray]  # running average of CLIP embeddings
    embedding_std: Optional[np.ndarray]
    
    def get_top_k_classes(self, k: int = 3) -> List[Tuple[str, float]]:
        """Return top K hypotheses"""
        return sorted(self.class_posterior.items(), key=lambda x: x[1], reverse=True)[:k]
    
    def update_with_observation(self, obs: EntityState, yolo_class_posterior: Dict[str, float]):
        """
        Bayesian update: combine prior + yolo + vlm evidence
        """
        # Likelihood from YOLO
        likelihood_yolo = yolo_class_posterior
        
        # Likelihood from CLIP embedding (compare to historical mean)
        if self.embedding_mean is not None:
            sim = np.dot(obs.embedding, self.embedding_mean) / (np.linalg.norm(obs.embedding) * np.linalg.norm(self.embedding_mean) + 1e-6)
            # High similarity → class consistent with history
            likelihood_embedding = {
                self.primary_class: min(1.0, sim * 1.2),
                **{c: max(0, sim * 0.5) for c in self.class_posterior if c != self.primary_class}
            }
        else:
            likelihood_embedding = likelihood_yolo
        
        # Combine: posterior ∝ prior × likelihood_yolo × likelihood_embedding
        prior = self.class_posterior
        posterior = {}
        
        for cls in set(list(prior.keys()) + list(likelihood_yolo.keys())):
            p = prior.get(cls, 0.01)
            l_yolo = likelihood_yolo.get(cls, 0.1)
            l_emb = likelihood_embedding.get(cls, 0.1)
            
            posterior[cls] = p * l_yolo * l_emb
        
        # Normalize
        total = sum(posterior.values())
        if total > 0:
            posterior = {k: v / total for k, v in posterior.items()}
        
        # Update state
        self.class_posterior = posterior
        self.primary_class = max(posterior.items(), key=lambda x: x[1])[0]
        self.primary_class_confidence = posterior[self.primary_class]
        
        # Update embedding running average
        if self.embedding_mean is None:
            self.embedding_mean = obs.embedding
        else:
            alpha = 0.1  # exponential moving average
            self.embedding_mean = (1 - alpha) * self.embedding_mean + alpha * obs.embedding
            self.embedding_mean /= np.linalg.norm(self.embedding_mean)  # re-normalize
```

### 2. Entity Tracker with 3D + Bayesian Beliefs

```python
class EntityTracker3D:
    """
    Track entities across frames using 3D + embeddings + Bayesian beliefs
    """
    
    def __init__(self, config: PerceptionConfig):
        self.config = config
        
        # === Active entities ===
        self.active_entities: Dict[str, BayesianEntityBelief] = {}
        self.entity_timelines: Dict[str, List[EntityState]] = {}
        self.entity_count = 0
        
        # === Disappeared entities (with TTL) ===
        self.disappeared_entities: Dict[str, Tuple[float, BayesianEntityBelief]] = {}
        self.ttl_seconds: float = 2.0  # indoor: 2s, outdoor: 5-10s (configurable)
        
        # === Tracking metrics ===
        self.id_switches = 0
        self.track_fragmentations = 0
    
    def track_frame(
        self,
        observations: List[EntityState],
        yolo_confidences: Dict[str, Dict[str, float]],  # {obs_id: {class: prob}}
        timestamp: float,
    ) -> Dict[str, BayesianEntityBelief]:
        """
        Track entities in current frame
        
        Args:
            observations: detections from YOLO + 3D in Phase 1
            yolo_confidences: class posterior from YOLO
            timestamp: current frame timestamp
        
        Returns:
            {entity_id: BayesianEntityBelief} for this frame
        """
        
        # Step 1: Match observations to active entities
        matched_pairs, unmatched_obs_ids, unmatched_entity_ids = self._match_observations(
            observations, self.active_entities, timestamp
        )
        
        frame_entities = {}
        
        # Step 2: Update matched entities
        for obs_id, entity_id in matched_pairs.items():
            obs = [o for o in observations if o.entity_id == obs_id][0]
            belief = self.active_entities[entity_id]
            
            # Bayesian update
            belief.update_with_observation(obs, yolo_confidences[obs_id])
            belief.recent_observations.append(obs)
            if len(belief.recent_observations) > 10:
                belief.recent_observations.pop(0)
            
            # Estimate motion
            if len(belief.recent_observations) >= 2:
                obs_prev = belief.recent_observations[-2]
                obs_curr = belief.recent_observations[-1]
                dt = obs_curr.timestamp - obs_prev.timestamp
                if dt > 0:
                    belief.velocity_3d = tuple(
                        (obs_curr.centroid_3d[i] - obs_prev.centroid_3d[i]) / dt
                        for i in range(3)
                    )
            
            # Update timeline
            self.entity_timelines[entity_id].append(obs)
            
            frame_entities[entity_id] = belief
        
        # Step 3: Try to re-identify disappeared entities
        reidentified = {}
        for obs_id in unmatched_obs_ids:
            obs = [o for o in observations if o.entity_id == obs_id][0]
            
            best_match_id = None
            best_score = 0.0
            
            for disappeared_id, (time_disappeared, disappeared_belief) in self.disappeared_entities.items():
                time_gap = timestamp - time_disappeared
                
                if time_gap > self.ttl_seconds:
                    continue  # Too old, skip
                
                # Score: embedding similarity + class match + motion plausibility
                emb_sim = np.dot(obs.embedding, disappeared_belief.embedding_mean) / (
                    np.linalg.norm(obs.embedding) * np.linalg.norm(disappeared_belief.embedding_mean) + 1e-6
                )
                
                class_match = 1.0 if obs.class_label == disappeared_belief.primary_class else 0.5
                
                # Motion plausibility
                if disappeared_belief.velocity_3d is not None:
                    predicted_pos = tuple(
                        disappeared_belief.recent_observations[-1].centroid_3d[i] + disappeared_belief.velocity_3d[i] * time_gap
                        for i in range(3)
                    )
                    dist = np.linalg.norm(np.array(obs.centroid_3d) - np.array(predicted_pos))
                    motion_score = max(0, 1 - dist / 500)  # 500mm = max plausible distance
                else:
                    motion_score = 1.0
                
                score = 0.5 * emb_sim + 0.3 * class_match + 0.2 * motion_score
                
                if score > best_score and score > 0.75:  # high threshold
                    best_score = score
                    best_match_id = disappeared_id
            
            if best_match_id:
                # Re-identification successful
                entity_id = best_match_id
                time_disappeared, belief = self.disappeared_entities[best_match_id]
                del self.disappeared_entities[best_match_id]
                
                belief.update_with_observation(obs, yolo_confidences[obs_id])
                belief.recent_observations.append(obs)
                self.entity_timelines[entity_id].append(obs)
                
                reidentified[obs_id] = entity_id
                frame_entities[entity_id] = belief
            else:
                # No good match, create new entity
                entity_id = f"entity_{self.entity_count}"
                self.entity_count += 1
                
                belief = BayesianEntityBelief(
                    entity_id=entity_id,
                    timestamp_updated=timestamp,
                    class_posterior=yolo_confidences[obs_id],
                    primary_class=max(yolo_confidences[obs_id].items(), key=lambda x: x[1])[0],
                    primary_class_confidence=max(yolo_confidences[obs_id].values()),
                    recent_observations=[obs],
                    embedding_mean=obs.embedding,
                )
                
                self.active_entities[entity_id] = belief
                self.entity_timelines[entity_id] = [obs]
                frame_entities[entity_id] = belief
        
        # Step 4: Handle unmatched entities (may disappear)
        for entity_id in unmatched_entity_ids:
            belief = self.active_entities[entity_id]
            del self.active_entities[entity_id]
            self.disappeared_entities[entity_id] = (timestamp, belief)
        
        # Step 5: Cleanup old disappeared entities
        current_disappeared = dict(self.disappeared_entities)
        for entity_id, (time_disappeared, belief) in current_disappeared.items():
            if timestamp - time_disappeared > self.ttl_seconds:
                del self.disappeared_entities[entity_id]
        
        return frame_entities
    
    def _match_observations(
        self,
        observations: List[EntityState],
        active_entities: Dict[str, BayesianEntityBelief],
        timestamp: float,
    ) -> Tuple[Dict, List, List]:
        """
        Match observations to active entities using Hungarian algorithm
        
        Returns:
            (matched_pairs, unmatched_obs_ids, unmatched_entity_ids)
        """
        from scipy.optimize import linear_sum_assignment
        
        n_obs = len(observations)
        n_ent = len(active_entities)
        
        if n_obs == 0 or n_ent == 0:
            return {}, list(range(n_obs)), list(active_entities.keys())
        
        # Build cost matrix
        cost_matrix = np.ones((n_obs, n_ent)) * 10.0  # high cost = no match
        
        obs_ids = [o.entity_id for o in observations]
        ent_ids = list(active_entities.keys())
        
        for i, obs in enumerate(observations):
            for j, entity_id in enumerate(ent_ids):
                belief = active_entities[entity_id]
                
                # Spatial distance (3D)
                dist_3d = np.linalg.norm(
                    np.array(obs.centroid_3d) - np.array(belief.recent_observations[-1].centroid_3d)
                )
                
                # Embedding similarity
                emb_sim = np.dot(obs.embedding, belief.embedding_mean) / (
                    np.linalg.norm(obs.embedding) * np.linalg.norm(belief.embedding_mean) + 1e-6
                )
                
                # Class consistency
                class_match = obs.class_label == belief.primary_class
                
                # Combined score (negative because we minimize cost)
                score = (
                    0.5 * emb_sim +
                    0.3 * (1 - min(dist_3d / 200, 1.0)) +  # 200mm = max distance for match
                    0.2 * (1.0 if class_match else 0.5)
                )
                
                cost_matrix[i, j] = -score  # negative for minimization
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_pairs = {}
        matched_obs = set()
        matched_ent = set()
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < -0.3:  # threshold for acceptance
                matched_pairs[obs_ids[i]] = ent_ids[j]
                matched_obs.add(i)
                matched_ent.add(j)
        
        unmatched_obs_ids = [i for i in range(n_obs) if i not in matched_obs]
        unmatched_entity_ids = [ent_ids[j] for j in range(n_ent) if j not in matched_ent]
        
        return matched_pairs, unmatched_obs_ids, unmatched_entity_ids
```

### 3. Object Permanence Tracker

```python
class ObjectPermanenceTracker:
    """
    Track object identities even during occlusion/off-screen periods
    """
    
    def __init__(self):
        self.permanent_entities: Dict[str, PermanentEntity] = {}  # entity_id -> PermanentEntity
    
    def register_entity(self, belief: BayesianEntityBelief, obs: EntityState):
        """
        Register an entity for permanent tracking
        """
        if belief.entity_id not in self.permanent_entities:
            self.permanent_entities[belief.entity_id] = PermanentEntity(
                entity_id=belief.entity_id,
                first_seen_timestamp=obs.timestamp,
                class_label=belief.primary_class,
                class_confidence=belief.primary_class_confidence,
                color=(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)),
            )
    
    def update_visibility(
        self,
        frame_entities: Dict[str, BayesianEntityBelief],
        all_entities: List[EntityState],
        timestamp: float,
    ):
        """
        Update visibility and occlusion status for all tracked entities
        """
        visible_entity_ids = set(frame_entities.keys())
        
        for entity_id, perm_entity in self.permanent_entities.items():
            if entity_id in visible_entity_ids:
                perm_entity.last_seen_timestamp = timestamp
                perm_entity.visibility_state = "VISIBLE"
            else:
                # Check if it's off-screen or occluded
                perm_entity.visibility_state = "HIDDEN"  # assume hidden if not detected
                perm_entity.occlusion_count += 1

@dataclass
class PermanentEntity:
    """
    Represents a persistently tracked entity across its entire lifetime
    """
    entity_id: str
    first_seen_timestamp: float
    last_seen_timestamp: float = 0
    class_label: str = ""
    class_confidence: float = 0.0
    color: Tuple[int, int, int] = (0, 0, 255)
    visibility_state: str = "VISIBLE"  # VISIBLE, HIDDEN, OFF_SCREEN
    occlusion_count: int = 0
    re_id_count: int = 0
```

---

## Tracking Metrics

```python
@dataclass
class TrackingMetrics:
    """
    MOTA-like metrics for tracking quality
    """
    num_frames: int
    num_ground_truth: int  # if GT available
    num_tracked: int
    
    # Identity metrics
    id_switches: int  # times an entity ID changed unexpectedly
    fragmentations: int  # times a track was broken and restarted
    
    # Completeness
    mostly_tracked: int  # tracks with >80% visibility
    mostly_lost: int  # tracks with <20% visibility
    
    # Re-id success
    successful_reids: int
    failed_reids: int
    
    @property
    def reid_success_rate(self) -> float:
        total = self.successful_reids + self.failed_reids
        return self.successful_reids / max(1, total)
    
    @property
    def id_switch_rate(self) -> float:
        """ID switches per 100 frames"""
        return (self.id_switches / max(1, self.num_frames)) * 100
```

---

## Testing & Validation

### Unit Tests
- **Bayesian update**: Prior + likelihood → posterior (check normalization)
- **Match algorithm**: Known correspondences → Hungarian output correct
- **Re-id matching**: Similar entities match, dissimilar don't

### Integration Tests
- **Multi-entity tracking**: 5+ objects in one video → unique IDs maintained
- **Occlusion handling**: Object disappears, reappears → same ID
- **Identity switches**: Quantify ID-SW rate; target <5%

### Dataset Evaluation
- **Ego4D with GT**: If hand-object interaction GT available, validate re-ids
- **Trajectory smoothness**: Motion should be continuous (no jumps)

---

## Performance Targets

| Task | Latency | Target |
|------|---------|--------|
| Frame matching | 30ms | Hungarian algorithm |
| Bayesian update | 10ms | Matrix ops |
| Disappearance handling | 5ms | Hashtable ops |
| **Total** | <50ms | Per-frame |

---

## Output Artifacts

- `tracking_result.pkl`: Entity timelines, trajectories, beliefs
- `tracking_metrics.json`: ID-SW, fragmentations, success rates
- `entity_trajectories.json`: 3D positions over time for each entity
- `re_id_log.jsonl`: All re-identification events with confidence scores

---

## Configuration

```yaml
tracking:
  ttl_seconds: 2.0  # indoor: 2, outdoor: 5-10
  
  # Re-id thresholds
  reid_threshold: 0.75
  reid_embedding_weight: 0.5
  reid_class_weight: 0.3
  reid_motion_weight: 0.2
  
  # Bayesian belief smoothing
  embedding_ema_alpha: 0.1  # exponential moving average
  
  # Occlusion parameters
  occlusion_margin_mm: 10
  occlusion_percentage_threshold: 0.3
  
  # Hungarian matching
  max_distance_3d_mm: 200  # max distance to consider a match
```

---

## CLI Integration

```bash
python -m orion.cli analyze \
  --video video.mp4 \
  --mode tracking \
  --config config.yaml \
  --output-dir results/phase2
```

---

## Next Phase (Phase 3)

- Compute CIS (Causal Influence Scores) from 3D trajectories
- Build Dense/Sparse scene graphs
- Detect state changes
- Prepare for QA

