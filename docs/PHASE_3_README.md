# PHASE 3: Enhanced CIS (3D + Hand Signals) + Scene Graph Building

**Objective**: Compute causal influence scores from 3D trajectories and hand interactions. Build Dense/Sparse scene graphs with context-aware indexing. Enable spatial-temporal-causal reasoning.

**Timeline**: Week 2–3

**Success Criteria**:
- ✅ 3D CIS formula implemented with spatial/motion/semantic/hand components
- ✅ Scene graph nodes created for entities, events, zones
- ✅ Dense and Sparse indices operational (context-aware switching)
- ✅ Causal links scored and indexed
- ✅ CIS precision/recall >85% vs GT causality (if available)
- ✅ Latency: <300ms/frame end-to-end
- ✅ Context detector operational (INDOOR vs OUTDOOR)

---

## Architecture Overview

```
TrackingResult (from Phase 2)
├─ entity_timelines: Dict[entity_id, List[EntityState]]
├─ trajectories: Dict[entity_id, Trajectory3D]
└─ frame_entities: Dict[timestamp, Dict[entity_id, Belief]]

    ↓
┌──────────────────────────────────────────────────┐
│  CONTEXT-AWARE PROCESSING                        │
├──────────────────────────────────────────────────┤
│  [Context Detection]                             │
│  ├─ Scene type (office, kitchen, street, etc.)   │
│  ├─ Camera motion (optical flow)                 │
│  ├─ GPS (optional)                              │
│  └─ Decide: INDOOR (dense) or OUTDOOR (sparse)   │
│                                                  │
│  [Route to appropriate index]                    │
│  ├─ DENSE: full causal graph                     │
│  └─ SPARSE: recent events only                   │
│                                                  │
└──────────────────────────────────────────────────┘

    ↓
┌──────────────────────────────────────────────────┐
│  CIS COMPUTATION (3D + Hand Signals)             │
├──────────────────────────────────────────────────┤
│                                                  │
│  For each temporal window:                       │
│  ├─ Temporal score: exp(-Δt / tau)               │
│  ├─ Spatial score: 3D distance-based             │
│  ├─ Motion score: velocity alignment (3D)        │
│  ├─ Semantic score: embedding + type compat      │
│  ├─ Hand bonus: if hand interacting              │
│  └─ CIS = weighted sum                           │
│                                                  │
└──────────────────────────────────────────────────┘

    ↓
┌──────────────────────────────────────────────────┐
│  SCENE GRAPH CONSTRUCTION                        │
├──────────────────────────────────────────────────┤
│                                                  │
│  [Entity Nodes]                                  │
│  ├─ {id, class, embedding, description}         │
│  └─ Aggregated from entire timeline             │
│                                                  │
│  [Event Nodes]                                   │
│  ├─ {timestamp, type, entities}                 │
│  └─ State changes, interactions                  │
│                                                  │
│  [Zone Nodes]                                    │
│  ├─ {zone_id, label, occupancy}                 │
│  └─ 3D spatial regions                           │
│                                                  │
│  [Edges: Causal Links]                          │
│  ├─ CIS > threshold → create edge                │
│  ├─ Weight: CIS score                            │
│  └─ Temporal direction: agent → patient         │
│                                                  │
│  [Edges: Spatial Relations]                      │
│  ├─ Near, inside, supporting, occluding          │
│  └─ Computed from 3D geometry                    │
│                                                  │
│  [Edges: Interaction Relations]                  │
│  ├─ Hand-object: grasping, touching, etc.       │
│  └─ Temporal extent                              │
│                                                  │
└──────────────────────────────────────────────────┘

    ↓
SemanticResult
├─ entities: List[SemanticEntity]
├─ events: List[SemanticEvent]
├─ causal_links: List[CausalLink]
├─ spatial_links: List[SpatialLink]
├─ interaction_links: List[InteractionLink]
├─ zones: List[SpatialZone]
└─ indices: {dense: DenseSceneGraph, sparse: SparseSceneGraph}
```

---

## Key Components

### 1. Context Detector

```python
class ContextDetector:
    """
    Detect context (INDOOR vs OUTDOOR, moving vs stationary)
    """
    
    def __init__(self):
        self.scene_classifier = SceneClassifier()
        self.motion_detector = MotionDetector()
    
    def detect(self, frame: np.ndarray, prev_frame: Optional[np.ndarray], gps_location: Optional[Tuple]) -> str:
        """
        Returns: "INDOOR", "OUTDOOR", "MOVING", or "UNKNOWN"
        """
        
        # Scene classification
        scene_type, scene_conf = self.scene_classifier.classify(frame)
        is_indoor = scene_type in ["office", "living_room", "kitchen", "bedroom", "bathroom"]
        is_outdoor = scene_type in ["street", "park", "outdoor"]
        
        # Camera motion
        if prev_frame is not None:
            motion_mag = self._compute_optical_flow_magnitude(prev_frame, frame)
        else:
            motion_mag = 0.0
        
        is_moving = motion_mag > 0.3  # threshold
        
        # GPS-based (if available)
        if gps_location:
            is_home = self._check_geofence(gps_location, "home")
            if is_home and scene_conf > 0.7:
                return "INDOOR"
            elif not is_home and is_outdoor and scene_conf > 0.7:
                return "OUTDOOR"
        
        # Fallback
        if is_moving:
            return "MOVING"
        elif is_indoor:
            return "INDOOR"
        elif is_outdoor:
            return "OUTDOOR"
        else:
            return "UNKNOWN"
    
    def _compute_optical_flow_magnitude(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute average motion between frames
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        return float(np.mean(mag))
```

### 2. Enhanced CIS (3D + Hand Signals)

```python
class CausalInfluenceScorer3D:
    """
    Compute CIS with 3D spatial/motion and hand interaction signals
    """
    
    def __init__(self, config: SemanticConfig):
        self.config = config
        # Weights from HPO or config
        self.weight_temporal = config.cis.weight_temporal or 0.30
        self.weight_spatial = config.cis.weight_spatial or 0.44
        self.weight_motion = config.cis.weight_motion or 0.21
        self.weight_semantic = config.cis.weight_semantic or 0.06
        # Hand interaction bonus (additive)
        self.hand_grasping_bonus = 0.30
        self.hand_touching_bonus = 0.15
        self.hand_near_bonus = 0.05
    
    def calculate_cis(
        self,
        agent: EntityState,
        patient: EntityState,
        time_delta: float,  # seconds
        scene_context: Dict = None,
    ) -> Tuple[float, Dict]:
        """
        Compute CIS between two entities
        
        Returns:
            (cis_score, components_dict)
        """
        
        # === 1. TEMPORAL SCORE ===
        T = self._temporal_score(time_delta)
        
        # === 2. SPATIAL SCORE (3D) ===
        S = self._spatial_score_3d(agent, patient)
        
        # === 3. MOTION SCORE (3D velocity alignment) ===
        M = self._motion_score_3d(agent, patient)
        
        # === 4. SEMANTIC SCORE ===
        Se = self._semantic_score(agent, patient)
        
        # === 5. HAND INTERACTION BONUS ===
        H = self._hand_bonus(agent, patient)
        
        # === COMBINE ===
        cis = (
            self.weight_temporal * T +
            self.weight_spatial * S +
            self.weight_motion * M +
            self.weight_semantic * Se +
            H
        )
        
        # Clip to [0, 1]
        cis = max(0, min(1, cis))
        
        # Apply scene context
        if scene_context:
            cis = self._apply_scene_context(cis, agent, patient, scene_context)
        
        components = {
            'temporal': T,
            'spatial': S,
            'motion': M,
            'semantic': Se,
            'hand_bonus': H,
            'cis': cis,
        }
        
        return cis, components
    
    def _temporal_score(self, time_delta: float) -> float:
        """
        Exponential decay with time
        τ = 4.0 seconds (configurable)
        """
        tau = self.config.cis.temporal_decay_seconds or 4.0
        score = np.exp(-time_delta / tau)
        return float(score)
    
    def _spatial_score_3d(self, agent: EntityState, patient: EntityState) -> float:
        """
        3D Euclidean distance-based scoring
        max_dist = 600mm (configurable)
        """
        if agent.centroid_3d is None or patient.centroid_3d is None:
            return 0.5  # unknown, neutral
        
        dx = agent.centroid_3d[0] - patient.centroid_3d[0]
        dy = agent.centroid_3d[1] - patient.centroid_3d[1]
        dz = agent.centroid_3d[2] - patient.centroid_3d[2]
        
        dist_3d = np.sqrt(dx**2 + dy**2 + dz**2)
        
        max_dist = self.config.cis.max_spatial_distance_mm or 600.0
        
        # Quadratic falloff
        score = max(0, 1 - (dist_3d / max_dist) ** 2)
        
        # Boost if very close (touching)
        if dist_3d < 10:  # 1cm
            score = min(1.0, score * 1.3)
        
        return float(score)
    
    def _motion_score_3d(self, agent: EntityState, patient: EntityState) -> float:
        """
        3D velocity alignment scoring
        """
        vel_agent = self._get_velocity_3d(agent)
        vel_patient = self._get_velocity_3d(patient)
        
        vel_mag_a = np.linalg.norm(vel_agent)
        vel_mag_p = np.linalg.norm(vel_patient)
        
        if vel_mag_a < 1e-6 or vel_mag_p < 1e-6:
            return 0.5  # neither moving strongly
        
        # Normalize and compute cosine similarity
        vel_a_norm = vel_agent / vel_mag_a
        vel_p_norm = vel_patient / vel_mag_p
        
        cos_angle = np.dot(vel_a_norm, vel_p_norm)
        
        # Map [-1, 1] → [0, 1]
        score = (cos_angle + 1) / 2
        
        # Boost if agent moving toward patient
        to_patient = np.array(patient.centroid_3d) - np.array(agent.centroid_3d)
        if np.dot(vel_agent, to_patient) > 0:
            score *= 1.2
        
        return float(min(1.0, score))
    
    def _semantic_score(self, agent: EntityState, patient: EntityState) -> float:
        """
        CLIP embedding similarity + type compatibility
        """
        if agent.embedding is None or patient.embedding is None:
            return 0.5
        
        # Embedding similarity
        emb_sim = np.dot(agent.embedding, patient.embedding) / (
            np.linalg.norm(agent.embedding) * np.linalg.norm(patient.embedding) + 1e-6
        )
        
        # Type compatibility heuristic
        type_compat = self._check_type_compatibility(agent.class_label, patient.class_label)
        
        score = 0.6 * emb_sim + 0.4 * type_compat
        
        return float(score)
    
    def _hand_bonus(self, agent: EntityState, patient: EntityState) -> float:
        """
        Bonus if agent is hand and interacting with patient
        """
        if not agent.is_hand or patient.hand_interaction_type is None:
            return 0.0
        
        interaction_type = patient.hand_interaction_type
        
        if interaction_type == "GRASPING":
            return self.hand_grasping_bonus
        elif interaction_type == "TOUCHING":
            return self.hand_touching_bonus
        elif interaction_type == "NEAR":
            return self.hand_near_bonus
        else:
            return 0.0
    
    def _get_velocity_3d(self, entity: EntityState) -> np.ndarray:
        """
        Extract 3D velocity from recent observations
        """
        if not hasattr(entity, 'velocity_3d') or entity.velocity_3d is None:
            return np.array([0, 0, 0])
        return np.array(entity.velocity_3d)
    
    def _check_type_compatibility(self, class_a: str, class_b: str) -> float:
        """
        Heuristic: are these classes likely to causally interact?
        """
        # Agent-patient compatible pairs
        compatible_pairs = {
            ('hand', 'cup'): 1.0,
            ('hand', 'door'): 1.0,
            ('hand', 'keyboard'): 0.8,
            ('person', 'door'): 0.9,
            ('cup', 'table'): 0.7,
            # ... more pairs
        }
        
        pair = tuple(sorted([class_a, class_b]))
        return compatible_pairs.get(pair, 0.5)
    
    def _apply_scene_context(self, cis: float, agent: EntityState, patient: EntityState, context: Dict) -> float:
        """
        Boost CIS based on scene type
        """
        scene_type = context.get('scene_type', 'unknown')
        
        scene_boosts = {
            'kitchen': {
                ('hand', 'knife'): 1.3,
                ('hand', 'oven'): 1.2,
                ('knife', 'food'): 1.3,
            },
            'office': {
                ('hand', 'keyboard'): 1.3,
                ('hand', 'mouse'): 1.3,
                ('hand', 'phone'): 1.2,
            },
            'living_room': {
                ('hand', 'remote'): 1.2,
                ('hand', 'tv'): 1.2,
            },
        }
        
        pair = (agent.class_label, patient.class_label)
        boost = scene_boosts.get(scene_type, {}).get(pair, 1.0)
        
        return min(1.0, cis * boost)
```

### 3. Dense vs Sparse Scene Graphs

```python
class DenseSceneGraph:
    """
    Complete graph for INDOOR environments
    """
    
    def __init__(self):
        self.entities: Dict[str, SemanticEntity] = {}
        self.events: List[SemanticEvent] = []
        self.causal_links: List[CausalLink] = []  # all links > threshold
        self.spatial_links: List[SpatialLink] = []
        self.interaction_links: List[InteractionLink] = []
        self.zones: Dict[str, SpatialZone] = {}
    
    def build_from_tracking(
        self,
        entity_timelines: Dict[str, List[EntityState]],
        hand_detections: List[Hand],
        cis_scorer: CausalInfluenceScorer3D,
        scene_context: Dict = None,
    ):
        """
        Construct complete scene graph from tracking data
        """
        
        # Step 1: Create entity nodes (aggregated)
        for entity_id, timeline in entity_timelines.items():
            self.entities[entity_id] = SemanticEntity(
                entity_id=entity_id,
                class_label=timeline[0].class_label,  # primary class
                first_seen=timeline[0].timestamp,
                last_seen=timeline[-1].timestamp,
                embedding_mean=np.mean([e.embedding for e in timeline], axis=0),
                trajectories_3d=[e.centroid_3d for e in timeline],
                descriptions=[e.description for e in timeline if e.description],
                hand_interactions=self._extract_hand_interactions(entity_id, hand_detections),
            )
        
        # Step 2: Detect state changes → events
        for entity_id, timeline in entity_timelines.items():
            for i in range(1, len(timeline)):
                state_change_mag = self._compute_state_change(timeline[i-1], timeline[i])
                if state_change_mag > 0.3:  # threshold
                    self.events.append(SemanticEvent(
                        timestamp=timeline[i].timestamp,
                        entity_id=entity_id,
                        event_type="STATE_CHANGE",
                        state_change_magnitude=state_change_mag,
                    ))
        
        # Step 3: Compute causal links
        entity_ids = list(entity_timelines.keys())
        for i, agent_id in enumerate(entity_ids):
            agent_timeline = entity_timelines[agent_id]
            
            for patient_id in entity_ids[i+1:]:
                if patient_id == agent_id:
                    continue
                
                patient_timeline = entity_timelines[patient_id]
                
                # Compute CIS for this pair
                agent_final = agent_timeline[-1]
                patient_final = patient_timeline[-1]
                
                time_delta = patient_final.timestamp - agent_final.timestamp
                
                cis_score, components = cis_scorer.calculate_cis(
                    agent_final, patient_final, abs(time_delta), scene_context
                )
                
                if cis_score > self.config.cis_threshold:
                    self.causal_links.append(CausalLink(
                        agent_id=agent_id,
                        patient_id=patient_id,
                        cis_score=cis_score,
                        components=components,
                    ))
        
        # Step 4: Spatial relationships
        for entity_id_a, entity_a in self.entities.items():
            for entity_id_b, entity_b in self.entities.items():
                if entity_id_a >= entity_id_b:
                    continue
                
                rel_type = self._compute_spatial_relation(entity_a, entity_b)
                if rel_type:
                    self.spatial_links.append(SpatialLink(
                        entity_a_id=entity_id_a,
                        entity_b_id=entity_id_b,
                        relation_type=rel_type,
                    ))
        
        # Step 5: Spatial zones (3D regions)
        self._detect_zones()

class SparseSceneGraph:
    """
    Lightweight graph for OUTDOOR environments
    """
    
    def __init__(self, max_history_seconds: float = 120.0):
        self.max_history_seconds = max_history_seconds
        self.recent_entities: Dict[str, SemanticEntity] = {}
        self.recent_events: List[SemanticEvent] = []
        self.recent_interactions: List[InteractionLink] = []
        self.obstacles: Dict[str, Obstacle] = {}
    
    def build_from_tracking(
        self,
        entity_timelines: Dict[str, List[EntityState]],
        current_timestamp: float,
        hand_detections: List[Hand],
    ):
        """
        Lightweight aggregation: only recent, high-value data
        """
        
        # Only entities seen in last max_history_seconds
        for entity_id, timeline in entity_timelines.items():
            if current_timestamp - timeline[-1].timestamp < self.max_history_seconds:
                # This entity is recent
                self.recent_entities[entity_id] = SemanticEntity(
                    entity_id=entity_id,
                    class_label=timeline[-1].class_label,
                    first_seen=timeline[0].timestamp,
                    last_seen=timeline[-1].timestamp,
                    embedding_mean=timeline[-1].embedding,  # just current
                    trajectories_3d=[e.centroid_3d for e in timeline],
                    descriptions=[timeline[-1].description] if timeline[-1].description else [],
                )
        
        # Only significant interactions (high confidence)
        for hand_det in hand_detections:
            for entity_id in self.recent_entities:
                # ... compute interaction, store if confidence > 0.7
                pass
        
        # Detect obstacles
        self.obstacles = self._detect_obstacles()

@dataclass
class SemanticEntity:
    entity_id: str
    class_label: str
    first_seen: float
    last_seen: float
    embedding_mean: np.ndarray
    trajectories_3d: List[Tuple[float, float, float]]
    descriptions: List[str]
    hand_interactions: List[Dict] = field(default_factory=list)

@dataclass
class CausalLink:
    agent_id: str
    patient_id: str
    cis_score: float
    components: Dict[str, float]  # temporal, spatial, motion, semantic, hand_bonus

@dataclass
class SpatialLink:
    entity_a_id: str
    entity_b_id: str
    relation_type: str  # "NEAR", "INSIDE", "SUPPORTING", "OCCLUDING"

@dataclass
class InteractionLink:
    hand_id: str
    object_id: str
    interaction_type: str  # "GRASPING", "TOUCHING", "POINTING", "PUSHING"
    duration_ms: float
    confidence: float
    timestamp: float
```

---

## Testing & Validation

### CIS Validation
- **Synthetic scenarios**: Hand grasps object → CIS should be 0.85+
- **Temporal**: Two events 1s apart → score > two events 10s apart
- **3D**: Objects 10cm apart → score > objects 1m apart
- **Ablations**: 2D vs 3D CIS, with/without hand signals

### Scene Graph Validation
- **Entity nodes**: Count matches tracking output
- **Event nodes**: State changes detected and timestamped
- **Causal links**: Ground truth comparison (if available)
- **Zone detection**: Visual inspection

---

## Performance Targets

| Task | Latency | Target |
|------|---------|--------|
| Context detection | 20ms | Lightweight |
| CIS computation (all pairs) | 100ms | O(n²) for n entities |
| Scene graph construction | 50ms | Graph building |
| **Total** | <200ms | Per-frame |

---

## Output Artifacts

- `scene_graph.pkl`: Entities, events, links, zones
- `causal_graph.json`: All causal links with CIS scores
- `spatial_zones.json`: Zone definitions and occupancy
- `cis_heatmap.csv`: CIS matrix for all entity pairs
- `semantic_result.pkl`: Full SemanticResult object

---

## Configuration

```yaml
semantic:
  cis:
    weight_temporal: 0.30
    weight_spatial: 0.44
    weight_motion: 0.21
    weight_semantic: 0.06
    temporal_decay_seconds: 4.0
    max_spatial_distance_mm: 600.0
    cis_threshold: 0.50  # minimum CIS to create causal link
  
  scene_graph:
    context_aware: true
    dense_ttl_seconds: 999999  # keep forever
    sparse_ttl_seconds: 120.0  # 2 minutes

context:
  enable_gps: false
  enable_motion_detection: true
```

---

## CLI Integration

```bash
python -m orion.cli analyze \
  --video video.mp4 \
  --mode semantic_3d \
  --config config.yaml \
  --output-dir results/phase3
```

---

## Next Phase (Phase 4)

- QA engine implementation
- HTML visualization
- End-to-end batch processing
- MVP demo

