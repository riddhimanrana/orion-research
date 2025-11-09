# PHASE 4: Visual SLAM Integration + Interactive Visualization

**Objective**: Integrate visual SLAM for accurate world-coordinate mapping. Fix zone detection to properly distinguish separate rooms. Implement interactive video overlay with clickable entities.

**Timeline**: Week 3–4

**Success Criteria**:
- ✅ Visual SLAM (ORB-SLAM3 or OpenCV SLAM) provides camera poses
- ✅ Transform all observations to consistent world frame
- ✅ Zone detection accurately detects 3-4 distinct rooms (not 8 viewpoints)
- ✅ Interactive OpenCV window: click entities to see info
- ✅ Spatial map shows true world coordinates with SLAM trajectory
- ✅ Entity positions persistent across camera movement
- ✅ Loop closure: recognize when returning to same room

---

## Phase 4 Architecture Overview

```
Video Frames
    ↓
┌──────────────────────────────────────────┐
│  VISUAL SLAM                             │
├──────────────────────────────────────────┤
│                                          │
│  [ORB-SLAM3 or OpenCV SLAM]             │
│  ├─ Feature extraction (ORB/SIFT)       │
│  ├─ Feature matching across frames      │
│  ├─ Motion estimation (5-point/8-point) │
│  ├─ Bundle adjustment                    │
│  └─ Loop closure detection               │
│                                          │
│  Output:                                 │
│  ├─ Camera poses: [R|t] per frame       │
│  ├─ 3D map points: world coordinates    │
│  ├─ Trajectory: camera path in 3D       │
│  └─ Covariance: pose uncertainty         │
│                                          │
└──────────────────────────────────────────┘
    ↓
Camera Poses (4×4 transformation matrices)

    ↓
┌──────────────────────────────────────────┐
│  COORDINATE TRANSFORMATION               │
├──────────────────────────────────────────┤
│                                          │
│  For each entity detection:              │
│  1. Get 3D position in camera frame     │
│  2. Apply camera pose transform:         │
│     P_world = R @ P_camera + t          │
│  3. Store in world coordinates          │
│                                          │
│  Benefits:                               │
│  ✓ Consistent coordinates across time   │
│  ✓ Same room = same location            │
│  ✓ Loop closure handled automatically   │
│                                          │
└──────────────────────────────────────────┘
    ↓
World-Coordinate Observations

    ↓
┌──────────────────────────────────────────┐
│  IMPROVED ZONE DETECTION                 │
├──────────────────────────────────────────┤
│                                          │
│  Now with world coordinates:             │
│  ├─ Bedroom 1 @ t=0s: (2.5, 0, 3.0) m   │
│  ├─ Bedroom 1 @ t=60s: (2.5, 0, 3.0) m  │
│  └─ Same location → Same zone! ✓        │
│                                          │
│  Zone Clustering:                        │
│  ├─ Aggregate entities by world pos     │
│  ├─ DBSCAN with eps=3m                   │
│  ├─ Zone re-ID based on spatial overlap │
│  └─ Expected: 3-4 zones for 3 rooms     │
│                                          │
└──────────────────────────────────────────┘
    ↓
Accurate Zone Map

    ↓
┌──────────────────────────────────────────┐
│  INTERACTIVE VISUALIZATION               │
├──────────────────────────────────────────┤
│                                          │
│  OpenCV Window Features:                 │
│  ├─ Main video with overlays             │
│  ├─ Click entity → show info panel       │
│  ├─ Hover → highlight entity path        │
│  ├─ Keyboard shortcuts:                  │
│  │  ├─ 'z' → toggle zone boundaries      │
│  │  ├─ 't' → toggle SLAM trajectory      │
│  │  ├─ 'd' → toggle depth heatmap        │
│  │  ├─ 'h' → toggle hand tracking        │
│  │  └─ 'space' → pause/play              │
│  └─ Spatial map (separate window):       │
│     ├─ Camera trajectory (SLAM)          │
│     ├─ Zone boundaries                   │
│     ├─ Entity positions (world coords)   │
│     └─ Click zone → highlight in video   │
│                                          │
└──────────────────────────────────────────┘
```

---

## Key Components

### 1. SLAM Integration Module

We'll use **OpenCV's SLAM** (simpler) or **ORB-SLAM3** (more accurate) for camera pose estimation.

```python
class SLAMEngine:
    """
    Visual SLAM for camera pose estimation
    """
    
    def __init__(
        self,
        method: str = "opencv",  # "opencv" or "orb-slam3"
        camera_params: Optional[Dict] = None
    ):
        self.method = method
        self.camera_params = camera_params
        
        if method == "opencv":
            self.slam = OpenCVSLAM(camera_params)
        elif method == "orb-slam3":
            self.slam = ORBSLAM3Wrapper(camera_params)
        else:
            raise ValueError(f"Unknown SLAM method: {method}")
        
        # State
        self.poses = []  # List of 4x4 transformation matrices
        self.map_points = []  # 3D map points in world frame
        self.trajectory = []  # Camera positions over time
    
    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_idx: int
    ) -> Optional[np.ndarray]:
        """
        Process frame and return camera pose
        
        Returns:
            4x4 transformation matrix [R|t] or None if tracking lost
        """
        pose = self.slam.track(frame, timestamp)
        
        if pose is not None:
            self.poses.append(pose)
            self.trajectory.append(pose[:3, 3])  # Translation component
        else:
            # Tracking lost, use last known pose
            if self.poses:
                pose = self.poses[-1]
            else:
                # Initialize with identity
                pose = np.eye(4)
        
        return pose
    
    def transform_to_world(
        self,
        point_camera: np.ndarray,  # (x, y, z) in camera frame
        pose: np.ndarray  # 4x4 transformation matrix
    ) -> np.ndarray:
        """
        Transform point from camera frame to world frame
        
        Args:
            point_camera: (x, y, z) in mm, camera frame
            pose: 4x4 [R|t] matrix
        
        Returns:
            (x, y, z) in mm, world frame
        """
        # Convert to homogeneous coordinates
        point_h = np.append(point_camera, 1.0)
        
        # Apply transformation
        point_world_h = pose @ point_h
        
        # Return 3D coordinates
        return point_world_h[:3]
    
    def get_trajectory(self) -> np.ndarray:
        """Get camera trajectory as (N, 3) array"""
        return np.array(self.trajectory)
    
    def save_trajectory(self, output_path: str):
        """Save trajectory in TUM format"""
        with open(output_path, 'w') as f:
            for i, pose in enumerate(self.poses):
                # Extract translation and rotation (quaternion)
                t = pose[:3, 3]
                R = pose[:3, :3]
                # Convert R to quaternion
                from scipy.spatial.transform import Rotation
                quat = Rotation.from_matrix(R).as_quat()  # (x, y, z, w)
                
                # TUM format: timestamp x y z qx qy qz qw
                f.write(f"{i} {t[0]} {t[1]} {t[2]} {quat[0]} {quat[1]} {quat[2]} {quat[3]}\n")


class OpenCVSLAM:
    """
    Simplified SLAM using OpenCV feature tracking + motion estimation
    """
    
    def __init__(self, camera_params: Optional[Dict] = None):
        # Feature detector
        self.detector = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Camera intrinsics
        if camera_params:
            self.K = np.array(camera_params['K']).reshape(3, 3)
        else:
            # Estimate from image size (will be set in track())
            self.K = None
        
        # State
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.current_pose = np.eye(4)  # Start at origin
        self.scale = 1.0  # Scale factor (unknown with monocular)
    
    def track(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Track camera motion using feature matching
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize camera matrix if needed
        if self.K is None:
            h, w = gray.shape
            fx = fy = w  # Approximate focal length
            cx, cy = w / 2, h / 2
            self.K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
        
        # Detect features
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if self.prev_frame is None:
            # First frame - initialize
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return self.current_pose
        
        if descriptors is None or len(keypoints) < 10:
            # Not enough features
            return self.current_pose
        
        # Match features
        matches = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)
        
        # Apply ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            # Not enough good matches
            return self.current_pose
        
        # Extract matched points
        pts1 = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([keypoints[m.trainIdx].pt for m in good_matches])
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is None:
            return self.current_pose
        
        # Recover pose
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        
        # Apply scale (unknown with monocular - use constant or estimate from depth)
        t_scaled = t * self.scale
        
        # Update current pose
        # T_new = T_old @ T_relative
        T_relative = np.eye(4)
        T_relative[:3, :3] = R
        T_relative[:3, 3] = t_scaled.flatten()
        
        self.current_pose = self.current_pose @ T_relative
        
        # Update previous frame
        self.prev_frame = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        
        return self.current_pose.copy()
```

### 2. World-Coordinate Tracking

```python
class WorldCoordinateTracker:
    """
    Tracker that maintains entity positions in world coordinates
    """
    
    def __init__(self, slam_engine: SLAMEngine):
        self.slam = slam_engine
        self.entities_world = {}  # entity_id → list of (timestamp, pos_world, pose_idx)
    
    def add_observation(
        self,
        entity_id: str,
        timestamp: float,
        pos_camera: np.ndarray,  # (x, y, z) in camera frame (mm)
        frame_idx: int
    ):
        """
        Add entity observation and transform to world coordinates
        """
        if frame_idx >= len(self.slam.poses):
            # No pose available yet
            return
        
        pose = self.slam.poses[frame_idx]
        pos_world = self.slam.transform_to_world(pos_camera, pose)
        
        if entity_id not in self.entities_world:
            self.entities_world[entity_id] = []
        
        self.entities_world[entity_id].append((timestamp, pos_world, frame_idx))
    
    def get_entity_world_centroid(self, entity_id: str) -> np.ndarray:
        """
        Get mean world position for entity
        """
        if entity_id not in self.entities_world:
            return None
        
        positions = [obs[1] for obs in self.entities_world[entity_id]]
        return np.mean(positions, axis=0)
    
    def get_all_entity_centroids(self) -> Dict[str, np.ndarray]:
        """
        Get world centroids for all entities
        """
        centroids = {}
        for entity_id in self.entities_world:
            centroids[entity_id] = self.get_entity_world_centroid(entity_id)
        return centroids
```

### 3. Interactive Visualization

```python
class InteractiveVisualizer:
    """
    Interactive OpenCV-based visualizer with clickable entities
    """
    
    def __init__(
        self,
        window_name: str = "Orion Interactive Viewer",
        spatial_map_size: Tuple[int, int] = (600, 600)
    ):
        self.window_name = window_name
        self.spatial_window_name = "Spatial Map (SLAM)"
        self.spatial_map_size = spatial_map_size
        
        # State
        self.selected_entity_id = None
        self.show_zones = True
        self.show_trajectory = True
        self.show_depth = False
        self.show_hands = True
        self.paused = False
        
        # Mouse callback state
        self.mouse_x = 0
        self.mouse_y = 0
        self.clicked = False
        
        # Create windows
        cv2.namedWindow(self.window_name)
        cv2.namedWindow(self.spatial_window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        cv2.setMouseCallback(self.spatial_window_name, self._spatial_mouse_callback)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events on main window"""
        self.mouse_x = x
        self.mouse_y = y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked = True
    
    def _spatial_mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events on spatial map"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Click on spatial map → highlight zone
            pass  # TODO: implement zone selection
    
    def visualize_frame(
        self,
        frame: np.ndarray,
        tracks: List,
        zones: List,
        slam_trajectory: np.ndarray,
        frame_number: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Visualize frame with interactive overlays
        
        Returns:
            (main_frame, spatial_map)
        """
        vis_frame = frame.copy()
        
        # Check for entity clicks
        if self.clicked:
            self.selected_entity_id = self._find_entity_at_click(tracks, self.mouse_x, self.mouse_y)
            self.clicked = False
        
        # Draw entity bounding boxes
        for track in tracks:
            is_selected = (track.entity_id == self.selected_entity_id)
            color = (0, 255, 0) if is_selected else (255, 0, 0)
            thickness = 3 if is_selected else 2
            
            x1, y1, x2, y2 = map(int, track.bbox)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw ID
            label = f"ID: {track.entity_id} | {track.most_likely_class}"
            cv2.putText(vis_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw zones if enabled
        if self.show_zones:
            vis_frame = self._draw_zones(vis_frame, zones)
        
        # Draw info panel for selected entity
        if self.selected_entity_id is not None:
            vis_frame = self._draw_info_panel(vis_frame, tracks)
        
        # Draw spatial map
        spatial_map = self._create_spatial_map_slam(
            tracks, zones, slam_trajectory, frame_number
        )
        
        # Draw instructions
        vis_frame = self._draw_instructions(vis_frame)
        
        return vis_frame, spatial_map
    
    def _find_entity_at_click(self, tracks: List, x: int, y: int) -> Optional[str]:
        """Find which entity was clicked"""
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            if x1 <= x <= x2 and y1 <= y <= y2:
                return track.entity_id
        return None
    
    def _draw_info_panel(self, frame: np.ndarray, tracks: List) -> np.ndarray:
        """Draw info panel for selected entity"""
        # Find selected track
        selected_track = None
        for track in tracks:
            if track.entity_id == self.selected_entity_id:
                selected_track = track
                break
        
        if selected_track is None:
            return frame
        
        # Draw panel on right side
        panel_width = 300
        panel = np.zeros((frame.shape[0], panel_width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        # Add text
        y_offset = 30
        lines = [
            f"Entity ID: {selected_track.entity_id}",
            f"Class: {selected_track.most_likely_class}",
            f"Confidence: {selected_track.detection_confidence:.2f}",
            f"World Pos: {selected_track.centroid_3d_world if hasattr(selected_track, 'centroid_3d_world') else 'N/A'}",
            f"First seen: {selected_track.first_seen_frame}",
            f"Last seen: {selected_track.last_seen_frame}",
        ]
        
        for line in lines:
            cv2.putText(panel, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 25
        
        # Concatenate panel to frame
        frame_with_panel = np.hstack([frame, panel])
        
        return frame_with_panel
    
    def _draw_instructions(self, frame: np.ndarray) -> np.ndarray:
        """Draw keyboard instructions"""
        instructions = [
            "Click entity for info",
            "z: toggle zones",
            "t: toggle trajectory",
            "d: toggle depth",
            "h: toggle hands",
            "SPACE: pause",
            "q: quit"
        ]
        
        y_offset = frame.shape[0] - 150
        for instr in instructions:
            cv2.putText(frame, instr, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
        
        return frame
    
    def _create_spatial_map_slam(
        self,
        tracks: List,
        zones: List,
        slam_trajectory: np.ndarray,
        frame_number: int
    ) -> np.ndarray:
        """
        Create spatial map with SLAM trajectory
        """
        map_h, map_w = self.spatial_map_size
        spatial_map = np.zeros((map_h, map_w, 3), dtype=np.uint8)
        spatial_map[:] = (30, 30, 30)
        
        # Draw grid (same as before but with world coordinates)
        # ... grid drawing code ...
        
        # Draw SLAM trajectory
        if self.show_trajectory and len(slam_trajectory) > 1:
            # Project trajectory to 2D (top-down)
            traj_2d = slam_trajectory[:, [0, 2]]  # x, z coordinates
            
            # Scale to map
            traj_min = traj_2d.min(axis=0)
            traj_max = traj_2d.max(axis=0)
            traj_range = traj_max - traj_min + 1e-6
            traj_scaled = (traj_2d - traj_min) / traj_range
            traj_scaled = traj_scaled * np.array([map_w - 40, map_h - 40]) + 20
            traj_scaled = traj_scaled.astype(int)
            
            # Draw trajectory line
            for i in range(len(traj_scaled) - 1):
                pt1 = tuple(traj_scaled[i])
                pt2 = tuple(traj_scaled[i + 1])
                cv2.line(spatial_map, pt1, pt2, (100, 200, 255), 2)
            
            # Draw current position
            if frame_number < len(traj_scaled):
                current_pos = tuple(traj_scaled[min(frame_number, len(traj_scaled) - 1)])
                cv2.circle(spatial_map, current_pos, 8, (0, 255, 0), -1)
        
        # Draw zones
        if self.show_zones:
            for zone in zones:
                # Draw zone boundary
                # ... zone drawing code ...
                pass
        
        # Draw entities in world coordinates
        for track in tracks:
            if hasattr(track, 'centroid_3d_world'):
                # Project to 2D map
                # ... entity drawing code ...
                pass
        
        return spatial_map
    
    def handle_keyboard(self) -> bool:
        """
        Handle keyboard input
        
        Returns:
            True if should continue, False if quit
        """
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return False
        elif key == ord('z'):
            self.show_zones = not self.show_zones
        elif key == ord('t'):
            self.show_trajectory = not self.show_trajectory
        elif key == ord('d'):
            self.show_depth = not self.show_depth
        elif key == ord('h'):
            self.show_hands = not self.show_hands
        elif key == ord(' '):
            self.paused = not self.paused
            if self.paused:
                print("PAUSED - Press SPACE to resume")
        
        return True
```

---

## Testing & Validation

### SLAM Accuracy
- Track known trajectory (e.g., walk in square, return to start)
- Measure loop closure error
- Compare with ground truth (if available)

### Zone Detection
- **Before SLAM**: 8 zones for 3-room path
- **After SLAM**: 3-4 zones (one per room)
- Validate zone persistence when returning to same room

### Performance
- SLAM overhead: <50ms per frame (OpenCV) or <100ms (ORB-SLAM3)
- Total pipeline: 1-2 FPS with SLAM

---

## Output Artifacts

- `slam_trajectory.txt`: Camera poses in TUM format
- `zones_world.json`: Zone definitions in world coordinates
- `entities_world.json`: Entity positions in world frame
- `slam_map.ply`: 3D map points (point cloud)

---

## CLI Integration

```
SemanticResult (from Phase 3)
├─ entities, events, causal_links
├─ indices: {dense, sparse}
└─ scene_graph: {nodes, edges}

    ↓
┌──────────────────────────────────────────┐
│  QA PIPELINE                             │
├──────────────────────────────────────────┤
│                                          │
│  [Question Input]                        │
│  ├─ "What did I hold?"                   │
│  ├─ "When did the cup spill?"            │
│  └─ "Did I touch the phone?"             │
│                                          │
│  [Question Classification]               │
│  ├─ Type: WHEN, WHERE, WHAT, HOW, WHY   │
│  ├─ Entities: extract mentioned objects │
│  ├─ Constraints: time/space filters     │
│  └─ LLM: small local model               │
│                                          │
│  [Route to Index]                        │
│  ├─ WHEN → temporal_index                │
│  ├─ WHERE → spatial_index                │
│  ├─ WHAT → semantic_index                │
│  └─ CAUSAL → memgraph/causal_index       │
│                                          │
│  [Query Execution]                       │
│  ├─ Fast path: in-memory lookups        │
│  ├─ Slow path: LLM generation            │
│  └─ Fallback: heuristic templates        │
│                                          │
│  [Answer Generation]                     │
│  ├─ Natural language synthesis           │
│  ├─ Evidence collection (entity IDs, ts) │
│  ├─ Video clip extraction (timestamps)   │
│  └─ Confidence scoring                   │
│                                          │
│  [Answer Return]                         │
│  ├─ Text: natural language               │
│  ├─ Clips: list of (start_s, end_s)     │
│  ├─ Highlighted entities: {id, ts}      │
│  └─ Evidence: structured JSON            │
│                                          │
└──────────────────────────────────────────┘

    ↓
Answer
├─ text: str
├─ confidence: float
├─ clips: List[(start_ms, end_ms)]
├─ highlighted_entities: List[str]
├─ evidence: Dict
└─ reasoning: str

    ↓
┌──────────────────────────────────────────┐
│  VISUALIZATION                           │
├──────────────────────────────────────────┤
│                                          │
│  [HTML Interactive Viewer]               │
│  ├─ Video player with timeline           │
│  ├─ Entity legend + colors               │
│  ├─ Overlay toggles:                     │
│  │  ├─ 3D bounding boxes                 │
│  │  ├─ Hand pose + interactions          │
│  │  ├─ Depth map heatmap                 │
│  │  └─ Zone boundaries                   │
│  ├─ Query input box                      │
│  ├─ Answer display                       │
│  └─ Clip browser                         │
│                                          │
│  [Video Clips]                           │
│  ├─ Extract segments from source         │
│  ├─ Encode MP4/WebM for web              │
│  ├─ Generate frame-by-frame overlays     │
│  └─ Store as PNG sequence + video        │
│                                          │
└──────────────────────────────────────────┘

    ↓
Outputs
├─ answers.json: all Q&A pairs
├─ interactive.html: web viewer
├─ clips/: video segments
├─ overlays/: PNG frames with boxes
└─ metrics.json: latency, quality
```

---

## Key Components

### 1. Question Classifier & Parser

```python
class QuestionClassifier:
    """
    Classify question type and extract entities/constraints
    """
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        if use_llm:
            # Small local LM for parsing (Llama2-7B or FLAN-UL2)
            self.lm = self._load_lm()
        
        # Regex-based fallback
        self.patterns = {
            'WHEN': r'when|what time|at what|during',
            'WHERE': r'where|what room|location|place',
            'WHAT': r'what|which|identify|describe',
            'HOW': r'how|do you|are you',
            'SEQUENCE': r'before|after|first|then|later',
            'CAUSALITY': r'cause|lead to|because|result|affect',
        }
    
    def classify(self, question: str) -> Dict:
        """
        Classify question and extract structured info
        
        Returns:
            {
                'q_type': str,  # WHEN, WHERE, etc.
                'confidence': float,
                'entities': List[str],  # mentioned objects
                'constraints': {
                    'time_range': (start_s, end_s),
                    'location': str,
                },
                'focus_entity': str,  # main subject
            }
        """
        
        if self.use_llm:
            result = self._classify_with_llm(question)
        else:
            result = self._classify_with_regex(question)
        
        return result
    
    def _classify_with_llm(self, question: str) -> Dict:
        """
        Use small LM for parsing
        """
        prompt = f"""Parse this question about a video analysis:
        
"{question}"

Return JSON with:
- q_type: one of WHEN, WHERE, WHAT, HOW, SEQUENCE, CAUSALITY
- entities: list of mentioned object names
- focus_entity: main subject
- time_constraint: e.g., "first 5 minutes" or null
- confidence: 0.0-1.0

JSON:
"""
        
        # Run LM (small, local)
        response = self.lm.generate(prompt, max_new_tokens=200)
        
        try:
            import json
            result_json = json.loads(response)
            return result_json
        except:
            # Fallback to regex
            return self._classify_with_regex(question)
    
    def _classify_with_regex(self, question: str) -> Dict:
        """
        Simple regex-based classification
        """
        q_lower = question.lower()
        
        for q_type, pattern in self.patterns.items():
            if re.search(pattern, q_lower):
                break
        else:
            q_type = 'WHAT'  # default
        
        # Extract entity mentions (simple: any noun-like word)
        entities = [word for word in q_lower.split() if len(word) > 3 and word not in self.patterns]
        
        return {
            'q_type': q_type,
            'confidence': 0.7,
            'entities': entities,
            'focus_entity': entities[0] if entities else None,
            'time_constraint': None,
        }
```

### 2. VideoQAEngine

```python
class VideoQAEngine:
    """
    Answer natural language questions about video
    """
    
    def __init__(
        self,
        semantic_result: SemanticResult,
        dense_index: DenseSceneGraph,
        sparse_index: SparseSceneGraph,
        use_llm: bool = True,
    ):
        self.semantic_result = semantic_result
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.classifier = QuestionClassifier(use_llm=use_llm)
        self.use_llm = use_llm
    
    def answer(self, question: str) -> Answer:
        """
        Answer a question about the video
        """
        
        # Step 1: Classify
        parsed = self.classifier.classify(question)
        q_type = parsed['q_type']
        entities = parsed['entities']
        focus_entity = parsed['focus_entity']
        
        # Step 2: Route and query
        if q_type == 'WHEN':
            answer_text, clips, evidence = self._answer_when(question, entities, focus_entity)
        
        elif q_type == 'WHERE':
            answer_text, clips, evidence = self._answer_where(question, entities, focus_entity)
        
        elif q_type == 'WHAT':
            answer_text, clips, evidence = self._answer_what(question, entities, focus_entity)
        
        elif q_type == 'SEQUENCE':
            answer_text, clips, evidence = self._answer_sequence(question, entities)
        
        elif q_type == 'CAUSALITY':
            answer_text, clips, evidence = self._answer_causality(question, entities)
        
        else:
            answer_text = f"I don't understand. Could you rephrase? (Detected: {q_type})"
            clips = []
            evidence = {}
        
        # Step 3: Return structured answer
        return Answer(
            text=answer_text,
            q_type=q_type,
            confidence=0.8,
            clips=clips,
            highlighted_entities=[e for e in entities if e in [ent.entity_id for ent in self.semantic_result.entities]],
            evidence=evidence,
            reasoning=f"Routed to {q_type} query path. Found evidence: {len(evidence)} items.",
        )
    
    def _answer_when(self, question: str, entities: List[str], focus_entity: str) -> Tuple[str, List, Dict]:
        """
        "When did X happen?" → temporal query
        """
        
        # Find entity
        matching_entities = [e for e in self.semantic_result.entities if focus_entity in e.entity_id.lower() or focus_entity in e.class_label.lower()]
        
        if not matching_entities:
            return f"Entity '{focus_entity}' not found in video", [], {}
        
        entity = matching_entities[0]
        
        # Get timeline
        answer_text = f"Entity '{entity.class_label}' (ID: {entity.entity_id}):\n"
        answer_text += f"  First seen: {entity.first_seen:.1f}s\n"
        answer_text += f"  Last seen: {entity.last_seen:.1f}s\n"
        answer_text += f"  Duration in view: {entity.last_seen - entity.first_seen:.1f}s\n"
        
        # Hand interactions
        if entity.hand_interactions:
            answer_text += f"  Hand interactions: {len(entity.hand_interactions)} times\n"
            for i, interaction in enumerate(entity.hand_interactions[:3]):  # top 3
                answer_text += f"    {i+1}. {interaction['type']} at {interaction['timestamp']:.1f}s\n"
        
        clips = [(entity.first_seen, entity.last_seen)]
        evidence = {'entity_id': entity.entity_id, 'events': entity.hand_interactions}
        
        return answer_text, clips, evidence
    
    def _answer_where(self, question: str, entities: List[str], focus_entity: str) -> Tuple[str, List, Dict]:
        """
        "Where is X?" → spatial query
        """
        
        matching_entities = [e for e in self.semantic_result.entities if focus_entity in e.entity_id.lower() or focus_entity in e.class_label.lower()]
        
        if not matching_entities:
            return f"Entity '{focus_entity}' not found", [], {}
        
        entity = matching_entities[0]
        
        # Spatial zones
        answer_text = f"Object '{entity.class_label}' was located in:\n"
        for zone in self.semantic_result.zones:
            if entity.entity_id in zone.entity_ids:
                answer_text += f"  - {zone.label} (centroid: {zone.centroid})\n"
        
        clips = [(entity.first_seen, entity.last_seen)]
        evidence = {'entity_id': entity.entity_id, 'zones': [z.label for z in self.semantic_result.zones if entity.entity_id in z.entity_ids]}
        
        return answer_text, clips, evidence
    
    def _answer_what(self, question: str, entities: List[str], focus_entity: str) -> Tuple[str, List, Dict]:
        """
        "What is X?" → semantic query
        """
        
        matching_entities = [e for e in self.semantic_result.entities if focus_entity in e.entity_id.lower() or focus_entity in e.class_label.lower()]
        
        if not matching_entities:
            return f"Entity '{focus_entity}' not found", [], {}
        
        entity = matching_entities[0]
        
        answer_text = f"Entity: {entity.class_label}\n"
        if entity.descriptions:
            answer_text += f"Descriptions:\n"
            for desc in entity.descriptions[:3]:  # first 3 descriptions
                answer_text += f"  - {desc}\n"
        
        clips = [(entity.first_seen, entity.last_seen)]
        evidence = {'entity_id': entity.entity_id, 'class': entity.class_label, 'descriptions': entity.descriptions}
        
        return answer_text, clips, evidence
    
    def _answer_sequence(self, question: str, entities: List[str]) -> Tuple[str, List, Dict]:
        """
        "Did X happen before Y?" → temporal ordering
        """
        
        if len(entities) < 2:
            return "Please mention two entities to compare order", [], {}
        
        entity_ids = []
        for ent_str in entities:
            matches = [e for e in self.semantic_result.entities if ent_str in e.entity_id.lower() or ent_str in e.class_label.lower()]
            if matches:
                entity_ids.append(matches[0].entity_id)
        
        if len(entity_ids) < 2:
            return "Could not find both entities", [], {}
        
        entity_a = [e for e in self.semantic_result.entities if e.entity_id == entity_ids[0]][0]
        entity_b = [e for e in self.semantic_result.entities if e.entity_id == entity_ids[1]][0]
        
        if entity_a.first_seen < entity_b.first_seen:
            answer_text = f"YES: {entity_a.class_label} appeared before {entity_b.class_label}\n"
            answer_text += f"  {entity_a.class_label}: {entity_a.first_seen:.1f}s\n"
            answer_text += f"  {entity_b.class_label}: {entity_b.first_seen:.1f}s\n"
            order_str = "before"
        else:
            answer_text = f"NO: {entity_b.class_label} appeared before {entity_a.class_label}\n"
            answer_text += f"  {entity_b.class_label}: {entity_b.first_seen:.1f}s\n"
            answer_text += f"  {entity_a.class_label}: {entity_a.first_seen:.1f}s\n"
            order_str = "after"
        
        clips = [(min(entity_a.first_seen, entity_b.first_seen), max(entity_a.last_seen, entity_b.last_seen))]
        evidence = {'order': order_str, 'entity_ids': entity_ids}
        
        return answer_text, clips, evidence
    
    def _answer_causality(self, question: str, entities: List[str]) -> Tuple[str, List, Dict]:
        """
        "Did X cause Y?" → causal link query
        """
        
        if len(entities) < 2:
            return "Please mention cause and effect entities", [], {}
        
        entity_ids = []
        for ent_str in entities:
            matches = [e for e in self.semantic_result.entities if ent_str in e.entity_id.lower() or ent_str in e.class_label.lower()]
            if matches:
                entity_ids.append(matches[0].entity_id)
        
        if len(entity_ids) < 2:
            return "Could not find both entities", [], {}
        
        cause_id, effect_id = entity_ids[0], entity_ids[1]
        
        # Look for causal links
        matching_links = [l for l in self.semantic_result.causal_links if l.agent_id == cause_id and l.patient_id == effect_id]
        
        if matching_links:
            link = matching_links[0]
            answer_text = f"YES, likely causal link detected:\n"
            answer_text += f"  Agent: {cause_id}\n"
            answer_text += f"  Patient: {effect_id}\n"
            answer_text += f"  CIS Score: {link.cis_score:.3f}\n"
            answer_text += f"  Components:\n"
            for comp_name, comp_score in link.components.items():
                answer_text += f"    {comp_name}: {comp_score:.3f}\n"
        else:
            answer_text = f"NO causal link found between {cause_id} and {effect_id}"
        
        clips = []
        evidence = {'causal_links': [l.to_dict() for l in matching_links]}
        
        return answer_text, clips, evidence

@dataclass
class Answer:
    text: str
    q_type: str
    confidence: float
    clips: List[Tuple[float, float]]  # (start_s, end_s)
    highlighted_entities: List[str]
    evidence: Dict
    reasoning: str = ""
```

### 3. Video Clip Extraction & Visualization

```python
class VideoClipExtractor:
    """
    Extract and encode video clips for web viewing
    """
    
    def __init__(self, video_path: str, output_dir: Path):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_clips(self, clips: List[Tuple[float, float]]) -> List[str]:
        """
        Extract clips and return paths
        
        Args:
            clips: list of (start_s, end_s)
        
        Returns:
            list of output video paths
        """
        
        output_paths = []
        
        for i, (start_s, end_s) in enumerate(clips):
            output_path = self.output_dir / f"clip_{i:03d}.mp4"
            
            # Use ffmpeg to extract segment
            import subprocess
            cmd = [
                'ffmpeg', '-i', str(self.video_path),
                '-ss', str(start_s),
                '-to', str(end_s),
                '-c', 'copy',  # fast copy, no re-encoding
                '-y',  # overwrite
                str(output_path),
            ]
            
            subprocess.run(cmd, capture_output=True)
            output_paths.append(str(output_path))
        
        return output_paths
    
    def extract_frame_overlays(
        self,
        clips: List[Tuple[float, float]],
        entities: Dict[str, SemanticEntity],
        entity_colors: Dict[str, Tuple[int, int, int]],
    ) -> List[str]:
        """
        Generate PNG frames with entity bounding boxes
        """
        
        overlay_paths = []
        
        for clip_idx, (start_s, end_s) in enumerate(clips):
            clip_dir = self.output_dir / f"clip_{clip_idx:03d}_frames"
            clip_dir.mkdir(exist_ok=True)
            
            # Extract frames
            import subprocess
            frame_pattern = str(clip_dir / "frame_%06d.png")
            cmd = [
                'ffmpeg', '-i', str(self.video_path),
                '-ss', str(start_s),
                '-to', str(end_s),
                '-vf', 'fps=2',  # 2 FPS for preview
                '-y',
                frame_pattern,
            ]
            
            subprocess.run(cmd, capture_output=True)
            
            overlay_paths.append(str(clip_dir))
        
        return overlay_paths

class HTMLViewer:
    """
    Generate interactive HTML viewer for analysis results
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
    
    def generate(
        self,
        video_path: str,
        semantic_result: SemanticResult,
        qa_pairs: List[Dict],  # [{'q': str, 'a': Answer}, ...]
    ) -> str:
        """
        Generate HTML viewer
        
        Returns:
            path to generated HTML file
        """
        
        html_path = self.output_dir / "viewer.html"
        
        html_content = self._build_html(video_path, semantic_result, qa_pairs)
        
        html_path.write_text(html_content)
        
        return str(html_path)
    
    def _build_html(self, video_path: str, semantic_result: SemanticResult, qa_pairs: List[Dict]) -> str:
        """
        Build HTML content
        """
        
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Orion Video Analysis Viewer</title>
    <style>
        body { font-family: Arial; background: #f5f5f5; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .video-section { margin-bottom: 20px; }
        video { width: 100%; max-width: 800px; border: 1px solid #ddd; border-radius: 4px; }
        .timeline { background: #eee; height: 40px; margin: 10px 0; position: relative; border-radius: 4px; }
        .timeline-event { position: absolute; height: 100%; background: #2196F3; opacity: 0.7; cursor: pointer; }
        .timeline-event:hover { opacity: 1; }
        .qa-section { margin: 20px 0; }
        .qa-item { background: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #2196F3; border-radius: 4px; }
        .question { font-weight: bold; color: #333; }
        .answer { color: #666; margin-top: 10px; }
        .entity-legend { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; margin: 20px 0; }
        .entity-item { padding: 10px; background: #f0f0f0; border-radius: 4px; border-left: 4px solid #333; }
        .entity-color { display: inline-block; width: 12px; height: 12px; margin-right: 8px; border-radius: 2px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Orion: Egocentric Video Understanding</h1>
        
        <div class="video-section">
            <h2>Video Analysis</h2>
            <video id="videoPlayer" controls>
                <source src="{video_path}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            
            <div class="timeline" id="timeline">
                <!-- Events will be placed here -->
            </div>
        </div>
        
        <div>
            <h2>Entity Legend</h2>
            <div class="entity-legend" id="entityLegend">
                <!-- Entities will be listed here -->
            </div>
        </div>
        
        <div class="qa-section">
            <h2>Q&A Analysis</h2>
            
            <div style="margin: 20px 0;">
                <input type="text" id="questionInput" placeholder="Ask a question about the video..." style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px;">
                <button onclick="askQuestion()" style="margin-top: 10px; padding: 10px 20px; background: #2196F3; color: white; border: none; border-radius: 4px; cursor: pointer;">Ask</button>
            </div>
            
            <div id="qaResults">
"""
        
        # Add pre-computed Q&A pairs
        for i, qa in enumerate(qa_pairs):
            html += f"""
                <div class="qa-item">
                    <div class="question">Q: {qa['q']}</div>
                    <div class="answer">A: {qa['a'].text}</div>
                    <div style="font-size: 12px; color: #999; margin-top: 10px;">
                        Confidence: {qa['a'].confidence:.2f} | Clips: {len(qa['a'].clips)}
                    </div>
                </div>
"""
        
        html += """
            </div>
        </div>
    </div>
    
    <script>
        function askQuestion() {
            const question = document.getElementById('questionInput').value;
            alert('Question asked: ' + question);
            // In real implementation, this would call backend API
        }
    </script>
</body>
</html>
"""
        
        return html
```

---

## Testing & Validation

### QA Quality
- **Accuracy**: Ground truth Q&A pairs (if available)
- **Latency**: Time per question (<1s target)
- **Coverage**: Answer rate (fraction of questions with non-null answers)

### Visualization
- **Correctness**: Overlays align with video frames
- **Usability**: Timeline interactive, players work, clips clickable

---

## Performance Targets

| Task | Latency | Target |
|------|---------|--------|
| Question classification | 50ms | LLM or regex |
| Index query | 100ms | In-memory lookup |
| LLM answer generation | 300ms | if using LLM |
| Clip extraction | 5s per clip | ffmpeg |
| **Total QA latency** | <500ms | End-to-end |

---

## Output Artifacts

- `answers.json`: all Q&A pairs with evidence
- `interactive_viewer.html`: web interface
- `clips/`: extracted video segments
- `qa_metrics.json`: latency, success rate, confidence distribution

---

## Configuration

```yaml
qa:
  use_llm: true
  lm_model: "flan-ul2"  # small, local
  
  question_types_enabled:
    - WHEN
    - WHERE
    - WHAT
    - SEQUENCE
    - CAUSALITY
  
  clip_extraction:
    codec: "libx264"
    bitrate: "2M"
    fps: 30
  
  html_viewer:
    theme: "light"
    enable_timeline: true
    enable_overlays: true
```

---

## CLI Integration

```bash
# Batch processing with QA
python -m orion.cli analyze-with-qa \
  --video video.mp4 \
  --config config.yaml \
  --output-dir results/phase4 \
  --questions \
    "What did I hold?" \
    "When did I pick up the phone?" \
    "Did the cup spill?" \
    "Was the door open when I entered the kitchen?"

# Launch viewer
open results/phase4/interactive_viewer.html
```

---

## Next Phase (Phase 5)

- Benchmarking on EASG, Ego4D, ActionGenome
- Ablation studies
- Scientific validation
- Historian engine roadmap

