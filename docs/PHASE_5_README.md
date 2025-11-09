# PHASE 5: Benchmarking, Evaluation, Scientific Validation & Historian Engine Roadmap

**Objective**: Rigorously evaluate system on public benchmarks (Ego4D, EASG, ActionGenome). Run ablation studies. Establish scientific novelty. Articulate historian model vision and roadmap.

**Timeline**: Week 4+ (ongoing research validation)

**Success Criteria**:
- ✅ Evaluated on Ego4D egocentric dataset with reported metrics
- ✅ Evaluated on ActionGenome with causal link precision/recall
- ✅ Ablation studies (2D vs 3D CIS, with/without hands)
- ✅ Results paper-ready (tables, figures, error analysis)
- ✅ Historian engine vision articulated with concrete roadmap
- ✅ WACV demo narrative refined with quantitative evidence

---

## Phase 5A: Benchmarking & Evaluation

### 1. Datasets & Adapters

```python
class Ego4DAdapter:
    """
    Adapt Ego4D dataset for Orion evaluation
    """
    
    def __init__(self, ego4d_root: Path):
        self.ego4d_root = ego4d_root
        self.split = "val"  # or "test"
    
    def load_video_clip(self, clip_id: str) -> Dict:
        """
        Load Ego4D video clip with GT annotations
        
        Returns:
            {
                'video_path': str,
                'frame_count': int,
                'fps': int,
                'objects': [{id, class, bbox_3d, trajectory}, ...],
                'interactions': [{hand_id, object_id, type, timestamps}, ...],
                'scenes': [{label, start_frame, end_frame}, ...],
            }
        """
        video_path = self.ego4d_root / f"{clip_id}" / "video.mp4"
        annotations_path = self.ego4d_root / f"{clip_id}" / "annotations.json"
        
        import json
        with open(annotations_path) as f:
            annotations = json.load(f)
        
        return {
            'video_path': str(video_path),
            'frame_count': annotations['frame_count'],
            'fps': annotations.get('fps', 30),
            'objects': annotations.get('objects', []),
            'interactions': annotations.get('hand_interactions', []),
            'scenes': annotations.get('scenes', []),
        }

class ActionGenomeAdapter:
    """
    Adapt Action Genome dataset for causal evaluation
    """
    
    def __init__(self, ag_root: Path):
        self.ag_root = ag_root
    
    def load_video(self, video_id: str) -> Dict:
        """
        Load AG video with GT causal relationships
        
        Returns:
            {
                'video_path': str,
                'objects': [{id, class, frames, locations}, ...],
                'actions': [{id, class, frames, agents, patients}, ...],
                'causal_links': [{cause_action, effect_action, confidence}, ...],
            }
        """
        # Load from AG format
        pass
```

### 2. Evaluation Metrics

```python
@dataclass
class EvaluationMetrics:
    """
    Comprehensive evaluation metrics
    """
    
    # === Detection Metrics ===
    detection_ap50: float  # Average Precision @ IoU 0.5
    detection_ap75: float  # AP @ IoU 0.75
    detection_map: float  # Mean AP across IoU thresholds
    detection_recall: float
    detection_precision: float
    
    # === Tracking Metrics (MOT challenge) ===
    mota: float  # Multiple Object Tracking Accuracy
    motp: float  # Multiple Object Tracking Precision
    id_switches: int
    fragmentations: int
    mostly_tracked: int
    mostly_lost: int
    
    # === Interaction Detection ===
    interaction_ap: float
    interaction_recall: float
    interaction_precision: float
    interaction_f1: float
    
    # === Hand-Object Interaction ===
    hand_object_interaction_f1: float  # GRASP, TOUCH, NEAR accuracy
    hand_object_interaction_confusion: np.ndarray  # confusion matrix
    
    # === Causal Reasoning ===
    causal_precision: float  # fraction of predicted links that are correct
    causal_recall: float  # fraction of GT links that are predicted
    causal_f1: float
    causal_auc_roc: float  # ROC curve under CIS threshold sweep
    
    # === Spatial Reasoning ===
    spatial_relation_accuracy: float  # NEAR, INSIDE, SUPPORTING accuracy
    spatial_zone_iou: float  # Intersection over Union of detected zones
    
    # === Latency & Efficiency ===
    avg_latency_ms_per_frame: float
    throughput_fps: float
    peak_memory_mb: float
    model_size_mb: float
    
    # === Ablations ===
    cis_2d_vs_3d_improvement: float  # (CIS_3D_F1 - CIS_2D_F1) / CIS_2D_F1
    hand_signal_improvement: float  # gain from hand interaction signals
    occlusion_model_improvement: float  # gain from occlusion detection
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

def compute_cis_metrics(predicted_cis_matrix: np.ndarray, gt_causal_links: Set[Tuple[str, str]], threshold: float = 0.5) -> Tuple[float, float, float]:
    """
    Compute precision, recall, F1 for causal link prediction
    
    Args:
        predicted_cis_matrix: (n_entities, n_entities) CIS scores
        gt_causal_links: set of (agent_id, patient_id) ground truth links
        threshold: CIS threshold for link prediction
    
    Returns:
        (precision, recall, f1)
    """
    
    # Predict links where CIS > threshold
    predicted_links = set()
    n = predicted_cis_matrix.shape[0]
    entity_ids = list(range(n))
    
    for i in range(n):
        for j in range(n):
            if i != j and predicted_cis_matrix[i, j] > threshold:
                predicted_links.add((entity_ids[i], entity_ids[j]))
    
    # Compute TP, FP, FN
    tp = len(predicted_links & gt_causal_links)
    fp = len(predicted_links - gt_causal_links)
    fn = len(gt_causal_links - predicted_links)
    
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    
    return precision, recall, f1

def compute_interaction_metrics(predicted_interactions: List[Dict], gt_interactions: List[Dict]) -> Dict:
    """
    Compute precision/recall for hand-object interaction detection
    
    Matching based on temporal and spatial overlap
    """
    
    # For each predicted interaction, find best GT match
    tp = 0
    fp = 0
    
    for pred in predicted_interactions:
        best_match = None
        best_iou = 0
        
        for gt in gt_interactions:
            # Temporal IoU
            pred_start, pred_end = pred['start_time'], pred['end_time']
            gt_start, gt_end = gt['start_time'], gt['end_time']
            
            intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
            union = max(pred_end, gt_end) - min(pred_start, gt_start)
            
            temporal_iou = intersection / union
            
            # Type match
            type_match = pred['type'] == gt['type']
            
            score = temporal_iou if type_match else 0
            
            if score > best_iou:
                best_iou = score
                best_match = gt
        
        if best_iou > 0.5:
            tp += 1
        else:
            fp += 1
    
    fn = len(gt_interactions) - tp
    
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    
    return {'precision': precision, 'recall': recall, 'f1': f1}
```

### 3. Evaluation Pipeline

```python
class EvaluationPipeline:
    """
    End-to-end evaluation on benchmark datasets
    """
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.pipeline = VideoPipeline.from_config(config.pipeline_config)
        self.results = []
    
    def evaluate_on_ego4d(self, split: str = "val") -> EvaluationMetrics:
        """
        Evaluate on Ego4D egocentric dataset
        """
        
        adapter = Ego4DAdapter(self.config.ego4d_root)
        clip_ids = self._get_ego4d_split_ids(split)
        
        all_metrics = []
        
        for clip_id in clip_ids:
            print(f"Processing {clip_id}...")
            
            clip_data = adapter.load_video_clip(clip_id)
            
            # Run Orion
            result = self.pipeline.process_video_with_qa(clip_data['video_path'], [])
            
            # Compare with GT
            metrics = self._compare_with_gt_ego4d(result, clip_data)
            all_metrics.append(metrics)
        
        # Aggregate
        return self._aggregate_metrics(all_metrics)
    
    def evaluate_on_action_genome(self) -> EvaluationMetrics:
        """
        Evaluate on Action Genome for causal reasoning
        """
        
        adapter = ActionGenomeAdapter(self.config.ag_root)
        video_ids = self._get_ag_video_ids()
        
        all_metrics = []
        
        for video_id in video_ids:
            print(f"Processing AG {video_id}...")
            
            video_data = adapter.load_video(video_id)
            
            # Run Orion
            result = self.pipeline.process_video_with_qa(video_data['video_path'], [])
            
            # Compute causal metrics
            metrics = self._compare_causal_with_gt(result, video_data)
            all_metrics.append(metrics)
        
        return self._aggregate_metrics(all_metrics)
    
    def run_ablation_study(self) -> Dict[str, EvaluationMetrics]:
        """
        Run ablations:
        1. CIS 2D vs 3D
        2. With/without hand signals
        3. With/without occlusion model
        """
        
        results = {}
        
        # Baseline: 2D CIS (no depth, no hand signals)
        print("\n=== Ablation 1: 2D CIS (baseline) ===")
        config_2d = self.config.copy()
        config_2d.perception.depth.enable = False
        config_2d.perception.hand_tracking.enable = False
        config_2d.semantic.cis.hand_bonus = 0.0
        
        pipeline_2d = VideoPipeline.from_config(config_2d.pipeline_config)
        results['2d_cis'] = self._eval_on_subset(pipeline_2d)
        
        # 3D CIS without hand signals
        print("\n=== Ablation 2: 3D CIS (no hand signals) ===")
        config_3d_no_hand = self.config.copy()
        config_3d_no_hand.perception.hand_tracking.enable = False
        config_3d_no_hand.semantic.cis.hand_bonus = 0.0
        
        pipeline_3d_no_hand = VideoPipeline.from_config(config_3d_no_hand.pipeline_config)
        results['3d_cis_no_hand'] = self._eval_on_subset(pipeline_3d_no_hand)
        
        # Full system (3D CIS + hand signals)
        print("\n=== Ablation 3: Full system (3D + hand) ===")
        pipeline_full = VideoPipeline.from_config(self.config.pipeline_config)
        results['full_system'] = self._eval_on_subset(pipeline_full)
        
        # No occlusion model
        print("\n=== Ablation 4: 3D + hand (no occlusion) ===")
        config_no_occ = self.config.copy()
        config_no_occ.perception.occlusion_detector.enable = False
        
        pipeline_no_occ = VideoPipeline.from_config(config_no_occ.pipeline_config)
        results['full_no_occlusion'] = self._eval_on_subset(pipeline_no_occ)
        
        return results
    
    def _compare_with_gt_ego4d(self, result: ProcessingResult, clip_data: Dict) -> EvaluationMetrics:
        """
        Compare Orion results with Ego4D ground truth
        """
        # Extract predicted entities and interactions
        predicted_objects = [{
            'id': e.entity_id,
            'class': e.class_label,
            'trajectory_3d': e.trajectories_3d,
        } for e in result.semantic_result.entities]
        
        predicted_interactions = [{
            'hand_id': i.hand_id,
            'object_id': i.object_id,
            'type': i.interaction_type,
            'start_time': min([obs.timestamp for obs in result.perception_result.raw_observations if obs.entity_id == i.object_id]),
            'end_time': max([obs.timestamp for obs in result.perception_result.raw_observations if obs.entity_id == i.object_id]),
        } for i in result.semantic_result.interaction_links]
        
        # Compare with GT
        metrics = EvaluationMetrics(
            detection_ap50=self._compute_detection_ap(predicted_objects, clip_data['objects'], iou_threshold=0.5),
            detection_ap75=self._compute_detection_ap(predicted_objects, clip_data['objects'], iou_threshold=0.75),
            # ... more metrics
            hand_object_interaction_f1=compute_interaction_metrics(predicted_interactions, clip_data['interactions'])['f1'],
        )
        
        return metrics
    
    def _compare_causal_with_gt(self, result: ProcessingResult, video_data: Dict) -> EvaluationMetrics:
        """
        Compare causal links with Action Genome ground truth
        """
        
        # Build CIS matrix
        entity_ids = [e.entity_id for e in result.semantic_result.entities]
        cis_matrix = np.zeros((len(entity_ids), len(entity_ids)))
        
        for link in result.semantic_result.causal_links:
            try:
                i = entity_ids.index(link.agent_id)
                j = entity_ids.index(link.patient_id)
                cis_matrix[i, j] = link.cis_score
            except ValueError:
                pass
        
        # GT causal links
        gt_links = set()
        for causal_link in video_data.get('causal_links', []):
            gt_links.add((causal_link['cause'], causal_link['effect']))
        
        # Compute metrics across threshold range
        precisions = []
        recalls = []
        thresholds = np.linspace(0, 1, 11)
        
        for threshold in thresholds:
            p, r, f1 = compute_cis_metrics(cis_matrix, gt_links, threshold)
            precisions.append(p)
            recalls.append(r)
        
        # Compute AUC-ROC
        from sklearn.metrics import auc
        auc_roc = auc(recalls, precisions)
        
        metrics = EvaluationMetrics(
            causal_precision=precisions[5],  # @ threshold 0.5
            causal_recall=recalls[5],
            causal_f1=2 * precisions[5] * recalls[5] / max(1e-6, precisions[5] + recalls[5]),
            causal_auc_roc=auc_roc,
        )
        
        return metrics
```

### 4. Results & Error Analysis

```python
class ResultsAnalyzer:
    """
    Analyze results and generate visualizations
    """
    
    def __init__(self, metrics_list: List[EvaluationMetrics]):
        self.metrics_list = metrics_list
    
    def generate_report(self, output_dir: Path):
        """
        Generate comprehensive evaluation report
        """
        
        import json
        import matplotlib.pyplot as plt
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Summary table
        summary_table = []
        for i, metrics in enumerate(self.metrics_list):
            summary_table.append({
                'clip_id': i,
                'detection_map': metrics.detection_map,
                'causal_f1': metrics.causal_f1,
                'interaction_f1': metrics.hand_object_interaction_f1,
                'latency_ms': metrics.avg_latency_ms_per_frame,
            })
        
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary_table, f, indent=2)
        
        # Plots
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist([m.detection_map for m in self.metrics_list], bins=20)
        plt.xlabel('Detection mAP')
        plt.ylabel('Count')
        plt.title('Detection Performance Distribution')
        
        plt.subplot(1, 2, 2)
        plt.hist([m.causal_f1 for m in self.metrics_list], bins=20)
        plt.xlabel('Causal F1')
        plt.ylabel('Count')
        plt.title('Causal Reasoning Performance Distribution')
        
        plt.tight_layout()
        plt.savefig(output_dir / "metrics_distribution.png")
        
        # Error analysis
        self._analyze_failure_modes(output_dir)
    
    def _analyze_failure_modes(self, output_dir: Path):
        """
        Identify and report failure modes
        """
        
        failure_modes = {
            'low_detection': [],  # clips with mAP < 0.5
            'poor_tracking': [],  # high ID switches
            'bad_causal': [],  # low causal F1
        }
        
        for i, metrics in enumerate(self.metrics_list):
            if metrics.detection_map < 0.5:
                failure_modes['low_detection'].append(i)
            if metrics.id_switches > 10:
                failure_modes['poor_tracking'].append(i)
            if metrics.causal_f1 < 0.3:
                failure_modes['bad_causal'].append(i)
        
        import json
        with open(output_dir / "failure_modes.json", 'w') as f:
            json.dump(failure_modes, f, indent=2)
```

---

## Phase 5B: Scientific Validation & Paper-Ready Results

### Key Claims & Evidence

**Claim 1**: "Monocular depth significantly improves spatial reasoning"

Evidence:
- Ablation: CIS F1 improves 15-25% with 3D vs 2D
- Visualizations: depth heatmaps showing scene structure

**Claim 2**: "Hand interaction signals are critical for egocentric understanding"

Evidence:
- Ablation: hand interaction bonus improves interaction detection F1 by 20-30%
- Confusion matrix: with hands, GRASP/TOUCH errors reduce 40%

**Claim 3**: "Occlusion detection improves object permanence"

Evidence:
- Ablation: ID switches reduce 30-40% with occlusion model
- Trajectory smoothness metrics improve

**Claim 4**: "Context-aware indexing enables efficient long-session understanding"

Evidence:
- Latency comparison: Dense vs Sparse modes
- QA response time vs session duration (linear vs quadratic)

### Paper Structure

```markdown
## Paper: "Historian Models for Egocentric Understanding: From 3D Perception to Causal Scene Graphs"

### Abstract
We propose Orion, a semantic uplift pipeline that transforms egocentric video into causally-aware knowledge graphs. By grounding perception in 3D space (monocular depth), incorporating hand tracking and interaction modeling, and employing context-aware indexing, we achieve accurate spatial-temporal-causal reasoning on long videos. Evaluated on Ego4D and ActionGenome, our system achieves X% causal link F1 and Y% hand-object interaction accuracy while maintaining real-time efficiency.

### Contributions
1. First integrated approach combining 3D perception + hand tracking + Bayesian re-identification for egocentric understanding
2. Novel CIS (Causal Influence Score) formula with 3D spatial/motion signals and hand interaction detection
3. Context-aware dual-index architecture (Dense for known, Sparse for unknown environments)
4. Comprehensive evaluation on public egocentric benchmarks with ablation studies

### Related Work
- Object tracking (SORT, DeepSORT, ...)
- Hand-object interaction (HCI papers)
- Scene graphs (Visual Genome, ...)
- Egocentric understanding (Ego4D baseline, ...)

### Method
[Section describing Phase 1-3 pipeline]

### Experiments
[Benchmark results, ablations, error analysis]

### Results
[Tables and figures]

### Discussion
- Historian model framing: system as "historian of events"
- Future work: real-time mobile, multi-modal fusion, ...

### Conclusion

### References
```

---

## Phase 5C: Historian Engine Roadmap

### Vision: The Historian Model

**Core Idea**: An AI system that acts as a "historian" of the camera holder's actions and interactions.

Like a human historian, it should:
1. **Observe**: Collect facts (perception)
2. **Contextualize**: Understand spatial-temporal-causal relationships (semantics)
3. **Remember**: Maintain long-term memory (graphs/indices)
4. **Reason**: Make inferences about causality and intent
5. **Narrate**: Answer questions and tell stories about the past

**Key Innovations**:
- Egocentric viewpoint (hand as primary agent)
- 3D scene understanding (depth, spatial relationships)
- Causal reasoning (not just correlation)
- Long-context memory (hours of data)
- Natural language interface

### Roadmap

#### Phase 1 (Current - WACV 2026)
- ✅ Monocular depth + hand tracking
- ✅ 3D scene graphs
- ✅ Causal influence scoring
- ✅ CLI batch processing (15 min videos)
- ✅ Video QA system

#### Phase 2 (Post-WACV)
- ⬜ Real-time streaming (1-hour sessions)
- ⬜ Temporal memory compression
- ⬜ Predictive modeling (what happens next?)
- ⬜ Multi-modal fusion (add audio, IMU)

#### Phase 3 (2026-2027)
- ⬜ iOS app (local processing on phone)
- ⬜ On-device models (quantized, lightweight)
- ⬜ Cloud-device hybrid (edge compute)
- ⬜ Ray-Ban AR display integration

#### Phase 4 (2027+)
- ⬜ On-device historian (full pipeline on glasses)
- ⬜ Real-time AR annotation
- ⬜ Social sharing (stories as scene graphs)
- ⬜ Multi-camera fusion (phone + external camera)

#### Phase 5 (Far Future)
- ⬜ Lifelong learning (adapts to user habits)
- ⬜ Reasoning about intent (why did user do this?)
- ⬜ Interactive guidance (suggests actions based on history)
- ⬜ Cross-modal reasoning (integrate video + text + semantics)

### Technical Roadmap

```
Phase 1 (Now)
├─ Depth: Monocular (ZoeDepth)
├─ Hand: MediaPipe (2D + 3D via depth)
├─ Inference: Desktop GPU, batch
├─ Model size: ~2GB
└─ Latency: 100-200ms/frame

    ↓ (optimize & quantize)

Phase 2 (6-12 months)
├─ Depth: Lightweight variant + temporal smoothing
├─ Hand: Distilled hand detector or heuristic
├─ Inference: Streaming on desktop
├─ Model size: 1.5GB (quantized)
└─ Latency: 30-50ms/frame

    ↓ (port to mobile)

Phase 3 (12-24 months)
├─ Depth: Mobile-optimized (MiDaS small or custom)
├─ Hand: On-device hand detector (100ms on iPhone)
├─ Inference: iOS phone, local only
├─ Model size: <500MB
└─ Latency: 100-200ms/frame (acceptable for background)

    ↓ (integrate AR glasses)

Phase 4 (24+ months)
├─ Depth: Phone depth sensor (if available) or monocular
├─ Hand: Glasses-based hand tracking (IMU + camera)
├─ Inference: On-glasses compute + phone offload
├─ Model size: 200-300MB (on-device core)
└─ Latency: <50ms for realtime AR overlay
```

### Example: Historian Queries (Future)

```python
# Current capabilities
q1 = "What did I hold?"
# A: "You held: phone (3 times), cup (2 times), pen (1 time)"

q2 = "Did the cup spill?"
# A: "No causal evidence of spilling. Cup was handled gently."

# Phase 2: Predictive
q3 = "What did I do after picking up the phone?"
# A: "Typically, you check the time, then put it back down. 78% likelihood of similar pattern."

# Phase 3: Intent reasoning
q4 = "Why did you open the drawer?"
# A: "Based on context: kitchen at 6pm, drawer contains utensils. Likely searching for fork (confirmed by subsequent 'fork in hand')."

# Phase 4: Lifelong learning
q5 = "Summarize my day"
# A: [Narrative of activities, time spent in each zone, key interactions, anomalies]

q6 = "How often do I forget where I put my keys?"
# A: "In the last 30 days: 7 times. Average search time: 2 minutes. Recommendation: designate key hook."
```

### Implementation Strategy

```python
class HistorianEngine:
    """
    Historian model: from historian frames/clips to narrative
    """
    
    def __init__(self):
        # Perception module
        self.perception = PerceptionEngine3D()
        
        # Semantic module
        self.semantics = SemanticEngine()
        
        # Long-term memory (could be Memgraph or distributed DB)
        self.memory = LongTermMemory()
        
        # Reasoning engine
        self.reasoning = CausalReasoningEngine()
        
        # Natural language interface
        self.nlp = NLPInterface()
    
    def ingest_stream(self, frames_stream):
        """
        Ingest continuous stream (from phone camera, Ray-Bans, etc.)
        """
        for frame, timestamp in frames_stream:
            # Perception
            perception_data = self.perception.process_frame(frame)
            
            # Semantic understanding
            semantic_data = self.semantics.update(perception_data)
            
            # Store in memory
            self.memory.append(semantic_data, timestamp)
            
            # Check for interesting events
            interesting = self.reasoning.detect_interesting_events(semantic_data)
            if interesting:
                self.memory.mark_event(interesting)
    
    def query(self, question: str) -> Answer:
        """
        Answer question using memory
        """
        # Can query across hours/days of accumulated history
        answer = self.nlp.answer(question, self.memory)
        return answer
    
    def get_narrative(self, time_range: Tuple[float, float] = None) -> str:
        """
        Generate narrative story of what happened
        """
        events = self.memory.get_events(time_range)
        narrative = self.nlp.generate_story(events)
        return narrative
```

### Research Opportunities

1. **Multi-modal reasoning**: Integrate audio, IMU, physiological signals
2. **Causal discovery**: Learn causal structure automatically (not just score pre-defined links)
3. **Counterfactual reasoning**: "What if I hadn't done X?"
4. **Social scene graphs**: Understand interaction between multiple agents
5. **Temporal abstraction**: Hierarchical event representation (micro/macro events)
6. **Interpretability**: Explain historian decisions in human-understandable terms

---

## Output Artifacts (Phase 5)

- `phase5_evaluation_report.md`: Comprehensive results
- `benchmark_results.json`: Metrics across datasets
- `ablation_study.json`: Component contributions
- `failure_analysis.md`: Error modes and mitigations
- `paper_draft.md`: Conference paper draft
- `historian_engine_roadmap.md`: 5-year vision

---

## Timeline & Deliverables

| Milestone | Timeline | Deliverables |
|-----------|----------|--------------|
| Phase 1 Complete | Week 1 | Depth + hand + 3D perception |
| Phase 2 Complete | Week 2 | Tracking + object permanence |
| Phase 3 Complete | Week 3 | CIS + scene graphs |
| Phase 4 Complete | Week 4 | QA engine + visualization |
| WACV Demo Ready | End Week 4 | MVP with all phases + demo script |
| Phase 5 Eval | Week 5-6 | Benchmarks, ablations, results |
| Paper Submission | Week 7-8 | Conference paper (if targeting late deadline) |

---

## Success Metrics for WACV 2026

- **Technical Innovation**: 3D + hand + context-aware indexing (novel combination)
- **Scientific Rigor**: Ablation studies showing each component's contribution
- **Practical Impact**: Demo on real egocentric video, working QA system
- **Vision**: Historian model framing resonates with audience, opens research directions
- **Reproducibility**: Code released, datasets cited, methodology clear

---

## Conclusion

The historian model represents a shift in how we think about video understanding: not just "what is in the video" but "what happened, why, and what does it mean." This roadmap takes us from WACV 2026 demo to a vision of intelligent glasses that understand the world through egocentric perspective.

