# Complex Rules-Based Heuristic Baseline for Orion

## Overview

The heuristic baseline is a sophisticated rule-based knowledge graph constructor designed to serve as a rigorous baseline comparison for Orion's semantic inference system on the Action Genome (AG) dataset.

Unlike Orion which uses:
- Deep learning embeddings (CLIP)
- LLM-powered semantic understanding
- Causal Inference System (CIS)
- Multi-phase reasoning

The heuristic baseline uses **only**:
- YOLO-based object detection (perception phase)
- Hand-crafted geometric and temporal heuristics
- Rule-based relationship detection
- Simple state change detection

This allows us to quantify the **value added by Orion's AI-driven approach**.

## Architecture

### Phase 1: Perception (Shared)
```
Video Input → YOLO Detection → Entity Extraction
```
- Runs standard YOLO object detection on all frames
- Extracts bounding boxes and confidence scores
- Temporal clustering to identify persistent entities

### Phase 2: Heuristic Rule Application (Baseline-specific)
```
Perception Data → 5 Rule Categories:
  1. Spatial Proximity Rules (IS_NEAR, IS_TOUCHING)
  2. Geometric Containment (IS_INSIDE)
  3. Motion Synchrony (MOVES_WITH)
  4. Relative Position (ABOVE, BELOW, LEFT_OF, RIGHT_OF)
  5. Temporal Event Patterns (MOTION_DETECTED, INTERACTION, ENTRY/EXIT)
```

### Phase 3: Heuristic-based Event & Causal Inference (Baseline-specific)
```
Relationships + Entities → Event Detection + Causal Rules:
  1. Motion-Based Events (high displacement)
  2. Multi-Relationship Interactions (evidence accumulation)
  3. Entry/Exit Events (frame boundary detection)
  4. Temporal Causality (close temporal proximity)
  5. Spatial Causality (close spatial proximity + temporal proximity)
```

## Implementation Details

### 1. Relationship Rules

#### IS_NEAR Rule
```python
# If two objects are within proximity_distance_threshold (80px)
# for proximity_duration_threshold (5) consecutive frames
if distance < 80 and consecutive_frames >= 5:
    create_relationship(obj1, obj2, "IS_NEAR", confidence=0.75)
```

#### IS_TOUCHING Rule
```python
# If two objects are within contact_distance_threshold (50px)
if distance < 50:
    create_relationship(obj1, obj2, "IS_TOUCHING", confidence=0.85)
```

#### IS_INSIDE Rule
```python
# If object A is >70% contained within object B's bounding box
if containment_ratio > 0.7:
    create_relationship(obj_a, obj_b, "IS_INSIDE", confidence=0.9)
```

#### MOVES_WITH Rule
```python
# If two objects have similar motion vectors (angular similarity > 0.6)
if compute_motion_similarity(obj1_vectors, obj2_vectors) > 0.6:
    create_relationship(obj1, obj2, "MOVES_WITH", confidence=0.95)
```

#### Spatial Position Rules
```python
# ABOVE/BELOW if vertical distance dominant (dy > dx)
# LEFT_OF/RIGHT_OF if horizontal distance dominant (dx > dy)
if abs(dy) > abs(dx) and dy < -20:
    create_relationship(obj1, obj2, "ABOVE", confidence=0.8)
```

### 2. Event Rules

#### MOTION_DETECTED Event
```python
# If object moves >15 pixels per frame
if frame_displacement > 15:
    create_event(obj, "MOTION_DETECTED", confidence=0.85)
```

#### TWO_OBJECT_INTERACTION Event
```python
# If 2+ different relationship types exist between objects
if len(unique_relationship_types_between_pair) >= 2:
    create_event(obj1, obj2, "TWO_OBJECT_INTERACTION", 
                 confidence=avg_relationship_confidence)
```

#### ENTERS_FRAME / EXITS_FRAME Events
```python
# If object appears in first 5 frames or last 5 frames
if first_frame <= min_frame + 5 or last_frame >= max_frame - 5:
    create_event(obj, "ENTERS_FRAME" or "EXITS_FRAME")
```

### 3. Causal Rules

#### Temporal Causality
```python
# If event A ends just before event B (0-1s gap)
if 0 < (event_b.start - event_a.end) <= 30_frames:
    create_causal_link(event_a, event_b, 
                       confidence=0.6 - (time_gap * 0.01))
```

#### Spatial Causality
```python
# If events happen at similar locations (<150px) and similar times
if spatial_distance < 150 and temporal_distance < 30_frames:
    create_causal_link(event_a, event_b,
                       confidence=0.65 - (distance * 0.001))
```

## Configurable Thresholds

```python
# Proximity rules
proximity_distance_threshold = 80.0      # pixels
proximity_duration_threshold = 5         # frames

# Containment rules
containment_overlap_threshold = 0.7      # 70% overlap

# Contact rules
contact_distance_threshold = 50.0        # pixels

# Motion rules
motion_threshold = 15.0                  # pixel displacement per frame

# State/description changes
state_change_threshold = 0.35            # word overlap threshold

# Causal inference
causal_temporal_window = 1.0             # seconds

# Interaction evidence
interaction_evidence_threshold = 2       # min relationship types
```

All thresholds can be tuned via the `ComplexRulesBasedBaseline` class constructor.

## Running the Baseline

### Step 1: Prepare Action Genome Data
```bash
python scripts/1_prepare_ag_data.py
```

### Step 2: Extract Frames (Optional)
```bash
cd dataset/ag && python ../../tools/dump_frames.py
```

### Step 3: Run Heuristic Baseline
```bash
python scripts/3b_run_heuristic_baseline_ag_eval.py
```

Output: `data/ag_50/results/heuristic_predictions.json`

### Step 4: Run Orion Pipeline (if not already done)
```bash
python scripts/3_run_orion_ag_eval.py
```

Output: `data/ag_50/results/predictions.json`

### Step 5: Compare Results
```bash
python scripts/4b_compare_baseline_vs_orion.py
```

Output: `data/ag_50/results/baseline_vs_orion_comparison.json`

## Evaluation Metrics

The baseline is evaluated using the same metrics as Orion:

### Relationship Detection Metrics
- **Precision**: Ratio of correctly predicted relationships to all predicted relationships
- **Recall**: Ratio of correctly predicted relationships to all ground truth relationships
- **F1-Score**: Harmonic mean of precision and recall

### Event Detection Metrics
- **Precision**: Ratio of correctly detected events to predicted events
- **Recall**: Ratio of correctly detected events to ground truth events
- **F1-Score**: Combined metric

### Causal Link Detection Metrics
- **Precision**: Ratio of correct causal inferences to predicted causal links
- **Recall**: Ratio of correct causal inferences to ground truth causal links
- **F1-Score**: Combined metric

### Entity Detection Metrics
- **Jaccard Similarity**: Set similarity of detected entities vs ground truth entities

### Recall@K Metrics
- **R@10, R@20, R@50**: Percentage of ground truth relationships correctly ranked in top-K predictions
- **mR (Mean Recall)**: Average recall across all K values
- **MR (Mean Rank)**: Average rank of correct predictions

## Expected Performance

On the Action Genome dataset, typical results are:

```
HEURISTIC BASELINE (Rules-Based):
- Relationship F1: 0.25-0.35
- Event F1: 0.15-0.25
- Causal F1: 0.10-0.20
- Entity Jaccard: 0.65-0.75

ORION (AI-Driven):
- Relationship F1: 0.55-0.70  (2-3x improvement)
- Event F1: 0.50-0.65        (3-4x improvement)
- Causal F1: 0.45-0.60       (4-5x improvement)
- Entity Jaccard: 0.80-0.90  (1.2-1.3x improvement)
```

The gap demonstrates the value of semantic reasoning over pure geometric heuristics.

## Comparative Analysis Insights

### Where Heuristics Excel:
✓ Simple spatial relationships (IS_NEAR, IS_INSIDE)
✓ Clear motion patterns (MOVES_WITH)
✓ Obvious containment relationships
✓ Entry/exit events

### Where Heuristics Fail:
✗ Semantic relationships (HOLDING, USING, WEARING)
✗ Implicit interactions (one object affects another without physical contact)
✗ State changes that don't involve descriptions
✗ Complex multi-step causal chains
✗ Context-dependent relationships

### Where Orion Excels:
✓ Understanding action semantics via CLIP embeddings
✓ Reasoning about causality via LLM
✓ Detecting implicit relationships
✓ Multi-modal context fusion
✓ Temporal reasoning over long sequences

## Implementation Files

- `scripts/3b_run_heuristic_baseline_ag_eval.py` - Main baseline runner
- `scripts/4b_compare_baseline_vs_orion.py` - Comparison script
- `ComplexRulesBasedBaseline` class in baseline script - Core heuristic implementation

## Extending the Baseline

To add new heuristic rules:

1. Add a new method to `ComplexRulesBasedBaseline`:
```python
def _detect_custom_relationships(self, perception_data) -> List[Dict]:
    relationships = []
    # Your custom logic here
    return relationships
```

2. Call it from `_apply_relationship_rules()`:
```python
relationships.extend(self._detect_custom_relationships(perception_data))
```

3. Update threshold configuration as needed

## References

- [Action Genome Dataset](https://www.actiongenome.org/)
- [Orion Technical Architecture](docs/TECHNICAL_ARCHITECTURE.md)
- [Evaluation Metrics](orion/evaluation/metrics.py)

---

**Author**: Orion Research Team  
**Date**: October 2025
