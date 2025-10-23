# Heuristic Baseline Quick Start Guide

## What is the Heuristic Baseline?

A sophisticated **rules-based alternative** to Orion that serves as a rigorous baseline for evaluating Orion's AI-driven approach on the Action Genome (AG) dataset.

- Uses **only** YOLO detection + hand-crafted geometric/temporal heuristics
- No embeddings, no LLMs, no causal inference system
- Demonstrates the **value-add** of Orion's semantic reasoning

## Quick Start (5 minutes)

### 1. Ensure AG data is prepared
```bash
python scripts/1_prepare_ag_data.py
```

### 2. Run the heuristic baseline
```bash
python scripts/3b_run_heuristic_baseline_ag_eval.py
```
- Processes 50 AG clips
- Outputs: `data/ag_50/results/heuristic_predictions.json`
- Includes: entities, relationships, events, causal links

### 3. Run Orion pipeline (if not already done)
```bash
python scripts/3_run_orion_ag_eval.py
```
- Outputs: `data/ag_50/results/predictions.json`

### 4. Compare the two approaches
```bash
python scripts/4b_compare_baseline_vs_orion.py
```
- Generates: `data/ag_50/results/baseline_vs_orion_comparison.json`
- Prints comparative metrics (precision, recall, F1, etc.)

## What It Does

### Relationship Detection (5 types of heuristics)
```
1. IS_NEAR       → Objects within 80px for 5+ consecutive frames
2. IS_TOUCHING   → Objects within 50px 
3. IS_INSIDE     → 70%+ spatial containment
4. MOVES_WITH    → Similar motion vectors (angular similarity > 0.6)
5. SPATIAL       → ABOVE, BELOW, LEFT_OF, RIGHT_OF based on centroid positions
```

### Event Detection (3 types)
```
1. MOTION_DETECTED  → Objects moving >15px per frame
2. INTERACTION      → 2+ relationship types between object pair
3. ENTRY/EXIT       → Objects appearing/disappearing at frame boundaries
```

### Causal Reasoning (2 rules)
```
1. Temporal        → Event A causing B if B starts <1s after A ends
2. Spatial         → Event A causing B if both occur <150px apart, <1s apart
```

## Key Performance Characteristics

### Strong Areas (baseline handles well)
- ✓ Simple proximity relationships
- ✓ Containment detection
- ✓ Motion-based events
- ✓ Frame entry/exit

### Weak Areas (where Orion excels)
- ✗ Semantic action understanding (HOLDING, USING, WEARING)
- ✗ Implicit interactions (non-contact relationships)
- ✗ Complex causal chains
- ✗ Context-dependent reasoning

## Expected Results

On Action Genome:

| Metric | Baseline | Orion | Improvement |
|--------|----------|-------|-------------|
| Edge F1 | 0.30 | 0.60 | **2.0x** |
| Event F1 | 0.20 | 0.55 | **2.7x** |
| Causal F1 | 0.15 | 0.50 | **3.3x** |
| Entity Jaccard | 0.70 | 0.85 | +0.15 |

## Output Format

```json
{
  "video_id": {
    "entities": {
      "entity_0": {
        "entity_id": "entity_0",
        "class": "person",
        "frames": [0, 1, 2, ...],
        "first_frame": 0,
        "last_frame": 150,
        "bboxes": {"0": [x1, y1, x2, y2], ...}
      }
    },
    "relationships": [
      {
        "subject": "entity_0",
        "object": "entity_1",
        "predicate": "IS_NEAR",
        "frame_id": 10,
        "confidence": 0.75,
        "evidence": "temporal_proximity"
      }
    ],
    "events": [
      {
        "event_id": "motion_entity_0_0",
        "type": "MOTION_DETECTED",
        "start_frame": 10,
        "end_frame": 12,
        "agents": ["entity_0"],
        "confidence": 0.85
      }
    ],
    "causal_links": [
      {
        "cause": "motion_entity_0_0",
        "effect": "interaction_entity_0_entity_1",
        "time_diff": 0.5,
        "confidence": 0.55,
        "method": "temporal_proximity"
      }
    ]
  }
}
```

## Customizing the Baseline

Edit thresholds in `ComplexRulesBasedBaseline.__init__()`:

```python
def __init__(self):
    self.proximity_distance_threshold = 80.0      # Adjust for IS_NEAR
    self.contact_distance_threshold = 50.0        # Adjust for IS_TOUCHING
    self.containment_overlap_threshold = 0.7      # Adjust for IS_INSIDE
    self.motion_threshold = 15.0                  # Motion sensitivity
    self.causal_temporal_window = 1.0             # Causal reasoning window
```

## Troubleshooting

### Baseline runs too slowly
- Reduce number of clips or frames processed
- Use `max_frames=500` parameter in `process_video_frames()`

### Too many/few relationships detected
- Increase/decrease proximity thresholds
- Adjust `proximity_duration_threshold` for temporal consistency

### Causal links not being generated
- Check `causal_temporal_window` (increase from 1.0 to 2.0)
- Check spatial proximity threshold (reduce from 150 to 100 for stricter)

## Understanding the Comparison Results

When you run `4b_compare_baseline_vs_orion.py`, you'll see:

```
RELATIONSHIP DETECTION:
  Orion       - F1: 0.6200
  Heuristic   - F1: 0.3100
  Improvement - +0.3100 (100.0%)
```

This means Orion achieves **2x the F1 score** on relationship detection, demonstrating the value of semantic reasoning.

## Files Generated

```
data/ag_50/results/
├── heuristic_predictions.json          # Baseline outputs
├── predictions.json                    # Orion outputs
├── metrics.json                        # Orion evaluation (from script 4)
└── baseline_vs_orion_comparison.json   # Side-by-side comparison
```

## Next Steps

1. **Analyze the comparison** - Which metric gap is largest?
2. **Review failure cases** - Look at clips where baseline struggles most
3. **Tune thresholds** - Try different configurations to improve baseline
4. **Extend baseline** - Add domain-specific rules for better performance
5. **Report results** - Use comparison data for research publication

## References

- [HEURISTIC_BASELINE_README.md](HEURISTIC_BASELINE_README.md) - Full technical details
- [TECHNICAL_ARCHITECTURE.md](docs/TECHNICAL_ARCHITECTURE.md) - Orion architecture
- [Action Genome Dataset](https://www.actiongenome.org/)

---

**Quick Commands Summary**:
```bash
# Prepare data
python scripts/1_prepare_ag_data.py

# Run baseline
python scripts/3b_run_heuristic_baseline_ag_eval.py

# Run Orion
python scripts/3_run_orion_ag_eval.py

# Compare
python scripts/4b_compare_baseline_vs_orion.py
```

Good luck! 🚀
