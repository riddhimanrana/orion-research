# Quick Start: Evaluation Framework

Get started with the Orion evaluation framework in 5 minutes.

## Prerequisites

```bash
# Ensure Orion is installed
cd orion-research
pip install -e .

# Install evaluation dependencies (if not already installed)
pip install hdbscan

# Start Ollama for LLM reasoning
ollama serve
ollama pull gemma3:4b
```

## 1. Evaluate a Single Video

Compare CIS+LLM method vs. heuristic baseline on your video:

```bash
python scripts/run_evaluation.py --video path/to/your/video.mp4
```

### What happens:
1. ✅ Perception engine extracts objects with OSNet embeddings + motion tracking
2. ✅ CIS+LLM builds knowledge graph using causal inference
3. ✅ Heuristic baseline builds graph using hand-crafted rules
4. ✅ Comparator generates detailed metrics

### Outputs (in `evaluation_output/`):
- `perception_log.json` - Raw perception data with motion info
- `graph_cis_llm.json` - Our method's knowledge graph
- `graph_heuristic.json` - Baseline method's knowledge graph  
- `comparison_report.json` - Precision, Recall, F1 metrics

## 2. View Results

```bash
# Check the comparison summary
cat evaluation_output/comparison_report.json | python -m json.tool

# Example output:
{
  "individual_metrics": {
    "cis_llm": {
      "structural": {
        "num_entities": 15,
        "num_relationships": 28,
        "num_events": 12,
        "graph_density": 0.1333
      },
      "causal": {
        "num_causal_links": 8,
        "avg_cis_score": 0.723
      }
    },
    "heuristic": {
      "structural": {
        "num_entities": 15,
        "num_relationships": 42,
        "num_events": 18
      }
    }
  },
  "pairwise_comparisons": {
    "cis_llm_vs_heuristic": {
      "edges": {
        "precision": 0.8214,
        "recall": 0.7143,
        "f1": 0.7639
      },
      "causal": {
        "precision": 0.8750,
        "recall": 0.7000,
        "f1": 0.7778
      }
    }
  }
}
```

## 3. Python API

Use the evaluation components programmatically:

```python
from orion.evaluation import HeuristicBaseline, GraphComparator
from orion.evaluation.metrics import evaluate_graph_quality
import json

# Load perception log
with open('evaluation_output/perception_log.json') as f:
    perception_objects = json.load(f)

# Run heuristic baseline
baseline = HeuristicBaseline()
graph = baseline.process_perception_log(perception_objects)

# Analyze graph quality
metrics = evaluate_graph_quality(graph)
print(f"Entities: {metrics.num_entities}")
print(f"Relationships: {metrics.num_relationships}")
print(f"Events: {metrics.num_events}")

# Compare multiple graphs
comparator = GraphComparator()
comparator.load_from_json("method_a", "graph_a.json")
comparator.load_from_json("method_b", "graph_b.json")
comparator.print_summary()
```

## 4. VSGR Benchmark (Optional)

If you have the VSGR dataset:

```bash
# Prepare VSGR dataset structure:
# vsgr_dataset/
#   ├── videos/
#   │   ├── clip_001.mp4
#   │   └── clip_002.mp4
#   ├── annotations/
#   │   ├── clip_001.json
#   │   └── clip_002.json
#   └── metadata.json

# Run evaluation
python scripts/run_evaluation.py \
    --mode benchmark \
    --benchmark vsgr \
    --dataset-path /path/to/vsgr_dataset
```

Results will show aggregated metrics across all clips:
- Average Edge F1
- Average Event F1  
- Average Causal F1

## 5. Understanding the Output

### Key Metrics

**Precision**: What % of predicted links are correct?
- High precision = few false alarms
- CIS+LLM should have higher precision (filtered by CIS)

**Recall**: What % of ground truth links were found?
- High recall = captures most relationships
- Heuristic might have higher recall (more aggressive)

**F1**: Harmonic mean of precision and recall
- Balanced metric
- Good F1 (>0.7) indicates solid performance

**Causal Precision**: Quality of causal link predictions
- Most important metric for research validation
- CIS+LLM should significantly outperform baseline

### Expected Results

Based on research hypothesis:

| Metric | CIS+LLM | Heuristic | Winner |
|--------|---------|-----------|--------|
| Edge Precision | 0.80-0.85 | 0.60-0.70 | ✅ CIS+LLM |
| Edge Recall | 0.70-0.75 | 0.75-0.85 | Baseline |
| Edge F1 | 0.75-0.80 | 0.65-0.75 | ✅ CIS+LLM |
| Causal Precision | 0.85-0.90 | 0.50-0.60 | ✅ CIS+LLM |
| Semantic Richness | High | Low | ✅ CIS+LLM |

## 6. Tuning CIS Parameters

Edit `src/orion/causal_inference.py`:

```python
class CausalConfig:
    # Adjust these weights based on your domain
    proximity_weight = 0.45    # How important is spatial proximity?
    motion_weight = 0.25       # How important is directed motion?
    temporal_weight = 0.20     # How important is temporal proximity?
    embedding_weight = 0.10    # How important is visual similarity?
    
    # Adjust thresholds
    min_score = 0.55           # Lower = more permissive (higher recall)
    max_pixel_distance = 600.0 # Proximity cutoff
```

Then re-run evaluation to see impact.

## 7. Troubleshooting

### "OSNet not available"
The system automatically falls back to ResNet50. For better Re-ID:
```bash
pip install timm --upgrade
```

### "HDBSCAN not found"
```bash
pip install hdbscan
```

### "Ollama connection refused"
```bash
# Start Ollama in another terminal
ollama serve

# Pull model
ollama pull gemma3:4b
```

### "No valid Cypher queries"
- Check Ollama is running
- Verify `gemma3:4b` is available (`ollama list`)
- System will fall back to simple queries if LLM fails

## 8. Next Steps

- Read [EVALUATION.md](EVALUATION.md) for complete documentation
- Check [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for architecture details
- Experiment with CIS parameters
- Create custom benchmarks for your domain
- Compare against your own baseline methods

## Need Help?

- Open an issue on GitHub
- Check logs in `evaluation_output/`
- Enable debug logging: `export ORION_LOG_LEVEL=DEBUG`
