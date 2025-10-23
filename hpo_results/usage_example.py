"""
Example: Using CIS Optimizer with Real Data
==========================================

This example shows how to optimize CIS weights from ground truth annotations.
"""

from pathlib import Path
import json
from orion.hpo.cis_optimizer import CISOptimizer, GroundTruthCausalPair
from orion.causal_inference import CausalInferenceEngine, CausalConfig

# === Step 1: Load Ground Truth Annotations ===
with open('data/ground_truth/causal_pairs.json') as f:
    annotations = json.load(f)

ground_truth = [GroundTruthCausalPair(**ann) for ann in annotations]
print(f"Loaded {len(ground_truth)} ground truth pairs")

# === Step 2: Load Agent Candidates and State Changes ===
# These come from your perception pipeline
# agent_candidates = [...]  # List[AgentCandidate] from your video
# state_changes = [...]     # List[StateChange] detected in your video

# === Step 3: Create Optimizer ===
optimizer = CISOptimizer(
    ground_truth=ground_truth,
    agent_candidates=agent_candidates,
    state_changes=state_changes
)

# === Step 4: Run Bayesian Optimization ===
print("Running optimization...")
result = optimizer.optimize(n_trials=100)

# === Step 5: View Results ===
print(f"\nBest Weights:")
for name, value in result.best_weights.items():
    print(f"  {name:20s}: {value:.4f}")

print(f"\nBest Threshold: {result.best_threshold:.4f}")
print(f"F1 Score: {result.best_score:.4f}")
print(f"Precision: {result.precision:.4f}")
print(f"Recall: {result.recall:.4f}")

# === Step 6: Save Optimized Config ===
config_path = Path('configs/cis_optimized.json')
with open(config_path, 'w') as f:
    json.dump({
        'best_weights': result.best_weights,
        'best_threshold': result.best_threshold,
        'metrics': {
            'f1': result.best_score,
            'precision': result.precision,
            'recall': result.recall
        },
        'sensitivity_analysis': result.sensitivity_analysis
    }, f, indent=2)

print(f"\nSaved optimized config to: {config_path}")

# === Step 7: Use in Production ===
# Load optimized weights into your CausalConfig
config = CausalConfig.from_hpo_result(str(config_path))
engine = CausalInferenceEngine(config)

# Now your engine uses scientifically justified weights!
print("\n✅ CIS weights are now learned from data via Bayesian optimization")
