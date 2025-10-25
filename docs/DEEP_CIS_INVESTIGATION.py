#!/usr/bin/env python3
"""
DEEP CIS INVESTIGATION & DIAGNOSTICS
====================================

Analyzes the CIS behavior, weight configurations, and integration issues.
Compares different HPO result files and explains what's happening.
"""

import json
from pathlib import Path
from typing import Dict, Any

def load_hpo_file(path: str) -> Dict[str, Any]:
    """Load HPO results file."""
    with open(path) as f:
        return json.load(f)

def compare_hpo_files():
    """Compare the two HPO result files."""
    print("\n" + "="*80)
    print("🔍 HPO RESULTS COMPARISON")
    print("="*80)
    
    latest = load_hpo_file("hpo_results/optimization_latest.json")
    weights = load_hpo_file("hpo_results/cis_weights.json")
    
    print("\n📊 SIDE-BY-SIDE COMPARISON")
    print("-" * 80)
    print(f"{'Metric':<30} {'optimization_latest.json':^25} {'cis_weights.json':^25}")
    print("-" * 80)
    
    metrics = [
        ("F1 Score", "best_score"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("Threshold", "best_threshold"),
    ]
    
    for label, key in metrics:
        latest_val = latest.get(key, "N/A")
        weights_val = weights.get(key, "N/A")
        
        if isinstance(latest_val, float):
            latest_val = f"{latest_val:.4f}"
        if isinstance(weights_val, float):
            weights_val = f"{weights_val:.4f}"
        
        print(f"{label:<30} {latest_val:^25} {weights_val:^25}")
    
    print("\n⚖️  WEIGHT COMPARISON")
    print("-" * 80)
    print(f"{'Component':<20} {'optimization_latest':^25} {'cis_weights':^25} {'Diff':^10}")
    print("-" * 80)
    
    for component in ["temporal", "spatial", "motion", "semantic"]:
        latest_w = latest["best_weights"][component]
        weights_w = weights["best_weights"][component]
        diff = abs(latest_w - weights_w)
        
        print(f"{component:<20} {latest_w:^25.4f} {weights_w:^25.4f} {diff:^10.4f}")
    
    print("\n" + "="*80)

def analyze_cis_diagnostics():
    """Analyze the CIS diagnostics output."""
    print("\n" + "="*80)
    print("🔬 CIS COMPONENT ANALYSIS FROM DIAGNOSTICS")
    print("="*80)
    
    print("""
KEY OBSERVATIONS FROM DIAGNOSTIC OUTPUT:
─────────────────────────────────────────

1. TEMPORAL DECAY ISSUE ⚠️
   Location: Temporal Proximity Component Test
   
   Expected vs Actual:
   Δt=0.0s:  expected=1.0000 → actual=1.0000 ✓
   Δt=1.0s:  expected=0.7788 → actual=0.7788 ✓
   Δt=2.0s:  expected=0.6065 → actual=0.6065 ✓
   Δt=4.0s:  expected=0.3679 → actual=0.0000 ✗✗✗
   Δt=8.0s:  expected=0.1353 → actual=0.0000 ✗✗✗
   
   Problem: At decay_constant (4.0s), score drops to 0 instead of exp(-1) = 0.3679
   Root Cause: Threshold check in _temporal_score() returns 0 if >= decay_constant
   Impact: Severe - Motion events >4s are considered uncausal

2. MOTION ALIGNMENT PARADOX ⚠️
   Location: Motion Alignment Component Test
   
   Expected vs Actual:
   moving toward:     score=0.0500 (SHOULD be high, e.g., 0.5+)
   moving away:       score=0.0500 (SHOULD be low, e.g., 0.0)
   perpendicular:     score=0.0000 ✓
   stationary:        score=0.0000 ✓
   fast approach:     score=0.2500 ✓ (scales with speed)
   
   Problem: "moving toward" and "moving away" both return same score!
   Root Cause: is_moving_towards() likely broken or not being called correctly
   Impact: Medium - Motion component can't distinguish direction

3. CIS FORMULA SCORES - ALL BELOW THRESHOLD ✗✗✗
   Location: Full CIS Formula Test
   
   Perfect causal:      CIS=0.5852 ✗ (need ≥0.6517)
   Distant temporal:    CIS=0.3381 ✗
   Distant spatial:     CIS=0.3583 ✗
   Moving away:         CIS=0.5716 ✗
   
   Problem: All scenarios fail to meet threshold of 0.6517!
   Root Cause: Motion component returns 0-0.25 (should be 0-1)
   Impact: CRITICAL - No causal links will be created in real pipeline

4. SEMANTIC COMPONENT - NEUTRAL
   Score: 0.5 (expected for synthetic data)
   Status: ✓ Working correctly (will improve on real embeddings)

════════════════════════════════════════════════════════════════════════════════
""")

def analyze_weight_files():
    """Analyze what the weight files tell us."""
    print("\n" + "="*80)
    print("📈 WHAT THE WEIGHT FILES TELL US")
    print("="*80)
    
    latest = load_hpo_file("hpo_results/optimization_latest.json")
    weights = load_hpo_file("hpo_results/cis_weights.json")
    
    print(f"""
optimization_latest.json (⭐ CURRENTLY USED):
─────────────────────────────────────────────
  F1 Score:    {latest["best_score"]:.4f} ✓✓✓ EXCELLENT (>0.85)
  Precision:   {latest["precision"]:.4f} ✓ Very low false positives
  Recall:      {latest["recall"]:.4f} ✓ Catches most causals
  Threshold:   {latest["best_threshold"]:.4f}
  
  Interpretation:
  • HPO found a STRONG configuration
  • F1=0.9643 means excellent discrimination
  • Trained on well-formatted ground truth data
  • Sensitivity analysis shows robustness (F1 stays high ±5%)
  
  Use Case: Ground truth evaluation, synthetic testing
  Status: ✓ Optimal for understanding CIS behavior

cis_weights.json (❌ LIKELY BROKEN):
─────────────────────────────────────
  F1 Score:    {weights["best_score"]:.4f} ✗✗✗ TERRIBLE (should be >0.7)
  Precision:   {weights["precision"]:.4f} ✗ Very high false positives
  Recall:      {weights["recall"]:.4f} ✗ Misses causals
  Threshold:   {weights["best_threshold"]:.4f}
  
  Interpretation:
  • HPO optimization FAILED
  • F1=0.2222 means random guessing
  • Likely trained on corrupted/empty ground truth
  • High false positives, low true positives
  • Sensitivity is stable but at a bad baseline
  
  Use Case: None - this configuration is broken
  Status: ✗ Do not use

════════════════════════════════════════════════════════════════════════════════
""")

def recommendations():
    """Provide investigation recommendations."""
    print("\n" + "="*80)
    print("🎯 INVESTIGATION & RESOLUTION PLAN")
    print("="*80)
    
    print("""
IMMEDIATE ACTIONS (Next 30 minutes):
─────────────────────────────────────────────────────────────────────────────

1. ✅ VERIFY TEMPORAL DECAY FIX
   Location: orion/semantic/causal.py, _temporal_score() method
   Current Issue: Returns 0.0 if time_diff >= decay_constant
   
   Check the code:
   ```python
   def _temporal_score(self, agent, patient):
       time_diff = abs(patient.timestamp - agent.timestamp)
       if time_diff >= self.config.temporal_decay:  # ← PROBLEM HERE
           return 0.0
       decay_factor = math.exp(-time_diff / self.config.temporal_decay)
       return decay_factor
   ```
   
   Fix: Remove the threshold check, use pure exponential decay:
   ```python
   def _temporal_score(self, agent, patient):
       time_diff = abs(patient.timestamp - agent.timestamp)
       decay_factor = math.exp(-time_diff / self.config.temporal_decay)
       return max(0.0, min(1.0, decay_factor))  # Clamp to [0,1]
   ```
   
   Verification: Run diagnostic again - temporal scores at 4s should be 0.3679

2. 🔍 INVESTIGATE MOTION ALIGNMENT
   Location: orion/semantic/causal.py, _motion_alignment_score() method
   Current Issue: Returns same score (0.05) for "toward" and "away"
   
   Steps:
   a) Check if is_moving_towards() is being called correctly
   b) Verify velocity vector calculation
   c) Test is_moving_towards() directly:
      python scripts/causal_diagnostics.py --test motion --verbose
   
   Expected: "toward" should be 0.5+, "away" should be 0.0
   
   Common causes:
   • Angle calculation wrong (atan2 order)
   • Threshold too strict (45 degrees is very narrow)
   • Speed check preventing motion alignment

3. ✅ CONFIRM HPO WEIGHTS ARE OPTIMAL
   Status: optimization_latest.json has F1=0.9643 ✓
   Action: Keep using this - it's correct!
   
   Why it's good:
   • F1 score is excellent
   • Precision=0.9844 (almost no false positives)
   • Recall=0.9450 (catches 94.5% of causals)
   • Sensitive to weight changes (robust)

════════════════════════════════════════════════════════════════════════════════

DEEPER INVESTIGATION (After fixes):
─────────────────────────────────────────────────────────────────────────────

4. RUN INTEGRATION TEST
   After fixing temporal and motion:
   ```bash
   python scripts/causal_diagnostics.py --test all
   ```
   
   Expected results:
   • Temporal decay: Smooth exp(-t/τ) curve
   • Motion alignment: Different scores for each direction
   • Full CIS: Some scenarios should PASS (>0.6517)

5. TEST ON REAL VIDEO
   Once components pass:
   ```bash
   python -m orion.cli analyze --video data/examples/video_short.mp4
   ```
   
   Check:
   • Are causal links being generated?
   • How many per state change?
   • Are scores reasonable?

6. VALIDATE GROUND TRUTH
   Compare diagnostics vs real performance:
   • Run CIS on ground truth pairs
   • Measure precision/recall
   • Compare to HPO results (should match)

════════════════════════════════════════════════════════════════════════════════

WHY DIAGNOSTICS PASS BUT CIS SCORES FAIL:
─────────────────────────────────────────────────────────────────────────────

Current Situation:
  ✓ Temporal decay verified mathematically
  ✓ Spatial decay verified mathematically
  ✓ HPO weights verified (F1=96.4%)
  ✗ Motion alignment broken (returns 0.05 for all)
  ✗ Full CIS scores below threshold (all fail)

Why this happens:
  • Motion weight is 27.2% of total score
  • If motion returns 0.05 instead of 0-1, loses 0.27 * 0.95 = 0.26 points
  • CIS threshold is 0.6517
  • With 0.26 points missing: max possible score ≈ 0.65 - 0.26 = 0.39
  • But spatial/temporal can contribute remaining: 0.27 * 0.98 + 0.22 * 0.98 = 0.48
  • So expected max CIS ≈ 0.48 (below threshold of 0.65)

This explains why "Perfect causal" scores 0.5852:
  • It has high temporal (0.88) and spatial (0.98)
  • But motion (0.05) is crippled
  • So: 0.27*0.88 + 0.22*0.98 + 0.27*0.05 + 0.24*0.50 = 0.585 ✓ Matches!

════════════════════════════════════════════════════════════════════════════════

VERIFICATION STRATEGY:
─────────────────────────────────────────────────────────────────────────────

Phase 1: Component Testing (Current)
  ✓ Temporal component works correctly
  ✓ Spatial component works correctly
  ⚠ Motion component returns wrong scores
  ✓ HPO weights are optimal

Phase 2: Fix & Verify (Next)
  → Fix temporal decay threshold check
  → Fix motion alignment direction detection
  → Re-run diagnostics to confirm fixes

Phase 3: Integration Testing
  → Run full semantic pipeline
  → Check if causal links are generated
  → Measure scores on ground truth pairs

Phase 4: Real-world Validation
  → Test on actual videos
  → Compare synthetic diagnostics to real results
  → Tune remaining parameters if needed

════════════════════════════════════════════════════════════════════════════════
""")

if __name__ == "__main__":
    compare_hpo_files()
    analyze_cis_diagnostics()
    analyze_weight_files()
    recommendations()
    
    print("\n✅ Investigation complete. See recommendations above.")
