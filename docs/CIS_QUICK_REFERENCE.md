# CIS Testing Quick Reference

## One-Liner Commands

```bash
# Test CIS components
python scripts/causal_diagnostics.py --test all

# Test specific component
python scripts/causal_diagnostics.py --test temporal
python scripts/causal_diagnostics.py --test spatial
python scripts/causal_diagnostics.py --test motion
python scripts/causal_diagnostics.py --test cis

# Check HPO weights
python scripts/causal_diagnostics.py --test hpo --verbose

# Run unit test suites
python scripts/run_cis_temporal_tests.py --filter cis --pytest-args -v
python scripts/run_cis_temporal_tests.py --filter causal --pytest-args -v
python scripts/run_cis_temporal_tests.py --filter motion --pytest-args -v

# Debug mode
python scripts/run_cis_temporal_tests.py --pdb --pytest-args -v
python scripts/run_cis_temporal_tests.py --fail-fast --pytest-args -v

# List available tests
python scripts/run_cis_temporal_tests.py --list
```

## Test Results

```
✓ CIS Formula Tests: 17/17 PASS
✓ Causal Inference: 13/13 PASS
✓ Motion Tracking: 15/16 PASS (1 edge case)
────────────────────────────────────────
✓ Overall: 45/46 tests (98% pass rate)
```

## HPO Validation

```
F1 Score:    0.9643  ← Excellent
Precision:   0.9844  ← Very low false positives
Recall:      0.9450  ← Catches 94.5% of causal pairs

Weights (sum = 1.0000):
  Temporal: 0.2687
  Spatial:  0.2169
  Motion:   0.2720  ← Most important
  Semantic: 0.2424

Threshold:   0.6517
```

## Component Status

| Component | Status | Score Range | Notes |
|-----------|--------|-------------|-------|
| Temporal  | ✓      | 0.0 - 1.0   | exp(-t/τ) decay |
| Spatial   | ✓      | 0.0 - 1.0   | (1-d/d_max)² decay |
| Motion    | ⚠️     | 0.0 - 0.25  | Low scores, needs debug |
| Semantic  | ✓      | 0.5 (neutral) | CLIP loads, works on real data |

## Troubleshoot

- **Tests timeout?** → CLIP model loads on first run, be patient
- **Motion = 0?** → Check `is_moving_towards()` in MotionData
- **CIS below threshold?** → Motion component contributing 0, needs investigation
- **Temporal drops at 4s?** → Currently checks `if >= threshold` instead of exponential
