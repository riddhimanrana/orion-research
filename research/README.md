# Research & Experiments

This directory contains research code, evaluation scripts, baselines, and hyperparameter optimization for the Orion project.

**These are experimental/research tools and are NOT part of the core `orion` library.**

## 📋 **NEW: Comprehensive Evaluation Plan**

⚠️ **Status**: Design complete, ready for implementation. Much of the existing code in this folder will be updated per the new plan.

### 📚 Essential Reading (in order)

1. **[RESEARCH_SUMMARY.md](./RESEARCH_SUMMARY.md)** — Start here! Quick overview and current status
2. **[DECISION_MATRIX.md](./DECISION_MATRIX.md)** — Strategic decisions and recommendations
3. **[EVALUATION_PLAN.md](./EVALUATION_PLAN.md)** — High-level strategy and expected results
4. **[IMPLEMENTATION_ROADMAP.md](./IMPLEMENTATION_ROADMAP.md)** — Detailed 5-7 week implementation plan

### 🎯 Quick Summary

We need to evaluate Orion on:
- **Datasets**: Action Genome (quantitative) + optionally VSGR/ASPIRe (cross-domain)
- **Baselines**: Heuristic (✅ exists), LLM-Only VoT (✅ exists), HyperGLM (❌ needs implementation)
- **Metrics**: Triplet F1, Entity Continuity (your unique strength!), Causal F1, Recall@K
- **Ablations**: No-tracking, No-LLM, No-semantic-uplift

**Recommended Timeline**: 4-6 weeks depending on baseline coverage

## Directory Structure

```
research/
├── EVALUATION_PLAN.md        # 📋 NEW: Comprehensive evaluation roadmap
├── evaluation/               # Benchmark evaluation scripts (needs refactor)
├── baselines/                # Baseline comparison implementations (needs implementation)
└── hpo/                      # Hyperparameter optimization (CIS weights)
```

## Evaluation

Run benchmark evaluations against standard datasets:

```bash
# Action Genome benchmark
python research/evaluation/benchmark_runner.py --dataset=action-genome --data-dir=data/ag_50

# With specific videos
python research/evaluation/benchmark_runner.py --dataset=action-genome --video-ids 1234 5678

# Limit number of videos
python research/evaluation/benchmark_runner.py --dataset=action-genome --max-videos=10
```

## Baselines

Baseline comparison implementations for research papers.

## HPO (Hyperparameter Optimization)

Tools for optimizing hyperparameters, particularly for the CIS (Contextual Inference System).

```bash
# Run CIS optimization
python research/hpo/cis_optimizer.py
```

## Notes

- These scripts expect the main `orion` library to be installed or in PYTHONPATH
- Evaluation results are typically saved to `evaluation_results/` or similar directories
- See individual module READMEs for more details
