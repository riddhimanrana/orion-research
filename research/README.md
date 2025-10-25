# Research & Experiments

This directory contains research code, evaluation scripts, baselines, and hyperparameter optimization for the Orion project.

**These are experimental/research tools and are NOT part of the core `orion` library.**

## Directory Structure

```
research/
├── evaluation/     # Benchmark evaluation scripts
├── baselines/      # Baseline comparison implementations
└── hpo/            # Hyperparameter optimization
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
