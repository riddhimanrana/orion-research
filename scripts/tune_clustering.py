#!/usr/bin/env python3
"""
Quick script to tune clustering parameters on saved embeddings.

Loads observations from a previous run and tries different HDBSCAN parameters
to find the best settings for your video.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import hdbscan
except ImportError:
    print("ERROR: hdbscan not installed. Run: pip install hdbscan")
    sys.exit(1)


def load_embeddings(results_file):
    """Load embeddings from tracking results"""
    with open(results_file) as f:
        data = json.load(f)
    
    # For now, this won't work since we need the actual embedding arrays
    # This is just a placeholder for future enhancement
    print(f"Loaded {data['total_observations']} observations")
    print(f"Current result: {data['total_entities']} entities")
    return None


def test_clustering_params(embeddings, min_cluster_size_range, epsilon_range):
    """Test different clustering parameters"""
    print("\nTesting clustering parameters...")
    print("="*80)
    
    best_config = None
    best_ratio = 0
    
    for min_size in min_cluster_size_range:
        for epsilon in epsilon_range:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_size,
                min_samples=1,
                metric='euclidean',
                cluster_selection_epsilon=epsilon
            )
            
            labels = clusterer.fit_predict(embeddings)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            n_entities = n_clusters + n_noise
            ratio = len(embeddings) / n_entities
            
            print(f"min_size={min_size:2d}, epsilon={epsilon:.2f}: "
                  f"{n_entities:3d} entities ({n_clusters} clusters, {n_noise} noise), "
                  f"ratio={ratio:.1f}x")
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_config = (min_size, epsilon, n_entities)
    
    print("="*80)
    if best_config:
        print(f"\nBest config: MIN_CLUSTER_SIZE={best_config[0]}, "
              f"CLUSTER_SELECTION_EPSILON={best_config[1]:.2f}")
        print(f"  → {best_config[2]} entities, {best_ratio:.1f}x reduction")
    
    return best_config


def main():
    parser = argparse.ArgumentParser(description="Tune clustering parameters")
    parser.add_argument(
        "--results",
        type=str,
        default="data/testing/tracking_results.json",
        help="Path to tracking results JSON"
    )
    
    args = parser.parse_args()
    
    results_file = Path(args.results)
    if not results_file.exists():
        print(f"ERROR: Results file not found: {results_file}")
        print("Run test_tracking.py first to generate observations")
        return 1
    
    print("="*80)
    print("CLUSTERING PARAMETER TUNER")
    print("="*80)
    
    # Load embeddings
    embeddings = load_embeddings(results_file)
    
    if embeddings is None:
        print("\nNOTE: This script needs to be enhanced to extract embeddings.")
        print("For now, manually tune parameters in tracking_engine.py Config class:")
        print("")
        print("  MIN_CLUSTER_SIZE: 2-5 (lower = more aggressive merging)")
        print("  CLUSTER_SELECTION_EPSILON: 0.8-1.5 (higher = merge more)")
        print("")
        print("Based on your sample cosine distances:")
        print("  min: 0.0195 → euclidean ~0.20")
        print("  max: 0.7519 → euclidean ~1.23") 
        print("  mean: 0.5692 → euclidean ~1.07")
        print("")
        print("Recommendation:")
        print("  MIN_CLUSTER_SIZE = 2")
        print("  CLUSTER_SELECTION_EPSILON = 1.0-1.3")
        return 0
    
    # Test different parameters
    min_sizes = [2, 3, 4, 5]
    epsilons = [0.8, 1.0, 1.2, 1.4, 1.6]
    
    best_config = test_clustering_params(embeddings, min_sizes, epsilons)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
