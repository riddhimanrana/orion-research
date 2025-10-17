#!/usr/bin/env python3
"""Quick diagnostic to check clustering behavior"""

import numpy as np
import hdbscan

# Simulate normalized 2048-dim embeddings
np.random.seed(42)
n_objects = 436
n_dims = 2048

# Create some clusters (simulate 4 actual objects seen many times each)
cluster_centers = np.random.randn(4, n_dims)
for i in range(4):
    cluster_centers[i] = cluster_centers[i] / np.linalg.norm(cluster_centers[i])

# Generate observations (each object appears ~100 times with slight variations)
embeddings = []
true_labels = []
for i in range(4):
    n_appearances = 109  # ~436/4
    for _ in range(n_appearances):
        # Add small noise
        noisy_emb = cluster_centers[i] + np.random.randn(n_dims) * 0.01
        noisy_emb = noisy_emb / np.linalg.norm(noisy_emb)  # Renormalize
        embeddings.append(noisy_emb)
        true_labels.append(i)

embeddings = np.array(embeddings)
print(f"Created {len(embeddings)} embeddings in {n_dims} dimensions")
print(f"True clusters: 4")

# Test with semantic_uplift.py parameters
print("\n" + "="*80)
print("Testing with semantic_uplift.py parameters:")
print("="*80)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=3,
    min_samples=2,
    metric='euclidean',
    cluster_selection_method='eom',
    cluster_selection_epsilon=0.15
)

labels = clusterer.fit_predict(embeddings)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Results:")
print(f"  Clusters found: {n_clusters}")
print(f"  Noise points: {n_noise}")
print(f"  Total entities: {n_clusters + n_noise}")

# Sample distances
sample_euc_dists = []
for i in range(min(10, len(embeddings))):
    for j in range(i+1, min(10, len(embeddings))):
        euc_dist = np.linalg.norm(embeddings[i] - embeddings[j])
        sample_euc_dists.append(euc_dist)

print(f"\nSample euclidean distances:")
print(f"  min: {min(sample_euc_dists):.4f}")
print(f"  max: {max(sample_euc_dists):.4f}")
print(f"  mean: {np.mean(sample_euc_dists):.4f}")
print(f"  epsilon: 0.15")
print(f"  Mean is {np.mean(sample_euc_dists)/0.15:.1f}x larger than epsilon")
