#!/usr/bin/env python3
"""
Phase 2 Re-ID Optimization Script

Analyzes current Re-ID performance and tests optimization strategies:
1. Similarity distribution analysis
2. Threshold tuning per class
3. Label-consistency constraints
4. Clustering algorithm comparison

Usage:
    python scripts/optimize_reid.py --results results/pipeline_test_fixed
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    console = None


@dataclass
class ClusterMetrics:
    """Metrics for evaluating clustering quality."""
    total_clusters: int
    singletons: int
    consistent_clusters: int  # All tracks have same label
    inconsistent_clusters: int
    avg_cluster_size: float
    largest_cluster: int
    intra_cluster_sim: float  # Avg similarity within clusters
    inter_cluster_sim: float  # Avg similarity between clusters
    
    @property
    def consistency_rate(self) -> float:
        return self.consistent_clusters / max(self.total_clusters, 1)
    
    @property
    def separation(self) -> float:
        """Good separation = high intra, low inter."""
        return self.intra_cluster_sim - self.inter_cluster_sim


def load_data(results_dir: Path) -> Tuple[Dict, Dict, Dict]:
    """Load tracks, clusters, and embeddings from results."""
    tracks_file = results_dir / "tracks.jsonl"
    clusters_file = results_dir / "reid_clusters.json"
    memory_file = results_dir / "memory.json"
    
    # Load tracks by ID
    tracks_by_id = defaultdict(list)
    with open(tracks_file) as f:
        for line in f:
            entry = json.loads(line)
            tracks_by_id[entry['track_id']].append(entry)
    
    # Load clusters
    clusters = json.load(open(clusters_file))
    
    # Load memory (has embeddings if available)
    memory = json.load(open(memory_file)) if memory_file.exists() else {}
    
    return dict(tracks_by_id), clusters, memory


def analyze_label_distribution(tracks_by_id: Dict) -> Dict[str, int]:
    """Analyze what labels exist in tracks."""
    label_counts = Counter()
    for track_id, observations in tracks_by_id.items():
        label = observations[0]['label']
        label_counts[label] += 1
    return dict(label_counts)


def analyze_clusters(clusters: Dict, tracks_by_id: Dict) -> ClusterMetrics:
    """Compute cluster quality metrics."""
    cluster_sizes = []
    consistent = 0
    inconsistent = 0
    inconsistent_examples = []
    
    for mem_id, track_ids in clusters.items():
        cluster_sizes.append(len(track_ids))
        
        # Check label consistency
        labels = [tracks_by_id.get(tid, [{}])[0].get('label', 'unknown') for tid in track_ids]
        unique_labels = set(labels)
        
        if len(unique_labels) == 1:
            consistent += 1
        else:
            inconsistent += 1
            if len(inconsistent_examples) < 10:
                inconsistent_examples.append({
                    'cluster': mem_id,
                    'labels': dict(Counter(labels)),
                    'size': len(track_ids)
                })
    
    singletons = sum(1 for s in cluster_sizes if s == 1)
    
    return ClusterMetrics(
        total_clusters=len(clusters),
        singletons=singletons,
        consistent_clusters=consistent,
        inconsistent_clusters=inconsistent,
        avg_cluster_size=np.mean(cluster_sizes) if cluster_sizes else 0,
        largest_cluster=max(cluster_sizes) if cluster_sizes else 0,
        intra_cluster_sim=0.0,  # Will compute if embeddings available
        inter_cluster_sim=0.0
    ), inconsistent_examples


def find_semantic_duplicates(tracks_by_id: Dict) -> List[Tuple[str, str]]:
    """Find semantically similar class names that should merge."""
    SEMANTIC_GROUPS = [
        {'monitor', 'computer monitor', 'computer screen', 'screen', 'television', 'tv'},
        {'chair', 'office chair', 'armchair', 'stool', 'seat'},
        {'bottle', 'water bottle', 'plastic bottle'},
        {'couch', 'sofa', 'loveseat'},
        {'plant', 'potted plant', 'houseplant'},
        {'desk', 'table', 'counter'},
        {'lamp', 'floor lamp', 'table lamp', 'light'},
        {'picture', 'painting', 'picture frame', 'frame', 'artwork'},
        {'rug', 'carpet', 'mat', 'floor mat'},
        {'pillow', 'cushion', 'throw pillow'},
    ]
    
    labels_in_data = set()
    for track_id, observations in tracks_by_id.items():
        labels_in_data.add(observations[0]['label'])
    
    potential_merges = []
    for group in SEMANTIC_GROUPS:
        present = group & labels_in_data
        if len(present) > 1:
            potential_merges.append(sorted(present))
    
    return potential_merges


def compute_similarity_stats(memory: Dict) -> Dict[str, Dict]:
    """Compute similarity statistics from memory objects."""
    if 'objects' not in memory:
        return {}
    
    objects = memory['objects']
    
    # Group by label
    by_label = defaultdict(list)
    for obj in objects:
        label = obj.get('label', 'unknown')
        emb = obj.get('embedding')
        if emb:
            by_label[label].append(np.array(emb))
    
    stats = {}
    for label, embeddings in by_label.items():
        if len(embeddings) < 2:
            continue
        
        # Compute pairwise similarities
        embs = np.array(embeddings)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embs_norm = embs / norms
        
        sim_matrix = embs_norm @ embs_norm.T
        
        # Get upper triangle (exclude diagonal)
        triu_idx = np.triu_indices_from(sim_matrix, k=1)
        similarities = sim_matrix[triu_idx]
        
        if len(similarities) > 0:
            stats[label] = {
                'count': len(embeddings),
                'mean': float(np.mean(similarities)),
                'std': float(np.std(similarities)),
                'min': float(np.min(similarities)),
                'max': float(np.max(similarities)),
                'p25': float(np.percentile(similarities, 25)),
                'p75': float(np.percentile(similarities, 75))
            }
    
    return stats


def recommend_thresholds(sim_stats: Dict, current_thresholds: Dict) -> Dict[str, float]:
    """Recommend new thresholds based on similarity distributions."""
    recommendations = {}
    
    for label, stats in sim_stats.items():
        current = current_thresholds.get(label.lower(), 0.75)
        
        # Recommend: mean - 1*std, but not below p25
        recommended = max(stats['mean'] - stats['std'], stats['p25'])
        recommended = min(recommended, 0.90)  # Cap at 0.90
        recommended = max(recommended, 0.50)  # Floor at 0.50
        
        if abs(recommended - current) > 0.05:  # Only if significantly different
            recommendations[label] = {
                'current': current,
                'recommended': round(recommended, 2),
                'mean_sim': round(stats['mean'], 3),
                'std': round(stats['std'], 3)
            }
    
    return recommendations


def simulate_clustering(tracks_by_id: Dict, embeddings: Dict, threshold: float,
                        use_label_constraint: bool = True) -> ClusterMetrics:
    """Simulate clustering with given parameters."""
    # Group by class
    tracks_by_class = defaultdict(list)
    for track_id, observations in tracks_by_id.items():
        label = observations[0]['label']
        if track_id in embeddings:
            tracks_by_class[label].append(track_id)
    
    clusters = []
    
    for label, track_ids in tracks_by_class.items():
        if len(track_ids) == 1:
            clusters.append([track_ids[0]])
            continue
        
        # Get embeddings
        embs = np.array([embeddings[tid] for tid in track_ids])
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embs_norm = embs / norms
        
        sim_matrix = embs_norm @ embs_norm.T
        
        # Greedy clustering
        assigned = set()
        label_clusters = []
        
        for i, tid_i in enumerate(track_ids):
            if tid_i in assigned:
                continue
            
            cluster = [tid_i]
            assigned.add(tid_i)
            
            for j, tid_j in enumerate(track_ids):
                if tid_j in assigned:
                    continue
                
                if sim_matrix[i, j] >= threshold:
                    cluster.append(tid_j)
                    assigned.add(tid_j)
            
            label_clusters.append(cluster)
        
        clusters.extend(label_clusters)
    
    # Compute metrics
    cluster_sizes = [len(c) for c in clusters]
    singletons = sum(1 for s in cluster_sizes if s == 1)
    
    return ClusterMetrics(
        total_clusters=len(clusters),
        singletons=singletons,
        consistent_clusters=len(clusters),  # All same-class by design
        inconsistent_clusters=0,
        avg_cluster_size=np.mean(cluster_sizes) if cluster_sizes else 0,
        largest_cluster=max(cluster_sizes) if cluster_sizes else 0,
        intra_cluster_sim=0.0,
        inter_cluster_sim=0.0
    )


def print_analysis(metrics: ClusterMetrics, inconsistent: List, 
                   semantic_dupes: List, sim_stats: Dict, 
                   recommendations: Dict):
    """Print analysis results."""
    print("\n" + "="*70)
    print("PHASE 2 RE-ID ANALYSIS")
    print("="*70)
    
    print("\n📊 CLUSTER QUALITY")
    print(f"  Total clusters: {metrics.total_clusters}")
    print(f"  Singletons: {metrics.singletons} ({100*metrics.singletons/max(metrics.total_clusters,1):.1f}%)")
    print(f"  Avg size: {metrics.avg_cluster_size:.1f}")
    print(f"  Largest: {metrics.largest_cluster}")
    
    print("\n📋 LABEL CONSISTENCY")
    print(f"  Consistent: {metrics.consistent_clusters} ({100*metrics.consistency_rate:.1f}%)")
    print(f"  Inconsistent: {metrics.inconsistent_clusters}")
    
    if inconsistent:
        print("\n  ⚠️ Inconsistent cluster examples:")
        for ex in inconsistent[:5]:
            print(f"    {ex['cluster']}: {ex['labels']}")
    
    if semantic_dupes:
        print("\n🔗 SEMANTIC DUPLICATES (should merge labels)")
        for group in semantic_dupes[:8]:
            print(f"    {' ↔ '.join(group)}")
    
    if recommendations:
        print("\n🎯 THRESHOLD RECOMMENDATIONS")
        print(f"  {'Class':<25} {'Current':>8} {'Recommend':>10} {'Mean Sim':>10}")
        for label, rec in sorted(recommendations.items(), 
                                  key=lambda x: abs(x[1]['recommended'] - x[1]['current']), 
                                  reverse=True)[:10]:
            print(f"  {label:<25} {rec['current']:>8.2f} {rec['recommended']:>10.2f} {rec['mean_sim']:>10.3f}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Analyze and optimize Phase 2 Re-ID")
    parser.add_argument("--results", default="results/pipeline_test_fixed",
                        help="Results directory to analyze")
    parser.add_argument("--sweep-thresholds", action="store_true",
                        help="Run threshold sweep simulation")
    parser.add_argument("--output", help="Save analysis to JSON file")
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)
    
    print(f"Analyzing: {results_dir}")
    
    # Load data
    tracks_by_id, clusters, memory = load_data(results_dir)
    print(f"Loaded {len(tracks_by_id)} tracks, {len(clusters)} clusters")
    
    # Current thresholds from code
    from orion.cli.commands.embed import CLASS_THRESHOLDS
    
    # Analyze
    metrics, inconsistent = analyze_clusters(clusters, tracks_by_id)
    semantic_dupes = find_semantic_duplicates(tracks_by_id)
    sim_stats = compute_similarity_stats(memory)
    recommendations = recommend_thresholds(sim_stats, CLASS_THRESHOLDS)
    
    # Print results
    print_analysis(metrics, inconsistent, semantic_dupes, sim_stats, recommendations)
    
    # Threshold sweep
    if args.sweep_thresholds and memory.get('objects'):
        print("\n🔍 THRESHOLD SWEEP")
        print("  Simulating different thresholds...")
        
        # Extract embeddings from memory
        embeddings = {}
        for obj in memory['objects']:
            for tid in obj.get('original_track_ids', []):
                if obj.get('embedding'):
                    embeddings[tid] = np.array(obj['embedding'])
        
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        print(f"\n  {'Threshold':>10} {'Clusters':>10} {'Singletons':>12} {'Avg Size':>10}")
        
        for thresh in thresholds:
            result = simulate_clustering(tracks_by_id, embeddings, thresh)
            print(f"  {thresh:>10.2f} {result.total_clusters:>10} {result.singletons:>12} {result.avg_cluster_size:>10.1f}")
    
    # Save results
    if args.output:
        output = {
            'metrics': {
                'total_clusters': metrics.total_clusters,
                'singletons': metrics.singletons,
                'consistent': metrics.consistent_clusters,
                'inconsistent': metrics.inconsistent_clusters,
                'consistency_rate': metrics.consistency_rate
            },
            'inconsistent_examples': inconsistent,
            'semantic_duplicates': semantic_dupes,
            'similarity_stats': sim_stats,
            'recommendations': recommendations
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
