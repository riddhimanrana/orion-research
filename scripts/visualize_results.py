#!/usr/bin/env python3
"""
Results Visualization Script
=============================

Creates publication-quality figures from evaluation results.

Author: Orion Research Team
Date: October 2025
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Visualizer")

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_f1_comparison(results: Dict, output_path: Path):
    """Bar chart comparing F1 scores across methods"""
    methods = []
    edge_f1 = []
    causal_f1 = []
    
    for method_name, metrics in results.items():
        methods.append(method_name)
        edge_f1.append(metrics.get('edge_f1', 0))
        causal_f1.append(metrics.get('causal_f1', 0))
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, edge_f1, width, label='Edge F1', color='steelblue')
    bars2 = ax.bar(x + width/2, causal_f1, width, label='Causal F1', color='coral')
    
    ax.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
    ax.set_title('F1 Score Comparison Across Methods', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved F1 comparison to {output_path}")
    plt.close()


def plot_precision_recall(results: Dict, output_path: Path):
    """Precision-Recall scatter plot"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for method_name, metrics in results.items():
        precision = metrics.get('causal_precision', 0)
        recall = metrics.get('causal_recall', 0)
        f1 = metrics.get('causal_f1', 0)
        
        # Size by F1
        size = f1 * 500
        
        ax.scatter(recall, precision, s=size, alpha=0.6, label=method_name)
        ax.text(recall + 0.02, precision, method_name, fontsize=10)
    
    # Add F1 iso-lines
    for f1_line in [0.3, 0.5, 0.7, 0.9]:
        x = np.linspace(0.01, 1, 100)
        y = (f1_line * x) / (2*x - f1_line)
        y = np.clip(y, 0, 1)
        ax.plot(x, y, '--', alpha=0.3, color='gray', linewidth=1)
        ax.text(0.9, (f1_line * 0.9) / (2*0.9 - f1_line), f'F1={f1_line}',
               fontsize=8, alpha=0.5)
    
    ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax.set_title('Precision-Recall Trade-off', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved P/R curve to {output_path}")
    plt.close()


def plot_component_ablation(ablation_results: Dict, output_path: Path):
    """Bar chart showing ablation study results"""
    components = []
    f1_scores = []
    
    for component, score in ablation_results.items():
        components.append(component)
        f1_scores.append(score)
    
    # Sort by F1 score
    sorted_pairs = sorted(zip(components, f1_scores), key=lambda x: x[1], reverse=True)
    components = [p[0] for p in sorted_pairs]
    f1_scores = [p[1] for p in sorted_pairs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['green' if 'Full' in c else 'orange' for c in components]
    bars = ax.barh(components, f1_scores, color=colors, alpha=0.7)
    
    ax.set_xlabel('Causal F1 Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Configuration', fontsize=14, fontweight='bold')
    ax.set_title('Ablation Study: Component Contribution', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1.0)
    
    # Add value labels
    for bar, score in zip(bars, f1_scores):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
               f'{score:.3f}',
               ha='left', va='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved ablation study to {output_path}")
    plt.close()


def plot_dataset_generalization(cross_dataset_results: Dict, output_path: Path):
    """Heatmap showing cross-dataset generalization"""
    # Prepare data
    datasets = list(cross_dataset_results.keys())
    metrics_matrix = []
    
    for dataset in datasets:
        row = [cross_dataset_results[dataset].get('causal_f1', 0)]
        metrics_matrix.append(row)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        metrics_matrix,
        annot=True,
        fmt='.3f',
        cmap='YlGnBu',
        xticklabels=['Causal F1'],
        yticklabels=datasets,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'F1 Score'},
        ax=ax
    )
    
    ax.set_title('Cross-Dataset Generalization', fontsize=16, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dataset', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved generalization heatmap to {output_path}")
    plt.close()


def plot_cis_distribution(cis_scores: List[float], output_path: Path):
    """Histogram of CIS scores"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(cis_scores, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add threshold line
    threshold = 0.55
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
              label=f'Threshold ({threshold})')
    
    ax.set_xlabel('CIS Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of Causal Influence Scores', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_cis = np.mean(cis_scores)
    std_cis = np.std(cis_scores)
    ax.text(0.98, 0.98, f'Mean: {mean_cis:.3f}\nStd: {std_cis:.3f}',
           transform=ax.transAxes, fontsize=12,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved CIS distribution to {output_path}")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], output_path: Path):
    """Confusion matrix for causal link prediction"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        ax=ax
    )
    
    ax.set_title('Causal Link Confusion Matrix', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved confusion matrix to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument(
        "--results-file",
        required=True,
        help="Path to evaluation results JSON"
    )
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Output directory for figures"
    )
    parser.add_argument(
        "--plots",
        nargs='+',
        default=['all'],
        choices=['all', 'f1', 'pr', 'ablation', 'generalization', 'cis', 'confusion'],
        help="Which plots to generate"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(args.results_file, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Loaded results from {args.results_file}")
    
    # Generate plots
    plots_to_generate = args.plots if 'all' not in args.plots else [
        'f1', 'pr', 'ablation', 'generalization', 'cis', 'confusion'
    ]
    
    if 'f1' in plots_to_generate and 'individual_metrics' in results:
        plot_f1_comparison(
            results['individual_metrics'],
            output_dir / 'f1_comparison.png'
        )
    
    if 'pr' in plots_to_generate and 'individual_metrics' in results:
        plot_precision_recall(
            results['individual_metrics'],
            output_dir / 'precision_recall.png'
        )
    
    if 'ablation' in plots_to_generate and 'ablation_results' in results:
        plot_component_ablation(
            results['ablation_results'],
            output_dir / 'ablation_study.png'
        )
    
    if 'generalization' in plots_to_generate and 'cross_dataset' in results:
        plot_dataset_generalization(
            results['cross_dataset'],
            output_dir / 'generalization.png'
        )
    
    if 'cis' in plots_to_generate and 'cis_scores' in results:
        plot_cis_distribution(
            results['cis_scores'],
            output_dir / 'cis_distribution.png'
        )
    
    if 'confusion' in plots_to_generate and 'confusion_matrix' in results:
        cm = np.array(results['confusion_matrix']['matrix'])
        labels = results['confusion_matrix']['labels']
        plot_confusion_matrix(
            cm,
            labels,
            output_dir / 'confusion_matrix.png'
        )
    
    logger.info(f"All figures saved to {output_dir}")
    print(f"\n✓ Visualization complete! Check {output_dir}/ for figures.")


if __name__ == "__main__":
    main()
