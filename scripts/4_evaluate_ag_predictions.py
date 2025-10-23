#!/usr/bin/env python3
"""
STEP 4: Evaluate Orion predictions using benchmark_evaluator.
Computes standard metrics (edge precision/recall/F1, entity Jaccard, Recall@K, MR)
"""

import json
import logging
import os
import sys
from typing import Dict, Any, List

sys.path.insert(0, '.')

from orion.evaluation.benchmark_evaluator import BenchmarkEvaluator
from orion.evaluation.metrics import compare_graphs
from orion.evaluation.recall_at_k import RecallAtK, compute_recall_at_k

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROUND_TRUTH_FILE = 'data/ag_50/ground_truth_graphs.json'
PREDICTIONS_FILE = 'data/ag_50/results/predictions.json'
METRICS_FILE = 'data/ag_50/results/metrics.json'

os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)


def extract_relationships_as_predictions(
    graph: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Convert graph relationships to prediction format for Recall@K computation.
    
    Format: [{'subject_bbox', 'object_bbox', 'predicate', 'confidence'}]
    """
    predictions = []
    
    relationships = graph.get('relationships', [])
    entities = graph.get('entities', {})
    
    for rel in relationships:
        try:
            subject_id = rel.get('subject')
            object_id = rel.get('object')
            
            subject = entities.get(str(subject_id), {})
            obj = entities.get(str(object_id), {})
            
            # Get a representative bbox from entities
            subject_bbox = subject.get('bboxes', {}).get(0, [0, 0, 0, 0])
            object_bbox = obj.get('bboxes', {}).get(0, [0, 0, 0, 0])
            
            if not subject_bbox or not object_bbox:
                continue
            
            pred = {
                'subject_bbox': subject_bbox,
                'object_bbox': object_bbox,
                'predicate': rel.get('predicate', 'unknown'),
                'confidence': rel.get('confidence', 0.5),
                'subject_id': subject_id,
                'object_id': object_id,
            }
            predictions.append(pred)
        except Exception as e:
            logger.debug(f"Error converting relationship: {e}")
            continue
    
    return predictions


def main():
    print("="*70)
    print("STEP 4: Evaluate Orion Predictions")
    print("="*70)
    
    # Load data
    print(f"\n1. Loading data...")
    
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"❌ Ground truth not found: {GROUND_TRUTH_FILE}")
        return False
    
    if not os.path.exists(PREDICTIONS_FILE):
        print(f"❌ Predictions not found: {PREDICTIONS_FILE}")
        return False
    
    with open(GROUND_TRUTH_FILE, 'r') as f:
        ground_truth_graphs = json.load(f)
    
    with open(PREDICTIONS_FILE, 'r') as f:
        predictions = json.load(f)
    
    print(f"   ✓ Ground truth clips: {len(ground_truth_graphs)}")
    print(f"   ✓ Predictions: {len(predictions)}")
    
    # Evaluate each clip
    print(f"\n2. Computing metrics...")
    all_metrics = {
        'dataset': 'Action Genome',
        'num_clips': len(predictions),
        'individual_results': {},
        'aggregated': {},
        'recall_at_k': {}
    }
    
    per_clip_metrics = []
    recall_at_k_metric = RecallAtK(k_values=[10, 20, 50])
    
    for clip_id, pred_graph in predictions.items():
        if clip_id not in ground_truth_graphs:
            logger.warning(f"Clip {clip_id} not in ground truth")
            continue
        
        gt_graph = ground_truth_graphs[clip_id]
        
        try:
            # Standard graph comparison metrics
            comparison = compare_graphs(pred_graph, gt_graph)
            all_metrics['individual_results'][clip_id] = comparison.to_dict()
            per_clip_metrics.append(comparison)
            
            # Recall@K metrics
            pred_rels = extract_relationships_as_predictions(pred_graph)
            gt_rels = extract_relationships_as_predictions(gt_graph)
            
            if pred_rels and gt_rels:
                recall_at_k_metric.update(pred_rels, gt_rels, iou_threshold=0.5)
        
        except Exception as e:
            logger.error(f"Error evaluating clip {clip_id}: {e}")
            continue
    
    print(f"   ✓ Evaluated {len(per_clip_metrics)} clips")
    
    # Aggregate metrics
    if per_clip_metrics:
        print(f"\n3. Aggregating standard metrics...")
        
        avg_edge_precision = sum(m.edge_precision for m in per_clip_metrics) / len(per_clip_metrics)
        avg_edge_recall = sum(m.edge_recall for m in per_clip_metrics) / len(per_clip_metrics)
        avg_edge_f1 = sum(m.edge_f1 for m in per_clip_metrics) / len(per_clip_metrics)
        
        avg_event_precision = sum(m.event_precision for m in per_clip_metrics) / len(per_clip_metrics)
        avg_event_recall = sum(m.event_recall for m in per_clip_metrics) / len(per_clip_metrics)
        avg_event_f1 = sum(m.event_f1 for m in per_clip_metrics) / len(per_clip_metrics)
        
        avg_causal_precision = sum(m.causal_precision for m in per_clip_metrics) / len(per_clip_metrics)
        avg_causal_recall = sum(m.causal_recall for m in per_clip_metrics) / len(per_clip_metrics)
        avg_causal_f1 = sum(m.causal_f1 for m in per_clip_metrics) / len(per_clip_metrics)
        
        avg_entity_jaccard = sum(m.entity_jaccard for m in per_clip_metrics) / len(per_clip_metrics)
        
        all_metrics['aggregated'] = {
            'edges': {
                'precision': round(avg_edge_precision, 4),
                'recall': round(avg_edge_recall, 4),
                'f1': round(avg_edge_f1, 4)
            },
            'events': {
                'precision': round(avg_event_precision, 4),
                'recall': round(avg_event_recall, 4),
                'f1': round(avg_event_f1, 4)
            },
            'causal': {
                'precision': round(avg_causal_precision, 4),
                'recall': round(avg_causal_recall, 4),
                'f1': round(avg_causal_f1, 4)
            },
            'entities': {
                'jaccard_similarity': round(avg_entity_jaccard, 4)
            }
        }
        
        # Compute Recall@K metrics
        print(f"\n4. Computing Recall@K metrics...")
        recall_metrics = recall_at_k_metric.compute()
        all_metrics['recall_at_k'] = {
            'R@10': round(recall_metrics.get('R@10', 0), 2),
            'R@20': round(recall_metrics.get('R@20', 0), 2),
            'R@50': round(recall_metrics.get('R@50', 0), 2),
            'mR': round(recall_metrics.get('mR', 0), 2),
            'MR': round(recall_metrics.get('MR', 0), 2),
        }
        
        if 'per_category' in recall_metrics:
            all_metrics['recall_at_k']['per_category'] = recall_metrics['per_category']
    
    # Save metrics
    print(f"\n5. Saving results...")
    with open(METRICS_FILE, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Print summary
    agg = all_metrics.get('aggregated', {})
    r_at_k = all_metrics.get('recall_at_k', {})
    
    print(f"\n" + "="*70)
    print(f"EVALUATION RESULTS")
    print(f"="*70)
    print(f"""
STANDARD METRICS:

Relationship (Edge) Detection:
  Precision: {agg.get('edges', {}).get('precision', 0):.4f}
  Recall: {agg.get('edges', {}).get('recall', 0):.4f}
  F1-Score: {agg.get('edges', {}).get('f1', 0):.4f}

Event Detection:
  Precision: {agg.get('events', {}).get('precision', 0):.4f}
  Recall: {agg.get('events', {}).get('recall', 0):.4f}
  F1-Score: {agg.get('events', {}).get('f1', 0):.4f}

Causal Link Detection:
  Precision: {agg.get('causal', {}).get('precision', 0):.4f}
  Recall: {agg.get('causal', {}).get('recall', 0):.4f}
  F1-Score: {agg.get('causal', {}).get('f1', 0):.4f}

Entity Detection:
  Jaccard Similarity: {agg.get('entities', {}).get('jaccard_similarity', 0):.4f}

RECALL@K METRICS (HyperGLM Protocol):
  R@10: {r_at_k.get('R@10', 0):.2f}%
  R@20: {r_at_k.get('R@20', 0):.2f}%
  R@50: {r_at_k.get('R@50', 0):.2f}%
  mR (Mean Recall): {r_at_k.get('mR', 0):.2f}%
  MR (Mean Rank): {r_at_k.get('MR', 0):.2f}

Summary:
  Clips evaluated: {len(per_clip_metrics)}/{all_metrics['num_clips']}

✓ Detailed metrics saved to: {METRICS_FILE}
""")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
