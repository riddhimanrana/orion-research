"""
Baseline Comparison Evaluator

Compares VoT baseline predictions against Orion and ground truth,
highlighting the effectiveness of structured reasoning vs LLM-only captions.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class BaselineMetrics:
    """Metrics comparing baseline vs Orion"""
    pipeline_name: str
    entity_precision: float = 0.0
    entity_recall: float = 0.0
    entity_f1: float = 0.0
    
    rel_precision: float = 0.0
    rel_recall: float = 0.0
    rel_f1: float = 0.0
    
    event_precision: float = 0.0
    event_recall: float = 0.0
    event_f1: float = 0.0
    
    causal_precision: float = 0.0
    causal_recall: float = 0.0
    causal_f1: float = 0.0
    
    avg_confidence: float = 0.0  # Average prediction confidence
    entity_continuity: float = 0.0  # How well entities are maintained across frames
    causal_chain_completeness: float = 0.0  # How complete causal chains are
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline": self.pipeline_name,
            "entities": {
                "precision": round(self.entity_precision, 4),
                "recall": round(self.entity_recall, 4),
                "f1": round(self.entity_f1, 4),
            },
            "relationships": {
                "precision": round(self.rel_precision, 4),
                "recall": round(self.rel_recall, 4),
                "f1": round(self.rel_f1, 4),
            },
            "events": {
                "precision": round(self.event_precision, 4),
                "recall": round(self.event_recall, 4),
                "f1": round(self.event_f1, 4),
            },
            "causal": {
                "precision": round(self.causal_precision, 4),
                "recall": round(self.causal_recall, 4),
                "f1": round(self.causal_f1, 4),
            },
            "confidence": round(self.avg_confidence, 4),
            "entity_continuity": round(self.entity_continuity, 4),
            "causal_chain_completeness": round(self.causal_chain_completeness, 4),
        }


class BaselineComparator:
    """Compare baseline (VoT) vs Orion predictions"""
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
    
    def compute_metrics(
        self, 
        ground_truth: Dict[str, Any],
        predictions: Dict[str, Any]
    ) -> BaselineMetrics:
        """Compute comparison metrics
        
        Args:
            ground_truth: Ground truth graph
            predictions: Predicted graph
            
        Returns:
            BaselineMetrics object
        """
        metrics = BaselineMetrics(pipeline_name=predictions.get("pipeline", "unknown"))
        
        # Extract entities
        gt_entities = ground_truth.get("entities", {})
        pred_entities = predictions.get("entities", {})
        
        # Entity matching
        entity_matches = self._match_entities(gt_entities, pred_entities)
        tp = len(entity_matches)
        fp = len(pred_entities) - tp
        fn = len(gt_entities) - tp
        
        metrics.entity_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics.entity_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics.entity_f1 = 2 * metrics.entity_precision * metrics.entity_recall / (
            metrics.entity_precision + metrics.entity_recall + 1e-6
        )
        
        # Relationship metrics
        gt_rels = ground_truth.get("relationships", [])
        pred_rels = predictions.get("relationships", [])
        
        tp_rel = 0
        for pred_rel in pred_rels:
            for gt_rel in gt_rels:
                if self._relationships_match(pred_rel, gt_rel, entity_matches):
                    tp_rel += 1
                    break
        
        fp_rel = len(pred_rels) - tp_rel
        fn_rel = len(gt_rels) - tp_rel
        
        metrics.rel_precision = tp_rel / (tp_rel + fp_rel) if (tp_rel + fp_rel) > 0 else 0.0
        metrics.rel_recall = tp_rel / (tp_rel + fn_rel) if (tp_rel + fn_rel) > 0 else 0.0
        metrics.rel_f1 = 2 * metrics.rel_precision * metrics.rel_recall / (
            metrics.rel_precision + metrics.rel_recall + 1e-6
        )
        
        # Event metrics
        gt_events = ground_truth.get("events", [])
        pred_events = predictions.get("events", [])
        
        tp_event = 0
        for pred_event in pred_events:
            for gt_event in gt_events:
                if self._events_match(pred_event, gt_event):
                    tp_event += 1
                    break
        
        fp_event = len(pred_events) - tp_event
        fn_event = len(gt_events) - tp_event
        
        metrics.event_precision = tp_event / (tp_event + fp_event) if (tp_event + fp_event) > 0 else 0.0
        metrics.event_recall = tp_event / (tp_event + fn_event) if (tp_event + fn_event) > 0 else 0.0
        metrics.event_f1 = 2 * metrics.event_precision * metrics.event_recall / (
            metrics.event_precision + metrics.event_recall + 1e-6
        )
        
        # Causal metrics
        gt_causal = ground_truth.get("causal_links", [])
        pred_causal = predictions.get("causal_links", [])
        
        tp_causal = 0
        for pred_link in pred_causal:
            for gt_link in gt_causal:
                if self._causal_match(pred_link, gt_link, entity_matches):
                    tp_causal += 1
                    break
        
        fp_causal = len(pred_causal) - tp_causal
        fn_causal = len(gt_causal) - tp_causal
        
        metrics.causal_precision = tp_causal / (tp_causal + fp_causal) if (tp_causal + fp_causal) > 0 else 0.0
        metrics.causal_recall = tp_causal / (tp_causal + fn_causal) if (tp_causal + fn_causal) > 0 else 0.0
        metrics.causal_f1 = 2 * metrics.causal_precision * metrics.causal_recall / (
            metrics.causal_precision + metrics.causal_recall + 1e-6
        )
        
        # Confidence
        all_confidences = []
        for rel in pred_rels:
            all_confidences.append(rel.get("confidence", 0.5))
        for event in pred_events:
            all_confidences.append(event.get("confidence", 0.5))
        
        metrics.avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
        
        # Entity continuity (how consistent entity presence across time)
        metrics.entity_continuity = self._compute_entity_continuity(pred_entities)
        
        # Causal chain completeness
        metrics.causal_chain_completeness = self._compute_causal_completeness(
            gt_causal, pred_causal, entity_matches
        )
        
        return metrics
    
    def _match_entities(
        self,
        gt_entities: Dict[str, Any],
        pred_entities: Dict[str, Any]
    ) -> Dict[str, str]:
        """Match predicted entities to ground truth
        
        Returns:
            Dict mapping pred_id -> gt_id
        """
        matches = {}
        used_gt = set()
        
        for pred_id, pred_ent in pred_entities.items():
            best_match = None
            best_score = 0.0
            
            for gt_id, gt_ent in gt_entities.items():
                if gt_id in used_gt:
                    continue
                
                # Simple class-based matching
                score = 1.0 if pred_ent.get("class") == gt_ent.get("class") else 0.0
                
                if score > best_score:
                    best_score = score
                    best_match = gt_id
            
            if best_match and best_score >= 0.5:
                matches[pred_id] = best_match
                used_gt.add(best_match)
        
        return matches
    
    def _relationships_match(
        self,
        pred_rel: Dict[str, Any],
        gt_rel: Dict[str, Any],
        entity_matches: Dict[str, str]
    ) -> bool:
        """Check if two relationships match"""
        pred_subj = pred_rel.get("subject")
        pred_obj = pred_rel.get("object")
        
        if pred_subj not in entity_matches or pred_obj not in entity_matches:
            return False
        
        mapped_subj = entity_matches[pred_subj]
        mapped_obj = entity_matches[pred_obj]
        
        gt_subj = gt_rel.get("subject")
        gt_obj = gt_rel.get("object")
        
        # Match subject/object and predicate type
        return (
            mapped_subj == gt_subj and
            mapped_obj == gt_obj and
            pred_rel.get("predicate") == gt_rel.get("predicate")
        )
    
    def _events_match(self, pred_event: Dict[str, Any], gt_event: Dict[str, Any]) -> bool:
        """Check if two events match"""
        # Match by temporal overlap
        pred_start = pred_event.get("start_frame", 0)
        pred_end = pred_event.get("end_frame", 0)
        gt_start = gt_event.get("start_frame", 0)
        gt_end = gt_event.get("end_frame", 0)
        
        intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
        union = max(pred_end, gt_end) - min(pred_start, gt_start)
        
        iou = intersection / union if union > 0 else 0.0
        
        return iou >= self.iou_threshold
    
    def _causal_match(
        self,
        pred_link: Dict[str, Any],
        gt_link: Dict[str, Any],
        entity_matches: Dict[str, str]
    ) -> bool:
        """Check if causal links match"""
        pred_cause = pred_link.get("cause")
        pred_effect = pred_link.get("effect")
        
        if pred_cause not in entity_matches or pred_effect not in entity_matches:
            return False
        
        mapped_cause = entity_matches[pred_cause]
        mapped_effect = entity_matches[pred_effect]
        
        gt_cause = gt_link.get("cause")
        gt_effect = gt_link.get("effect")
        
        return mapped_cause == gt_cause and mapped_effect == gt_effect
    
    def _compute_entity_continuity(self, entities: Dict[str, Any]) -> float:
        """Compute how continuous entities are across time
        
        Higher = more consistent entity tracking across frames
        """
        if not entities:
            return 0.0
        
        continuity_scores = []
        
        for ent_id, ent in entities.items():
            frames = ent.get("frames", [])
            if len(frames) < 2:
                continuity_scores.append(0.0)
                continue
            
            # Compute gaps in frame sequences
            frames_sorted = sorted(frames)
            gaps = []
            for i in range(1, len(frames_sorted)):
                gap = frames_sorted[i] - frames_sorted[i-1]
                gaps.append(gap)
            
            # Penalize large gaps (indicates dropped tracking)
            max_gap = max(gaps) if gaps else 0
            avg_gap = np.mean(gaps) if gaps else 0
            
            # Continuity: lower gap = higher continuity
            continuity = 1.0 / (1.0 + avg_gap / 10.0)
            continuity_scores.append(continuity)
        
        return np.mean(continuity_scores) if continuity_scores else 0.0
    
    def _compute_causal_completeness(
        self,
        gt_causal: List[Dict],
        pred_causal: List[Dict],
        entity_matches: Dict[str, str]
    ) -> float:
        """Compute how complete causal chains are
        
        Measures if chains form connected sequences
        """
        if not gt_causal:
            return 1.0 if not pred_causal else 0.0
        
        if not pred_causal:
            return 0.0
        
        # Build graphs
        gt_graph = {}
        for link in gt_causal:
            cause = link.get("cause")
            effect = link.get("effect")
            if cause not in gt_graph:
                gt_graph[cause] = []
            gt_graph[cause].append(effect)
        
        pred_graph = {}
        for link in pred_causal:
            cause = link.get("cause")
            effect = link.get("effect")
            if cause not in pred_graph:
                pred_graph[cause] = []
            pred_graph[cause].append(effect)
        
        # Measure coverage
        matched_links = 0
        for pred_cause, pred_effects in pred_graph.items():
            if pred_cause in entity_matches:
                mapped_cause = entity_matches[pred_cause]
                if mapped_cause in gt_graph:
                    gt_effects = gt_graph[mapped_cause]
                    for pred_effect in pred_effects:
                        if pred_effect in entity_matches:
                            mapped_effect = entity_matches[pred_effect]
                            if mapped_effect in gt_effects:
                                matched_links += 1
        
        total_pred_links = sum(len(e) for e in pred_graph.values())
        
        return matched_links / total_pred_links if total_pred_links > 0 else 0.0


def print_baseline_comparison(
    orion_metrics: BaselineMetrics,
    baseline_metrics: BaselineMetrics
):
    """Print comparison table between Orion and baseline"""
    
    console.print("\n[bold cyan]Baseline Comparison: Orion vs VoT[/bold cyan]\n")
    
    table = Table(title="Model Performance Comparison", show_header=True)
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Orion", style="green", justify="right", width=12)
    table.add_column("VoT Baseline", style="yellow", justify="right", width=12)
    table.add_column("Improvement", style="magenta", justify="right", width=12)
    
    # Entity metrics
    entity_improvement = (
        (orion_metrics.entity_f1 - baseline_metrics.entity_f1) / 
        (baseline_metrics.entity_f1 + 1e-6)
    ) * 100
    table.add_row(
        "Entity F1",
        f"{orion_metrics.entity_f1:.3f}",
        f"{baseline_metrics.entity_f1:.3f}",
        f"+{entity_improvement:.1f}%" if entity_improvement > 0 else f"{entity_improvement:.1f}%"
    )
    
    # Relationship metrics
    rel_improvement = (
        (orion_metrics.rel_f1 - baseline_metrics.rel_f1) / 
        (baseline_metrics.rel_f1 + 1e-6)
    ) * 100
    table.add_row(
        "Relationship F1",
        f"{orion_metrics.rel_f1:.3f}",
        f"{baseline_metrics.rel_f1:.3f}",
        f"+{rel_improvement:.1f}%" if rel_improvement > 0 else f"{rel_improvement:.1f}%"
    )
    
    # Event metrics
    event_improvement = (
        (orion_metrics.event_f1 - baseline_metrics.event_f1) / 
        (baseline_metrics.event_f1 + 1e-6)
    ) * 100
    table.add_row(
        "Event F1",
        f"{orion_metrics.event_f1:.3f}",
        f"{baseline_metrics.event_f1:.3f}",
        f"+{event_improvement:.1f}%" if event_improvement > 0 else f"{event_improvement:.1f}%"
    )
    
    # Causal metrics
    causal_improvement = (
        (orion_metrics.causal_f1 - baseline_metrics.causal_f1) / 
        (baseline_metrics.causal_f1 + 1e-6)
    ) * 100
    table.add_row(
        "Causal F1",
        f"{orion_metrics.causal_f1:.3f}",
        f"{baseline_metrics.causal_f1:.3f}",
        f"+{causal_improvement:.1f}%" if causal_improvement > 0 else f"{causal_improvement:.1f}%"
    )
    
    # Entity continuity
    continuity_improvement = (
        (orion_metrics.entity_continuity - baseline_metrics.entity_continuity) / 
        (baseline_metrics.entity_continuity + 1e-6)
    ) * 100
    table.add_row(
        "Entity Continuity",
        f"{orion_metrics.entity_continuity:.3f}",
        f"{baseline_metrics.entity_continuity:.3f}",
        f"+{continuity_improvement:.1f}%" if continuity_improvement > 0 else f"{continuity_improvement:.1f}%"
    )
    
    # Causal chain completeness
    chain_improvement = (
        (orion_metrics.causal_chain_completeness - baseline_metrics.causal_chain_completeness) / 
        (baseline_metrics.causal_chain_completeness + 1e-6)
    ) * 100
    table.add_row(
        "Causal Chain Completeness",
        f"{orion_metrics.causal_chain_completeness:.3f}",
        f"{baseline_metrics.causal_chain_completeness:.3f}",
        f"+{chain_improvement:.1f}%" if chain_improvement > 0 else f"{chain_improvement:.1f}%"
    )
    
    console.print(table)
    
    # Print insights
    console.print("\n[bold]Key Findings:[/bold]")
    
    if entity_improvement > 20:
        console.print(f"  • Orion provides {entity_improvement:.0f}% better entity detection")
        console.print(f"    → Demonstrates value of structured embedding-based tracking")
    
    if rel_improvement > 20:
        console.print(f"  • Orion provides {rel_improvement:.0f}% better relationship extraction")
        console.print(f"    → Shows importance of semantic uplift over free-form reasoning")
    
    if continuity_improvement > 20:
        console.print(f"  • Orion provides {continuity_improvement:.0f}% better entity continuity")
        console.print(f"    → Proves value of temporal tracking vs caption-only approach")
    
    if causal_improvement > 20:
        console.print(f"  • Orion provides {causal_improvement:.0f}% better causal understanding")
        console.print(f"    → Validates structured causal inference vs LLM reasoning")


__all__ = ['BaselineComparator', 'BaselineMetrics', 'print_baseline_comparison']
