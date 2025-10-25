"""
Evaluation Framework for Causal Understanding and Event Linking
================================================================

This module provides a comprehensive evaluation pipeline for benchmarking
Orion's causal understanding, event detection, and relationship extraction
against standard video scene graph datasets.

Supported Datasets:
- Action Genome: Dense spatio-temporal scene graphs with actions
- VSGR (Video Scene Graph): Multi-object tracking with relationships
- PVSG (Panoptic Video Scene Graph): Panoptic segmentation + scene graphs

Key Metrics:
1. Relationship Detection:
   - Precision, Recall, F1 for spatial relationships
   - Per-relationship-type breakdown
   
2. Event/Action Detection:
   - Event detection F1 score
   - Temporal IoU (tIoU) for event boundaries
   - Event classification accuracy
   
3. Causal Understanding:
   - Causal link precision/recall
   - Temporal ordering accuracy
   - Causal chain completeness

4. Scene Graph Quality:
   - Graph Edit Distance (GED)
   - Entity matching accuracy
   - Relationship type accuracy

Author: Orion Research Team
Date: October 2025
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track

from ..knowledge_graph import KnowledgeGraphBuilder
from .metrics import GraphMetrics, ComparisonMetrics

logger = logging.getLogger("orion.evaluation")
console = Console()


@dataclass
class GroundTruthGraph:
    """
    Standardized representation of ground truth scene graph
    
    This unifies different dataset formats into a common structure
    that can be compared against Orion's output.
    """
    video_id: str
    entities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    causal_links: List[Dict[str, Any]] = field(default_factory=list)
    
    # Temporal info
    fps: float = 30.0
    total_frames: int = 0
    
    # Dataset-specific metadata
    dataset_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionGraph:
    """
    Orion's predicted scene graph in standardized format
    """
    video_id: str
    entities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    causal_links: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EvaluationMetrics:
    """Comprehensive metrics for a single video evaluation"""
    video_id: str
    
    # Relationship metrics
    rel_precision: float = 0.0
    rel_recall: float = 0.0
    rel_f1: float = 0.0
    rel_per_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Event metrics
    event_precision: float = 0.0
    event_recall: float = 0.0
    event_f1: float = 0.0
    event_tiou: float = 0.0  # Temporal IoU
    
    # Causal metrics
    causal_precision: float = 0.0
    causal_recall: float = 0.0
    causal_f1: float = 0.0
    causal_temporal_accuracy: float = 0.0
    
    # Entity metrics
    entity_precision: float = 0.0
    entity_recall: float = 0.0
    entity_class_accuracy: float = 0.0
    
    # Overall graph quality
    graph_edit_distance: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "relationships": {
                "precision": round(self.rel_precision, 3),
                "recall": round(self.rel_recall, 3),
                "f1": round(self.rel_f1, 3),
                "per_type": self.rel_per_type,
            },
            "events": {
                "precision": round(self.event_precision, 3),
                "recall": round(self.event_recall, 3),
                "f1": round(self.event_f1, 3),
                "temporal_iou": round(self.event_tiou, 3),
            },
            "causal": {
                "precision": round(self.causal_precision, 3),
                "recall": round(self.causal_recall, 3),
                "f1": round(self.causal_f1, 3),
                "temporal_accuracy": round(self.causal_temporal_accuracy, 3),
            },
            "entities": {
                "precision": round(self.entity_precision, 3),
                "recall": round(self.entity_recall, 3),
                "class_accuracy": round(self.entity_class_accuracy, 3),
            },
            "graph_edit_distance": round(self.graph_edit_distance, 2),
        }


class BenchmarkEvaluator:
    """
    Evaluates Orion's predictions against ground truth from benchmarks
    """
    
    def __init__(self, iou_threshold: float = 0.5, tiou_threshold: float = 0.3):
        """
        Args:
            iou_threshold: IoU threshold for entity matching
            tiou_threshold: Temporal IoU threshold for event matching
        """
        self.iou_threshold = iou_threshold
        self.tiou_threshold = tiou_threshold
    
    def evaluate(
        self,
        ground_truth: GroundTruthGraph,
        prediction: PredictionGraph,
    ) -> EvaluationMetrics:
        """
        Compare prediction against ground truth
        
        Returns:
            Comprehensive evaluation metrics
        """
        metrics = EvaluationMetrics(video_id=ground_truth.video_id)
        
        # 1. Entity matching
        entity_matches = self._match_entities(ground_truth, prediction)
        metrics.entity_precision, metrics.entity_recall, metrics.entity_class_accuracy = \
            self._compute_entity_metrics(ground_truth, prediction, entity_matches)
        
        # 2. Relationship evaluation
        rel_metrics = self._evaluate_relationships(
            ground_truth, prediction, entity_matches
        )
        metrics.rel_precision = rel_metrics["precision"]
        metrics.rel_recall = rel_metrics["recall"]
        metrics.rel_f1 = rel_metrics["f1"]
        metrics.rel_per_type = rel_metrics["per_type"]
        
        # 3. Event evaluation
        event_metrics = self._evaluate_events(ground_truth, prediction, entity_matches)
        metrics.event_precision = event_metrics["precision"]
        metrics.event_recall = event_metrics["recall"]
        metrics.event_f1 = event_metrics["f1"]
        metrics.event_tiou = event_metrics["tiou"]
        
        # 4. Causal link evaluation
        causal_metrics = self._evaluate_causal_links(
            ground_truth, prediction, entity_matches
        )
        metrics.causal_precision = causal_metrics["precision"]
        metrics.causal_recall = causal_metrics["recall"]
        metrics.causal_f1 = causal_metrics["f1"]
        metrics.causal_temporal_accuracy = causal_metrics["temporal_accuracy"]
        
        # 5. Graph-level metrics
        metrics.graph_edit_distance = self._compute_graph_edit_distance(
            ground_truth, prediction
        )
        
        return metrics
    
    def _match_entities(
        self,
        gt: GroundTruthGraph,
        pred: PredictionGraph,
    ) -> Dict[str, str]:
        """
        Match predicted entities to ground truth entities
        
        Uses greedy matching based on:
        - Temporal overlap
        - Spatial IoU
        - Class similarity
        
        Returns:
            Dictionary mapping pred_id -> gt_id
        """
        matches = {}
        
        # Build cost matrix
        pred_ids = list(pred.entities.keys())
        gt_ids = list(gt.entities.keys())
        
        cost_matrix = np.zeros((len(pred_ids), len(gt_ids)))
        
        for i, pred_id in enumerate(pred_ids):
            pred_ent = pred.entities[pred_id]
            for j, gt_id in enumerate(gt_ids):
                gt_ent = gt.entities[gt_id]
                
                # Compute similarity score
                score = self._entity_similarity(pred_ent, gt_ent)
                cost_matrix[i, j] = score
        
        # Greedy matching (can be improved with Hungarian algorithm)
        used_gt = set()
        for i in range(len(pred_ids)):
            if cost_matrix[i].max() >= self.iou_threshold:
                j = cost_matrix[i].argmax()
                if j not in used_gt:
                    matches[pred_ids[i]] = gt_ids[j]
                    used_gt.add(j)
        
        return matches
    
    def _entity_similarity(self, pred_ent: Dict, gt_ent: Dict) -> float:
        """
        Compute similarity between predicted and ground truth entity
        
        Combines:
        - Class match (0.4 weight)
        - Temporal overlap (0.3 weight)
        - Spatial IoU (0.3 weight)
        """
        score = 0.0
        
        # Class match
        if pred_ent.get("class") == gt_ent.get("class"):
            score += 0.4
        
        # Temporal overlap
        pred_frames = set(pred_ent.get("frames", []))
        gt_frames = set(gt_ent.get("frames", []))
        if pred_frames and gt_frames:
            temporal_iou = len(pred_frames & gt_frames) / len(pred_frames | gt_frames)
            score += 0.3 * temporal_iou
        
        # Spatial IoU (average across frames)
        pred_bboxes = pred_ent.get("bboxes", {})
        gt_bboxes = gt_ent.get("bboxes", {})
        common_frames = set(pred_bboxes.keys()) & set(gt_bboxes.keys())
        if common_frames:
            ious = []
            for frame in common_frames:
                iou = self._bbox_iou(pred_bboxes[frame], gt_bboxes[frame])
                ious.append(iou)
            score += 0.3 * np.mean(ious)
        
        return score
    
    @staticmethod
    def _bbox_iou(box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two bounding boxes [x1, y1, x2, y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_entity_metrics(
        self,
        gt: GroundTruthGraph,
        pred: PredictionGraph,
        matches: Dict[str, str],
    ) -> Tuple[float, float, float]:
        """
        Compute entity-level metrics
        
        Returns:
            (precision, recall, class_accuracy)
        """
        tp = len(matches)
        fp = len(pred.entities) - tp
        fn = len(gt.entities) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Class accuracy for matched entities
        correct_class = 0
        for pred_id, gt_id in matches.items():
            if pred.entities[pred_id].get("class") == gt.entities[gt_id].get("class"):
                correct_class += 1
        
        class_accuracy = correct_class / len(matches) if matches else 0.0
        
        return precision, recall, class_accuracy
    
    def _evaluate_relationships(
        self,
        gt: GroundTruthGraph,
        pred: PredictionGraph,
        entity_matches: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Evaluate relationship detection
        
        Returns:
            Dict with precision, recall, f1, and per-type metrics
        """
        # Map relationships to matched entities
        gt_rels_mapped = self._map_relationships(gt.relationships, {v: k for k, v in entity_matches.items()})
        pred_rels_mapped = self._map_relationships(pred.relationships, entity_matches)
        
        # Convert to tuple format (subj, pred, obj)
        gt_rel_set = set()
        for rel in gt_rels_mapped:
            gt_rel_set.add((rel["subject"], rel["predicate"], rel["object"]))
        
        pred_rel_set = set()
        for rel in pred_rels_mapped:
            pred_rel_set.add((rel["subject"], rel["predicate"], rel["object"]))
        
        # Compute metrics
        tp = len(gt_rel_set & pred_rel_set)
        fp = len(pred_rel_set - gt_rel_set)
        fn = len(gt_rel_set - pred_rel_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Per-type metrics
        per_type = self._compute_per_type_metrics(gt.relationships, pred.relationships, entity_matches)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "per_type": per_type,
        }
    
    @staticmethod
    def _map_relationships(
        relationships: List[Dict],
        entity_map: Dict[str, str],
    ) -> List[Dict]:
        """Map relationships through entity matching"""
        mapped = []
        for rel in relationships:
            subj = rel.get("subject")
            obj = rel.get("object")
            
            if subj in entity_map and obj in entity_map:
                mapped.append({
                    "subject": entity_map[subj],
                    "predicate": rel.get("predicate", "related_to"),
                    "object": entity_map[obj],
                })
        
        return mapped
    
    def _compute_per_type_metrics(
        self,
        gt_rels: List[Dict],
        pred_rels: List[Dict],
        entity_matches: Dict[str, str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute precision/recall/f1 for each relationship type"""
        per_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        
        # Group by type
        gt_by_type = defaultdict(set)
        for rel in gt_rels:
            rel_type = rel.get("predicate", "related_to")
            subj = rel.get("subject")
            obj = rel.get("object")
            # Reverse map through entity_matches
            reverse_map = {v: k for k, v in entity_matches.items()}
            if subj in reverse_map and obj in reverse_map:
                gt_by_type[rel_type].add((reverse_map[subj], reverse_map[obj]))
        
        pred_by_type = defaultdict(set)
        for rel in pred_rels:
            rel_type = rel.get("predicate", "related_to")
            subj = rel.get("subject")
            obj = rel.get("object")
            if subj in entity_matches and obj in entity_matches:
                pred_by_type[rel_type].add((entity_matches[subj], entity_matches[obj]))
        
        # Compute metrics per type
        all_types = set(gt_by_type.keys()) | set(pred_by_type.keys())
        result = {}
        
        for rel_type in all_types:
            gt_set = gt_by_type[rel_type]
            pred_set = pred_by_type[rel_type]
            
            tp = len(gt_set & pred_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            result[rel_type] = {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
            }
        
        return result
    
    def _evaluate_events(
        self,
        gt: GroundTruthGraph,
        pred: PredictionGraph,
        entity_matches: Dict[str, str],
    ) -> Dict[str, float]:
        """
        Evaluate event detection
        
        Uses temporal IoU for matching events
        """
        # Match events based on temporal overlap and involved entities
        matched_events = self._match_events(gt.events, pred.events, entity_matches)
        
        tp = len(matched_events)
        fp = len(pred.events) - tp
        fn = len(gt.events) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Compute average temporal IoU for matched events
        if matched_events:
            tious = [self._temporal_iou(gt.events[gt_idx], pred.events[pred_idx])
                     for pred_idx, gt_idx in matched_events]
            avg_tiou = np.mean(tious)
        else:
            avg_tiou = 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tiou": avg_tiou,
        }
    
    def _match_events(
        self,
        gt_events: List[Dict],
        pred_events: List[Dict],
        entity_matches: Dict[str, str],
    ) -> List[Tuple[int, int]]:
        """Match predicted events to ground truth events"""
        matches = []
        used_gt = set()
        
        for i, pred_event in enumerate(pred_events):
            best_match = None
            best_score = 0.0
            
            for j, gt_event in enumerate(gt_events):
                if j in used_gt:
                    continue
                
                # Compute match score based on:
                # 1. Temporal IoU
                # 2. Event type match
                # 3. Involved entities overlap
                
                tiou = self._temporal_iou(pred_event, gt_event)
                
                type_match = 1.0 if pred_event.get("type") == gt_event.get("type") else 0.0
                
                # Entity overlap
                pred_entities = set(pred_event.get("entities", []))
                gt_entities = set(gt_event.get("entities", []))
                
                # Map predicted entities to ground truth space
                mapped_pred_entities = {entity_matches.get(e, e) for e in pred_entities}
                
                if mapped_pred_entities and gt_entities:
                    entity_overlap = len(mapped_pred_entities & gt_entities) / len(mapped_pred_entities | gt_entities)
                else:
                    entity_overlap = 0.0
                
                # Combined score
                score = 0.5 * tiou + 0.3 * type_match + 0.2 * entity_overlap
                
                if score > best_score and tiou >= self.tiou_threshold:
                    best_score = score
                    best_match = j
            
            if best_match is not None:
                matches.append((i, best_match))
                used_gt.add(best_match)
        
        return matches
    
    @staticmethod
    def _temporal_iou(event1: Dict, event2: Dict) -> float:
        """Compute temporal IoU between two events"""
        start1 = event1.get("start_frame", 0)
        end1 = event1.get("end_frame", 0)
        start2 = event2.get("start_frame", 0)
        end2 = event2.get("end_frame", 0)
        
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        
        if intersection_end < intersection_start:
            return 0.0
        
        intersection = intersection_end - intersection_start
        union = (end1 - start1) + (end2 - start2) - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _evaluate_causal_links(
        self,
        gt: GroundTruthGraph,
        pred: PredictionGraph,
        entity_matches: Dict[str, str],
    ) -> Dict[str, float]:
        """
        Evaluate causal link detection
        
        Considers:
        - Correct cause/effect entity pairing
        - Temporal ordering (cause before effect)
        """
        # Map causal links through entity matching
        gt_causal_mapped = self._map_causal_links(gt.causal_links, {v: k for k, v in entity_matches.items()})
        pred_causal_mapped = self._map_causal_links(pred.causal_links, entity_matches)
        
        # Convert to set of (cause, effect) tuples
        gt_causal_set = {(c["cause"], c["effect"]) for c in gt_causal_mapped}
        pred_causal_set = {(c["cause"], c["effect"]) for c in pred_causal_mapped}
        
        tp = len(gt_causal_set & pred_causal_set)
        fp = len(pred_causal_set - gt_causal_set)
        fn = len(gt_causal_set - pred_causal_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Temporal ordering accuracy
        temporal_correct = 0
        temporal_total = 0
        for pred_link in pred_causal_mapped:
            cause_id = pred_link["cause"]
            effect_id = pred_link["effect"]
            
            # Find corresponding GT link
            for gt_link in gt_causal_mapped:
                if gt_link["cause"] == cause_id and gt_link["effect"] == effect_id:
                    temporal_total += 1
                    # Check if temporal ordering is preserved
                    pred_time_diff = pred_link.get("time_diff", 0)
                    gt_time_diff = gt_link.get("time_diff", 0)
                    
                    if pred_time_diff > 0 and gt_time_diff > 0:
                        temporal_correct += 1
                    break
        
        temporal_accuracy = temporal_correct / temporal_total if temporal_total > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "temporal_accuracy": temporal_accuracy,
        }
    
    @staticmethod
    def _map_causal_links(
        causal_links: List[Dict],
        entity_map: Dict[str, str],
    ) -> List[Dict]:
        """Map causal links through entity matching"""
        mapped = []
        for link in causal_links:
            cause = link.get("cause")
            effect = link.get("effect")
            
            if cause in entity_map and effect in entity_map:
                mapped.append({
                    "cause": entity_map[cause],
                    "effect": entity_map[effect],
                    "time_diff": link.get("time_diff", 0),
                })
        
        return mapped
    
    def _compute_graph_edit_distance(
        self,
        gt: GroundTruthGraph,
        pred: PredictionGraph,
    ) -> float:
        """
        Approximate Graph Edit Distance
        
        Simplified metric based on node and edge differences
        """
        # Node differences
        node_diff = abs(len(gt.entities) - len(pred.entities))
        
        # Edge differences
        edge_diff = abs(len(gt.relationships) - len(pred.relationships))
        
        # Event differences
        event_diff = abs(len(gt.events) - len(pred.events))
        
        # Simple weighted sum
        ged = node_diff + 0.5 * edge_diff + 0.3 * event_diff
        
        return ged


def print_evaluation_results(metrics: EvaluationMetrics, detailed: bool = True):
    """
    Pretty-print evaluation results using Rich tables
    """
    console.print(f"\n[bold cyan]Evaluation Results: {metrics.video_id}[/bold cyan]\n")
    
    # Entity metrics table
    entity_table = Table(title="Entity Detection", show_header=True)
    entity_table.add_column("Metric", style="cyan")
    entity_table.add_column("Value", style="green", justify="right")
    
    entity_table.add_row("Precision", f"{metrics.entity_precision:.3f}")
    entity_table.add_row("Recall", f"{metrics.entity_recall:.3f}")
    entity_table.add_row("Class Accuracy", f"{metrics.entity_class_accuracy:.3f}")
    
    console.print(entity_table)
    
    # Relationship metrics table
    rel_table = Table(title="Relationship Detection", show_header=True)
    rel_table.add_column("Metric", style="cyan")
    rel_table.add_column("Value", style="green", justify="right")
    
    rel_table.add_row("Precision", f"{metrics.rel_precision:.3f}")
    rel_table.add_row("Recall", f"{metrics.rel_recall:.3f}")
    rel_table.add_row("F1 Score", f"{metrics.rel_f1:.3f}")
    
    console.print(rel_table)
    
    # Per-type breakdown if detailed
    if detailed and metrics.rel_per_type:
        type_table = Table(title="Per-Type Relationship Metrics", show_header=True)
        type_table.add_column("Relationship Type", style="cyan")
        type_table.add_column("Precision", style="green", justify="right")
        type_table.add_column("Recall", style="green", justify="right")
        type_table.add_column("F1", style="green", justify="right")
        
        for rel_type, type_metrics in metrics.rel_per_type.items():
            type_table.add_row(
                rel_type,
                f"{type_metrics['precision']:.3f}",
                f"{type_metrics['recall']:.3f}",
                f"{type_metrics['f1']:.3f}",
            )
        
        console.print(type_table)
    
    # Event metrics table
    event_table = Table(title="Event Detection", show_header=True)
    event_table.add_column("Metric", style="cyan")
    event_table.add_column("Value", style="green", justify="right")
    
    event_table.add_row("Precision", f"{metrics.event_precision:.3f}")
    event_table.add_row("Recall", f"{metrics.event_recall:.3f}")
    event_table.add_row("F1 Score", f"{metrics.event_f1:.3f}")
    event_table.add_row("Temporal IoU", f"{metrics.event_tiou:.3f}")
    
    console.print(event_table)
    
    # Causal metrics table
    causal_table = Table(title="Causal Understanding", show_header=True)
    causal_table.add_column("Metric", style="cyan")
    causal_table.add_column("Value", style="green", justify="right")
    
    causal_table.add_row("Precision", f"{metrics.causal_precision:.3f}")
    causal_table.add_row("Recall", f"{metrics.causal_recall:.3f}")
    causal_table.add_row("F1 Score", f"{metrics.causal_f1:.3f}")
    causal_table.add_row("Temporal Accuracy", f"{metrics.causal_temporal_accuracy:.3f}")
    
    console.print(causal_table)
    
    # Overall quality
    console.print(f"\n[yellow]Graph Edit Distance:[/yellow] {metrics.graph_edit_distance:.2f}\n")


if __name__ == "__main__":
    # Example usage
    console.print("[bold cyan]Orion Evaluation Framework[/bold cyan]")
    console.print("Use this module to benchmark against Action Genome, VSGR, and other datasets")
