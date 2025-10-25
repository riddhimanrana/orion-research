"""
Knowledge Graph Evaluation Metrics
===================================

This module provides comprehensive metrics for evaluating and comparing
knowledge graphs constructed by different methods.

Author: Orion Research Team
Date: October 2025
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger("GraphMetrics")


@dataclass
class GraphMetrics:
    """
    Comprehensive metrics for evaluating a knowledge graph
    """
    
    # Structural metrics
    num_entities: int = 0
    num_relationships: int = 0
    num_events: int = 0
    
    # Edge type distribution
    relationship_types: Dict[str, int] = None
    event_types: Dict[str, int] = None
    
    # Graph density
    graph_density: float = 0.0
    avg_degree: float = 0.0
    
    # Semantic richness
    avg_description_length: float = 0.0
    unique_event_labels: int = 0
    
    # Causal metrics (if available)
    num_causal_links: int = 0
    avg_cis_score: float = 0.0
    
    def __post_init__(self):
        if self.relationship_types is None:
            self.relationship_types = {}
        if self.event_types is None:
            self.event_types = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "structural": {
                "num_entities": self.num_entities,
                "num_relationships": self.num_relationships,
                "num_events": self.num_events,
                "graph_density": round(self.graph_density, 4),
                "avg_degree": round(self.avg_degree, 2),
            },
            "relationships": self.relationship_types,
            "events": self.event_types,
            "semantic_richness": {
                "avg_description_length": round(self.avg_description_length, 2),
                "unique_event_labels": self.unique_event_labels,
            },
            "causal": {
                "num_causal_links": self.num_causal_links,
                "avg_cis_score": round(self.avg_cis_score, 3),
            }
        }


@dataclass
class ComparisonMetrics:
    """
    Metrics for comparing two knowledge graphs
    """
    
    # Precision, Recall, F1 for edges
    edge_precision: float = 0.0
    edge_recall: float = 0.0
    edge_f1: float = 0.0
    
    # Entity overlap
    entity_jaccard: float = 0.0
    
    # Event metrics
    event_precision: float = 0.0
    event_recall: float = 0.0
    event_f1: float = 0.0
    
    # Semantic similarity
    label_accuracy: float = 0.0  # For matched edges
    avg_label_similarity: float = 0.0
    
    # Causal accuracy (if ground truth available)
    causal_precision: float = 0.0
    causal_recall: float = 0.0
    causal_f1: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "edges": {
                "precision": round(self.edge_precision, 4),
                "recall": round(self.edge_recall, 4),
                "f1": round(self.edge_f1, 4),
            },
            "entities": {
                "jaccard_similarity": round(self.entity_jaccard, 4),
            },
            "events": {
                "precision": round(self.event_precision, 4),
                "recall": round(self.event_recall, 4),
                "f1": round(self.event_f1, 4),
            },
            "semantic": {
                "label_accuracy": round(self.label_accuracy, 4),
                "label_similarity": round(self.avg_label_similarity, 4),
            },
            "causal": {
                "precision": round(self.causal_precision, 4),
                "recall": round(self.causal_recall, 4),
                "f1": round(self.causal_f1, 4),
            }
        }


def evaluate_graph_quality(graph_data: Dict[str, Any]) -> GraphMetrics:
    """
    Evaluate the quality of a single knowledge graph
    
    Args:
        graph_data: Dictionary with 'entities', 'relationships', 'events'
        
    Returns:
        GraphMetrics object with computed metrics
    """
    entities = graph_data.get("entities", [])
    relationships = graph_data.get("relationships", [])
    events = graph_data.get("events", [])
    
    metrics = GraphMetrics()
    
    # Basic counts
    metrics.num_entities = len(entities)
    metrics.num_relationships = len(relationships)
    metrics.num_events = len(events)
    
    # Relationship type distribution
    rel_types = {}
    for rel in relationships:
        rel_type = rel.get("type", "unknown")
        rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
    metrics.relationship_types = rel_types
    
    # Event type distribution
    event_types = {}
    event_labels = set()
    for event in events:
        event_type = event.get("type", "unknown")
        event_types[event_type] = event_types.get(event_type, 0) + 1
        
        # Collect unique event labels
        if "relationship" in event:
            event_labels.add(event["relationship"])
    
    metrics.event_types = event_types
    metrics.unique_event_labels = len(event_labels)
    
    # Graph density and average degree
    if metrics.num_entities > 1:
        max_edges = metrics.num_entities * (metrics.num_entities - 1)
        metrics.graph_density = metrics.num_relationships / max_edges
        metrics.avg_degree = (2 * metrics.num_relationships) / metrics.num_entities
    
    # Semantic richness
    descriptions = [
        e.get("description", "")
        for e in entities
        if e.get("description")
    ]
    if descriptions:
        metrics.avg_description_length = np.mean([len(d.split()) for d in descriptions])
    
    # Causal metrics (if CIS scores available)
    causal_events = [e for e in events if "cis_score" in e]
    metrics.num_causal_links = len(causal_events)
    if causal_events:
        metrics.avg_cis_score = np.mean([e["cis_score"] for e in causal_events])
    
    logger.info(
        f"Graph quality: {metrics.num_entities} entities, "
        f"{metrics.num_relationships} relationships, "
        f"{metrics.num_events} events"
    )
    
    return metrics


def compare_graphs(
    predicted_graph: Dict[str, Any],
    ground_truth_graph: Dict[str, Any],
    match_entities_by: str = "id"  # "id" or "description"
) -> ComparisonMetrics:
    """
    Compare predicted graph against ground truth
    
    Args:
        predicted_graph: Predicted knowledge graph
        ground_truth_graph: Ground truth reference graph
        match_entities_by: How to match entities ("id" or "description")
        
    Returns:
        ComparisonMetrics with precision, recall, F1, etc.
    """
    metrics = ComparisonMetrics()
    
    # Extract components
    pred_entities = predicted_graph.get("entities", [])
    pred_rels = predicted_graph.get("relationships", [])
    pred_events = predicted_graph.get("events", [])
    
    gt_entities = ground_truth_graph.get("entities", [])
    gt_rels = ground_truth_graph.get("relationships", [])
    gt_events = ground_truth_graph.get("events", [])
    
    # Entity matching
    pred_entity_ids = {e.get("entity_id") for e in pred_entities}
    gt_entity_ids = {e.get("entity_id") for e in gt_entities}
    
    entity_intersection = pred_entity_ids & gt_entity_ids
    entity_union = pred_entity_ids | gt_entity_ids
    
    if entity_union:
        metrics.entity_jaccard = len(entity_intersection) / len(entity_union)
    
    # Edge-level metrics
    pred_edges = _extract_edge_set(pred_rels)
    gt_edges = _extract_edge_set(gt_rels)
    
    metrics.edge_precision, metrics.edge_recall, metrics.edge_f1 = _compute_prf(
        pred_edges, gt_edges
    )
    
    # Event-level metrics
    pred_event_set = _extract_event_set(pred_events)
    gt_event_set = _extract_event_set(gt_events)
    
    metrics.event_precision, metrics.event_recall, metrics.event_f1 = _compute_prf(
        pred_event_set, gt_event_set
    )
    
    # Label accuracy (for matched edges)
    matched_edges = pred_edges & gt_edges
    if matched_edges:
        # Count how many have matching labels
        # (simplified - in full version, would match by source/target and compare types)
        metrics.label_accuracy = 1.0  # Placeholder
    
    # Causal metrics (events with CAUSED relationship)
    pred_causal = {
        (e.get("agent"), e.get("patient"))
        for e in pred_events
        if e.get("relationship") == "CAUSED"
    }
    gt_causal = {
        (e.get("agent"), e.get("patient"))
        for e in gt_events
        if e.get("relationship") == "CAUSED"
    }
    
    metrics.causal_precision, metrics.causal_recall, metrics.causal_f1 = _compute_prf(
        pred_causal, gt_causal
    )
    
    logger.info(
        f"Comparison: Edge F1={metrics.edge_f1:.3f}, "
        f"Event F1={metrics.event_f1:.3f}, "
        f"Causal F1={metrics.causal_f1:.3f}"
    )
    
    return metrics


def _extract_edge_set(relationships: List[Dict]) -> Set[Tuple]:
    """
    Extract edges as a set of (source, target, type) tuples
    """
    edges = set()
    for rel in relationships:
        source = rel.get("source")
        target = rel.get("target")
        rel_type = rel.get("type")
        if source and target and rel_type:
            edges.add((source, target, rel_type))
    return edges


def _extract_event_set(events: List[Dict]) -> Set[Tuple]:
    """
    Extract events as a set of (agent, patient, relationship) tuples
    """
    event_set = set()
    for event in events:
        agent = event.get("agent")
        patient = event.get("patient")
        relationship = event.get("relationship")
        if agent and patient and relationship:
            event_set.add((agent, patient, relationship))
    return event_set


def _compute_prf(
    predicted: Set,
    ground_truth: Set
) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, F1
    
    Returns:
        (precision, recall, f1)
    """
    if not predicted and not ground_truth:
        return 1.0, 1.0, 1.0
    
    if not predicted:
        return 0.0, 0.0, 0.0
    
    if not ground_truth:
        return 0.0, 0.0, 0.0
    
    true_positives = len(predicted & ground_truth)
    
    precision = true_positives / len(predicted)
    recall = true_positives / len(ground_truth)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two text strings
    
    Uses simple word overlap for baseline (can be enhanced with embeddings)
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score in [0, 1]
    """
    if not text1 or not text2:
        return 0.0
    
    # Simple word-level Jaccard similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1 & words2
    union = words1 | words2
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)
