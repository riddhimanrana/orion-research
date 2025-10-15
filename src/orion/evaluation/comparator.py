"""
Knowledge Graph Comparator
===========================

Tools for comparing and visualizing differences between knowledge graphs
constructed by different methods.

Author: Orion Research Team
Date: October 2025
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from .metrics import (
    GraphMetrics,
    ComparisonMetrics,
    evaluate_graph_quality,
    compare_graphs,
)

logger = logging.getLogger("GraphComparator")


class GraphComparator:
    """
    Compare knowledge graphs from different construction methods
    """
    
    def __init__(self):
        self.graphs: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, GraphMetrics] = {}
    
    def add_graph(self, name: str, graph_data: Dict[str, Any]):
        """
        Add a graph for comparison
        
        Args:
            name: Identifier for this graph (e.g., "heuristic", "cis_llm")
            graph_data: Graph data dictionary
        """
        self.graphs[name] = graph_data
        self.metrics[name] = evaluate_graph_quality(graph_data)
        logger.info(f"Added graph '{name}' for comparison")
    
    def load_from_json(self, name: str, json_path: str):
        """
        Load a graph from JSON file
        
        Args:
            name: Identifier for this graph
            json_path: Path to JSON file
        """
        with open(json_path, 'r') as f:
            graph_data = json.load(f)
        self.add_graph(name, graph_data)
    
    def compare(
        self,
        predicted_name: str,
        ground_truth_name: str
    ) -> ComparisonMetrics:
        """
        Compare two graphs
        
        Args:
            predicted_name: Name of predicted graph
            ground_truth_name: Name of ground truth graph
            
        Returns:
            ComparisonMetrics
        """
        if predicted_name not in self.graphs:
            raise ValueError(f"Graph '{predicted_name}' not found")
        if ground_truth_name not in self.graphs:
            raise ValueError(f"Graph '{ground_truth_name}' not found")
        
        predicted = self.graphs[predicted_name]
        ground_truth = self.graphs[ground_truth_name]
        
        return compare_graphs(predicted, ground_truth)
    
    def generate_report(
        self,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report
        
        Args:
            output_path: Optional path to save JSON report
            
        Returns:
            Report dictionary
        """
        report = {
            "individual_metrics": {},
            "pairwise_comparisons": {},
        }
        
        # Individual graph metrics
        for name, metrics in self.metrics.items():
            report["individual_metrics"][name] = metrics.to_dict()
        
        # Pairwise comparisons
        graph_names = list(self.graphs.keys())
        for i, name1 in enumerate(graph_names):
            for name2 in graph_names[i + 1:]:
                comparison_key = f"{name1}_vs_{name2}"
                comparison = self.compare(name1, name2)
                report["pairwise_comparisons"][comparison_key] = comparison.to_dict()
        
        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved comparison report to {output_path}")
        
        return report
    
    def print_summary(self):
        """Print a human-readable summary of all graphs"""
        print("\n" + "="*80)
        print("KNOWLEDGE GRAPH COMPARISON SUMMARY")
        print("="*80)
        
        for name, metrics in self.metrics.items():
            print(f"\n{name.upper()}")
            print("-" * 40)
            print(f"  Entities:       {metrics.num_entities}")
            print(f"  Relationships:  {metrics.num_relationships}")
            print(f"  Events:         {metrics.num_events}")
            print(f"  Graph Density:  {metrics.graph_density:.4f}")
            print(f"  Avg Degree:     {metrics.avg_degree:.2f}")
            
            if metrics.num_causal_links > 0:
                print(f"  Causal Links:   {metrics.num_causal_links}")
                print(f"  Avg CIS Score:  {metrics.avg_cis_score:.3f}")
            
            print(f"\n  Relationship Types:")
            for rel_type, count in sorted(
                metrics.relationship_types.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"    {rel_type}: {count}")
        
        # Pairwise comparisons
        graph_names = list(self.graphs.keys())
        if len(graph_names) >= 2:
            print("\n" + "="*80)
            print("PAIRWISE COMPARISONS")
            print("="*80)
            
            for i, name1 in enumerate(graph_names):
                for name2 in graph_names[i + 1:]:
                    print(f"\n{name1.upper()} vs {name2.upper()}")
                    print("-" * 40)
                    
                    comparison = self.compare(name1, name2)
                    
                    print(f"  Edge F1:        {comparison.edge_f1:.4f}")
                    print(f"  Event F1:       {comparison.event_f1:.4f}")
                    print(f"  Causal F1:      {comparison.causal_f1:.4f}")
                    print(f"  Entity Jaccard: {comparison.entity_jaccard:.4f}")
        
        print("\n" + "="*80 + "\n")
    
    def find_discrepancies(
        self,
        name1: str,
        name2: str
    ) -> Dict[str, Any]:
        """
        Find specific discrepancies between two graphs
        
        Args:
            name1: First graph name
            name2: Second graph name
            
        Returns:
            Dictionary of discrepancies
        """
        graph1 = self.graphs[name1]
        graph2 = self.graphs[name2]
        
        # Entity differences
        entities1 = {e["entity_id"] for e in graph1.get("entities", [])}
        entities2 = {e["entity_id"] for e in graph2.get("entities", [])}
        
        # Relationship differences
        rels1 = {
            (r["source"], r["target"], r["type"])
            for r in graph1.get("relationships", [])
        }
        rels2 = {
            (r["source"], r["target"], r["type"])
            for r in graph2.get("relationships", [])
        }
        
        # Event differences
        events1 = {
            (e.get("agent"), e.get("patient"), e.get("relationship"))
            for e in graph1.get("events", [])
            if e.get("agent") and e.get("patient")
        }
        events2 = {
            (e.get("agent"), e.get("patient"), e.get("relationship"))
            for e in graph2.get("events", [])
            if e.get("agent") and e.get("patient")
        }
        
        discrepancies = {
            "entities_only_in_" + name1: list(entities1 - entities2),
            "entities_only_in_" + name2: list(entities2 - entities1),
            "relationships_only_in_" + name1: list(rels1 - rels2),
            "relationships_only_in_" + name2: list(rels2 - rels1),
            "events_only_in_" + name1: list(events1 - events2),
            "events_only_in_" + name2: list(events2 - events1),
        }
        
        return discrepancies
