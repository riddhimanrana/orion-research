"""
Integration Tests for Evaluation Pipeline
==========================================

Tests the complete evaluation workflow including:
- Dataset loading
- Perception engine
- CIS calculation
- Graph construction
- Metrics computation

Author: Orion Research Team
Date: October 2025
"""

import json
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from orion.evaluation import HeuristicBaseline, GraphComparator
from orion.evaluation.metrics import evaluate_graph_quality, compare_graphs
from orion.evaluation.hyperparameter_tuning import HyperparameterTuner, CISHyperparameters


class TestEvaluationPipeline:
    """Test full evaluation pipeline"""
    
    @pytest.fixture
    def sample_perception_log(self):
        """Create sample perception log"""
        return [
            {
                "timestamp": 0.0,
                "frame_number": 0,
                "bounding_box": [100, 100, 150, 150],
                "visual_embedding": [0.1] * 512,
                "detection_confidence": 0.95,
                "object_class": "person",
                "temp_id": "det_001",
                "rich_description": "a person wearing blue shirt",
                "centroid": (125.0, 125.0),
                "velocity": (0.0, 0.0),
                "speed": 0.0,
                "direction": 0.0,
            },
            {
                "timestamp": 0.5,
                "frame_number": 15,
                "bounding_box": [200, 150, 250, 200],
                "visual_embedding": [0.2] * 512,
                "detection_confidence": 0.88,
                "object_class": "cup",
                "temp_id": "det_002",
                "rich_description": "a white coffee cup",
                "centroid": (225.0, 175.0),
                "velocity": (0.0, 0.0),
                "speed": 0.0,
                "direction": 0.0,
            },
        ]
    
    @pytest.fixture
    def sample_graph(self):
        """Create sample knowledge graph"""
        return {
            "entities": [
                {
                    "entity_id": "entity_1",
                    "class": "person",
                    "description": "person in blue",
                },
                {
                    "entity_id": "entity_2",
                    "class": "cup",
                    "description": "white cup",
                },
            ],
            "relationships": [
                {
                    "source": "entity_1",
                    "target": "entity_2",
                    "type": "holding",
                }
            ],
            "events": [
                {
                    "type": "pick_up",
                    "agent": "entity_1",
                    "patient": "entity_2",
                    "relationship": "CAUSED",
                    "timestamp": 0.5,
                }
            ],
        }
    
    def test_heuristic_baseline(self, sample_perception_log):
        """Test heuristic baseline can process perception log"""
        baseline = HeuristicBaseline()
        graph = baseline.process_perception_log(sample_perception_log)
        
        assert "entities" in graph
        assert "relationships" in graph
        assert "events" in graph
        assert len(graph["entities"]) > 0
    
    def test_graph_metrics(self, sample_graph):
        """Test graph quality metrics calculation"""
        metrics = evaluate_graph_quality(sample_graph)
        
        assert metrics.num_entities == 2
        assert metrics.num_relationships == 1
        assert metrics.num_events == 1
        assert metrics.graph_density >= 0.0
    
    def test_graph_comparison(self, sample_graph):
        """Test comparing two graphs"""
        # Create slightly modified graph
        graph2 = {
            "entities": sample_graph["entities"].copy(),
            "relationships": sample_graph["relationships"].copy(),
            "events": [],  # Different events
        }
        
        metrics = compare_graphs(sample_graph, graph2)
        
        assert 0.0 <= metrics.edge_precision <= 1.0
        assert 0.0 <= metrics.edge_recall <= 1.0
        assert 0.0 <= metrics.edge_f1 <= 1.0
    
    def test_graph_comparator(self, sample_graph):
        """Test GraphComparator with multiple graphs"""
        comparator = GraphComparator()
        
        # Add graphs
        comparator.add_graph("method_a", sample_graph)
        comparator.add_graph("method_b", sample_graph)  # Same for testing
        
        # Generate report
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            report_path = f.name
        
        report = comparator.generate_report(report_path)
        
        assert "individual_metrics" in report
        assert "pairwise_comparisons" in report
        assert "method_a" in report["individual_metrics"]
        
        # Clean up
        Path(report_path).unlink()
    
    def test_hyperparameter_validation(self):
        """Test hyperparameter validation"""
        # Valid params
        params = CISHyperparameters(
            proximity_weight=0.45,
            motion_weight=0.25,
            temporal_weight=0.20,
            embedding_weight=0.10,
        )
        assert params.validate() is True
        
        # Invalid params (weights don't sum to 1.0)
        params_invalid = CISHyperparameters(
            proximity_weight=0.9,
            motion_weight=0.5,
            temporal_weight=0.2,
            embedding_weight=0.1,
        )
        assert params_invalid.validate() is False


class TestDatasetLoaders:
    """Test dataset loaders"""
    
    def test_sample_dataset_structure(self, tmp_path):
        """Test sample dataset can be created"""
        # Create sample dataset structure
        dataset_dir = tmp_path / "sample_dataset"
        dataset_dir.mkdir()
        (dataset_dir / "videos").mkdir()
        (dataset_dir / "annotations").mkdir()
        
        # Create annotation
        annotation = {
            "video_id": "test_001",
            "objects": [
                {
                    "id": "obj_1",
                    "class": "person",
                    "bbox": [0, 0, 100, 100],
                    "frame_id": 0,
                }
            ],
            "relationships": [],
            "actions": [],
        }
        
        ann_path = dataset_dir / "annotations" / "test_001.json"
        with open(ann_path, 'w') as f:
            json.dump(annotation, f)
        
        # Verify structure
        assert (dataset_dir / "annotations" / "test_001.json").exists()
        assert (dataset_dir / "videos").is_dir()


class TestEndToEnd:
    """End-to-end integration tests"""
    
    def test_full_pipeline_sample_data(self, tmp_path):
        """Test complete pipeline on tiny sample"""
        # This is a simplified test - full test would require video
        
        # Step 1: Create sample perception log
        perception_log = [
            {
                "timestamp": i * 0.1,
                "frame_number": i,
                "bounding_box": [100 + i*10, 100, 150 + i*10, 150],
                "visual_embedding": [0.1 * i] * 512,
                "detection_confidence": 0.9,
                "object_class": "person",
                "temp_id": f"det_{i:03d}",
                "rich_description": f"person at position {i}",
                "centroid": (125.0 + i*10, 125.0),
                "velocity": (10.0, 0.0),
                "speed": 10.0,
                "direction": 0.0,
            }
            for i in range(5)
        ]
        
        # Step 2: Run heuristic baseline
        baseline = HeuristicBaseline()
        heuristic_graph = baseline.process_perception_log(perception_log)
        
        # Step 3: Evaluate quality
        metrics = evaluate_graph_quality(heuristic_graph)
        
        assert metrics.num_entities > 0
        assert metrics.num_relationships >= 0
        assert metrics.num_events >= 0
        
        # Step 4: Save to temp file
        output_file = tmp_path / "test_graph.json"
        baseline.export_to_json(str(output_file))
        
        assert output_file.exists()
        
        # Step 5: Load and verify
        with open(output_file, 'r') as f:
            loaded_graph = json.load(f)
        
        assert "entities" in loaded_graph
        assert "relationships" in loaded_graph
        assert "events" in loaded_graph


class TestMetricsCalculation:
    """Test detailed metrics calculations"""
    
    def test_precision_recall_f1(self):
        """Test P/R/F1 calculation"""
        from orion.evaluation.metrics import _compute_prf
        
        # Perfect match
        predicted = {(1, 2, "rel"), (3, 4, "rel2")}
        ground_truth = {(1, 2, "rel"), (3, 4, "rel2")}
        
        p, r, f1 = _compute_prf(predicted, ground_truth)
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0
        
        # Partial match
        predicted = {(1, 2, "rel"), (3, 4, "rel2"), (5, 6, "rel3")}
        ground_truth = {(1, 2, "rel"), (3, 4, "rel2")}
        
        p, r, f1 = _compute_prf(predicted, ground_truth)
        assert p == 2/3  # 2 correct out of 3 predicted
        assert r == 1.0  # 2 correct out of 2 ground truth
        assert f1 == 2 * (p * r) / (p + r)
        
        # No match
        predicted = {(1, 2, "rel")}
        ground_truth = {(3, 4, "rel2")}
        
        p, r, f1 = _compute_prf(predicted, ground_truth)
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
