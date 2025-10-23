"""
Tests for HDBSCAN Fallback and Embedding Similarity Guards
===========================================================

Tests the fixes for:
1. HDBSCAN fallback breaking object permanence (2 cups → 1 entity)
2. Missing embedding similarity guards ("laptop" vs "computer")
"""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.tracking_engine import EntityTracker, Observation, Entity
from orion.config import OrionConfig


class TestHDBSCANFallback:
    """Test that fallback clustering preserves object permanence"""
    
    def create_mock_observation(
        self, 
        frame: int, 
        class_name: str, 
        embedding: np.ndarray,
        timestamp: float = None
    ) -> Observation:
        """Helper to create mock observation"""
        if timestamp is None:
            timestamp = frame / 30.0  # 30 FPS
        
        return Observation(
            frame_number=frame,
            timestamp=timestamp,
            bbox=[100, 100, 200, 200],
            class_name=class_name,
            confidence=0.9,
            embedding=embedding,
            crop=np.zeros((100, 100, 3), dtype=np.uint8),
            frame_width=1920,
            frame_height=1080
        )
    
    def test_fallback_distinguishes_multiple_cups(self):
        """Test that fallback doesn't merge 2 distinct cups into 1 entity"""
        # Create 2 cups with different embeddings (different visual appearance)
        # Cup 1: Red cup (embedding centered at [1, 0, 0, ...])
        cup1_embedding = np.random.randn(512).astype(np.float32)
        cup1_embedding[0] = 5.0  # Strong red signal
        cup1_embedding = cup1_embedding / np.linalg.norm(cup1_embedding)
        
        # Cup 2: Blue cup (embedding centered at [0, 1, 0, ...])
        cup2_embedding = np.random.randn(512).astype(np.float32)
        cup2_embedding[1] = 5.0  # Strong blue signal
        cup2_embedding = cup2_embedding / np.linalg.norm(cup2_embedding)
        
        # Verify they're dissimilar
        similarity = float(np.dot(cup1_embedding, cup2_embedding))
        assert similarity < 0.7, f"Test setup error: cups should be dissimilar (got {similarity:.3f})"
        
        # Create observations: 3 of cup1, 3 of cup2
        observations = [
            self.create_mock_observation(1, "cup", cup1_embedding, 0.1),
            self.create_mock_observation(2, "cup", cup2_embedding, 0.2),
            self.create_mock_observation(3, "cup", cup1_embedding, 0.3),
            self.create_mock_observation(4, "cup", cup2_embedding, 0.4),
            self.create_mock_observation(5, "cup", cup1_embedding, 0.5),
            self.create_mock_observation(6, "cup", cup2_embedding, 0.6),
        ]
        
        # Mock tracker
        config = OrionConfig()
        tracker = EntityTracker(config)
        
        # Call the new embedding-aware fallback
        entities = tracker._cluster_by_embedding_similarity(
            observations, 
            "cup",
            similarity_threshold=0.85
        )
        
        # Should create 2 entities (one per cup)
        assert len(entities) == 2, f"Expected 2 entities, got {len(entities)}"
        
        # Each entity should have 3 observations
        for entity in entities:
            assert len(entity.observations) == 3, \
                f"Entity {entity.id} should have 3 observations, got {len(entity.observations)}"
        
        print("✓ Fallback preserves object permanence (2 cups → 2 entities)")
    
    def test_fallback_groups_same_object_appearances(self):
        """Test that fallback correctly groups appearances of same object"""
        # Same laptop appearing 5 times with very similar embeddings
        base_embedding = np.random.randn(512).astype(np.float32)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        
        # Create 5 slightly noisy versions (same laptop, slight viewing angle changes)
        observations = []
        for i in range(5):
            # Add very small noise to stay above similarity threshold (0.85)
            noisy_embedding = base_embedding + np.random.randn(512).astype(np.float32) * 0.01
            noisy_embedding = noisy_embedding / np.linalg.norm(noisy_embedding)
            
            # Verify similarity is high enough
            sim = float(np.dot(base_embedding, noisy_embedding))
            assert sim > 0.85, f"Test setup: similarity {sim:.3f} should be > 0.85"
            
            obs = self.create_mock_observation(
                frame=i * 10,
                class_name="laptop",
                embedding=noisy_embedding,
                timestamp=i * 0.5
            )
            observations.append(obs)
        
        # Mock tracker
        config = OrionConfig()
        tracker = EntityTracker(config)
        
        # Call embedding-aware fallback
        entities = tracker._cluster_by_embedding_similarity(
            observations,
            "laptop",
            similarity_threshold=0.85
        )
        
        # Should create 1 entity (same laptop)
        assert len(entities) == 1, f"Expected 1 entity, got {len(entities)}"
        assert len(entities[0].observations) == 5, \
            f"Entity should have 5 observations, got {len(entities[0].observations)}"
        
        print("✓ Fallback correctly groups same object appearances")
    
    def test_old_fallback_fails_object_permanence(self):
        """Demonstrate that old fallback incorrectly merges distinct objects"""
        # Create 2 distinct cups (as before)
        cup1_embedding = np.random.randn(512).astype(np.float32)
        cup1_embedding[0] = 5.0
        cup1_embedding = cup1_embedding / np.linalg.norm(cup1_embedding)
        
        cup2_embedding = np.random.randn(512).astype(np.float32)
        cup2_embedding[1] = 5.0
        cup2_embedding = cup2_embedding / np.linalg.norm(cup2_embedding)
        
        observations = [
            self.create_mock_observation(1, "cup", cup1_embedding),
            self.create_mock_observation(2, "cup", cup2_embedding),
            self.create_mock_observation(3, "cup", cup1_embedding),
        ]
        
        # OLD approach: group by class only
        from collections import defaultdict
        class_groups = defaultdict(list)
        for obs in observations:
            class_groups[obs.class_name].append(obs)
        
        # Old approach creates 1 entity for ALL cups
        old_entities = []
        for class_name, obs_list in class_groups.items():
            entity = Entity(
                id=f"class_{class_name}_000",
                class_name=class_name,
                observations=obs_list
            )
            old_entities.append(entity)
        
        # OLD: Incorrectly creates 1 entity
        assert len(old_entities) == 1, "Old approach should create 1 entity (incorrect)"
        assert len(old_entities[0].observations) == 3, "Old approach merges all 3 observations"
        
        print("✓ Confirmed: Old fallback breaks object permanence")


class TestSemanticSimilarityGuards:
    """Test CLIP-based semantic similarity guards"""
    
    @patch('orion.tracking_engine.ModelManager')
    def test_laptop_computer_semantic_compatibility(self, mock_model_manager):
        """Test that 'laptop' (YOLO) and 'computer' (VLM) are semantically compatible"""
        # Setup mock CLIP
        mock_clip = Mock()
        
        # Mock CLIP embeddings for "laptop" and "computer"
        # They should have high cosine similarity
        laptop_embedding = np.array([0.8, 0.3, 0.5])
        laptop_embedding = laptop_embedding / np.linalg.norm(laptop_embedding)
        
        computer_embedding = np.array([0.75, 0.35, 0.48])
        computer_embedding = computer_embedding / np.linalg.norm(computer_embedding)
        
        # Similarity should be high (around 0.95)
        similarity = float(np.dot(laptop_embedding, computer_embedding))
        assert similarity > 0.9, f"Test setup: laptop-computer similarity should be high (got {similarity:.3f})"
        
        def mock_encode_text(text, normalize=True):
            if "laptop" in text.lower():
                return laptop_embedding
            elif "computer" in text.lower():
                return computer_embedding
            else:
                return np.random.randn(3) / 3.0
        
        mock_clip.encode_text = mock_encode_text
        mock_model_manager.get_instance.return_value.clip = mock_clip
        
        # Create describer
        from orion.tracking_engine import SmartDescriber
        config = OrionConfig()
        describer = SmartDescriber(config)
        describer.model_manager = mock_model_manager.get_instance.return_value
        
        # Check compatibility
        is_compatible, score = describer._check_semantic_compatibility(
            yolo_class="laptop",
            vlm_description="This is a sleek silver computer with a keyboard and screen",
            confidence=0.9
        )
        
        assert is_compatible, "laptop and computer should be semantically compatible"
        assert score > 0.75, f"Similarity score should be high (got {score:.3f})"
        
        print("✓ Semantic guard: laptop ↔ computer compatible")
    
    @patch('orion.tracking_engine.ModelManager')
    def test_cup_phone_semantic_incompatibility(self, mock_model_manager):
        """Test that 'cup' (YOLO) and 'phone' (VLM) are semantically incompatible"""
        # Setup mock CLIP
        mock_clip = Mock()
        
        # Mock CLIP embeddings for "cup" and "phone"
        # They should have low cosine similarity
        cup_embedding = np.array([0.9, 0.1, 0.1])
        cup_embedding = cup_embedding / np.linalg.norm(cup_embedding)
        
        phone_embedding = np.array([0.1, 0.9, 0.1])
        phone_embedding = phone_embedding / np.linalg.norm(phone_embedding)
        
        # Similarity should be low
        similarity = float(np.dot(cup_embedding, phone_embedding))
        assert similarity < 0.5, f"Test setup: cup-phone similarity should be low (got {similarity:.3f})"
        
        def mock_encode_text(text, normalize=True):
            if "cup" in text.lower():
                return cup_embedding
            elif "phone" in text.lower() or "smartphone" in text.lower():
                return phone_embedding
            else:
                return np.random.randn(3) / 3.0
        
        mock_clip.encode_text = mock_encode_text
        mock_model_manager.get_instance.return_value.clip = mock_clip
        
        # Create describer
        from orion.tracking_engine import SmartDescriber
        config = OrionConfig()
        describer = SmartDescriber(config)
        describer.model_manager = mock_model_manager.get_instance.return_value
        
        # Check compatibility
        is_compatible, score = describer._check_semantic_compatibility(
            yolo_class="cup",
            vlm_description="This is a black smartphone with a touchscreen",
            confidence=0.65
        )
        
        assert not is_compatible, "cup and phone should be semantically incompatible"
        assert score < 0.75, f"Similarity score should be low (got {score:.3f})"
        
        print("✓ Semantic guard: cup ↔ phone incompatible")
    
    def test_semantic_guard_fallback_on_error(self):
        """Test that semantic guard gracefully falls back on CLIP errors"""
        from orion.tracking_engine import SmartDescriber
        
        config = OrionConfig()
        describer = SmartDescriber(config)
        
        # Mock CLIP to raise error
        describer.model_manager = Mock()
        describer.model_manager.clip.encode_text.side_effect = Exception("CLIP failed")
        
        # Should fallback to string matching without crashing
        is_compatible, score = describer._check_semantic_compatibility(
            yolo_class="laptop",
            vlm_description="This is a laptop computer",
            confidence=0.9
        )
        
        # Fallback to simple string matching
        assert is_compatible, "Fallback should find 'laptop' in description"
        assert score == 0.0, "Fallback should return 0.0 score"
        
        print("✓ Semantic guard gracefully handles CLIP errors")


class TestIntegration:
    """Integration tests for both fixes together"""
    
    def test_complete_pipeline_with_fixes(self):
        """Test full pipeline with both HDBSCAN fallback fix and semantic guards"""
        # This would require full pipeline setup
        # For now, just verify both fixes are integrated
        
        from orion.tracking_engine import EntityTracker, SmartDescriber
        config = OrionConfig()
        
        # Verify new methods exist
        tracker = EntityTracker(config)
        assert hasattr(tracker, '_cluster_by_embedding_similarity'), \
            "New embedding-aware fallback method should exist"
        
        describer = SmartDescriber(config)
        assert hasattr(describer, '_check_semantic_compatibility'), \
            "New semantic compatibility check should exist"
        
        print("✓ Both fixes integrated into pipeline")


if __name__ == "__main__":
    # Run tests
    print("="*80)
    print("TESTING HDBSCAN FALLBACK & SEMANTIC SIMILARITY GUARDS")
    print("="*80)
    
    print("\n1. HDBSCAN Fallback Tests")
    print("-" * 80)
    test_fallback = TestHDBSCANFallback()
    test_fallback.test_fallback_distinguishes_multiple_cups()
    test_fallback.test_fallback_groups_same_object_appearances()
    test_fallback.test_old_fallback_fails_object_permanence()
    
    print("\n2. Semantic Similarity Guard Tests")
    print("-" * 80)
    test_semantic = TestSemanticSimilarityGuards()
    test_semantic.test_laptop_computer_semantic_compatibility()
    test_semantic.test_cup_phone_semantic_incompatibility()
    test_semantic.test_semantic_guard_fallback_on_error()
    
    print("\n3. Integration Tests")
    print("-" * 80)
    test_integration = TestIntegration()
    test_integration.test_complete_pipeline_with_fixes()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
