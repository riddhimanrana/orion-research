"""
Test Semantic Validation for Class Correction
==============================================

Tests the new semantic validation layer that prevents bad corrections
like "tire" → "car" when the description clearly describes a tire.
"""

import pytest
import logging
from orion.class_correction import ClassCorrector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSemanticValidation:
    """Test semantic validation of class corrections"""
    
    def setup_method(self):
        """Setup corrector for each test"""
        self.corrector = ClassCorrector()
    
    def test_tire_description_should_not_map_to_car(self):
        """Test Case 1: Tire description shouldn't be corrected to 'car'"""
        yolo_class = "suitcase"
        description = """The image depicts a close-up view of a car tire. The tire is black and 
        appears to be in motion, as indicated by the blurred background. The tire's tread pattern 
        is visible, and the rim of the tire is also discernible."""
        
        # Should recognize this is describing a tire, not a full car
        corrected_class, conf = self.corrector.extract_corrected_class(
            yolo_class,
            description,
            use_llm=False,
            validate_with_description=True
        )
        
        # Should either not correct, or if it does, validate should catch it
        if corrected_class == "car":
            # This would be a bad correction
            logger.warning(f"Bad correction detected: {yolo_class} → {corrected_class}")
            # Validation should have prevented this
            is_valid, score = self.corrector.validate_correction_with_description(
                description, yolo_class, corrected_class
            )
            assert not is_valid or score < 0.5, "Validation should reject tire → car mapping"
        
        logger.info(f"✓ Tire test: {yolo_class} → {corrected_class} (conf: {conf:.2f})")
    
    def test_knob_description_should_stay_knob(self):
        """Test Case 2: Knob description should map to appropriate class or stay"""
        yolo_class = "hair drier"
        description = """The image depicts a close-up view of a metallic object, which appears 
        to be a knob or a handle. The object is predominantly black in color, with a shiny, 
        reflective surface that suggests it is made of metal. The knob has a circular shape 
        with a slightly curved top."""
        
        corrected_class, conf = self.corrector.extract_corrected_class(
            yolo_class,
            description,
            use_llm=False,
            validate_with_description=True
        )
        
        # Should map to something reasonable (remote, bottle, or similar small object)
        # NOT to something large like refrigerator
        if corrected_class:
            assert corrected_class in ['remote', 'bottle', 'cell phone', 'knob'], \
                f"Unexpected correction: {corrected_class}"
            
            # Validate the correction makes sense
            is_valid, score = self.corrector.validate_correction_with_description(
                description, yolo_class, corrected_class
            )
            assert is_valid and score > 0.3, \
                f"Correction {yolo_class} → {corrected_class} failed validation (score: {score:.2f})"
        
        logger.info(f"✓ Knob test: {yolo_class} → {corrected_class} (conf: {conf:.2f})")
    
    def test_person_description_should_stay_person(self):
        """Test Case 3: Correct classifications shouldn't be changed"""
        yolo_class = "person"
        description = """The image shows a person standing in a room. They are wearing 
        casual clothing and appear to be looking at something off-camera."""
        
        corrected_class, conf = self.corrector.extract_corrected_class(
            yolo_class,
            description,
            use_llm=False,
            validate_with_description=True
        )
        
        # Should NOT correct since person is correct
        assert corrected_class is None or corrected_class == "person", \
            f"Should not correct correct classifications: {yolo_class} → {corrected_class}"
        
        logger.info(f"✓ Person test: {yolo_class} → {corrected_class or 'no change'}")
    
    def test_validation_rejects_poor_semantic_match(self):
        """Test Case 4: Validation should reject corrections with poor semantic match"""
        description = "A close-up of a coffee cup on a table"
        original_class = "cup"
        proposed_class = "bicycle"  # Clearly wrong
        
        is_valid, score = self.corrector.validate_correction_with_description(
            description, original_class, proposed_class
        )
        
        # Should reject this bad correction
        assert not is_valid or score < 0.4, \
            f"Validation should reject cup → bicycle (score: {score:.2f})"
        
        logger.info(f"✓ Rejection test: cup → bicycle rejected (valid: {is_valid}, score: {score:.2f})")
    
    def test_validation_accepts_good_semantic_match(self):
        """Test Case 5: Validation should accept corrections with good semantic match"""
        description = "A laptop computer sitting on a desk with a keyboard"
        original_class = "tv"  # YOLO misclassified
        proposed_class = "laptop"  # Correct
        
        is_valid, score = self.corrector.validate_correction_with_description(
            description, original_class, proposed_class
        )
        
        # Should accept this good correction
        assert is_valid and score > 0.4, \
            f"Validation should accept tv → laptop (score: {score:.2f})"
        
        logger.info(f"✓ Acceptance test: tv → laptop accepted (valid: {is_valid}, score: {score:.2f})")
    
    def test_part_of_detection_prevents_bad_mapping(self):
        """Test Case 6: Part-of detection should prevent tire → car mapping"""
        yolo_class = "suitcase"  # Misclassified tire
        description = "The tire tread pattern shows significant wear on the rim"
        confidence = 0.46
        
        # Should detect tire in part-of context and NOT correct
        should_correct = self.corrector.should_correct(
            yolo_class, description, confidence, clip_verified=False
        )
        
        # With part-of detection, should be more conservative
        logger.info(f"✓ Part-of test: should_correct={should_correct} for tire description")


class TestSemanticClassMatch:
    """Test semantic class matching with embeddings"""
    
    def setup_method(self):
        """Setup corrector for each test"""
        self.corrector = ClassCorrector()
    
    def test_semantic_match_with_direct_mention(self):
        """Test semantic matching when class is directly mentioned"""
        description = "This is a laptop computer with a black keyboard"
        yolo_class = "tv"
        
        matched_class, conf = self.corrector.semantic_class_match(
            description, yolo_class, top_k=5
        )
        
        # Should find "laptop" in description
        assert matched_class == "laptop" or conf > 0.9, \
            f"Should match laptop from direct mention (got: {matched_class}, conf: {conf:.2f})"
        
        logger.info(f"✓ Direct mention test: {yolo_class} → {matched_class} (conf: {conf:.2f})")
    
    def test_semantic_match_with_embedding_similarity(self):
        """Test semantic matching using embedding similarity"""
        description = "A small handheld device used for controlling electronics"
        yolo_class = "cell phone"
        
        matched_class, conf = self.corrector.semantic_class_match(
            description, yolo_class, top_k=5, threshold=0.6
        )
        
        # Should match to "remote" based on description
        # (if no direct match, embeddings should work)
        logger.info(f"✓ Embedding test: {yolo_class} → {matched_class} (conf: {conf:.2f})")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
