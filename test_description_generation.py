#!/usr/bin/env python3
"""
Test script to verify temporal description generation.

This script creates a synthetic entity with 2 observations and verifies
that the TemporalDescriptionGenerator produces at least 2 descriptions.
"""

import sys
import logging
import numpy as np
from orion.semantic.temporal_description_generator import TemporalDescriptionGenerator
from orion.semantic.types import SemanticEntity
from orion.perception.types import Observation, BoundingBox, ObjectClass

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info("Testing Temporal Description Generation")
    logger.info("="*80)
    
    # Create a test entity with 2 observations
    entity = SemanticEntity(
        entity_id="test_entity_1",
        object_class="person",
    )
    
    # Add 2 observations at different timestamps
    bbox1 = BoundingBox(x1=100, y1=100, x2=200, y2=300)
    obs1 = Observation(
        bounding_box=bbox1,
        centroid=(150, 200),
        object_class=ObjectClass.PERSON,
        confidence=0.95,
        visual_embedding=np.random.randn(512).astype(np.float32),
        timestamp=0.0,
        frame_number=0,
        temp_id="temp_1",
    )
    
    bbox2 = BoundingBox(x1=150, y1=120, x2=250, y2=320)
    obs2 = Observation(
        bounding_box=bbox2,
        centroid=(200, 220),
        object_class=ObjectClass.PERSON,
        confidence=0.93,
        visual_embedding=np.random.randn(512).astype(np.float32),
        timestamp=1.0,
        frame_number=30,
        temp_id="temp_2",
    )
    
    entity.observations = [obs1, obs2]
    
    logger.info(f"\nCreated test entity: {entity.entity_id}")
    logger.info(f"  Class: {entity.object_class}")
    logger.info(f"  Observations: {len(entity.observations)}")
    logger.info(f"  Obs 1: frame={obs1.frame_number}, time={obs1.timestamp}s, bbox={bbox1.to_list()}")
    logger.info(f"  Obs 2: frame={obs2.frame_number}, time={obs2.timestamp}s, bbox={bbox2.to_list()}")
    
    # Create description generator (without models for this test)
    generator = TemporalDescriptionGenerator(
        clip_model=None,
        vlm_model=None,
        sample_interval=0.5,  # Sample every 0.5 seconds
        min_samples_per_entity=2,
    )
    
    logger.info("\nRunning description generation...")
    
    # Generate descriptions
    entities = [entity]
    generator.generate_temporal_descriptions(entities)
    
    # Check results
    logger.info("\n" + "="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    
    entity = entities[0]  # Get the updated entity
    num_descriptions = len(entity.descriptions)
    
    logger.info(f"Entity: {entity.entity_id}")
    logger.info(f"Descriptions generated: {num_descriptions}")
    
    if num_descriptions == 0:
        logger.error("❌ FAILED: No descriptions generated!")
        return 1
    
    for i, desc in enumerate(entity.descriptions):
        logger.info(f"  [{i+1}] time={desc['timestamp']}s, frame={desc['frame']}")
        logger.info(f"      text: {desc['text']}")
    
    if num_descriptions < 2:
        logger.error(f"❌ FAILED: Only {num_descriptions} description(s) generated (expected at least 2)")
        return 1
    
    logger.info(f"\n✅ SUCCESS: Generated {num_descriptions} descriptions for entity with 2 observations")
    return 0

if __name__ == "__main__":
    sys.exit(main())
