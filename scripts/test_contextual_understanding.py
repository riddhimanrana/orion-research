#!/usr/bin/env python3
"""
Test script for LLM-enhanced contextual understanding.

This demonstrates how the system now:
1. Correctly identifies "hair drier" as "door_knob" using full context
2. Detects actions like "opening door", "entering room"
3. Generates rich narratives with spatial awarenesson how that
4. Uses LLM reasoning to explain its decisions

Usage:
    python scripts/test_contextual_understanding.py
    python scripts/test_contextual_understanding.py --results path/to/results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orion.config import OrionConfig
from orion.model_manager import ModelManager
from orion.llm_contextual_understanding import EnhancedContextualUnderstandingEngine
from orion.class_correction import ClassCorrector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test LLM-enhanced contextual understanding")
    parser.add_argument(
        '--results',
        type=str,
        default='data/testing/tracking_results_save1.json',
        help='Path to tracking results JSON'
    )
    parser.add_argument(
        '--use-corrected',
        action='store_true',
        help='Use corrected results if available'
    )
    parser.add_argument(
        '--skip-correction',
        action='store_true',
        help='Skip class correction step'
    )
    
    args = parser.parse_args()
    
    # Load config
    logger.info("📋 Loading configuration...")
    config = OrionConfig()
    
    # Initialize model manager
    logger.info("🔧 Initializing model manager...")
    model_manager = ModelManager.get_instance()
    
    # Load tracking results
    logger.info(f"📂 Loading tracking results from {args.results}...")
    results_path = Path(args.results)
    
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return
    
    with open(results_path, 'r') as f:
        tracking_results = json.load(f)
    
    entities = tracking_results.get('entities', [])
    logger.info(f"   Loaded {len(entities)} entities")
    
    # Step 1: Apply class corrections (optional)
    corrected_entities = None
    if not args.skip_correction:
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Class Correction")
        logger.info("="*80)
        
        corrector = ClassCorrector(config, model_manager)
        corrected_entities, correction_map = corrector.apply_corrections(
            entities,
            use_llm=config.correction.use_llm,  # default False
        )
        
        if correction_map:
            logger.info(f"\n✅ Applied {len(correction_map)} corrections:")
            for entity_id, (old_class, new_class, conf) in correction_map.items():
                logger.info(f"   {entity_id}: {old_class} → {new_class} (confidence: {conf:.2f})")
        else:
            logger.info("   No corrections needed")
    
    # Step 2: Build contextual understanding with LLM reasoning
    logger.info("\n" + "="*80)
    logger.info("STEP 2: LLM-Enhanced Contextual Understanding")
    logger.info("="*80)
    
    engine = EnhancedContextualUnderstandingEngine(config, model_manager)
    
    understanding = engine.understand_scene(
        tracking_results=tracking_results,
        corrected_entities=corrected_entities,
    )
    
    # Step 3: Display results
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Results")
    logger.info("="*80)
    
    # Extended objects
    logger.info("\n🎯 EXTENDED OBJECTS (What objects truly are):")
    logger.info("-" * 80)
    for obj in understanding['extended_objects'][:10]:  # First 10
        logger.info(f"\n{obj.object_id}:")
        logger.info(f"  Type: {obj.object_type}")
        logger.info(f"  Confidence: {obj.confidence:.2f}")
        logger.info(f"  Location: {obj.spatial_zone}")
        logger.info(f"  Nearby: {', '.join(obj.proximity_objects[:3])}" if obj.proximity_objects else "  Nearby: none")
        if obj.reasoning:
            logger.info(f"  Reasoning:")
            for reason in obj.reasoning[:2]:  # First 2 reasons
                logger.info(f"    • {reason}")
    
    # Detected actions
    logger.info("\n🎬 DETECTED ACTIONS:")
    logger.info("-" * 80)
    if understanding['detected_actions']:
        for action in understanding['detected_actions']:
            logger.info(f"\n{action.action_type}:")
            logger.info(f"  Confidence: {action.confidence:.2f}")
            logger.info(f"  Frames: {action.frame_range[0]} to {action.frame_range[1]}")
            if action.reasoning_steps:
                logger.info(f"  Reasoning:")
                for step in action.reasoning_steps[:2]:
                    logger.info(f"    • {step}")
    else:
        logger.info("  No actions detected yet (temporal analysis in progress)")
    
    # Narrative
    logger.info("\n📖 SCENE NARRATIVE:")
    logger.info("-" * 80)
    narrative_text = understanding.get('narrative_text', 'No narrative generated')
    logger.info(f"\n{narrative_text}\n")
    
    # Summary
    logger.info("\n📊 SUMMARY:")
    logger.info("-" * 80)
    logger.info(understanding.get('summary', 'No summary available'))
    
    # Save results
    output_path = Path('data/testing/contextual_understanding_results.json')
    logger.info(f"\n💾 Saving results to {output_path}...")
    
    # Convert to serializable format
    output_data = {
        'extended_objects': [
            {
                'object_id': obj.object_id,
                'object_type': obj.object_type,
                'confidence': obj.confidence,
                'spatial_zone': obj.spatial_zone,
                'reasoning': obj.reasoning,
            }
            for obj in understanding['extended_objects']
        ],
        'detected_actions': [
            {
                'action_type': action.action_type,
                'confidence': action.confidence,
                'frame_range': list(action.frame_range),
                'reasoning': action.reasoning_steps,
            }
            for action in understanding['detected_actions']
        ],
        'narrative_text': understanding.get('narrative_text', ''),
        'summary': understanding.get('summary', ''),
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info("✅ Done!")
    
    # Example comparison
    logger.info("\n" + "="*80)
    logger.info("BEFORE vs AFTER COMPARISON")
    logger.info("="*80)
    
    # Find the "hair drier" example
    hair_drier_entity = next(
        (e for e in entities if e.get('class') == 'hair drier'),
        None
    )
    
    if hair_drier_entity:
        entity_id = hair_drier_entity.get('entity_id')
        original_class = hair_drier_entity.get('class')
        description = hair_drier_entity.get('description', '')
        
        # Find corrected/extended version
        extended_obj = next(
            (obj for obj in understanding['extended_objects'] if obj.object_id == entity_id),
            None
        )
        
        logger.info(f"\nExample: {entity_id}")
        logger.info("-" * 80)
        logger.info(f"BEFORE (YOLO):")
        logger.info(f"  Class: {original_class}")
        logger.info(f"  Description: {description[:100]}...")
        logger.info(f"  ❌ Problem: User would ask about 'hair drier' but it's not actually a hair drier!")
        
        if extended_obj:
            logger.info(f"\nAFTER (LLM Contextual Understanding):")
            logger.info(f"  Class: {extended_obj.object_type}")
            logger.info(f"  Confidence: {extended_obj.confidence:.2f}")
            logger.info(f"  Spatial Context: {extended_obj.spatial_zone}, near {', '.join(extended_obj.proximity_objects[:2])}")
            logger.info(f"  ✅ Now correctly identified as: {extended_obj.object_type.replace('_', ' ')}")
            logger.info(f"\nReasoning:")
            for reason in extended_obj.reasoning[:3]:
                logger.info(f"  • {reason}")
    else:
        logger.info("\nNo 'hair drier' misclassification found in this video")


if __name__ == '__main__':
    main()
