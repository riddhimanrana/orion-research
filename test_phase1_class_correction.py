#!/usr/bin/env python3
"""
Phase 1 Test: Unbiased Description & Class Correction
======================================================

This script tests the new Phase 1 enhancements:
1. Unbiased description generation (no YOLO hints)
2. Semantic class correction using sentence transformers
3. Multi-strategy correction (keyword, semantic, fuzzy)

Expected improvements:
- "HAIR_DRIER" misclassifications should be corrected
- Duplicate BED entities should be identified
- More accurate object classifications overall

Author: Orion Research Team
Date: October 2025
"""

import json
import logging
from pathlib import Path

from orion.pipeline import VideoPipeline, PipelineConfig
from orion.perception.config import get_fast_config
from orion.semantic.config import get_fast_semantic_config
from orion.settings import OrionSettings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

def main():
    print("=" * 80)
    print("PHASE 1 TEST: UNBIASED DESCRIPTION & CLASS CORRECTION")
    print("=" * 80)
    print()
    
    # Load Neo4j password
    try:
        settings = OrionSettings.load()
        neo4j_password = settings.get_neo4j_password()
        print(f"[CONFIG] Neo4j password loaded: {'*' * len(neo4j_password)}")
    except Exception as e:
        print(f"[WARNING] Could not load Neo4j password: {e}")
        neo4j_password = "password"
    
    print()
    
    # Create pipeline config with Phase 1 enhancements enabled
    perception_config = get_fast_config()
    
    # Note: Class correction is enabled by default in EntityDescriber
    # The unbiased prompts are also automatic now
    
    config = PipelineConfig(
        perception_config=perception_config,
        semantic_config=get_fast_semantic_config(),
        neo4j_uri="neo4j://127.0.0.1:7687",
        neo4j_user="neo4j",
        neo4j_password=neo4j_password,
    )
    
    print("[INIT] Creating pipeline with Phase 1 enhancements...")
    print("  ✓ Unbiased description generation (no YOLO hints in prompts)")
    print("  ✓ Semantic class correction (sentence transformers)")
    print("  ✓ Multi-strategy matching (keyword, semantic, fuzzy)")
    print()
    
    pipeline = VideoPipeline(config)
    print(f"[INIT] Pipeline created successfully")
    print(f"[INIT] Neo4j connected: {pipeline.neo4j_manager.driver is not None if pipeline.neo4j_manager else False}")
    print()
    
    # Process video
    video_path = "data/examples/video_short.mp4"
    scene_id = "phase1_test"
    
    print(f"[VIDEO] Processing: {video_path}")
    print(f"[VIDEO] Scene ID: {scene_id}")
    print("-" * 80)
    print()
    
    results = pipeline.process_video(video_path, scene_id)
    
    # Analyze results
    print()
    print("=" * 80)
    print("PHASE 1 RESULTS ANALYSIS")
    print("=" * 80)
    print()
    
    perception = results.get("perception", {})
    entities = perception.get("entities", [])
    
    print(f"Entity Detection:")
    print(f"  Total entities: {len(entities)}")
    print()
    
    # Check for corrections
    print("Class Corrections Applied:")
    corrections = []
    for entity in entities:
        if "corrected_class" in entity and entity["corrected_class"]:
            corrections.append({
                "entity_id": entity["entity_id"],
                "original": entity["class"],
                "corrected": entity["corrected_class"],
                "confidence": entity.get("correction_confidence", 0.0),
                "description_preview": entity.get("description", "")[:100]
            })
    
    if corrections:
        for i, corr in enumerate(corrections, 1):
            print(f"  {i}. {corr['entity_id']}")
            print(f"     Original:  {corr['original']}")
            print(f"     Corrected: {corr['corrected']} (confidence: {corr['confidence']:.2f})")
            print(f"     Desc: {corr['description_preview']}...")
            print()
    else:
        print("  No corrections applied (all YOLO detections validated)")
    
    print()
    
    # Check for specific issues from original results
    print("Known Issue Analysis:")
    print()
    
    # Issue 1: HAIR_DRIER misclassification
    hair_drier_entities = [e for e in entities if e["class"].upper() == "HAIR_DRIER"]
    if hair_drier_entities:
        print(f"  ⚠️ HAIR_DRIER detection(s) found: {len(hair_drier_entities)}")
        for entity in hair_drier_entities:
            desc = entity.get("description", "")[:150]
            corrected = entity.get("corrected_class")
            if corrected:
                print(f"    ✓ Corrected to: {corrected}")
            else:
                print(f"    ✗ Not corrected - Description: {desc}...")
    else:
        print(f"  ✓ No HAIR_DRIER misclassifications")
    
    print()
    
    # Issue 2: Duplicate BED entities
    bed_entities = [e for e in entities if e["class"].upper() == "BED"]
    if len(bed_entities) > 1:
        print(f"  ⚠️ Multiple BED entities detected: {len(bed_entities)}")
        print(f"    Note: Entity deduplication is Phase 4 - not yet implemented")
    else:
        print(f"  ✓ BED entity count: {len(bed_entities)}")
    
    print()
    print("=" * 80)
    print("COMPARING WITH ORIGINAL RESULTS")
    print("=" * 80)
    print()
    
    # Load original results for comparison
    original_results_path = Path("test_results/full_pipeline_results.json")
    if original_results_path.exists():
        with open(original_results_path) as f:
            original_results = json.load(f)
        
        original_entities = original_results.get("perception", {}).get("entities", [])
        
        print("Original Results (WITHOUT Phase 1):")
        print(f"  Total entities: {len(original_entities)}")
        print(f"  HAIR_DRIER detections: {sum(1 for e in original_entities if e['class'].upper() == 'HAIR_DRIER')}")
        print(f"  BED detections: {sum(1 for e in original_entities if e['class'].upper() == 'BED')}")
        print()
        
        print("Phase 1 Results (WITH unbiased descriptions + correction):")
        print(f"  Total entities: {len(entities)}")
        print(f"  HAIR_DRIER detections: {sum(1 for e in entities if e['class'].upper() == 'HAIR_DRIER')}")
        print(f"  BED detections: {sum(1 for e in entities if e['class'].upper() == 'BED')}")
        print(f"  Corrections applied: {len(corrections)}")
    else:
        print("  (Original results not found - skipping comparison)")
    
    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print()
    print(f"Results saved to: test_results/phase1_test_results.json")
    
    # Save results
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "phase1_test_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()

if __name__ == "__main__":
    main()
