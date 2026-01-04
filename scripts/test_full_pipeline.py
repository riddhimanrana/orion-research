"""Test full pipeline: Perception + Semantic + CLIP class correction"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.perception.config import get_accurate_config
from orion.perception.engine import PerceptionEngine
from orion.semantic.engine import SemanticEngine
from orion.semantic.config import SemanticConfig
from orion.managers.model_manager import ModelManager
import json
from collections import Counter
import numpy as np

print("="*80)
print("FULL PIPELINE TEST: Perception + Semantic + CLIP Class Correction")
print("="*80)

# Test video
video_path = "data/examples/room.mp4"
output_dir = "results/full_pipeline"

print(f"\nVideo: {video_path}")
print(f"Output: {output_dir}\n")

# ====================================================================
# STEP 1: Perception (with CLIP class correction)
# ====================================================================
print("STEP 1: Running Perception Engine...")
print("-" * 80)

config = get_accurate_config()
config.target_fps = 0.25  # Process 1 frame every 4 seconds
config.enable_3d = False  # Skip 3D for faster processing
config.enable_tracking = True
config.yolo_model = "yolo11n"  # Use faster nano model
# Use CLIP backend for embeddings (needed for CLIP class correction)
config.embedding.backend = "clip"

engine = PerceptionEngine(config)
perception_result = engine.process_video(video_path, save_visualizations=True, output_dir=output_dir)

print(f"\n✓ Perception complete:")
print(f"  Detections: {perception_result.total_detections}")
print(f"  Entities: {perception_result.unique_entities}")

# ====================================================================
# STEP 2: CLIP Class Correction
# ====================================================================
print("\n" + "="*80)
print("STEP 2: CLIP-based Class Correction")
print("-" * 80)

# Get CLIP model
from orion.managers.model_manager import ModelManager
model_manager = ModelManager.get_instance()
clip_model = model_manager.clip

# Define common misclassifications to check
class_corrections = {
    "tv": ["monitor", "television", "TV screen", "computer display"],
    "cat": ["dog", "teddy bear", "stuffed animal", "toy animal"],  
    "couch": ["sofa", "chair", "seat"],
    "dining table": ["desk", "table", "workstation"],
}

corrected_count = 0
for entity in perception_result.entities:
    original_class = entity.object_class.value if hasattr(entity.object_class, 'value') else str(entity.object_class)
    
    # Check if this class needs verification
    if original_class in class_corrections:
        try:
            # Get candidate labels
            candidate_labels = [original_class] + class_corrections[original_class]
            
            # Get CLIP text embeddings
            text_embeddings = []
            for label in candidate_labels:
                text_emb = clip_model.encode_text(label, normalize=True)
                text_embeddings.append(text_emb)
            
            text_embeddings = np.array(text_embeddings)
            
            # Compare visual embedding with text embeddings
            visual_emb = entity.average_embedding
            if visual_emb is not None:
                # Ensure normalized
                visual_emb = visual_emb / (np.linalg.norm(visual_emb) + 1e-8)
                
                # Compute cosine similarities
                similarities = text_embeddings @ visual_emb
                best_idx = similarities.argmax()
                best_label = candidate_labels[best_idx]
                confidence = float(similarities[best_idx])
                
                if best_label != original_class and confidence > 0.28:
                    print(f"  ✓ Correction: '{original_class}' → '{best_label}' (confidence: {confidence:.3f})")
                    entity.corrected_class = best_label
                    entity.correction_confidence = confidence
                    corrected_count += 1
                else:
                    print(f"  ✓ Verified: '{original_class}' (confidence: {similarities[0]:.3f})")
            else:
                print(f"  ⚠ No embedding for {original_class}")
                
        except Exception as e:
            print(f"  ⚠ Error checking '{original_class}': {e}")

print(f"\n✓ Class correction complete: {corrected_count} corrections made")

# ====================================================================
# STEP 3: Semantic Analysis
# ====================================================================
print("\n" + "="*80)
print("STEP 3: Semantic Analysis (Zones, State Changes, Events)")
print("-" * 80)

try:
    semantic_config = SemanticConfig(
        verbose=True,
        enable_graph_ingestion=False,  # Skip graph for now
    )
    semantic_engine = SemanticEngine(semantic_config)
    
    # Run semantic analysis
    semantic_result = semantic_engine.process(perception_result)
    
    print(f"\n✓ Semantic analysis complete:")
    print(f"  Entities tracked: {len(semantic_result.entities)}")
    print(f"  State changes detected: {len(semantic_result.state_changes)}")
    print(f"  Events composed: {len(semantic_result.events)}")
    
    # Display state changes
    if semantic_result.state_changes:
        print("\nState Changes (first 15):")
        for sc in semantic_result.state_changes[:15]:
            entity_class = sc.entity_id.split('_')[0] if '_' in sc.entity_id else 'unknown'
            print(f"  - Frame {sc.frame_after}: [{entity_class}] {sc.change_type}")
    else:
        print("\n⚠ No state changes detected")
    
    # Display events
    if semantic_result.events:
        print("\nEvents Detected (first 10):")
        for i, event in enumerate(semantic_result.events[:10]):
            participants = ', '.join(event.involved_entities[:3])  # First 3 entities
            if len(event.involved_entities) > 3:
                participants += f" (+{len(event.involved_entities)-3} more)"
            print(f"  {i+1}. {event.event_type}: {participants}")
            if hasattr(event, 'description') and event.description:
                print(f"      \"{event.description[:80]}...\"")
    else:
        print("\n⚠ No events detected")
            
except Exception as e:
    print(f"⚠ Semantic analysis error: {e}")
    import traceback
    traceback.print_exc()
    semantic_result = None

# ====================================================================
# STEP 4: Summary & Export
# ====================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

# ====================================================================
# FINAL SUMMARY
# ====================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print("\nEntity Class Distribution (after CLIP correction):")
class_counts = Counter()
for entity in perception_result.entities:
    final_class = getattr(entity, 'corrected_class', None) or (
        entity.object_class.value if hasattr(entity.object_class, 'value') else str(entity.object_class)
    )
    class_counts[final_class] += 1

for cls in sorted(class_counts.keys()):
    print(f"  {cls}: {class_counts[cls]}")

print("\n" + "="*80)
print("✓ Full pipeline test complete!")
print(f"  Results saved to: {output_dir}/")
print("="*80)
