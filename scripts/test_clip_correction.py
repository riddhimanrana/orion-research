"""Test CLIP class correction on existing entities"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.managers.model_manager import ModelManager
import json
import numpy as np
from collections import Counter

print("="*80)
print("CLIP CLASS CORRECTION TEST")
print("="*80)

# Load entities from previous run
entities_path = "results/full_pipeline/entities.json"
print(f"\nLoading entities from: {entities_path}")

with open(entities_path) as f:
    entities_data = json.load(f)

print(f"Loaded {len(entities_data)} entities")

# Get CLIP model
print("\nLoading CLIP model...")
model_manager = ModelManager.get_instance()
clip_model = model_manager.clip

# Define common misclassifications to check
class_corrections = {
    "tv": ["monitor", "television", "TV screen", "computer display"],
    "cat": ["dog", "teddy bear", "stuffed animal", "toy animal", "plush toy"],  
    "couch": ["sofa", "chair", "seat"],
    "dining table": ["desk", "table", "workstation"],
}

print("\n" + "="*80)
print("CLIP Class Verification")
print("-" * 80)

corrections = {}
corrected_count = 0

for entity_data in entities_data:
    entity_id = entity_data["entity_id"]
    original_class = entity_data["object_class"]
    
    # Check if this class needs verification
    if original_class in class_corrections:
        # Check if entity has average_embedding (DINO) or clip_embedding
        embedding = None
        embedding_type = None
        
        if "average_embedding" in entity_data and entity_data["average_embedding"]:
            embedding = np.array(entity_data["average_embedding"])
            embedding_type = "DINO (1024-dim)"
        elif "clip_embedding" in entity_data and entity_data["clip_embedding"]:
            embedding = np.array(entity_data["clip_embedding"])
            embedding_type = "CLIP visual (512-dim)"
        
        if embedding is None:
            print(f"\n{entity_id} ({original_class}): No embedding available")
            continue
            
        print(f"\n{entity_id} ({original_class}) - using {embedding_type}:")
        
        # Get candidate labels
        candidate_labels = [original_class] + class_corrections[original_class]
        
        # For DINO embeddings (1024-dim), we can't directly compare with CLIP text (512-dim)
        # We need to re-encode the image with CLIP, or use a different approach
        if embedding.shape[0] == 1024:
            print(f"  ⚠ DINO embeddings not compatible with CLIP text embeddings")
            print(f"    Need to re-extract CLIP visual embeddings from images")
            continue
        
        try:
            # Get CLIP text embeddings for each candidate
            text_embeddings = []
            for label in candidate_labels:
                text_emb = clip_model.encode_text(label, normalize=True)
                text_embeddings.append(text_emb)
            
            text_embeddings = np.array(text_embeddings)
            
            # Ensure visual embedding is normalized
            visual_emb = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            # Compute cosine similarities
            similarities = text_embeddings @ visual_emb
            best_idx = similarities.argmax()
            best_label = candidate_labels[best_idx]
            confidence = float(similarities[best_idx])
            
            print(f"  Similarities:")
            for i, (label, sim) in enumerate(zip(candidate_labels, similarities)):
                marker = " ✓" if i == best_idx else ""
                print(f"    {label}: {sim:.3f}{marker}")
            
            if best_label != original_class and confidence > 0.28:
                print(f"  ⟹ CORRECTION: '{original_class}' → '{best_label}' (confidence: {confidence:.3f})")
                corrections[entity_id] = {
                    "original": original_class,
                    "corrected": best_label,
                    "confidence": confidence
                }
                corrected_count += 1
            else:
                print(f"  ⟹ VERIFIED: '{original_class}' (confidence: {similarities[0]:.3f})")
                
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            import traceback
            traceback.print_exc()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nTotal entities checked: {len([e for e in entities_data if e['object_class'] in class_corrections])}")
print(f"Corrections made: {corrected_count}")

if corrections:
    print("\nCorrections:")
    for entity_id, corr in corrections.items():
        print(f"  {entity_id}: {corr['original']} → {corr['corrected']} ({corr['confidence']:.3f})")
else:
    print("\n⚠ No corrections made (entities likely use DINO embeddings)")
    print("  Recommendation: Re-run perception with config.embedding.backend = 'clip'")
