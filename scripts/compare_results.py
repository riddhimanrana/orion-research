import json
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# Paths
workspace_root = Path(__file__).resolve().parents[1]
pipeline_output_path = workspace_root / "results/test_run/pipeline_output.json"
gemini_output_path = workspace_root / "results/test_run/gemini_analysis.json"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    if not pipeline_output_path.exists():
        print(f"Error: Pipeline output not found at {pipeline_output_path}")
        return
    if not gemini_output_path.exists():
        print(f"Error: Gemini output not found at {gemini_output_path}")
        return

    print("Loading data...")
    pipeline_data = load_json(pipeline_output_path)
    gemini_data = load_json(gemini_output_path)

    pipeline_entities = pipeline_data.get("entities", [])
    gemini_entities = gemini_data.get("entities", [])

    print(f"Pipeline found {len(pipeline_entities)} entities.")
    print(f"Gemini found {len(gemini_entities)} entities.")

    print("\nLoading SentenceTransformer for semantic comparison...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Prepare descriptions
    pipeline_descs = [e.get("description", "") for e in pipeline_entities]
    gemini_descs = [e.get("description", "") for e in gemini_entities]

    # Compute embeddings
    pipeline_embeddings = model.encode(pipeline_descs, convert_to_tensor=True)
    gemini_embeddings = model.encode(gemini_descs, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_scores = util.cos_sim(gemini_embeddings, pipeline_embeddings)

    print("\n--- Matching Gemini Entities to Pipeline Entities ---\n")

    matches = []

    for i, gemini_entity in enumerate(gemini_entities):
        best_match_idx = cosine_scores[i].argmax().item()
        score = cosine_scores[i][best_match_idx].item()
        
        pipeline_entity = pipeline_entities[best_match_idx]
        
        matches.append({
            "gemini_class": gemini_entity.get("object_class"),
            "gemini_desc": gemini_entity.get("description"),
            "pipeline_id": pipeline_entity.get("entity_id"),
            "pipeline_class": pipeline_entity.get("object_class"),
            "pipeline_desc": pipeline_entity.get("description"),
            "score": score
        })

    # Sort by score descending
    matches.sort(key=lambda x: x["score"], reverse=True)

    for m in matches:
        print(f"Match Score: {m['score']:.4f}")
        print(f"  Gemini:   {m['gemini_class']} - {m['gemini_desc'][:100]}...")
        print(f"  Pipeline: {m['pipeline_class']} ({m['pipeline_id']}) - {m['pipeline_desc'][:100]}...")
        print("-" * 80)

if __name__ == "__main__":
    main()
