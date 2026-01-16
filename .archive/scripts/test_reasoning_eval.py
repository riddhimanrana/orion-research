#!/usr/bin/env python3
import time
import json
import logging
from typing import List, Dict, Any
from orion.query.rag_v2 import OrionRAG
from orion.query.reasoning import ReasoningConfig

logging.basicConfig(level=logging.INFO)

def run_reasoning_eval(episode_id: str, questions: List[str]):
    rag = OrionRAG(
        ollama_url="http://localhost:11434",
        llm_model="qwen2.5:14b-instruct-q8_0",
        enable_llm=True
    )
    
    results = []
    print(f"\nEvaluating Stage 6 Reasoning for episode: {episode_id}")
    print("=" * 80)
    
    for q in questions:
        print(f"\nQuestion: {q}")
        start_time = time.time()
        
        # We use query which handles Retrieval + Synthesis
        result = rag.query(q, use_llm=True)
        
        duration = time.time() - start_time
        print(f"Answer: {result.answer}")
        print(f"Latency: {duration:.2f}s")
        
        results.append({
            "question": q,
            "answer": result.answer,
            "cypher": result.cypher_query,
            "evidence_count": len(result.evidence),
            "latency": duration
        })
        
    return results

if __name__ == "__main__":
    # Test questions for test.mp4 (which is currently in Memgraph)
    questions = [
        "What objects were detected in the video?",
        "Was there any interaction between a person and a laptop?",
        "When was the table first seen?",
        "Tell me about the spatial relationships between objects."
    ]
    
    results = run_reasoning_eval("eval_test", questions)
    
    with open("results/reasoning_eval_test.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Reasoning evaluation complete. Results saved to results/reasoning_eval_test.json")
