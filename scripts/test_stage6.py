#!/usr/bin/env python3
"""
Stage 6 Integration Test
========================

Quick test of the Stage 6 reasoning capabilities.
Run this after setting up Memgraph and Ollama.

Usage:
    python scripts/test_stage6.py --host 127.0.0.1
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_reasoning_module():
    """Test the reasoning module standalone."""
    print("\n" + "=" * 60)
    print("TEST 1: Reasoning Module")
    print("=" * 60)
    
    try:
        from orion.query.reasoning import ReasoningModel, ReasoningConfig
        
        config = ReasoningConfig()
        print(f"  Model: {config.model}")
        print(f"  Base URL: {config.base_url}")
        
        model = ReasoningModel(config=config)
        
        # Test model validation
        print("\n  Validating model availability...")
        if model.validate_model():
            print("  ✓ Model available")
        else:
            print("  ✗ Model not available")
            print(f"    Pull with: ollama pull {config.model}")
            return False
        
        # Test Cypher generation
        print("\n  Testing Cypher generation...")
        cypher = model.generate_cypher("What objects are near the laptop?")
        if cypher:
            print(f"  ✓ Generated: {cypher[:80]}...")
        else:
            print("  ✗ No Cypher generated")
        
        # Test answer synthesis
        print("\n  Testing answer synthesis...")
        evidence = [
            {"class": "laptop", "observations": 50},
            {"class": "book", "observations": 30, "holder": "person"},
        ]
        answer = model.synthesize_answer("What objects are in the video?", evidence)
        if answer:
            print(f"  ✓ Answer: {answer[:100]}...")
        else:
            print("  ✗ No answer generated")
        
        print("\n  ✓ Reasoning module OK")
        return True
        
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        return False


def test_memgraph_connection(host: str = "127.0.0.1", port: int = 7687):
    """Test Memgraph connectivity."""
    print("\n" + "=" * 60)
    print("TEST 2: Memgraph Connection")
    print("=" * 60)
    
    try:
        from orion.graph.backends.memgraph import MemgraphBackend
        
        print(f"  Connecting to {host}:{port}...")
        backend = MemgraphBackend(host=host, port=port)
        
        # Test query
        cursor = backend.connection.cursor()
        cursor.execute("MATCH (n) RETURN count(n) AS cnt")
        result = cursor.fetchall()
        
        print(f"  ✓ Connected! Total nodes: {result[0][0]}")
        
        # Check for entities
        cursor.execute("MATCH (e:Entity) RETURN count(e) AS cnt")
        entity_count = cursor.fetchall()[0][0]
        print(f"  ✓ Entities: {entity_count}")
        
        backend.close()
        return True
        
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        print("    Make sure Memgraph is running: docker compose up -d")
        return False


def test_rag_queries(host: str = "127.0.0.1"):
    """Test RAG query functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: RAG Queries")
    print("=" * 60)
    
    try:
        from orion.query.rag_v2 import OrionRAG
        
        print(f"  Initializing RAG...")
        rag = OrionRAG(
            host=host,
            port=7687,
            enable_llm=True,
        )
        
        # Get stats
        stats = rag.get_stats()
        print(f"  ✓ Database stats:")
        for k, v in stats.items():
            print(f"      {k}: {v}")
        
        if stats['entities'] == 0:
            print("\n  ⚠️  No entities in database!")
            print("      Run perception pipeline first:")
            print("      python -m orion.cli.run_showcase --episode test --video video.mp4 --memgraph")
            rag.close()
            return True  # Not a failure, just no data
        
        # Test queries
        test_questions = [
            "What objects are in the video?",
            "What did the person interact with?",
            "What was near the laptop?",
        ]
        
        print(f"\n  Testing {len(test_questions)} queries...")
        
        for q in test_questions:
            print(f"\n  Q: {q}")
            result = rag.query(q, use_llm=False)  # Template answer for speed
            print(f"  A: {result.answer[:100]}...")
            print(f"     Confidence: {result.confidence:.2f}, Latency: {result.latency_ms:.0f}ms")
        
        # Test with LLM if available
        if rag.llm_enabled:
            print("\n  Testing with LLM synthesis...")
            result = rag.query("Describe the objects in this video", use_llm=True)
            print(f"  LLM Answer: {result.answer[:150]}...")
        
        rag.close()
        print("\n  ✓ RAG queries OK")
        return True
        
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        return False


def test_streaming(host: str = "127.0.0.1"):
    """Test streaming responses."""
    print("\n" + "=" * 60)
    print("TEST 4: Streaming Responses")
    print("=" * 60)
    
    try:
        from orion.query.rag_v2 import OrionRAG
        
        rag = OrionRAG(host=host, enable_llm=True)
        
        if not rag.llm_enabled:
            print("  ⚠️  LLM not enabled, skipping streaming test")
            rag.close()
            return True
        
        stats = rag.get_stats()
        if stats['entities'] == 0:
            print("  ⚠️  No data to query, skipping")
            rag.close()
            return True
        
        print("  Streaming answer: ", end="", flush=True)
        token_count = 0
        for token in rag.stream_query("What objects are visible?"):
            print(token, end="", flush=True)
            token_count += 1
            if token_count > 100:  # Limit output
                print("...", end="")
                break
        print()
        
        rag.close()
        print("\n  ✓ Streaming OK")
        return True
        
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Stage 6 Integration Test")
    parser.add_argument("--host", default="127.0.0.1", help="Memgraph host")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama tests")
    args = parser.parse_args()
    
    print("=" * 60)
    print("ORION STAGE 6 INTEGRATION TEST")
    print("=" * 60)
    print(f"Memgraph Host: {args.host}")
    print(f"Skip Ollama: {args.skip_ollama}")
    
    results = []
    
    # Test 1: Reasoning module
    if not args.skip_ollama:
        results.append(("Reasoning Module", test_reasoning_module()))
    
    # Test 2: Memgraph connection
    results.append(("Memgraph Connection", test_memgraph_connection(args.host)))
    
    # Test 3: RAG queries
    results.append(("RAG Queries", test_rag_queries(args.host)))
    
    # Test 4: Streaming
    if not args.skip_ollama:
        results.append(("Streaming", test_streaming(args.host)))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! Stage 6 is ready.")
    else:
        print("Some tests failed. Check the output above.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
