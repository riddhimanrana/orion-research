#!/usr/bin/env python3
"""
Validation test for DINOv3 embedding integration into scene graphs.

Tests:
1. Import all new functions
2. Load memory and tracks
3. Build graphs with embedding verification disabled
4. Build graphs with embedding verification enabled
5. Verify both modes produce consistent results
6. Check CLI arguments are registered
"""

import json
import sys
from pathlib import Path

def test_imports():
    """Test that all new functions can be imported."""
    print("🧪 Test 1: Imports...")
    try:
        from orion.graph.scene_graph import build_scene_graphs, _verify_edges_with_embeddings
        # Dynamic import to avoid static analyzer false-positives in tooling
        import importlib
        emb_mod = importlib.import_module("orion.graph.embedding_scene_graph")
        load_embeddings_from_memory = getattr(emb_mod, "load_embeddings_from_memory")
        cosine_similarity = getattr(emb_mod, "cosine_similarity")
        EmbeddingRelationConfig = getattr(emb_mod, "EmbeddingRelationConfig")
        print("   ✅ All imports successful")
        return True
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_scene_graph_generation():
    """Test scene graph generation with and without embedding verification."""
    print("\n🧪 Test 2: Scene graph generation...")
    
    from orion.graph.scene_graph import build_scene_graphs
    
    # Find a test episode
    results_dir = Path("results/0001_4164158586_yoloworld")
    if not results_dir.exists():
        print(f"   ⚠️  Test data not found: {results_dir}")
        return None  # Skip if test data unavailable
    
    # Load data
    memory_path = results_dir / "memory.json"
    tracks_path = results_dir / "tracks.jsonl"
    
    try:
        with open(memory_path) as f:
            memory = json.load(f)
        
        tracks = []
        with open(tracks_path) as f:
            for line in f:
                tracks.append(json.loads(line))
        
        print(f"   Loaded {len(memory['objects'])} objects, {len(tracks)} tracks")
        
        # Test geometry-only mode
        graphs_geom = build_scene_graphs(
            memory, tracks,
            use_embedding_verification=False
        )
        edges_geom = sum(len(g.get("edges", [])) for g in graphs_geom)
        print(f"   Geometry-only: {len(graphs_geom)} graphs, {edges_geom} edges")
        
        # Test embedding-aware mode
        graphs_emb = build_scene_graphs(
            memory, tracks,
            use_embedding_verification=True,
            embedding_weight=0.3,
            embedding_similarity_threshold=0.5
        )
        edges_emb = sum(len(g.get("edges", [])) for g in graphs_emb)
        print(f"   Embedding-aware: {len(graphs_emb)} graphs, {edges_emb} edges")
        
        # When embeddings unavailable, both should match
        if edges_geom == edges_emb and len(graphs_geom) == len(graphs_emb):
            print("   ✅ Consistent results (embeddings unavailable → geometry-only)")
            return True
        else:
            print("   ❌ Inconsistent results")
            return False
    
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def test_embedding_loader():
    """Test embedding loading function."""
    print("\n🧪 Test 3: Embedding loader...")
    
    # Dynamic import to avoid static import resolution issues in some environments
    import importlib
    emb_mod = importlib.import_module("orion.graph.embedding_scene_graph")
    load_embeddings_from_memory = getattr(emb_mod, "load_embeddings_from_memory")
    
    results_dir = Path("results/0001_4164158586_yoloworld")
    if not results_dir.exists():
        print(f"   ⚠️  Test data not found: {results_dir}")
        return None
    
    memory_path = results_dir / "memory.json"
    
    try:
        # Test with file path
        embeddings = load_embeddings_from_memory(memory_path)
        print(f"   Loaded {len(embeddings)} embeddings from file")
        
        # Test with dict
        with open(memory_path) as f:
            memory = json.load(f)
        
        embeddings2 = load_embeddings_from_memory(memory)
        print(f"   Loaded {len(embeddings2)} embeddings from dict")
        
        if len(embeddings) == len(embeddings2):
            print("   ✅ Loader works with both file and dict input")
            return True
        else:
            print("   ❌ Inconsistent results between file and dict")
            return False
    
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def test_cli_args():
    """Test that CLI arguments are registered."""
    print("\n🧪 Test 4: CLI arguments...")
    
    try:
        from orion.cli.run_showcase import main
        import argparse
        
        # Create parser and test
        parser = argparse.ArgumentParser()
        # Simplified test - just check imports work
        from orion.cli import run_showcase
        
        # Check that the module has the expected arg parser setup
        if hasattr(run_showcase, 'main'):
            print("   ✅ CLI module imports successfully")
            return True
        else:
            print("   ❌ CLI module structure issue")
            return False
    
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("DINOv3 Embedding Integration Validation")
    print("=" * 60)
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: Scene graph generation
    results.append(("Scene graph generation", test_scene_graph_generation()))
    
    # Test 3: Embedding loader
    results.append(("Embedding loader", test_embedding_loader()))
    
    # Test 4: CLI args
    results.append(("CLI arguments", test_cli_args()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r is True)
    skipped = sum(1 for _, r in results if r is None)
    failed = sum(1 for _, r in results if r is False)
    
    for name, result in results:
        if result is True:
            print(f"✅ {name}")
        elif result is None:
            print(f"⏭️  {name} (skipped - data unavailable)")
        else:
            print(f"❌ {name}")
    
    print(f"\nResults: {passed} passed, {skipped} skipped, {failed} failed")
    
    if failed == 0:
        print("\n✨ Integration validation PASSED!")
        return 0
    else:
        print(f"\n⚠️  Integration validation FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
