"""orion query - Query memory with natural language using Ollama"""

import json
import logging
from pathlib import Path
import requests

logger = logging.getLogger(__name__)


def run_query(args) -> int:
    """Query memory with natural language."""
    
    episode_dir = Path("results") / args.episode
    memory_path = episode_dir / "memory.json"
    
    if not memory_path.exists():
        print(f"No memory found for episode: {args.episode}")
        print("Run: orion analyze --episode <name> first")
        return 1
    
    # Load memory
    with open(memory_path) as f:
        memory = json.load(f)
    
    # Load scene graph if available
    scene_graph_path = episode_dir / "scene_graph.jsonl"
    scene_contexts = []
    if scene_graph_path.exists():
        with open(scene_graph_path) as f:
            for line in f:
                if line.strip():
                    scene_contexts.append(json.loads(line))
    
    # Load VLM scenes if available
    vlm_path = episode_dir / "vlm_scene.jsonl"
    vlm_scenes = []
    if vlm_path.exists():
        with open(vlm_path) as f:
            for line in f:
                if line.strip():
                    vlm_scenes.append(json.loads(line))
    
    # Build context for LLM
    context_parts = []
    
    # Add memory objects
    context_parts.append("## Memory Objects")
    for obj in memory.get("objects", []):
        desc = obj.get("description", "")
        attrs = obj.get("attributes", {})
        context_parts.append(f"- **{obj['canonical_label']}** (ID: {obj['id']})")
        if desc:
            context_parts.append(f"  Description: {desc}")
        if attrs:
            context_parts.append(f"  Attributes: {attrs}")
        context_parts.append(f"  Seen {obj['total_observations']} times, frames {obj['first_frame']}-{obj['last_frame']}")
    
    # Add scene summaries
    if vlm_scenes:
        context_parts.append("\n## Scene Descriptions")
        for scene in vlm_scenes[:5]:  # Limit to 5 scenes
            context_parts.append(f"- Frame {scene.get('frame_id', '?')}: {scene.get('scene_caption', '')}")
    
    # Add relationships
    if scene_contexts:
        context_parts.append("\n## Object Relationships")
        relations = set()
        for ctx in scene_contexts[:10]:  # Sample relationships
            for edge in ctx.get("edges", []):
                relations.add(f"{edge['subject']} {edge['predicate']} {edge['object']}")
        for rel in list(relations)[:20]:
            context_parts.append(f"- {rel}")
    
    context = "\n".join(context_parts)
    
    # Build prompt
    system_prompt = """You are an AI assistant that answers questions about video content based on memory.
You have access to:
1. Memory Objects: tracked objects with descriptions and temporal information
2. Scene Descriptions: VLM-generated descriptions of key frames
3. Object Relationships: spatial/temporal relationships between objects

Answer questions concisely based on the provided context. If you're unsure, say so.
Always cite which objects or scenes your answer is based on."""
    
    user_prompt = f"""Context from video memory:

{context}

Question: {args.query}

Answer:"""
    
    # Query Ollama
    ollama_url = args.ollama_url or "http://localhost:11434"
    model = args.model or "qwen2.5:14b-instruct-q8_0"
    
    print(f"\n  Querying {model}...")
    print(f"  Question: {args.query}\n")
    
    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_ctx": 8192,
                }
            },
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        answer = result.get("response", "No response")
        
        print("  Answer:")
        print("  " + "─" * 60)
        for line in answer.split("\n"):
            print(f"  {line}")
        print("  " + "─" * 60)
        
        # Show timing
        if "total_duration" in result:
            duration_ms = result["total_duration"] / 1_000_000
            print(f"\n  Generated in {duration_ms:.0f}ms")
        
        return 0
        
    except requests.exceptions.ConnectionError:
        print(f"  ✗ Cannot connect to Ollama at {ollama_url}")
        print("  Make sure Ollama is running: ollama serve")
        return 1
    except requests.exceptions.Timeout:
        print("  ✗ Ollama request timed out")
        return 1
    except Exception as e:
        print(f"  ✗ Error querying Ollama: {e}")
        return 1


def run_query_cypher(args) -> int:
    """Query Memgraph with Cypher."""
    
    print("\n  Cypher Query Interface")
    print("  " + "─" * 60)
    
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("  ✗ neo4j package not installed")
        print("  Run: pip install neo4j")
        return 1
    
    memgraph_uri = args.memgraph_uri or "bolt://localhost:7687"
    
    try:
        driver = GraphDatabase.driver(memgraph_uri, auth=("", ""))
        
        with driver.session() as session:
            result = session.run(args.cypher)
            records = list(result)
            
            if not records:
                print("  No results")
                return 0
            
            # Print as table
            keys = records[0].keys()
            print("  " + " | ".join(keys))
            print("  " + "-+-".join(["-" * 20 for _ in keys]))
            
            for record in records[:50]:  # Limit output
                values = [str(record[k])[:20] for k in keys]
                print("  " + " | ".join(values))
            
            if len(records) > 50:
                print(f"\n  ... and {len(records) - 50} more rows")
        
        driver.close()
        return 0
        
    except Exception as e:
        print(f"  ✗ Memgraph error: {e}")
        return 1
