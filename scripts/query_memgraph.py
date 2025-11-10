"""
Interactive Query Interface for Memgraph Video Understanding Graph

Usage:
    python scripts/query_memgraph.py --interactive
    python scripts/query_memgraph.py --query "What color was the book?"
    python scripts/query_memgraph.py --stats

Real-time graph queries powered by Memgraph (C++ native, 1000+ TPS)
Query-time FastVLM captioning: Captions generated on-demand (<300ms)
"""

import argparse
import sys
import time
from pathlib import Path

# Add orion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.graph.memgraph_backend import MemgraphBackend


# Global FastVLM instance (lazy loaded)
_fastvlm = None


def get_fastvlm():
    """Lazy load FastVLM model"""
    global _fastvlm
    if _fastvlm is None:
        print("  Loading FastVLM model (first query only)...")
        try:
            from orion.backends.fastvlm_backend import FastVLMBackend
            _fastvlm = FastVLMBackend(model_name="fastvlm-0.5b")
            print("  ✓ FastVLM loaded")
        except Exception as e:
            print(f"  ⚠️  Failed to load FastVLM: {e}")
            _fastvlm = None
    return _fastvlm


def generate_caption_on_demand(crop_path: str) -> str:
    """Generate caption for a crop on-demand using FastVLM"""
    try:
        start = time.time()
        
        # Load FastVLM
        fastvlm = get_fastvlm()
        if fastvlm is None:
            return None
        
        # Load crop image
        import cv2
        from PIL import Image
        crop = cv2.imread(crop_path)
        if crop is None:
            return None
        
        # Convert BGR → RGB → PIL
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        
        # Generate caption
        prompt = "Describe this object in detail: what is it, what color, what's it doing?"
        caption = fastvlm.generate_description(
            crop_pil,
            prompt,
            max_tokens=128,
            temperature=0.3
        )
        
        elapsed = time.time() - start
        print(f"  ⏱️  Caption generated in {elapsed*1000:.0f}ms")
        
        return caption
        
    except Exception as e:
        print(f"  ⚠️  Caption generation failed: {e}")
        return None


def parse_natural_language_query(query: str):
    """
    Parse natural language query into graph operations
    
    Supported patterns:
    - "What color was the [object]?" → Find object, extract color from caption
    - "Where was the [object]?" → Find object, return zone info
    - "What objects were near [object]?" → Find spatial relationships
    - "What happened after [object] appeared?" → Temporal queries
    """
    query_lower = query.lower()
    
    # Extract object class
    import re
    
    # Color queries
    if "what color" in query_lower or "color of" in query_lower:
        match = re.search(r"(?:the |a )?(\w+)", query_lower.split("color")[-1])
        if match:
            object_class = match.group(1).strip()
            return {
                "type": "color",
                "object_class": object_class
            }
    
    # Location queries
    if "where" in query_lower:
        match = re.search(r"(?:the |a )?(\w+)", query_lower.split("where")[-1])
        if match:
            object_class = match.group(1).strip()
            return {
                "type": "location",
                "object_class": object_class
            }
    
    # Spatial relationship queries
    if "near" in query_lower or "next to" in query_lower:
        match = re.search(r"(?:near|next to) (?:the |a )?(\w+)", query_lower)
        if match:
            object_class = match.group(1).strip()
            return {
                "type": "spatial",
                "object_class": object_class
            }
    
    # Temporal queries
    if "after" in query_lower or "before" in query_lower:
        match = re.search(r"(?:after|before) (?:the |a )?(\w+)", query_lower)
        if match:
            object_class = match.group(1).strip()
            return {
                "type": "temporal",
                "object_class": object_class
            }
    
    return {
        "type": "unknown",
        "raw_query": query
    }


def execute_query(backend: MemgraphBackend, parsed_query: dict):
    """Execute parsed query against Memgraph"""
    
    query_type = parsed_query["type"]
    
    if query_type == "color":
        object_class = parsed_query["object_class"]
        print(f"\n🔍 Searching for {object_class}...")
        
        results = backend.query_entity_by_class(object_class, limit=5)
        
        if not results:
            print(f"❌ No {object_class} found in video")
            return
        
        # Check for captions with color information
        for entity in results:
            entity_id = entity["entity_id"]
            observations = entity["observations"]
            
            # Look for observation with caption
            caption_found = False
            for obs in observations:
                if obs.get("caption"):
                    caption_found = True
                    print(f"\n✅ Found {object_class} (entity #{entity_id})")
                    print(f"   Frame: {obs['frame_idx']}")
                    print(f"   Caption: {obs['caption']}")
                    
                    # Extract color from caption (simple heuristic)
                    caption_lower = obs['caption'].lower()
                    colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 
                             'brown', 'gray', 'grey', 'orange', 'purple', 'pink']
                    found_colors = [c for c in colors if c in caption_lower]
                    
                    if found_colors:
                        print(f"   💡 Color: {', '.join(found_colors).upper()}")
                    return
            
            # No caption found - try query-time captioning
            if not caption_found:
                print(f"\n🔍 Found {object_class} (entity #{entity_id}) - no cached caption")
                print(f"   Generating caption on-demand...")
                
                # Find observation with crop_path
                for obs in observations:
                    crop_path = obs.get('crop_path')
                    if crop_path and Path(crop_path).exists():
                        caption = generate_caption_on_demand(crop_path)
                        
                        if caption:
                            print(f"   ✓ Generated caption: {caption}")
                            
                            # Extract color
                            caption_lower = caption.lower()
                            colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 
                                     'brown', 'gray', 'grey', 'orange', 'purple', 'pink']
                            found_colors = [c for c in colors if c in caption_lower]
                            
                            if found_colors:
                                print(f"   💡 Color: {', '.join(found_colors).upper()}")
                            
                            # Cache caption in Memgraph for next time
                            # (Could implement backend.update_caption() here)
                            return
                        else:
                            print(f"   ⚠️  Caption generation failed")
                            break
                
                print(f"   First seen: {entity['first_seen']:.1f}s")
                print(f"   Observations: {len(observations)}")
    
    elif query_type == "location":
        object_class = parsed_query["object_class"]
        print(f"\n🔍 Searching for {object_class} location...")
        
        results = backend.query_entity_by_class(object_class, limit=5)
        
        if not results:
            print(f"❌ No {object_class} found in video")
            return
        
        for entity in results:
            entity_id = entity["entity_id"]
            observations = entity["observations"]
            
            print(f"\n✅ {object_class.upper()} (entity #{entity_id})")
            print(f"   First seen: {entity['first_seen']:.1f}s")
            print(f"   Last seen: {entity['last_seen']:.1f}s")
            print(f"   Observations: {len(observations)}")
            
            # Show zones
            zones = set(obs.get('zone_id') for obs in observations if obs.get('zone_id') is not None)
            if zones:
                print(f"   Zones: {', '.join(map(str, zones))}")
    
    elif query_type == "spatial":
        object_class = parsed_query["object_class"]
        print(f"\n🔍 Finding objects near {object_class}...")
        
        # First find the target object
        results = backend.query_entity_by_class(object_class, limit=1)
        
        if not results:
            print(f"❌ No {object_class} found in video")
            return
        
        entity_id = results[0]["entity_id"]
        
        # Query spatial relationships
        relationships = backend.query_spatial_relationships(entity_id)
        
        if not relationships:
            print(f"⚠️  No spatial relationships recorded for {object_class}")
            return
        
        print(f"\n✅ Objects near {object_class}:")
        for rel in relationships:
            print(f"   {rel['relationship']}: {rel['related_class']} (conf: {rel['confidence']:.2f})")
    
    elif query_type == "temporal":
        object_class = parsed_query["object_class"]
        print(f"\n🔍 Finding objects that appeared with {object_class}...")
        
        # First find the target object
        results = backend.query_entity_by_class(object_class, limit=1)
        
        if not results:
            print(f"❌ No {object_class} found in video")
            return
        
        entity_id = results[0]["entity_id"]
        
        # Query temporal coexistence
        coexisting = backend.query_temporal_coexistence(entity_id, time_window=5.0)
        
        if not coexisting:
            print(f"⚠️  No other objects appeared with {object_class}")
            return
        
        print(f"\n✅ Objects that appeared with {object_class}:")
        for item in coexisting[:10]:
            print(f"   {item['class_name']}: appeared together {item['coexistence_count']} times")
    
    else:
        print(f"❌ Could not parse query: {parsed_query.get('raw_query', '')}")
        print("Try queries like:")
        print("  - What color was the book?")
        print("  - Where was the laptop?")
        print("  - What objects were near the person?")


def interactive_mode(backend: MemgraphBackend):
    """Interactive query loop"""
    print("\n" + "="*80)
    print("🎯 MEMGRAPH INTERACTIVE QUERY INTERFACE")
    print("="*80)
    print("\nConnected to video understanding graph")
    
    # Show statistics
    stats = backend.get_statistics()
    print(f"\n📊 Graph Statistics:")
    print(f"   Entities: {stats['entities']}")
    print(f"   Frames: {stats['frames']}")
    print(f"   Zones: {stats['zones']}")
    print(f"   Observations: {stats['observations']}")
    print(f"   Spatial Relationships: {stats['spatial_relationships']}")
    
    print("\n💡 Example queries:")
    print("   - What color was the book?")
    print("   - Where was the laptop?")
    print("   - What objects were near the person?")
    print("   - What happened after the book appeared?")
    print("\nType 'quit' or 'exit' to quit\n")
    
    while True:
        try:
            query = input("❓ Query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Parse and execute
            parsed = parse_natural_language_query(query)
            execute_query(backend, parsed)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Query Memgraph video understanding graph"
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive query mode'
    )
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Single query to execute'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show graph statistics'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Memgraph host (default: localhost)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7687,
        help='Memgraph port (default: 7687)'
    )
    
    args = parser.parse_args()
    
    try:
        # Connect to Memgraph
        print(f"Connecting to Memgraph at {args.host}:{args.port}...")
        backend = MemgraphBackend(host=args.host, port=args.port)
        
        if args.stats:
            stats = backend.get_statistics()
            print("\n📊 Graph Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        elif args.query:
            parsed = parse_natural_language_query(args.query)
            execute_query(backend, parsed)
        
        elif args.interactive:
            interactive_mode(backend)
        
        else:
            parser.print_help()
        
        backend.close()
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure Memgraph is running:")
        print("  cd memgraph-platform")
        print("  docker compose up -d")
        sys.exit(1)


if __name__ == '__main__':
    main()
