#!/usr/bin/env python3
"""
Interactive Query System with Gemma 3.2-1B (MLX)

Integrates:
- Spatial memory retrieval
- Memgraph graph database queries
- LLM-based conversational interface
- Context-aware question answering
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from typing import List, Dict, Any, Optional
import numpy as np

# MLX for Gemma
try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("⚠️  mlx_lm not available. Install with: pip install mlx-lm")

from orion.graph.spatial_memory import SpatialMemorySystem


class SpatialIntelligenceAssistant:
    """
    Conversational assistant that answers questions about spatial scene understanding
    
    Features:
    - Retrieves relevant entities from spatial memory
    - Queries Memgraph for relationships
    - Uses Gemma LLM for natural language responses
    - Maintains conversation context
    """
    
    def __init__(
        self,
        memory_dir: Path,
        model_name: str = "mlx-community/gemma-2-2b-it-4bit",
        use_memgraph: bool = False
    ):
        self.memory_dir = memory_dir
        self.model_name = model_name
        self.use_memgraph = use_memgraph
        
        # Load spatial memory
        print(f"📚 Loading spatial memory from {memory_dir}...")
        self.spatial_memory = SpatialMemorySystem(memory_dir=memory_dir)
        stats = self.spatial_memory.get_statistics()
        print(f"   ✓ Loaded: {stats['total_entities']} entities, {stats['total_captions']} captions")
        print(f"   ✓ Total observations: {sum(e.observations_count for e in self.spatial_memory.entities.values())}")
        
        # Load Memgraph backend (optional)
        self.memgraph = None
        if use_memgraph:
            try:
                from orion.graph.memgraph_backend import MemgraphBackend
                self.memgraph = MemgraphBackend()
                print(f"   ✓ Connected to Memgraph")
            except Exception as e:
                print(f"   ⚠️  Memgraph not available: {e}")
                self.use_memgraph = False
        
        # Load Gemma LLM
        print(f"\n🧠 Loading Gemma model: {model_name}...")
        if not MLX_AVAILABLE:
            print("   ⚠️  MLX not available, running in context-only mode")
            self.model = None
            self.tokenizer = None
        else:
            try:
                self.model, self.tokenizer = load(model_name)
                print(f"   ✓ Gemma loaded (MLX optimized for Mac)")
            except Exception as e:
                print(f"   ⚠️  Failed to load Gemma: {e}")
                self.model = None
                self.tokenizer = None
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        self.last_retrieved_entities: List[int] = []
        
        print("\n✅ Assistant ready!\n")
    
    def retrieve_relevant_entities(
        self,
        query: str,
        top_k: int = 5
    ) -> List[tuple]:
        """
        Retrieve entities relevant to the query
        
        Simple keyword-based retrieval for now.
        In production, could use CLIP embeddings or semantic search.
        
        Returns:
            List of (entity_id, entity, relevance_score) tuples
        """
        query_lower = query.lower()
        
        # Extract potential entity types from query
        entity_keywords = {
            'person': ['person', 'people', 'human', 'man', 'woman'],
            'chair': ['chair', 'seat', 'sitting'],
            'table': ['table', 'desk', 'surface'],
            'tv': ['tv', 'television', 'screen', 'display'],
            'couch': ['couch', 'sofa', 'furniture'],
            'bed': ['bed', 'bedroom', 'sleep'],
            'laptop': ['laptop', 'computer'],
            'keyboard': ['keyboard'],
            'mouse': ['mouse'],
            'bottle': ['bottle', 'container'],
            'cup': ['cup', 'mug', 'drink'],
            'sink': ['sink', 'kitchen', 'bathroom'],
            'refrigerator': ['refrigerator', 'fridge'],
            'potted plant': ['plant', 'plants'],
            'dining table': ['dining', 'table']
        }
        
        # Position keywords
        position_keywords = {
            'left': ['left', 'leftmost'],
            'right': ['right', 'rightmost'],
            'center': ['center', 'middle', 'central'],
            'near': ['near', 'close', 'nearby', 'closest'],
            'far': ['far', 'distant', 'away'],
            'front': ['front', 'foreground'],
            'back': ['back', 'background']
        }
        
        scored_entities = []
        
        for entity_id, entity in self.spatial_memory.entities.items():
            score = 0.0
            
            # Class name matching
            for class_name, keywords in entity_keywords.items():
                if entity.class_name == class_name:
                    for keyword in keywords:
                        if keyword in query_lower:
                            score += 2.0
                            break
            
            # Caption matching (if exists)
            if entity.captions:
                for caption in entity.captions:
                    caption_lower = caption.lower()
                    # Count matching words
                    query_words = set(query_lower.split())
                    caption_words = set(caption_lower.split())
                    overlap = query_words & caption_words
                    score += len(overlap) * 0.5
            
            # Tracking duration bonus (longer tracked = more important)
            score += entity.observations_count * 0.01
            
            # Zone matching
            if 'zone' in query_lower:
                score += 1.0
            
            if score > 0:
                scored_entities.append((entity_id, entity, score))
        
        # Sort by score
        scored_entities.sort(key=lambda x: x[2], reverse=True)
        
        # Return top K
        return scored_entities[:top_k]
    
    def format_entity_context(self, entities: List[tuple]) -> str:
        """Format retrieved entities as context for LLM"""
        if not entities:
            return "No relevant entities found in the scene."
        
        context_parts = ["RETRIEVED ENTITIES FROM SPATIAL MEMORY:"]
        context_parts.append("=" * 60)
        
        for i, (entity_id, entity, score) in enumerate(entities, 1):
            context_parts.append(f"\n{i}. Entity #{entity_id} ({entity.class_name})")
            context_parts.append(f"   Tracked for: {entity.observations_count} frames")
            
            if entity.last_known_position_3d:
                x, y, z = entity.last_known_position_3d
                context_parts.append(f"   Position: ({x:.0f}, {y:.0f}, {z:.0f}) mm")
                context_parts.append(f"   Distance: {z/1000:.1f}m from camera")
            
            if hasattr(entity, 'zone_id'):
                context_parts.append(f"   Zone: {entity.zone_id}")
            
            if entity.captions:
                context_parts.append(f"   Description: {entity.captions[0]}")
            
            context_parts.append(f"   Relevance: {score:.2f}")
        
        context_parts.append("=" * 60)
        
        return "\n".join(context_parts)
    
    def get_scene_statistics(self) -> str:
        """Get overall scene statistics"""
        stats = self.spatial_memory.get_statistics()
        
        # Count by class
        class_counts = {}
        for entity in self.spatial_memory.entities.values():
            class_name = entity.class_name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Sort by count
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        
        stats_parts = [
            "SCENE STATISTICS:",
            "=" * 60,
            f"Total entities detected: {stats['total_entities']}",
            f"Total observations: {sum(e.observations_count for e in self.spatial_memory.entities.values())}",
            f"Entities with captions: {stats['total_captions']}",
            "",
            "Entity types detected:",
        ]
        
        for class_name, count in sorted_classes[:10]:
            stats_parts.append(f"  - {class_name}: {count}")
        
        if len(sorted_classes) > 10:
            stats_parts.append(f"  ... and {len(sorted_classes) - 10} more types")
        
        stats_parts.append("=" * 60)
        
        return "\n".join(stats_parts)
    
    def generate_response(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """Generate response using Gemma LLM"""
        
        if self.model is None:
            # Fallback: return context only
            return f"[LLM not available]\n\n{context}\n\nQuery: {query}"
        
        # Build prompt
        system_prompt = """You are a helpful AI assistant analyzing spatial scene understanding data from a video.
You have access to detected entities, their positions, tracking information, and descriptions.

Your role is to:
1. Answer questions about what's in the scene
2. Describe spatial relationships between objects
3. Provide insights about object positions and movements
4. Ask clarifying questions when needed
5. Make reasonable assumptions based on the data

Be conversational, helpful, and precise. Cite entity IDs when referencing specific objects."""

        # Format conversation history
        history_text = ""
        if conversation_history:
            for turn in conversation_history[-3:]:  # Last 3 turns
                history_text += f"User: {turn['query']}\nAssistant: {turn['response']}\n\n"
        
        # Full prompt
        full_prompt = f"""{system_prompt}

{context}

{history_text}User: {query}
Assistant:"""

        try:
            # Generate response
            response = generate(
                self.model,
                self.tokenizer,
                prompt=full_prompt,
                max_tokens=300,
                temp=0.7,
                top_p=0.95,
                verbose=False
            )
            
            return response.strip()
        
        except Exception as e:
            return f"Error generating response: {e}\n\nContext:\n{context}"
    
    def query(self, user_query: str) -> str:
        """
        Process a user query and return response
        
        Steps:
        1. Retrieve relevant entities from spatial memory
        2. Query Memgraph for relationships (if available)
        3. Format context
        4. Generate LLM response
        5. Update conversation history
        """
        
        print(f"\n🔍 Processing query: \"{user_query}\"")
        
        # Handle meta queries (statistics, help, etc.)
        if any(word in user_query.lower() for word in ['statistics', 'stats', 'summary', 'overview']):
            context = self.get_scene_statistics()
            response = self.generate_response(user_query, context, self.conversation_history)
        else:
            # Retrieve relevant entities
            relevant_entities = self.retrieve_relevant_entities(user_query, top_k=5)
            self.last_retrieved_entities = [eid for eid, _, _ in relevant_entities]
            
            # Format context
            context = self.format_entity_context(relevant_entities)
            
            # Query Memgraph for relationships (if available)
            if self.memgraph and relevant_entities:
                # TODO: Add Memgraph relationship queries
                pass
            
            # Generate response
            response = self.generate_response(user_query, context, self.conversation_history)
        
        # Update conversation history
        self.conversation_history.append({
            'query': user_query,
            'response': response,
            'retrieved_entities': self.last_retrieved_entities
        })
        
        return response
    
    def interactive_session(self):
        """Run interactive Q&A session"""
        print("="*80)
        print("🤖 ORION SPATIAL INTELLIGENCE ASSISTANT")
        print("="*80)
        print("\nAsk questions about the scene! Examples:")
        print("  - What objects are in the scene?")
        print("  - Where is the TV?")
        print("  - How many chairs are there?")
        print("  - What's near the couch?")
        print("  - Show me statistics")
        print("\nType 'quit' or 'exit' to end the session.")
        print("="*80 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                # Process query
                response = self.query(user_input)
                
                print(f"\n🤖 Assistant:\n{response}\n")
                print("-"*80 + "\n")
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description='Interactive Spatial Intelligence Query System')
    parser.add_argument(
        '--memory-dir',
        type=str,
        default='memory/spatial_intelligence',
        help='Path to spatial memory directory'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='mlx-community/gemma-2-2b-it-4bit',
        help='Gemma model to use (MLX format)'
    )
    parser.add_argument(
        '--use-memgraph',
        action='store_true',
        help='Enable Memgraph graph database queries'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Single query mode (non-interactive)'
    )
    
    args = parser.parse_args()
    
    # Create assistant
    assistant = SpatialIntelligenceAssistant(
        memory_dir=Path(args.memory_dir),
        model_name=args.model,
        use_memgraph=args.use_memgraph
    )
    
    # Single query mode or interactive
    if args.query:
        response = assistant.query(args.query)
        print(f"\n🤖 Assistant:\n{response}\n")
    else:
        assistant.interactive_session()


if __name__ == '__main__':
    main()
