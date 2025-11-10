"""
Interactive Spatial Intelligence System

This is the "historian" interface - an intelligent query system that:
- Remembers everything across sessions
- Asks clarifying questions when needed
- Maintains conversation context
- Provides rich spatial and semantic understanding

Think of this as talking to a model that has perfect memory and spatial understanding
of everything that happened in the space.
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint
import readline  # For command history

# Add orion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.graph.memgraph_backend import MemgraphBackend
from orion.graph.spatial_memory import SpatialMemorySystem


console = Console()


class SpatialIntelligenceAssistant:
    """
    Intelligent assistant with persistent spatial memory
    
    Features:
    - Contextual understanding ("it", "that", "the same one")
    - Asks clarifying questions
    - Remembers conversation history
    - Rich spatial and semantic reasoning
    """
    
    def __init__(self):
        # Connect to real-time graph
        try:
            self.backend = MemgraphBackend()
            console.print("✓ Connected to Memgraph", style="green")
        except Exception as e:
            console.print(f"⚠️  Could not connect to Memgraph: {e}", style="yellow")
            console.print("   Starting with memory-only mode", style="yellow")
            self.backend = None
        
        # Load persistent memory
        self.memory = SpatialMemorySystem(memory_dir=Path("memory/spatial_intelligence"))
        
        stats = self.memory.get_statistics()
        if stats['total_entities'] > 0:
            console.print(f"✓ Loaded memory: {stats['total_entities']} entities, "
                         f"{stats['total_captions']} captions", style="green")
        else:
            console.print("📝 Starting fresh memory", style="cyan")
    
    def sync_from_memgraph(self):
        """Sync latest observations from Memgraph into persistent memory"""
        if not self.backend:
            return
        
        console.print("\n🔄 Syncing from Memgraph...", style="cyan")
        
        try:
            # Get all entities from Memgraph
            stats = self.backend.get_statistics()
            console.print(f"   Found {stats['entities']} entities in graph", style="dim")
            
            # Query each entity class
            # (In production, we'd have a better bulk query)
            synced = 0
            
            # Get all unique classes (we'd cache this)
            common_classes = ['person', 'book', 'laptop', 'cup', 'phone', 'chair', 'desk']
            
            for class_name in common_classes:
                results = self.backend.query_entity_by_class(class_name, limit=100)
                
                for entity in results:
                    entity_id = entity['entity_id']
                    
                    # Add to memory with all observations
                    for obs in entity['observations']:
                        self.memory.add_entity_observation(
                            entity_id=entity_id,
                            class_name=entity['class_name'],
                            timestamp=obs['timestamp'],
                            position_3d=None,  # Would extract from bbox if needed
                            zone_id=obs.get('zone_id'),
                            caption=obs.get('caption'),
                            confidence=obs['confidence']
                        )
                    
                    synced += 1
            
            # Save updated memory
            self.memory.save()
            console.print(f"✓ Synced {synced} entities to persistent memory", style="green")
        
        except Exception as e:
            console.print(f"⚠️  Sync failed: {e}", style="yellow")
    
    def query(self, user_query: str) -> None:
        """Process a user query with full context and intelligence"""
        
        # Use persistent memory for intelligent response
        result = self.memory.query_with_context(user_query)
        
        # Check if clarification needed
        if result['clarification_needed']:
            console.print(Panel(
                f"[yellow]❓ {result['clarification_question']}[/yellow]",
                title="Clarification Needed",
                border_style="yellow"
            ))
            return
        
        # Display answer
        if result['answer']:
            console.print(Panel(
                f"[green]{result['answer']}[/green]",
                title="Answer",
                border_style="green"
            ))
            
            # Show confidence
            confidence = result['confidence']
            if confidence < 0.7:
                console.print(f"   [dim](Confidence: {confidence:.0%} - I might be uncertain)[/dim]")
        
        # Show related entities if any
        if result['entities_mentioned']:
            console.print(f"\n[dim]Related entities: {', '.join(map(str, result['entities_mentioned']))}[/dim]")
    
    def show_memory_stats(self):
        """Display current memory statistics"""
        stats = self.memory.get_statistics()
        
        table = Table(title="Spatial Memory Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Entities", str(stats['total_entities']))
        table.add_row("Total Zones", str(stats['total_zones']))
        table.add_row("Captions Generated", str(stats['total_captions']))
        table.add_row("Frames Processed", str(stats['frames_processed']))
        table.add_row("Session Duration", f"{stats['session_duration_seconds']:.1f}s")
        table.add_row("Conversation Turns", str(stats['conversation_turns']))
        
        console.print(table)
    
    def show_entities(self, limit: int = 10):
        """Show entities in memory"""
        if not self.memory.entities:
            console.print("No entities in memory yet", style="yellow")
            return
        
        table = Table(title=f"Entities in Memory (showing {min(limit, len(self.memory.entities))})")
        table.add_column("ID", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Semantic Label", style="yellow")
        table.add_column("Observations", style="magenta")
        table.add_column("Captions", style="blue")
        
        for i, (eid, entity) in enumerate(list(self.memory.entities.items())[:limit]):
            semantic_label = self.memory.generate_semantic_label(eid)
            table.add_row(
                str(eid),
                entity.class_name,
                semantic_label,
                str(entity.observations_count),
                str(len(entity.captions))
            )
        
        console.print(table)
    
    def interactive_mode(self):
        """Start interactive query session"""
        console.print("\n" + "="*60, style="bold cyan")
        console.print("🤖 SPATIAL INTELLIGENCE ASSISTANT", style="bold cyan")
        console.print("="*60 + "\n", style="bold cyan")
        
        console.print("I'm your spatial intelligence historian. I remember everything", style="cyan")
        console.print("about the space, can understand context, and ask clarifying questions.\n", style="cyan")
        
        # Show what's in memory
        stats = self.memory.get_statistics()
        if stats['total_entities'] > 0:
            console.print(f"📊 Memory: {stats['total_entities']} entities, "
                         f"{stats['total_captions']} captions", style="dim")
        
        console.print("\n[dim]Commands:[/dim]")
        console.print("  [cyan]sync[/cyan]     - Sync from Memgraph")
        console.print("  [cyan]stats[/cyan]    - Show memory statistics")
        console.print("  [cyan]entities[/cyan] - List entities in memory")
        console.print("  [cyan]help[/cyan]     - Show this help")
        console.print("  [cyan]exit[/cyan]     - Exit (saves memory)\n")
        
        console.print("[dim]Or just ask me anything! Examples:[/dim]")
        console.print("  [green]What color was the book?[/green]")
        console.print("  [green]Where was it?[/green] (references previous question)")
        console.print("  [green]What objects were near the laptop?[/green]")
        console.print("  [green]Tell me about the person[/green]\n")
        
        while True:
            try:
                # Get user input
                user_input = console.input("[bold yellow]You:[/bold yellow] ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == 'exit':
                    console.print("\n💾 Saving memory...", style="cyan")
                    self.memory.save()
                    console.print("👋 Goodbye!", style="green")
                    break
                
                elif user_input.lower() == 'sync':
                    self.sync_from_memgraph()
                    continue
                
                elif user_input.lower() == 'stats':
                    self.show_memory_stats()
                    continue
                
                elif user_input.lower() == 'entities':
                    self.show_entities()
                    continue
                
                elif user_input.lower() == 'help':
                    console.print("\n[cyan]Available commands:[/cyan]")
                    console.print("  sync     - Sync from Memgraph")
                    console.print("  stats    - Show memory statistics")
                    console.print("  entities - List entities")
                    console.print("  help     - Show help")
                    console.print("  exit     - Exit and save\n")
                    continue
                
                # Process as query
                console.print()
                self.query(user_input)
                console.print()
            
            except KeyboardInterrupt:
                console.print("\n\n💾 Saving memory...", style="cyan")
                self.memory.save()
                console.print("👋 Goodbye!", style="green")
                break
            
            except Exception as e:
                console.print(f"\n❌ Error: {e}", style="red")
                console.print("[dim]Try rephrasing your question[/dim]\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Spatial Intelligence Assistant - Your persistent spatial memory historian"
    )
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Start interactive mode'
    )
    parser.add_argument(
        '--sync',
        action='store_true',
        help='Sync from Memgraph and exit'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show memory statistics'
    )
    parser.add_argument(
        '-q', '--query',
        type=str,
        help='Single query to process'
    )
    
    args = parser.parse_args()
    
    # Create assistant
    assistant = SpatialIntelligenceAssistant()
    
    if args.sync:
        assistant.sync_from_memgraph()
    
    elif args.stats:
        assistant.show_memory_stats()
    
    elif args.query:
        assistant.query(args.query)
    
    elif args.interactive:
        assistant.interactive_mode()
    
    else:
        # Default to interactive
        assistant.interactive_mode()


if __name__ == '__main__':
    main()
