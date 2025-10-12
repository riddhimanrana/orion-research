"""
Interactive Video Question Answering System

Uses the knowledge graph built from video analysis to answer questions
about the video content using a local LLM (Gemini 3B).
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import ollama

logger = logging.getLogger(__name__)


class VideoQASystem:
    """Interactive question answering system for analyzed videos"""
    
    def __init__(self, 
                 neo4j_uri: str = "neo4j://127.0.0.1:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password",
                 model: str = "gemma3:4b"):
        """
        Initialize QA system
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            model: Ollama model to use (gemma3:4b for better quality)
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.model = model
        self.driver: Optional[GraphDatabase.driver] = None
        
    def connect(self) -> bool:
        """Connect to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("✓ Connected to Neo4j for Q&A")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to connect to Neo4j: {e}")
            return False
    
    def get_video_context(self) -> str:
        """
        Retrieve relevant context from the knowledge graph
        
        Returns:
            Formatted context string for the LLM
        """
        if not self.driver:
            return "No knowledge graph available"
        
        context_parts = []
        
        try:
            with self.driver.session() as session:
                # Get entities (use correct property names: id instead of entity_id, appearance_count instead of appearances)
                entities = session.run("""
                    MATCH (e:Entity)
                    RETURN e.label as label, e.id as id, 
                           e.appearance_count as appearances
                    ORDER BY e.appearance_count DESC
                    LIMIT 20
                """).data()
                
                if entities:
                    context_parts.append("## Detected Objects/Entities:")
                    for ent in entities:
                        context_parts.append(f"- {ent['label']}: appears {ent['appearances']} times")
                
                # Get events/state changes
                events = session.run("""
                    MATCH (ev:Event)
                    RETURN ev.description as description, ev.timestamp as timestamp
                    ORDER BY ev.timestamp
                    LIMIT 10
                """).data()
                
                if events:
                    context_parts.append("\n## Timeline of Events:")
                    for event in events:
                        ts = event.get('timestamp', 'unknown')
                        desc = event.get('description', 'No description')
                        context_parts.append(f"- At {ts}s: {desc}")
                
                # Get relationships
                relationships = session.run("""
                    MATCH (e1:Entity)-[r]->(e2:Entity)
                    RETURN e1.label as from, type(r) as relation, e2.label as to
                    LIMIT 15
                """).data()
                
                if relationships:
                    context_parts.append("\n## Object Relationships:")
                    for rel in relationships:
                        context_parts.append(f"- {rel['from']} {rel['relation']} {rel['to']}")
        
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return "Error retrieving video context"
        
        return "\n".join(context_parts) if context_parts else "No video analysis available"
    
    def ask_question(self, question: str) -> str:
        """
        Answer a question about the video using the knowledge graph and LLM
        
        Args:
            question: User's question
            
        Returns:
            Answer from the LLM
        """
        # Get context from knowledge graph
        context = self.get_video_context()
        
        # Build prompt
        prompt = f"""You are analyzing a video based on automated visual analysis. 
You have access to the following information extracted from the video:

{context}

Based on this information, please answer the following question:
Question: {question}

Provide a clear, concise answer based only on the available data. If the information isn't available, say so.

Answer:"""
        
        try:
            # Query Ollama
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }]
            )
            
            answer = response['message']['content']
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error: {str(e)}"
    
    def start_interactive_session(self):
        """Start an interactive Q&A session"""
        from rich.console import Console
        from rich.panel import Panel
        from rich.markdown import Markdown
        
        console = Console()
        
        # Check Ollama is available
        try:
            ollama.list()
        except Exception as e:
            console.print(f"[red]✗ Ollama not available: {e}[/red]")
            console.print("[yellow]Install Ollama from: https://ollama.ai[/yellow]")
            return
        
        # Check model is available
        try:
            models = [m['name'] for m in ollama.list()['models']]
            if self.model not in models:
                console.print(f"[yellow]Downloading {self.model} model...[/yellow]")
                ollama.pull(self.model)
        except Exception as e:
            console.print(f"[red]Error checking models: {e}[/red]")
        
        # Connect to Neo4j
        if not self.connect():
            console.print("[red]✗ Cannot connect to Neo4j. Run the pipeline first![/red]")
            return
        
        console.print(Panel.fit(
            "[bold cyan]Video Question Answering System[/bold cyan]\n"
            f"Using model: {self.model}\n"
            "Type 'quit' or 'exit' to end session",
            border_style="cyan"
        ))
        
        while True:
            try:
                question = console.input("\n[bold green]Ask a question:[/bold green] ")
                
                if question.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                if not question.strip():
                    continue
                
                console.print("[dim]Thinking...[/dim]")
                answer = self.ask_question(question)
                
                console.print("\n[bold blue]Answer:[/bold blue]")
                console.print(Panel(answer, border_style="blue"))
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Session ended[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        
        if self.driver:
            self.driver.close()
    
    def close(self):
        """Close connections"""
        if self.driver:
            self.driver.close()


def main():
    """Run interactive Q&A session"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ask questions about an analyzed video")
    parser.add_argument("--model", default="gemma3:4b", help="Ollama model to use")
    parser.add_argument("--neo4j-password", default="orion123", help="Neo4j password")
    
    args = parser.parse_args()
    
    qa = VideoQASystem(
        model=args.model,
        neo4j_password=args.neo4j_password
    )
    
    qa.start_interactive_session()


if __name__ == "__main__":
    main()
