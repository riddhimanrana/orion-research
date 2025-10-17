"""
Enhanced Video Question Answering System
=========================================

Advanced QA system that leverages:
- Scene/room understanding
- Spatial relationships
- Contextual embeddings
- Causal reasoning
- Multi-modal retrieval

Author: Orion Research Team
Date: October 2025
"""

import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ollama
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from .model_manager import ModelManager
from .config import OrionConfig

logger = logging.getLogger('orion.video_qa')


class EnhancedVideoQASystem:
    """Advanced QA system with scene understanding and spatial reasoning"""
    
    def __init__(
        self,
        config: Optional[OrionConfig] = None,
        neo4j_uri: str = "neo4j://127.0.0.1:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "orion123",
        llm_model: str = "gemma3:4b"
    ):
        self.config = config or OrionConfig()
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.llm_model = llm_model
        self.driver = None
        self.model_manager = ModelManager.get_instance()
    
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
    
    def ask_question(self, question: str) -> str:
        """
        Answer a question about the video using enhanced context
        
        Args:
            question: User's question
            
        Returns:
            Answer from the LLM
        """
        if not self.driver:
            if not self.connect():
                return "Error: Cannot connect to knowledge graph"
        
        # Step 1: Analyze question to determine retrieval strategy
        question_type = self._classify_question(question)
        
        # Step 2: Retrieve relevant context based on question type
        context = self._retrieve_context(question, question_type)
        
        # Step 3: Generate answer using LLM
        answer = self._generate_answer(question, context)
        
        return answer
    
    def _classify_question(self, question: str) -> str:
        """
        Classify question type to determine retrieval strategy
        
        Types:
        - spatial: "Where is...", "What's near...", "What's on the left..."
        - scene: "What room...", "What type of place...", "What's the setting..."
        - temporal: "When did...", "What happened after...", "How long..."
        - causal: "Why did...", "What caused...", "What led to..."
        - entity: "Tell me about...", "What is the...", "Describe the..."
        - general: Everything else
        """
        question_lower = question.lower()
        
        # Spatial indicators
        spatial_keywords = [
            'where', 'near', 'next to', 'beside', 'left', 'right', 
            'above', 'below', 'in front', 'behind', 'location'
        ]
        if any(kw in question_lower for kw in spatial_keywords):
            return 'spatial'
        
        # Scene indicators
        scene_keywords = [
            'room', 'place', 'setting', 'scene', 'environment',
            'office', 'kitchen', 'bedroom', 'outdoor'
        ]
        if any(kw in question_lower for kw in scene_keywords):
            return 'scene'
        
        # Temporal indicators
        temporal_keywords = [
            'when', 'time', 'after', 'before', 'during', 'first',
            'last', 'how long', 'duration'
        ]
        if any(kw in question_lower for kw in temporal_keywords):
            return 'temporal'
        
        # Causal indicators
        causal_keywords = [
            'why', 'cause', 'reason', 'led to', 'result', 'because',
            'what made', 'what happened'
        ]
        if any(kw in question_lower for kw in causal_keywords):
            return 'causal'
        
        # Entity indicators
        entity_keywords = [
            'what is', 'tell me about', 'describe', 'show me',
            'information about'
        ]
        if any(kw in question_lower for kw in entity_keywords):
            return 'entity'
        
        return 'general'
    
    def _retrieve_context(self, question: str, question_type: str) -> str:
        """Retrieve relevant context based on question type"""
        context_parts = []
        
        try:
            with self.driver.session() as session:
                if question_type == 'spatial':
                    context_parts.append(self._retrieve_spatial_context(session, question))
                
                elif question_type == 'scene':
                    context_parts.append(self._retrieve_scene_context(session, question))
                
                elif question_type == 'temporal':
                    context_parts.append(self._retrieve_temporal_context(session, question))
                
                elif question_type == 'causal':
                    context_parts.append(self._retrieve_causal_context(session, question))
                
                elif question_type == 'entity':
                    context_parts.append(self._retrieve_entity_context(session, question))
                
                else:  # general
                    # Retrieve a mix of context
                    context_parts.append(self._retrieve_overview_context(session))
        
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return "Error retrieving video context"
        
        return "\n\n".join(filter(None, context_parts))
    
    def _retrieve_spatial_context(self, session, question: str) -> str:
        """Retrieve spatial relationships context"""
        parts = []
        
        # Get entities with spatial relationships
        query = """
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[r:SPATIAL_REL]->(other:Entity)
        WITH e, collect({
            other: other.class,
            relationship: r.type,
            confidence: r.confidence
        }) AS relationships
        WHERE size(relationships) > 0
        RETURN e.id AS id,
               e.class AS class,
               e.spatial_zone AS zone,
               relationships
        LIMIT 10
        """
        
        results = session.run(query).data()
        
        if results:
            parts.append("**Spatial Relationships:**")
            for record in results:
                entity_class = record['class']
                zone = record.get('zone', 'unknown')
                rels = record.get('relationships', [])
                
                parts.append(f"\n- {entity_class} (located in {zone}):")
                for rel in rels[:3]:  # Top 3 relationships
                    other = rel['other']
                    rel_type = rel['relationship']
                    parts.append(f"  - {rel_type} {other}")
        
        return "\n".join(parts)
    
    def _retrieve_scene_context(self, session, question: str) -> str:
        """Retrieve scene/room context"""
        parts = []
        
        # Get scene information
        query = """
        MATCH (s:Scene)
        RETURN s.id AS id,
               s.scene_type AS type,
               s.confidence AS confidence,
               s.description AS description,
               s.dominant_objects AS objects
        ORDER BY s.frame_start
        LIMIT 5
        """
        
        results = session.run(query).data()
        
        if results:
            parts.append("**Scenes/Rooms Detected:**")
            for record in results:
                scene_type = record.get('type', 'unknown')
                confidence = record.get('confidence', 0)
                description = record.get('description', '')
                objects = record.get('objects', [])
                
                parts.append(f"\n- {scene_type.title()} (confidence: {confidence:.2f})")
                parts.append(f"  {description}")
                if objects:
                    parts.append(f"  Key objects: {', '.join(objects[:5])}")
        
        return "\n".join(parts)
    
    def _retrieve_temporal_context(self, session, question: str) -> str:
        """Retrieve temporal context"""
        parts = []
        
        # Get scene timeline
        query = """
        MATCH (s:Scene)
        RETURN s.scene_type AS type,
               s.timestamp_start AS start,
               s.timestamp_end AS end,
               s.dominant_objects AS objects
        ORDER BY s.timestamp_start
        """
        
        results = session.run(query).data()
        
        if results:
            parts.append("**Timeline:**")
            for record in results:
                scene_type = record.get('type', 'unknown')
                start = record.get('start', 0)
                end = record.get('end', 0)
                objects = record.get('objects', [])
                
                duration = end - start
                parts.append(
                    f"\n- {start:.1f}s - {end:.1f}s ({duration:.1f}s): "
                    f"{scene_type} with {', '.join(objects[:3])}"
                )
        
        return "\n".join(parts)
    
    def _retrieve_causal_context(self, session, question: str) -> str:
        """Retrieve causal relationships context"""
        parts = []
        
        # Get causal chains
        query = """
        MATCH (cause:Entity)-[r:POTENTIALLY_CAUSED]->(effect:Entity)
        RETURN cause.class AS cause_class,
               effect.class AS effect_class,
               r.confidence AS confidence,
               r.temporal_gap AS temporal_gap,
               r.cause_state AS cause_state,
               r.effect_state AS effect_state
        ORDER BY r.confidence DESC
        LIMIT 5
        """
        
        results = session.run(query).data()
        
        if results:
            parts.append("**Causal Relationships:**")
            for record in results:
                cause = record.get('cause_class', 'unknown')
                effect = record.get('effect_class', 'unknown')
                confidence = record.get('confidence', 0)
                gap = record.get('temporal_gap', 0)
                
                parts.append(
                    f"\n- {cause} → {effect} "
                    f"(confidence: {confidence:.2f}, gap: {gap:.1f}s)"
                )
                
                cause_state = record.get('cause_state')
                effect_state = record.get('effect_state')
                if cause_state:
                    parts.append(f"  Cause: {cause_state[:80]}...")
                if effect_state:
                    parts.append(f"  Effect: {effect_state[:80]}...")
        
        return "\n".join(parts)
    
    def _retrieve_entity_context(self, session, question: str) -> str:
        """Retrieve detailed entity context"""
        parts = []
        
        # Try to extract entity mention from question
        # Simple approach: look for YOLO class names
        common_classes = [
            'person', 'car', 'dog', 'cat', 'tv', 'laptop', 'phone',
            'keyboard', 'mouse', 'chair', 'table', 'cup', 'book'
        ]
        
        mentioned_class = None
        question_lower = question.lower()
        for cls in common_classes:
            if cls in question_lower:
                mentioned_class = cls
                break
        
        # Get entity details
        if mentioned_class:
            query = """
            MATCH (e:Entity)
            WHERE toLower(e.class) = $class
            OPTIONAL MATCH (e)-[:APPEARS_IN]->(s:Scene)
            WITH e, collect(DISTINCT s.scene_type) AS scenes
            OPTIONAL MATCH (e)-[r:SPATIAL_REL]-(other:Entity)
            WITH e, scenes, collect({
                other: other.class,
                type: r.type
            }) AS relationships
            RETURN e.id AS id,
                   e.class AS class,
                   e.description AS description,
                   e.appearance_count AS count,
                   scenes,
                   relationships
            LIMIT 3
            """
            
            results = session.run(query, {'class': mentioned_class}).data()
            
            if results:
                parts.append(f"**Information about {mentioned_class}:**")
                for record in results:
                    desc = record.get('description', '')
                    count = record.get('count', 0)
                    scenes = record.get('scenes', [])
                    rels = record.get('relationships', [])
                    
                    parts.append(f"\n- Appeared {count} times")
                    if scenes:
                        parts.append(f"- Found in: {', '.join(set(scenes))}")
                    if desc:
                        parts.append(f"- Description: {desc[:200]}...")
                    if rels:
                        rel_summary = ', '.join(f"{r['type']} {r['other']}" for r in rels[:3])
                        parts.append(f"- Relationships: {rel_summary}")
        else:
            # No specific entity mentioned, return top entities
            parts.append(self._retrieve_overview_context(session))
        
        return "\n".join(parts)
    
    def _retrieve_overview_context(self, session) -> str:
        """Retrieve general overview context"""
        parts = []
        
        # Get top entities
        query = """
        MATCH (e:Entity)
        RETURN e.class AS class,
               e.appearance_count AS count
        ORDER BY e.appearance_count DESC
        LIMIT 5
        """
        
        results = session.run(query).data()
        
        if results:
            parts.append("**Most Prominent Objects:**")
            for record in results:
                cls = record['class']
                count = record['count']
                parts.append(f"- {cls} ({count} appearances)")
        
        # Get scene types
        query = """
        MATCH (s:Scene)
        RETURN DISTINCT s.scene_type AS type, count(*) AS count
        ORDER BY count DESC
        """
        
        results = session.run(query).data()
        
        if results:
            parts.append("\n**Scene Types:**")
            for record in results:
                scene_type = record['type']
                count = record['count']
                parts.append(f"- {scene_type} ({count} scenes)")
        
        return "\n".join(parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM with retrieved context"""
        prompt = f"""You are analyzing a video based on automated visual analysis with scene understanding and spatial reasoning.

You have access to the following information extracted from the video:

{context}

Based on this information, please answer the following question:
Question: {question}

Provide a clear, concise answer based only on the available data. If the information isn't available, say so.
Be specific and reference the spatial relationships, scenes, and temporal information when relevant.

Answer:"""
        
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            answer = response["message"]["content"]
            return answer
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error: {str(e)}"
    
    def start_interactive_session(self):
        """Start an interactive Q&A session"""
        try:
            from rich.console import Console
            from rich.markdown import Markdown
            from rich.panel import Panel
        except ImportError:
            print("Rich library not found. Install with: pip install rich")
            return
        
        console = Console()
        
        # Check Ollama
        try:
            ollama.list()
        except Exception as e:
            console.print(f"[red]✗ Ollama not available: {e}[/red]")
            console.print("[yellow]Install from: https://ollama.ai[/yellow]")
            return
        
        # Connect to Neo4j
        if not self.connect():
            console.print("[red]✗ Cannot connect to Neo4j[/red]")
            console.print("[yellow]Run the enhanced knowledge graph builder first![/yellow]")
            return
        
        console.print(Panel.fit(
            "[bold cyan]Enhanced Video Question Answering System[/bold cyan]\n"
            f"Using model: {self.llm_model}\n"
            "Features: Scene understanding, Spatial reasoning, Causal inference\n"
            "Type 'quit' or 'exit' to end session",
            border_style="cyan"
        ))
        
        console.print("\n[dim]Example questions:[/dim]")
        console.print("[dim]- What type of room is this?[/dim]")
        console.print("[dim]- What objects are near the laptop?[/dim]")
        console.print("[dim]- What happened in the video?[/dim]")
        console.print("[dim]- Why did X change?[/dim]")
        
        while True:
            try:
                question = console.input("\n[bold green]Ask a question:[/bold green] ")
                
                if question.lower() in ["quit", "exit", "q"]:
                    console.print("[yellow]Session ended[/yellow]")
                    break
                
                if not question.strip():
                    continue
                
                console.print("[dim]Analyzing...[/dim]")
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(name)s | %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description="Ask questions about an analyzed video (enhanced version)"
    )
    parser.add_argument(
        "--model",
        default="gemma3:4b",
        help="Ollama model to use"
    )
    parser.add_argument(
        "--neo4j-password",
        default="orion123",
        help="Neo4j password"
    )
    
    args = parser.parse_args()
    
    qa = EnhancedVideoQASystem(
        llm_model=args.model,
        neo4j_password=args.neo4j_password
    )
    
    qa.start_interactive_session()


if __name__ == "__main__":
    main()
