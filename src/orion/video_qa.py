"""
Interactive Video Question Answering System

Uses the knowledge graph built from video analysis to answer questions
about the video content using a local LLM (Gemini 3B).
"""

import logging
from typing import List, Dict, Any, Optional

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError

import ollama

from .embedding_model import create_embedding_model

logger = logging.getLogger(__name__)


class VideoQASystem:
    """Interactive question answering system for analyzed videos"""
    
    def __init__(
        self,
        neo4j_uri: str = "neo4j://127.0.0.1:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "orion123",
        model: str = "gemma3:4b",
        embedding_backend: str = "auto",
        embedding_model: str = "embeddinggemma",
        embedding_fallback: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize QA system
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            model: Ollama model to use (gemma3:4b for better quality)
            embedding_backend: Embedding backend ('auto', 'ollama', or 'sentence-transformer')
            embedding_model: Preferred embedding model identifier
            embedding_fallback: Sentence-transformer fallback when using auto embeddings
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.model = model
        self.driver: Optional[Driver] = None
        self.embedding_model: Optional[Any] = None
        self.embedding_backend_choice = embedding_backend
        self.embedding_model_name = embedding_model
        self.embedding_fallback_name = embedding_fallback
        self.vector_index_name = "entity_embedding"
        self.vector_search_enabled = True
        self.vector_index_dimensions: Optional[int] = None
        
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

    def ensure_embedding_model(self) -> bool:
        """Ensure an embedding model is available for vector search."""
        if self.embedding_model is not None:
            return True

        try:
            prefer_ollama = self.embedding_backend_choice in {"auto", "ollama"}
            self.embedding_model = create_embedding_model(
                prefer_ollama=prefer_ollama,
                backend=self.embedding_backend_choice,
                model=self.embedding_model_name,
                fallback=self.embedding_fallback_name,
            )
            model_obj = self.embedding_model
            if model_obj is not None:
                try:
                    self.vector_index_dimensions = model_obj.get_embedding_dimension()
                except Exception:  # pragma: no cover - dimension query failed
                    self.vector_index_dimensions = None
            return True
        except Exception as exc:
            logger.warning(
                "Embedding model unavailable, falling back to keyword retrieval: %s",
                exc,
            )
            self.embedding_model = None
            return False
    
    def get_video_context(self, question: Optional[str] = None, top_k: int = 5) -> str:
        """Retrieve relevant context from the knowledge graph."""
        if not self.driver:
            return "No knowledge graph available"

        context_parts: List[str] = []

        try:
            with self.driver.session() as session:
                relevant_entities: List[Dict[str, Any]] = []

                if question:
                    relevant_entities = self._vector_search_entities(session, question, top_k)

                if not relevant_entities:
                    relevant_entities = self._fetch_top_entities(session, top_k)

                entity_details = self._fetch_entity_details(session, relevant_entities)
                recent_events = self._fetch_recent_events(session, limit=8)

                all_event_ids = {
                    event.get("id")
                    for entity in entity_details
                    for event in entity.get("events", [])
                    if event.get("id")
                }
                all_event_ids.update(
                    event.get("id")
                    for event in recent_events
                    if event.get("id")
                )

                event_context_lookup = self._fetch_event_context(session, list(all_event_ids))

                for entity in entity_details:
                    for event in entity.get("events", []):
                        event_id = event.get("id")
                        if not event_id:
                            continue
                        metadata = event_context_lookup.get(event_id, {})
                        if metadata.get("participants"):
                            event["participants"] = metadata["participants"]
                        if metadata.get("locations"):
                            event["locations"] = metadata["locations"]

                for event in recent_events:
                    event_id = event.get("id")
                    if not event_id:
                        continue
                    metadata = event_context_lookup.get(event_id, {})
                    if metadata.get("participants"):
                        event["participants"] = metadata["participants"]
                    if metadata.get("locations"):
                        event["locations"] = metadata["locations"]

                if entity_details:
                    context_parts.append("## Focused Entities:")
                    for entity in entity_details:
                        header = f"- {entity['label']} (id: {entity['id']}, seen {entity['appearances']} times"
                        if entity.get("score") is not None:
                            header += f", relevance {entity['score']:.3f}"
                        header += ")"
                        context_parts.append(header)

                        if entity.get("description"):
                            context_parts.append(f"  Summary: {entity['description']}")

                        if entity.get("scenes"):
                            for scene in entity["scenes"][:3]:
                                scene_line = f"  Scene link ({scene['relation']}): {scene['name']}"
                                if scene.get("description"):
                                    scene_line += f" — {scene['description']}"
                                context_parts.append(scene_line)

                        if entity.get("relations"):
                            context_parts.append("  Relationships:")
                            for rel in entity["relations"][:5]:
                                context_parts.append(
                                    f"    - {rel['relation']} -> {rel['target_label']} (id: {rel['target_id']})"
                                )

                        if entity.get("events"):
                            context_parts.append("  Key Events:")
                            for event in entity["events"][:5]:
                                line = (
                                    f"    - [{event['timestamp']}] {event.get('type', 'event')}: "
                                    f"{event.get('description', 'No description')}"
                                )
                                participants = event.get("participants")
                                if participants:
                                    line += f" | Participants: {', '.join(participants)}"
                                locations = event.get("locations")
                                if locations:
                                    line += f" | Locations: {', '.join(locations)}"
                                context_parts.append(line)

                if recent_events:
                    context_parts.append("\n## Global Timeline Highlights:")
                    for event in recent_events:
                        line = (
                            f"- [{event['timestamp']}] {event.get('type', 'event')}: "
                            f"{event.get('description', 'No description')}"
                        )
                        participants = event.get("participants")
                        if participants:
                            line += f" | Participants: {', '.join(participants)}"
                        locations = event.get("locations")
                        if locations:
                            line += f" | Locations: {', '.join(locations)}"
                        context_parts.append(line)

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
        context = self.get_video_context(question)
        
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
            models_response = ollama.list()
            models: List[str] = []

            if isinstance(models_response, dict) and 'models' in models_response:
                iterable = models_response['models']
            elif isinstance(models_response, list):
                iterable = models_response
            else:
                iterable = []

            for item in iterable:
                if isinstance(item, dict):
                    name = item.get('name') or item.get('model')
                    if name:
                        models.append(name)
                elif isinstance(item, (list, tuple)) and item:
                    models.append(str(item[0]))
                elif isinstance(item, str):
                    models.append(item)
            
            if self.model not in models:
                console.print(f"[yellow]Downloading {self.model} model...[/yellow]")
                ollama.pull(self.model)
        except Exception as e:
            console.print(f"[red]Error checking models: {e}[/red]")
            # Continue anyway - model might still work
        
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

    @staticmethod
    def _format_timestamp(value: Any) -> str:
        if value is None:
            return "unknown"

        try:
            if hasattr(value, "to_native"):
                native = value.to_native()
                return native.isoformat()
            if isinstance(value, (int, float)):
                return f"{value:.2f}s"
        except Exception:  # pragma: no cover - defensive formatting
            pass

        return str(value)

    def _vector_search_entities(self, session, question: str, top_k: int) -> List[Dict[str, Any]]:
        if not self.vector_search_enabled:
            return []

        if not self.ensure_embedding_model():
            return []

        if self.embedding_model is None:
            return []

        if self.vector_index_dimensions is None:
            self.vector_index_dimensions = self._get_vector_index_dimensions(session)

        model_dim = None
        if hasattr(self.embedding_model, "get_embedding_dimension"):
            try:
                model_dim = int(self.embedding_model.get_embedding_dimension())
            except Exception:
                model_dim = None

        if (
            self.vector_index_dimensions is not None
            and model_dim is not None
            and self.vector_index_dimensions != model_dim
        ):
            logger.warning(
                "Vector index expects dimension %s but embedding model returns %s; disabling vector search",
                self.vector_index_dimensions,
                model_dim,
            )
            self.vector_search_enabled = False
            return []

        try:
            embedding = self.embedding_model.encode([question])[0]
        except Exception as exc:
            logger.warning(f"Failed to encode question for vector search: {exc}")
            return []

        vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

        cypher = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
        YIELD node, score
        RETURN node.id AS id,
               node.label AS label,
               node.first_description AS description,
               node.appearance_count AS appearances,
               score
        """

        try:
            records = session.run(
                cypher,
                {
                    "index_name": self.vector_index_name,
                    "top_k": top_k,
                    "embedding": vector,
                },
            )

            results: List[Dict[str, Any]] = []
            for record in records:
                node_id = record.get("id")
                if not node_id:
                    continue
                results.append(
                    {
                        "id": node_id,
                        "label": record.get("label", "Unknown"),
                        "description": record.get("description"),
                        "appearances": record.get("appearances", 0),
                        "score": float(record.get("score", 0.0)),
                    }
                )

            return results

        except Neo4jError as exc:
            logger.warning(f"Vector search failed (Neo4jError): {exc}")
            if "dimensions" in str(exc) or "embedding" in str(exc):
                logger.warning("Disabling vector search due to incompatible embedding configuration")
                self.vector_search_enabled = False
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(f"Vector search encountered an error: {exc}")
            if "dimensions" in str(exc):
                logger.warning("Disabling vector search due to incompatible embedding configuration")
                self.vector_search_enabled = False

        return []

    def _get_vector_index_dimensions(self, session) -> Optional[int]:
        try:
            record = session.run(
                """
                CALL db.indexes()
                YIELD name, type, config
                WHERE type = 'VECTOR' AND (name = $name OR $name IS NULL)
                RETURN config['vector.dimensions'] AS dims
                ORDER BY CASE WHEN name = $name THEN 0 ELSE 1 END
                LIMIT 1
                """,
                {"name": self.vector_index_name},
            ).single()
        except Neo4jError as exc:
            logger.debug(f"Failed to inspect vector index metadata: {exc}")
            return None
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(f"Error inspecting vector index metadata: {exc}")
            return None

        if not record:
            return None

        dims_value = record.get("dims")

        candidates: List[Any]
        if isinstance(dims_value, (list, tuple)):
            candidates = list(dims_value)
        else:
            candidates = [dims_value]

        for item in candidates:
            if isinstance(item, (int, float)):
                return int(item)
            if isinstance(item, str):
                try:
                    return int(float(item))
                except ValueError:
                    continue

        return None

    @staticmethod
    def _fetch_top_entities(session, limit: int) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (e:Entity)
        RETURN e.id AS id,
               e.label AS label,
               e.appearance_count AS appearances,
               e.first_description AS description
        ORDER BY e.appearance_count DESC
        LIMIT $limit
        """

        records = session.run(cypher, {"limit": limit})

        return [
            {
                "id": record["id"],
                "label": record.get("label", "Unknown"),
                "description": record.get("description"),
                "appearances": record.get("appearances", 0),
            }
            for record in records
            if record.get("id")
        ]

    def _fetch_entity_details(self, session, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not entities:
            return []

        ids = [entity["id"] for entity in entities if entity.get("id")]
        if not ids:
            return []

        cypher = """
        MATCH (e:Entity)
        WHERE e.id IN $ids
        OPTIONAL MATCH (e)-[:PARTICIPATED_IN]->(ev:Event)
        WITH e, collect(DISTINCT ev) AS events
        OPTIONAL MATCH (e)-[r]->(other:Entity)
        WITH e, events, collect(DISTINCT {relation: type(r), other: other}) AS rels
        OPTIONAL MATCH (e)-[sceneRel]->(sceneNode)
        WHERE any(label IN labels(sceneNode) WHERE label IN ['Scene','Location','Place'])
        WITH e, events, rels, collect(DISTINCT {relation: type(sceneRel), node: sceneNode}) AS scenes
        RETURN e, events, rels, scenes
        """

        records = session.run(cypher, {"ids": ids})

        score_lookup = {entity["id"]: entity.get("score") for entity in entities}

        details: List[Dict[str, Any]] = []

        for record in records:
            node = record.get("e")
            if node is None:
                continue

            entity_id = node.get("id")
            label = node.get("label", "Unknown")
            appearances = node.get("appearance_count", 0)
            description = node.get("first_description")

            events_data = []
            for ev in record.get("events") or []:
                if ev is None:
                    continue
                events_data.append(
                    {
                        "id": ev.get("id"),
                        "type": ev.get("type"),
                        "description": ev.get("description"),
                        "timestamp": self._format_timestamp(ev.get("timestamp")),
                    }
                )

            relations_data = []
            for rel in record.get("rels") or []:
                other = rel.get("other") if isinstance(rel, dict) else None
                if other is None or other.get("id") is None:
                    continue
                relations_data.append(
                    {
                        "relation": rel.get("relation", "RELATED_TO"),
                        "target_id": other.get("id"),
                        "target_label": other.get("label", "Unknown"),
                    }
                )

            scenes_data = []
            for scene in record.get("scenes") or []:
                node_ref = scene.get("node") if isinstance(scene, dict) else None
                if node_ref is None:
                    continue
                props = dict(node_ref)
                scene_name = props.get("name") or props.get("label") or props.get("title")
                scenes_data.append(
                    {
                        "relation": scene.get("relation", "ASSOCIATED_WITH"),
                        "name": scene_name or "Unknown",
                        "description": props.get("description"),
                    }
                )

            details.append(
                {
                    "id": entity_id,
                    "label": label,
                    "description": description,
                    "appearances": appearances,
                    "events": sorted(events_data, key=lambda ev: ev.get("timestamp", "")),
                    "relations": relations_data,
                    "scenes": scenes_data,
                    "score": score_lookup.get(entity_id),
                }
            )

        return sorted(
            details,
            key=lambda item: (
                -item.get("score", 0.0) if item.get("score") is not None else -item.get("appearances", 0)
            ),
        )

    def _fetch_event_context(self, session, event_ids: List[str]) -> Dict[str, Dict[str, List[str]]]:
        if not event_ids:
            return {}

        cypher = """
        MATCH (ev:Event)
        WHERE ev.id IN $event_ids
        OPTIONAL MATCH (ev)<-[:PARTICIPATED_IN]-(participant:Entity)
        WITH ev, collect(DISTINCT participant.label) AS participants
        OPTIONAL MATCH (ev)-[r]->(place)
        WHERE any(label IN labels(place) WHERE label IN ['Scene','Location','Place'])
        WITH ev, participants, collect(DISTINCT place) AS places
        RETURN ev.id AS event_id, participants, places
        """

        records = session.run(cypher, {"event_ids": event_ids})

        context: Dict[str, Dict[str, List[str]]] = {}

        for record in records:
            event_id = record.get("event_id")
            if not event_id:
                continue

            participants = [p for p in (record.get("participants") or []) if p]

            locations: List[str] = []
            for place in record.get("places") or []:
                if place is None:
                    continue
                props = dict(place)
                name = props.get("name") or props.get("label") or props.get("title")
                if name:
                    locations.append(name)

            context[event_id] = {
                "participants": participants,
                "locations": locations,
            }

        return context

    def _fetch_recent_events(self, session, limit: int = 8) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (ev:Event)
        RETURN ev.id AS id,
               ev.type AS type,
               ev.description AS description,
               ev.timestamp AS timestamp
        ORDER BY ev.timestamp
        LIMIT $limit
        """

        records = session.run(cypher, {"limit": limit})

        return [
            {
                "id": record.get("id"),
                "type": record.get("type"),
                "description": record.get("description"),
                "timestamp": self._format_timestamp(record.get("timestamp")),
            }
            for record in records
        ]


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
