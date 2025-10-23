"""Unified Video Question Answering system."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import ollama
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from ..embedding_model import EmbeddingModel, create_embedding_model
from ..model_manager import ModelManager
from ..vector_indexing import query_similar_entities, query_similar_scenes
from ..settings import OrionSettings
from .config import VideoQAConfig

logger = logging.getLogger("orion.video_qa")


class VideoQASystem:
    """Unified question answering system for analyzed videos."""

    def __init__(
        self,
        config: Optional[VideoQAConfig] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_backend: str = "auto",
        embedding_model: Optional[str] = None,
    ) -> None:
        self.config = config or VideoQAConfig()

        # Use centralized settings as defaults, but allow overrides
        settings = OrionSettings.load()
        self.neo4j_uri = neo4j_uri or settings.neo4j_uri
        self.neo4j_user = neo4j_user or settings.neo4j_user
        self.neo4j_password = neo4j_password or settings.get_neo4j_password()
        self.llm_model = llm_model or settings.qa_model
        self.embedding_backend_choice = embedding_backend
        self.embedding_model_name = (
            embedding_model or "openai/clip-vit-base-patch32"
        )

        self.driver: Optional[Driver] = None
        self.embedding_model: Optional[EmbeddingModel] = None
        self.vector_index_dimensions: Optional[int] = None
        self.model_manager = ModelManager.get_instance()
        self.conversation_history: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Connection + model bootstrap
    # ------------------------------------------------------------------
    def connect(self) -> bool:
        """
        Connect to Neo4j database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if not self.neo4j_uri:
                logger.error("No Neo4j URI configured")
                return False
            
            self.driver = GraphDatabase.driver(
                self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
            )
            
            # Test connection with a simple query
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                result.single()
            
            logger.info("✓ Connected to Neo4j for Q&A at %s", self.neo4j_uri)
            return True
            
        except ServiceUnavailable as exc:
            logger.error("✗ Neo4j service unavailable at %s: %s", self.neo4j_uri, exc)
            logger.info("  Please ensure Neo4j is running: `neo4j start` or Docker")
            return False
        except Exception as exc:  # pragma: no cover - connection errors are environment driven
            logger.error("✗ Failed to connect to Neo4j: %s", exc)
            logger.info("  Check connection settings in .env or config")
            return False

    def close(self) -> None:
        """Close Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            self.driver = None
    
    def check_data_available(self) -> Dict[str, Any]:
        """
        Check if Neo4j database contains video analysis data.
        
        Returns:
            Dictionary with availability status and counts
        """
        if not self.driver and not self.connect():
            return {
                "available": False,
                "error": "Cannot connect to Neo4j",
                "entities": 0,
                "scenes": 0,
                "events": 0
            }
        
        try:
            with self.driver.session() as session:
                # Count different node types
                entity_count = session.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
                scene_count = session.run("MATCH (s:Scene) RETURN count(s) as count").single()["count"]
                event_count = session.run("MATCH (ev:Event) RETURN count(ev) as count").single()["count"]
                
                return {
                    "available": entity_count > 0 or scene_count > 0,
                    "entities": entity_count,
                    "scenes": scene_count,
                    "events": event_count
                }
        except Exception as exc:
            logger.error("Error checking data availability: %s", exc)
            return {
                "available": False,
                "error": str(exc),
                "entities": 0,
                "scenes": 0,
                "events": 0
            }

    def _ensure_embedding_model(self) -> bool:
        if self.embedding_model is not None:
            return True
        try:
            prefer_ollama = self.embedding_backend_choice in {"auto", "ollama"}
            self.embedding_model = create_embedding_model(
                prefer_ollama=prefer_ollama,
                backend=self.embedding_backend_choice,
                model=self.embedding_model_name,
            )
            if self.embedding_model is not None:
                try:
                    self.vector_index_dimensions = (
                        self.embedding_model.get_embedding_dimension()
                    )
                except Exception:  # pragma: no cover - backend specific
                    self.vector_index_dimensions = None
            return True
        except Exception as exc:
            logger.warning(
                "Embedding model unavailable, falling back to keyword retrieval: %s",
                exc,
            )
            self.embedding_model = None
            self.vector_index_dimensions = None
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ask_question(self, question: str) -> str:
        """
        Ask a question about the analyzed video.
        
        Args:
            question: User's question
            
        Returns:
            Answer string (or user-friendly error message)
        """
        question = question.strip()
        if not question:
            return "Please provide a question."

        # Get context with error handling
        try:
            context = self.get_video_context(question)
        except Exception as exc:
            logger.error("Error retrieving context: %s", exc, exc_info=True)
            return (
                "I'm having trouble accessing the video analysis data. "
                "Please ensure the video has been analyzed and Neo4j is running."
            )
        
        # Check if we got meaningful context
        if not context or context in {"No knowledge graph available", "Error retrieving video context", "No video analysis available"}:
            return (
                "No video analysis is available yet. "
                "Please run the pipeline on a video first: "
                "`python -m orion.run_pipeline --video your_video.mp4`"
            )

        history_block = self._format_history()

        prompt = (
            "You are a helpful video assistant answering questions about a video the user just watched.\n\n"
            "Important instructions:\n"
            "- Answer naturally in 2-3 sentences.\n"
            "- Use spatial language when relevant (e.g., 'on the bed').\n"
            "- Do not mention internal IDs or database terms.\n"
            "- Incorporate previous conversation turns when helpful.\n\n"
            "Video Analysis Data:\n"
            f"{context}\n"
            f"{history_block}"
            f"Current Question: {question}\n\n"
            "Answer:"
        )

        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.7},
            )
            answer = response["message"]["content"]
        except ConnectionError as exc:
            logger.error("Ollama connection error: %s", exc)
            return (
                "Cannot connect to Ollama. Please ensure Ollama is running: "
                "`ollama serve` or install from https://ollama.ai"
            )
        except KeyError as exc:
            logger.error("Unexpected Ollama response format: %s", exc)
            return (
                "Received unexpected response from Ollama. "
                "Please check your Ollama installation and try again."
            )
        except Exception as exc:  # pragma: no cover - depends on local Ollama install
            logger.error("Error generating answer: %s", exc, exc_info=True)
            return (
                f"Error generating answer: {type(exc).__name__}. "
                "Please ensure Ollama is running and the model is available."
            )

        self.conversation_history.append({"question": question, "answer": answer})
        history_limit = self.config.conversation_history_size
        if len(self.conversation_history) > history_limit:
            self.conversation_history = self.conversation_history[-history_limit:]
        return answer

    def get_video_context(self, question: Optional[str] = None, top_k: int = 5) -> str:
        """
        Retrieve video analysis context from Neo4j knowledge graph.
        
        Args:
            question: Optional user question to guide context retrieval
            top_k: Number of top entities to include
            
        Returns:
            Formatted context string (or error message)
        """
        # Check Neo4j connection
        if not self.driver:
            if not self.connect():
                logger.warning("No Neo4j connection available")
                return "No knowledge graph available"
        
        if not self.driver:
            return "No knowledge graph available"

        # Classify question to determine what context to retrieve
        try:
            question_type = self._classify_question(question) if question else "general"
        except Exception as exc:
            logger.warning("Error classifying question: %s", exc)
            question_type = "general"
        
        sections: List[str] = []
        retrieval_errors = []

        try:
            with self.driver.session() as session:
                # Semantic context (vector search)
                if self.config.enable_semantic and question:
                    try:
                        semantic_ctx = self._retrieve_semantic_context(session, question)
                        if semantic_ctx:
                            sections.append(semantic_ctx)
                    except Exception as exc:
                        logger.warning("Error retrieving semantic context: %s", exc)
                        retrieval_errors.append("semantic")

                # Entity overview
                if self.config.enable_overview:
                    try:
                        overview_ctx = self._build_overview_context(
                            session, question, top_k=self.config.overview_top_entities
                        )
                        if overview_ctx:
                            sections.append(overview_ctx)
                    except Exception as exc:
                        logger.warning("Error building overview context: %s", exc)
                        retrieval_errors.append("overview")

                # Spatial context
                if self.config.enable_spatial and question_type in {"spatial", "general"}:
                    try:
                        spatial_ctx = self._retrieve_spatial_context(session)
                        if spatial_ctx:
                            sections.append(spatial_ctx)
                    except Exception as exc:
                        logger.warning("Error retrieving spatial context: %s", exc)
                        retrieval_errors.append("spatial")

                # Scene context
                if self.config.enable_scene and question_type in {"scene", "general"}:
                    try:
                        scene_ctx = self._retrieve_scene_context(session)
                        if scene_ctx:
                            sections.append(scene_ctx)
                    except Exception as exc:
                        logger.warning("Error retrieving scene context: %s", exc)
                        retrieval_errors.append("scene")

                # Temporal context
                if self.config.enable_temporal and question_type in {"temporal", "general"}:
                    try:
                        temporal_ctx = self._retrieve_temporal_context(session)
                        if temporal_ctx:
                            sections.append(temporal_ctx)
                    except Exception as exc:
                        logger.warning("Error retrieving temporal context: %s", exc)
                        retrieval_errors.append("temporal")

                # Causal context
                if self.config.enable_causal and question_type in {"causal", "general"}:
                    try:
                        causal_ctx = self._retrieve_causal_context(session)
                        if causal_ctx:
                            sections.append(causal_ctx)
                    except Exception as exc:
                        logger.warning("Error retrieving causal context: %s", exc)
                        retrieval_errors.append("causal")

                # Event context
                if self.config.enable_events and question_type in {"causal", "temporal", "general"}:
                    try:
                        event_ctx = self._retrieve_event_context(session)
                        if event_ctx:
                            sections.append(event_ctx)
                    except Exception as exc:
                        logger.warning("Error retrieving event context: %s", exc)
                        retrieval_errors.append("events")

        except Neo4jError as exc:
            logger.error("Neo4j error retrieving context: %s", exc, exc_info=True)
            return "Error retrieving video context"
        except Exception as exc:
            logger.error("Unexpected error retrieving context: %s", exc, exc_info=True)
            return "Error retrieving video context"

        # Assemble result
        result = "\n\n".join(section for section in sections if section)
        
        # If we had errors but got some context, log it but continue
        if retrieval_errors and result:
            logger.info("Retrieved partial context (errors in: %s)", ", ".join(retrieval_errors))
        
        return result or "No video analysis available"

    def start_interactive_session(self) -> None:
        """
        Start an interactive Q&A session in the terminal.
        Handles errors gracefully and provides helpful feedback.
        """
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel

        console = Console()

        # Verify Ollama availability
        try:
            ollama.list()
        except ConnectionError as exc:
            console.print(f"[red]✗ Cannot connect to Ollama service[/red]")
            console.print("[yellow]Please start Ollama: `ollama serve`[/yellow]")
            console.print("[yellow]Or install from: https://ollama.ai[/yellow]")
            return
        except Exception as exc:  # pragma: no cover - environment specific
            console.print(f"[red]✗ Ollama error: {exc}[/red]")
            console.print("[yellow]Install Ollama from: https://ollama.ai[/yellow]")
            return

        # Check if model is available
        try:
            models = ollama.list()
            model_names = [m.get('name', '').split(':')[0] for m in models.get('models', [])]
            if self.llm_model.split(':')[0] not in model_names:
                console.print(f"[yellow]⚠ Model '{self.llm_model}' not found locally[/yellow]")
                console.print(f"[yellow]Pull it with: `ollama pull {self.llm_model}`[/yellow]")
                console.print("[yellow]Continuing anyway (will download if needed)...[/yellow]")
        except Exception:
            # Non-fatal, continue anyway
            pass

        # Connect to Neo4j
        if not self.connect():
            console.print("[red]✗ Cannot connect to Neo4j[/red]")
            console.print("[yellow]Please ensure Neo4j is running and the database contains video analysis data[/yellow]")
            console.print("[yellow]Run pipeline first: `python -m orion.run_pipeline --video your_video.mp4`[/yellow]")
            return

        console.print(
            Panel.fit(
                "[bold cyan]Video Question Answering System[/bold cyan]\n"
                f"Using model: {self.llm_model}\n"
                "Type 'quit' to exit",
                border_style="cyan",
            )
        )

        error_count = 0
        max_errors = 3  # Exit after 3 consecutive errors

        while True:
            try:
                question = console.input("\n[bold green]Ask a question:[/bold green] ")
                if question.lower() in {"quit", "exit", "q"}:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                if not question.strip():
                    continue
                    
                console.print("[dim]Thinking...[/dim]")
                answer = self.ask_question(question)
                console.print("\n[bold blue]Answer:[/bold blue]")
                console.print(Panel(answer, border_style="blue"))
                
                # Reset error count on successful interaction
                error_count = 0
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Session ended[/yellow]")
                break
            except Exception as exc:  # pragma: no cover - console rendering errors
                error_count += 1
                console.print(f"[red]Error: {exc}[/red]")
                
                if error_count >= max_errors:
                    console.print(f"[red]Too many errors ({max_errors}). Exiting.[/red]")
                    break

        self.close()

    # ------------------------------------------------------------------
    # Context builders
    # ------------------------------------------------------------------
    def _build_overview_context(
        self, session, question: Optional[str], top_k: int
    ) -> str:
        relevant_entities: List[Dict[str, Any]] = []
        if question and self._ensure_embedding_model():
            relevant_entities = self._vector_search_entities(
                session, question, k=min(top_k, self.config.vector_top_k_entities)
            )

        if not relevant_entities:
            relevant_entities = self._fetch_top_entities(session, top_k)

        entity_details = self._fetch_entity_details(session, relevant_entities)
        recent_events = []
        if self.config.enable_events:
            recent_events = self._fetch_recent_events(
                session, limit=self.config.overview_recent_events
            )

        all_event_ids = {
            event.get("id")
            for entity in entity_details
            for event in entity.get("events", [])
            if event.get("id")
        }
        all_event_ids.update(
            event.get("id") for event in recent_events if event.get("id")
        )

        if all_event_ids:
            event_context_lookup = self._fetch_event_context_details(
                session, list(all_event_ids)
            )
            self._enrich_events(entity_details, event_context_lookup)
            self._enrich_events([{"events": recent_events}], event_context_lookup)

        parts: List[str] = []
        if entity_details:
            parts.append("## Focused Entities:")
            for entity in entity_details:
                header = (
                    f"- {entity['label']} (id: {entity['id']}, seen {entity['appearances']} times"
                )
                if entity.get("score") is not None:
                    header += f", relevance {entity['score']:.3f}"
                header += ")"
                parts.append(header)

                if entity.get("description"):
                    parts.append(f"  Summary: {entity['description']}")

                for scene in entity.get("scenes", [])[:3]:
                    scene_line = f"  Scene link ({scene['relation']}): {scene['name']}"
                    if scene.get("description"):
                        scene_line += f" — {scene['description']}"
                    parts.append(scene_line)

                if entity.get("relations"):
                    parts.append("  Relationships:")
                    for rel in entity["relations"][:5]:
                        parts.append(
                            f"    - {rel['relation']} -> {rel['target_label']} (id: {rel['target_id']})"
                        )

                if entity.get("events"):
                    parts.append("  Key Events:")
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
                        parts.append(line)

        if recent_events:
            parts.append("\n## Global Timeline Highlights:")
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
                parts.append(line)

        return "\n".join(parts)

    def _retrieve_semantic_context(self, session, question: str) -> str:
        if not self._ensure_embedding_model() or not self.embedding_model:
            return ""
        try:
            vec = self.embedding_model.encode([question])[0].tolist()
        except Exception as exc:
            logger.debug("Embedding failed: %s", exc)
            return ""

        sections: List[str] = []
        try:
            entity_hits = query_similar_entities(
                session, vec, k=self.config.vector_top_k_entities
            )
            scene_hits = query_similar_scenes(
                session, vec, k=self.config.vector_top_k_scenes
            )
        except Exception as exc:
            logger.debug("Vector search failed: %s", exc)
            return ""

        if entity_hits:
            sections.append("**Semantically similar entities:**")
            for hit in entity_hits:
                label = hit.get("class") or "unknown"
                canonical = hit.get("canonical")
                score = hit.get("score", 0.0)
                if canonical:
                    sections.append(f"- {label} (canonical: {canonical}) [score: {score:.3f}]")
                else:
                    sections.append(f"- {label} [score: {score:.3f}]")

        if scene_hits:
            sections.append("\n**Semantically similar scenes:**")
            for hit in scene_hits:
                scene_type = hit.get("type") or "unknown"
                score = hit.get("score", 0.0)
                sections.append(f"- {scene_type} [score: {score:.3f}]")

        return "\n".join(sections)

    def _retrieve_spatial_context(self, session) -> str:
        parts = []
        
        # First, retrieve spatial co-location zones
        zone_query = """
        MATCH (z:SpatialZone)
        OPTIONAL MATCH (e:Entity)-[r:IN_SPATIAL_ZONE]->(z)
        WITH z, collect(DISTINCT e.class) AS entity_classes, count(DISTINCT e) AS entity_count
        RETURN z.id AS zone_id,
               z.location_descriptor AS location,
               z.frame_start AS frame_start,
               z.frame_end AS frame_end,
               z.max_concurrent_entities AS max_concurrent,
               entity_classes,
               entity_count
        ORDER BY z.frame_start
        LIMIT 5
        """
        try:
            zone_records = session.run(zone_query).data()
            if zone_records:
                parts.append("**Spatial Co-Location Zones:**")
                for record in zone_records:
                    location = record.get("location", "unknown")
                    entity_classes = record.get("entity_classes", [])
                    entity_count = record.get("entity_count", 0)
                    max_concurrent = record.get("max_concurrent", 0)
                    frame_start = record.get("frame_start", 0)
                    frame_end = record.get("frame_end", 0)
                    
                    if entity_classes:
                        parts.append(
                            f"\n- {location} (frames {frame_start}-{frame_end}): "
                            f"{entity_count} entities ({', '.join(entity_classes[:5])}), "
                            f"up to {max_concurrent} concurrent"
                        )
        except Exception as e:
            logger.debug(f"Could not retrieve spatial zones: {e}")
        
        # Then retrieve spatial relationships
        query = """
        MATCH (e:Entity)-[r:SPATIAL_REL]->(other:Entity)
        RETURN e.class AS class,
               e.spatial_zone AS zone,
               collect({other: other.class, relationship: r.type, confidence: r.confidence}) AS relationships
        LIMIT 10
        """
        records = session.run(query).data()
        if records:
            if parts:
                parts.append("\n")
            parts.append("**Entity Spatial Relationships:**")
            for record in records:
                entity_class = record.get("class", "unknown")
                zone = record.get("zone", "unknown")
                parts.append(f"\n- {entity_class} (located in {zone}):")
                for rel in (record.get("relationships") or [])[:3]:
                    other = rel.get("other", "unknown")
                    rel_type = rel.get("relationship", "related to")
                    parts.append(f"  - {rel_type} {other}")
        
        return "\n".join(parts) if parts else ""

    def _retrieve_scene_context(self, session) -> str:
        query = """
        MATCH (s:Scene)
        RETURN s.scene_type AS type,
               s.confidence AS confidence,
               s.description AS description,
               s.dominant_objects AS objects
        ORDER BY s.frame_start
        LIMIT 5
        """
        records = session.run(query).data()
        if not records:
            return ""
        parts = ["**Scenes/Rooms Detected:**"]
        for record in records:
            scene_type = (record.get("type") or "unknown").title()
            confidence = record.get("confidence") or 0.0
            description = record.get("description") or ""
            objects = record.get("objects") or []
            parts.append(f"\n- {scene_type} (confidence: {confidence:.2f})")
            if description:
                parts.append(f"  {description}")
            if objects:
                parts.append(f"  Key objects: {', '.join(objects[:5])}")
        return "\n".join(parts)

    def _retrieve_temporal_context(self, session) -> str:
        query = """
        MATCH (s:Scene)
        RETURN s.scene_type AS type,
               s.timestamp_start AS start,
               s.timestamp_end AS end,
               s.dominant_objects AS objects
        ORDER BY s.timestamp_start
        """
        records = session.run(query).data()
        if not records:
            return ""
        parts = ["**Timeline:**"]
        for record in records:
            scene_type = record.get("type") or "unknown"
            start = record.get("start") or 0.0
            end = record.get("end") or 0.0
            duration = max(end - start, 0.0)
            objects = record.get("objects") or []
            obj_summary = ", ".join(objects[:3]) if objects else ""
            parts.append(
                f"\n- {start:.1f}s - {end:.1f}s ({duration:.1f}s): {scene_type}"
                + (f" with {obj_summary}" if obj_summary else "")
            )
        return "\n".join(parts)

    def _retrieve_causal_context(self, session) -> str:
        query = """
        MATCH (cause:Entity)-[r:POTENTIALLY_CAUSED]->(effect:Entity)
        RETURN cause.class AS cause_class,
               effect.class AS effect_class,
               coalesce(r.confidence, 0.0) AS confidence
        ORDER BY confidence DESC
        LIMIT 5
        """
        records = session.run(query).data()
        if not records:
            return ""
        parts = ["**Causal Relationships:**"]
        for record in records:
            cause = record.get("cause_class", "unknown")
            effect = record.get("effect_class", "unknown")
            confidence = record.get("confidence") or 0.0
            parts.append(
                f"\n- {cause} \u2192 {effect} (confidence: {confidence:.2f})"
            )
        return "\n".join(parts)

    def _retrieve_event_context(self, session) -> str:
        query = """
        MATCH (e:Event)
        WITH e
        ORDER BY e.timestamp DESC
        LIMIT 8
        OPTIONAL MATCH (e)<-[:PARTICIPATED_IN]-(participant:Entity)
        OPTIONAL MATCH (e)-[:OCCURRED_AT]->(scene:Scene)
        RETURN e.id AS id,
               e.type AS type,
               e.description AS description,
               e.timestamp AS timestamp,
               collect(DISTINCT participant.class) AS participants,
               collect(DISTINCT scene.scene_type) AS locations
        """
        records = session.run(query).data()
        if not records:
            return ""
        parts = ["**Recent Events:**"]
        for record in records:
            line = (
                f"\n- [{self._format_timestamp(record.get('timestamp'))}] "
                f"{record.get('type', 'event')}: "
                f"{record.get('description', 'No description')}"
            )
            participants = [p for p in record.get("participants") or [] if p]
            locations = [l for l in record.get("locations") or [] if l]
            if participants:
                line += f" | Participants: {', '.join(participants)}"
            if locations:
                line += f" | Locations: {', '.join(locations)}"
            parts.append(line)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _format_history(self) -> str:
        if not self.conversation_history:
            return ""
        history_lines = ["\nPrevious conversation:"]
        for turn in self.conversation_history[-3:]:
            history_lines.append(f"Q: {turn['question']}")
            history_lines.append(f"A: {turn['answer']}")
        return "\n".join(history_lines) + "\n\n"

    def _classify_question(self, question: Optional[str]) -> str:
        if not question:
            return "general"
        q = question.lower()
        if any(kw in q for kw in ["where", "near", "left", "right", "beside", "behind", "in front", "location"]):
            return "spatial"
        if any(kw in q for kw in ["room", "place", "scene", "setting", "environment", "kitchen", "bedroom", "office"]):
            return "scene"
        if any(kw in q for kw in ["when", "after", "before", "during", "first", "last", "how long", "duration", "time"]):
            return "temporal"
        if any(kw in q for kw in ["why", "cause", "reason", "led to", "because", "what made", "result"]):
            return "causal"
        if any(kw in q for kw in ["what is", "tell me about", "describe", "information about", "show me"]):
            return "entity"
        return "general"

    def _format_timestamp(self, value: Any) -> str:
        if value is None:
            return "unknown"
        try:
            if hasattr(value, "to_native"):
                native = value.to_native()
                return native.isoformat()
            if isinstance(value, (int, float)):
                return f"{value:.2f}s"
        except Exception:  # pragma: no cover - defensive
            pass
        return str(value)

    def _vector_search_entities(self, session, question: str, k: int) -> List[Dict[str, Any]]:
        if not self._ensure_embedding_model() or not self.embedding_model:
            return []
        try:
            vec = self.embedding_model.encode_text(question).tolist()
        except Exception as exc:
            logger.debug("Question embedding failed: %s", exc)
            return []
        try:
            return query_similar_entities(session, vec, k=k)
        except Exception as exc:
            logger.debug("Vector entity search failed: %s", exc)
            return []

    def _fetch_top_entities(self, session, top_k: int) -> List[Dict[str, Any]]:
        query = """
        MATCH (e:Entity)
        RETURN e.id AS id,
               e.object_class AS label,
               e.appearance_count AS appearances,
               e.description AS description
        ORDER BY e.appearance_count DESC
        LIMIT $limit
        """
        return session.run(query, {"limit": int(top_k)}).data()

    def _fetch_entity_details(
        self, session, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not entities:
            return []
        ids = [entity["id"] for entity in entities if entity.get("id")]
        if not ids:
            return []
        query = """
        MATCH (e:Entity)
        WHERE e.id IN $ids
    OPTIONAL MATCH (e)-[scene_rel:APPEARS_IN]->(s:Scene)
    OPTIONAL MATCH (e)-[r:SPATIAL_REL]->(target:Entity)
        OPTIONAL MATCH (ev:Event)-[:INVOLVES]->(e)
        RETURN e.id AS id,
               e.class AS label,
               e.description AS description,
               e.appearance_count AS appearances,
               collect(DISTINCT {
                   relation: r.type,
                   target_id: target.id,
                   target_label: target.class
               }) AS relations,
               collect(DISTINCT {
                   relation: type(scene_rel),
                   name: s.scene_type,
                   description: s.description
               }) AS scenes,
               collect(DISTINCT {
                   id: ev.id,
                   type: ev.type,
                   timestamp: ev.timestamp,
                   description: ev.description
               }) AS events
        """
        records = session.run(
            query,
            {"ids": ids},
        ).data()
        records_by_id = {record["id"]: record for record in records}
        ordered: List[Dict[str, Any]] = []
        for entity in entities:
            eid = entity.get("id")
            if not eid:
                continue
            detailed = records_by_id.get(eid)
            if not detailed:
                continue
            detailed["score"] = entity.get("score")
            ordered.append(detailed)
        return ordered

    def _fetch_recent_events(self, session, limit: int) -> List[Dict[str, Any]]:
        query = """
        MATCH (e:Event)
        WITH e
        ORDER BY e.timestamp DESC
        LIMIT $limit
        OPTIONAL MATCH (participant:Entity)-[:PARTICIPATED_IN]->(e)
        RETURN e.id AS id,
               e.type AS type,
               e.timestamp AS timestamp,
               e.description AS description,
               collect(DISTINCT participant.class) AS participants
        """
        return session.run(query, {"limit": int(limit)}).data()

    def _fetch_event_context_details(
        self, session, event_ids: List[str]
    ) -> Dict[str, Dict[str, List[str]]]:
        if not event_ids:
            return {}
        query = """
        MATCH (e:Event)
        WHERE e.id IN $ids
        OPTIONAL MATCH (participant:Entity)-[:PARTICIPATED_IN]->(e)
        OPTIONAL MATCH (participant)-[:APPEARS_IN]->(scene:Scene)
        RETURN e.id AS id,
               collect(DISTINCT participant.class) AS participants,
               collect(DISTINCT scene.scene_type) AS locations
        """
        records = session.run(query, {"ids": event_ids}).data()
        return {record["id"]: record for record in records}

    def _enrich_events(self, items: List[Dict[str, Any]], lookup: Dict[str, Dict[str, Any]]) -> None:
        for item in items:
            for event in item.get("events", []):
                event_id = event.get("id")
                if not event_id:
                    continue
                metadata = lookup.get(event_id) or {}
                participants = [p for p in metadata.get("participants") or [] if p]
                locations = [l for l in metadata.get("locations") or [] if l]
                if participants:
                    event["participants"] = participants
                if locations:
                    event["locations"] = locations
