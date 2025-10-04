"""
Part 3: Agent Implementations
==============================

This module implements the three question-answering agents:
- Agent A: Gemini Baseline (vision-only)
- Agent B: Heuristic Graph Baseline (graph-only)
- Agent C: Augmented SOTA (vision + graph + reasoning)

Each agent has different capabilities and serves as a comparison point.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from PIL import Image
import io

# Import base classes from part3_query_evaluation
from production.part3_query_evaluation import (
    Agent, Question, Answer, QuestionType, Config,
    extract_video_clip, format_timestamp
)

# Google Gemini
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("google-generativeai not available")

# Neo4j
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# AGENT A: GEMINI BASELINE
# ============================================================================

class AgentA_GeminiBaseline(Agent):
    """
    Baseline agent using only Gemini vision API.
    
    No knowledge graph - relies purely on visual understanding of video clips.
    This establishes the baseline performance for pure vision-language models.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini baseline agent.
        
        Args:
            api_key: Google Gemini API key (uses Config.GEMINI_API_KEY if not provided)
        """
        super().__init__("Agent A: Gemini Baseline")
        
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai required for Agent A")
        
        self.api_key = api_key or Config.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter"
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        
        # Safety settings - allow all to avoid blocks
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        logger.info(f"Initialized {self.name} with model: {Config.GEMINI_MODEL}")
    
    def answer_question(
        self,
        question: Question,
        video_path: Optional[str] = None,
        neo4j_driver: Optional[Any] = None,
    ) -> Answer:
        """
        Answer question using only Gemini vision.
        
        Args:
            question: Question to answer
            video_path: Path to video (uses question.video_path if not provided)
            neo4j_driver: Not used by this agent
        
        Returns:
            Answer object
        """
        start_time = time.time()
        
        # Get video path
        vid_path = video_path or question.video_path
        if not vid_path:
            return Answer(
                question_id=question.id,
                agent_name=self.name,
                answer_text="[Error: No video path provided]",
                confidence=0.0,
                processing_time=0.0
            )
        
        # Extract video clip
        frames = extract_video_clip(
            vid_path,
            start_time=question.timestamp_start,
            end_time=question.timestamp_end,
            max_frames=Config.CLIP_MAX_FRAMES,
            sample_rate=Config.CLIP_FRAME_SAMPLE_RATE
        )
        
        if not frames:
            return Answer(
                question_id=question.id,
                agent_name=self.name,
                answer_text="[Error: Could not extract video frames]",
                confidence=0.0,
                processing_time=time.time() - start_time
            )
        
        # Convert frames to PIL Images
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        # Build prompt
        prompt = self._build_prompt(question, len(pil_frames))
        
        # Query Gemini
        try:
            # Prepare content with images and text
            content = pil_frames + [prompt]
            
            response = self.model.generate_content(
                content,
                safety_settings=self.safety_settings,
                generation_config=genai.GenerationConfig(
                    temperature=Config.GEMINI_TEMPERATURE,
                    max_output_tokens=Config.GEMINI_MAX_TOKENS,
                )
            )
            
            answer_text = response.text if response.text else "[No response from model]"
            
            # Try to extract confidence (not directly available, estimate from response)
            confidence = 0.8  # Default moderate confidence
            
            reasoning = f"Analyzed {len(pil_frames)} frames from video using Gemini vision"
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            answer_text = f"[Error: {str(e)}]"
            confidence = 0.0
            reasoning = f"Failed to query Gemini: {str(e)}"
        
        processing_time = time.time() - start_time
        
        return Answer(
            question_id=question.id,
            agent_name=self.name,
            answer_text=answer_text,
            confidence=confidence,
            reasoning=reasoning,
            processing_time=processing_time,
            evidence=[{
                'type': 'visual_frames',
                'num_frames': len(pil_frames),
                'timestamp_range': f"{question.timestamp_start or 0:.1f}s - {question.timestamp_end or 'end'}s"
            }]
        )
    
    def _build_prompt(self, question: Question, num_frames: int) -> str:
        """
        Build prompt for Gemini based on question type.
        
        Args:
            question: The question
            num_frames: Number of frames being analyzed
        
        Returns:
            Prompt string
        """
        base_prompt = f"""You are analyzing {num_frames} frames from a video. 

Question: {question.question_text}

Please provide a clear, concise answer based on what you observe in the video frames. 
If you're uncertain, indicate your confidence level.

Answer:"""
        
        # Add question-type specific guidance
        if question.question_type == QuestionType.WHEN_START or question.question_type == QuestionType.WHEN_END:
            base_prompt += "\nNote: Provide timestamp in MM:SS format if possible."
        
        elif question.question_type == QuestionType.WHO_COUNT:
            base_prompt += "\nNote: Provide an exact count."
        
        elif question.question_type == QuestionType.TEMPORAL_ORDER:
            base_prompt += "\nNote: Describe the sequence of events in chronological order."
        
        return base_prompt


# ============================================================================
# AGENT B: HEURISTIC GRAPH BASELINE
# ============================================================================

class AgentB_HeuristicGraph(Agent):
    """
    Baseline agent using only the knowledge graph with heuristic queries.
    
    No vision input - relies purely on the structured knowledge graph.
    Uses hand-crafted Cypher queries based on question type.
    No LLM for query planning.
    """
    
    def __init__(self, neo4j_uri: Optional[str] = None, neo4j_user: Optional[str] = None, neo4j_password: Optional[str] = None):
        """
        Initialize heuristic graph agent.
        
        Args:
            neo4j_uri: Neo4j URI (uses Config if not provided)
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        super().__init__("Agent B: Heuristic Graph")
        
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j driver required for Agent B")
        
        self.neo4j_uri = neo4j_uri or Config.NEO4J_URI
        self.neo4j_user = neo4j_user or Config.NEO4J_USER
        self.neo4j_password = neo4j_password or Config.NEO4J_PASSWORD
        
        # Don't create driver here - will use passed driver or create per-query
        logger.info(f"Initialized {self.name}")
    
    def answer_question(
        self,
        question: Question,
        video_path: Optional[str] = None,
        neo4j_driver: Optional[Any] = None,
    ) -> Answer:
        """
        Answer question using only knowledge graph queries.
        
        Args:
            question: Question to answer
            video_path: Not used by this agent
            neo4j_driver: Neo4j driver instance
        
        Returns:
            Answer object
        """
        start_time = time.time()
        
        # Get or create driver
        if neo4j_driver:
            driver = neo4j_driver
            should_close = False
        else:
            try:
                driver = GraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password)
                )
                should_close = True
            except Exception as e:
                return Answer(
                    question_id=question.id,
                    agent_name=self.name,
                    answer_text=f"[Error: Could not connect to Neo4j: {e}]",
                    confidence=0.0,
                    processing_time=time.time() - start_time
                )
        
        # Build Cypher query based on question type
        cypher_query = self._build_cypher_query(question)
        
        # Execute query
        try:
            with driver.session() as session:
                result = session.run(cypher_query)
                records = list(result)
            
            # Format answer from results
            answer_text = self._format_answer(question, records)
            confidence = 0.7 if records else 0.3
            reasoning = f"Executed Cypher query, found {len(records)} results"
            evidence = [{'type': 'graph_query', 'cypher': cypher_query, 'num_results': len(records)}]
            
        except Exception as e:
            logger.error(f"Neo4j query error: {e}")
            answer_text = f"[Error: {str(e)}]"
            confidence = 0.0
            reasoning = f"Failed to query graph: {str(e)}"
            evidence = []
        
        finally:
            if should_close:
                driver.close()
        
        processing_time = time.time() - start_time
        
        return Answer(
            question_id=question.id,
            agent_name=self.name,
            answer_text=answer_text,
            confidence=confidence,
            reasoning=reasoning,
            processing_time=processing_time,
            evidence=evidence
        )
    
    def _build_cypher_query(self, question: Question) -> str:
        """
        Build Cypher query based on question type using heuristics.
        
        Args:
            question: The question
        
        Returns:
            Cypher query string
        """
        q_type = question.question_type
        
        # What queries
        if q_type == QuestionType.WHAT_ACTION:
            return """
            MATCH (e:Entity)-[:PARTICIPATED_IN]->(ev:Event)
            RETURN e.label, ev.description, ev.timestamp
            ORDER BY ev.timestamp
            LIMIT 10
            """
        
        elif q_type == QuestionType.WHAT_OBJECT:
            return """
            MATCH (e:Entity)
            RETURN e.label, count(e) as count, e.first_description
            ORDER BY count DESC
            LIMIT 10
            """
        
        # When queries
        elif q_type == QuestionType.WHEN_START:
            return """
            MATCH (e:Entity)
            RETURN e.label, e.first_seen
            ORDER BY e.first_seen
            LIMIT 1
            """
        
        elif q_type == QuestionType.WHEN_END:
            return """
            MATCH (e:Entity)
            RETURN e.label, e.last_seen
            ORDER BY e.last_seen DESC
            LIMIT 1
            """
        
        # Who queries
        elif q_type == QuestionType.WHO_IDENTITY:
            return """
            MATCH (e:Entity {label: 'person'})
            RETURN e.id, e.first_description, e.appearance_count
            ORDER BY e.appearance_count DESC
            LIMIT 5
            """
        
        elif q_type == QuestionType.WHO_COUNT:
            return """
            MATCH (e:Entity {label: 'person'})
            RETURN count(DISTINCT e) as person_count
            """
        
        # Temporal queries
        elif q_type == QuestionType.TEMPORAL_ORDER:
            return """
            MATCH (e:Entity)-[:PARTICIPATED_IN]->(ev:Event)
            RETURN e.label, ev.description, ev.timestamp
            ORDER BY ev.timestamp
            LIMIT 20
            """
        
        # State change queries
        elif q_type == QuestionType.STATE_CHANGE:
            return """
            MATCH (e:Entity)-[:PARTICIPATED_IN]->(ev:Event {type: 'state_change'})
            RETURN e.label, ev.description, ev.timestamp
            ORDER BY ev.timestamp
            LIMIT 10
            """
        
        # Default: get all entities and events
        else:
            return """
            MATCH (e:Entity)
            OPTIONAL MATCH (e)-[:PARTICIPATED_IN]->(ev:Event)
            RETURN e.label, e.first_description, ev.description, ev.timestamp
            LIMIT 10
            """
    
    def _format_answer(self, question: Question, records: List[Any]) -> str:
        """
        Format graph query results into natural language answer.
        
        Args:
            question: The question
            records: Neo4j query results
        
        Returns:
            Formatted answer string
        """
        if not records:
            return "No relevant information found in the knowledge graph."
        
        q_type = question.question_type
        
        # WHO_COUNT - return exact number
        if q_type == QuestionType.WHO_COUNT:
            count = records[0]['person_count']
            return f"{count} people"
        
        # WHEN queries - return timestamp
        elif q_type == QuestionType.WHEN_START:
            timestamp = records[0]['e.first_seen']
            return f"At {format_timestamp(timestamp)}"
        
        elif q_type == QuestionType.WHEN_END:
            timestamp = records[0]['e.last_seen']
            return f"At {format_timestamp(timestamp)}"
        
        # TEMPORAL_ORDER - list events chronologically
        elif q_type == QuestionType.TEMPORAL_ORDER:
            events = []
            for record in records[:5]:  # Limit to 5 events
                label = record.get('e.label', 'entity')
                desc = record.get('ev.description', 'unknown event')
                events.append(f"{label}: {desc}")
            return "; ".join(events)
        
        # STATE_CHANGE - list state changes
        elif q_type == QuestionType.STATE_CHANGE:
            changes = []
            for record in records[:3]:
                label = record.get('e.label', 'entity')
                desc = record.get('ev.description', 'changed state')
                changes.append(f"{label} {desc}")
            return "; ".join(changes) if changes else "No state changes detected"
        
        # Default: describe first few entities/events
        else:
            items = []
            for record in records[:3]:
                label = record.get('e.label', 'entity')
                desc = record.get('e.first_description') or record.get('ev.description', '')
                if desc:
                    items.append(f"{label}: {desc}")
                else:
                    items.append(label)
            
            return "; ".join(items) if items else "Multiple entities and events found"


# ============================================================================
# AGENT C: AUGMENTED SOTA
# ============================================================================

class AgentC_AugmentedSOTA(Agent):
    """
    State-of-the-art agent combining vision, graph, and reasoning.
    
    Uses:
    1. Gemini vision for visual context
    2. Neo4j knowledge graph for structured memory
    3. LLM-powered query planning and result fusion
    
    This is the main system being evaluated.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None
    ):
        """
        Initialize augmented SOTA agent.
        
        Args:
            api_key: Gemini API key
            neo4j_uri: Neo4j URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        super().__init__("Agent C: Augmented SOTA")
        
        # Initialize Gemini (if available and enabled)
        self.gemini_enabled = Config.USE_VISION_CONTEXT and GEMINI_AVAILABLE
        if self.gemini_enabled:
            self.api_key = api_key or Config.GEMINI_API_KEY
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
                self.safety_settings = {
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            else:
                self.gemini_enabled = False
                logger.warning("Gemini API key not provided, vision context disabled")
        
        # Initialize Neo4j connection info
        self.graph_enabled = Config.USE_GRAPH_CONTEXT and NEO4J_AVAILABLE
        if self.graph_enabled:
            self.neo4j_uri = neo4j_uri or Config.NEO4J_URI
            self.neo4j_user = neo4j_user or Config.NEO4J_USER
            self.neo4j_password = neo4j_password or Config.NEO4J_PASSWORD
        
        logger.info(f"Initialized {self.name}")
        logger.info(f"  Vision context: {'enabled' if self.gemini_enabled else 'disabled'}")
        logger.info(f"  Graph context: {'enabled' if self.graph_enabled else 'disabled'}")
    
    def answer_question(
        self,
        question: Question,
        video_path: Optional[str] = None,
        neo4j_driver: Optional[Any] = None,
    ) -> Answer:
        """
        Answer question using augmented retrieval.
        
        Strategy:
        1. Query knowledge graph for structured information
        2. Extract relevant video clips for visual context
        3. Combine both sources with LLM reasoning
        
        Args:
            question: Question to answer
            video_path: Path to video
            neo4j_driver: Neo4j driver instance
        
        Returns:
            Answer object
        """
        start_time = time.time()
        
        # Get video path
        vid_path = video_path or question.video_path
        
        # Collect evidence from multiple sources
        evidence = []
        graph_context = ""
        visual_context = ""
        
        # 1. Query knowledge graph
        if self.graph_enabled:
            graph_results = self._query_graph(question, neo4j_driver)
            if graph_results:
                graph_context = self._format_graph_results(graph_results)
                evidence.append({
                    'type': 'graph_query',
                    'num_results': len(graph_results),
                    'context': graph_context[:200]  # Truncate for evidence
                })
        
        # 2. Extract visual context (if enabled and video available)
        if self.gemini_enabled and vid_path:
            visual_context = self._get_visual_context(question, vid_path)
            if visual_context:
                evidence.append({
                    'type': 'visual_analysis',
                    'context': visual_context[:200]
                })
        
        # 3. Combine contexts and generate answer
        answer_text = self._generate_answer(question, graph_context, visual_context)
        
        # Estimate confidence based on available evidence
        confidence = 0.5  # Base
        if graph_context:
            confidence += 0.2
        if visual_context:
            confidence += 0.3
        confidence = min(confidence, 1.0)
        
        reasoning = f"Combined {len(evidence)} sources: "
        reasoning += ", ".join(e['type'] for e in evidence)
        
        processing_time = time.time() - start_time
        
        return Answer(
            question_id=question.id,
            agent_name=self.name,
            answer_text=answer_text,
            confidence=confidence,
            reasoning=reasoning,
            processing_time=processing_time,
            evidence=evidence
        )
    
    def _query_graph(self, question: Question, neo4j_driver: Optional[Any]) -> List[Dict[str, Any]]:
        """Query Neo4j graph for relevant information."""
        if not self.graph_enabled:
            return []
        
        # Use provided driver or create new one
        if neo4j_driver:
            driver = neo4j_driver
            should_close = False
        else:
            try:
                driver = GraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password)
                )
                should_close = True
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                return []
        
        try:
            # Build query based on question type (reuse heuristic approach)
            agent_b = AgentB_HeuristicGraph()
            cypher_query = agent_b._build_cypher_query(question)
            
            with driver.session() as session:
                result = session.run(cypher_query)
                records = [dict(record) for record in result]
            
            return records[:Config.MAX_GRAPH_RESULTS]
        
        except Exception as e:
            logger.error(f"Graph query error: {e}")
            return []
        
        finally:
            if should_close:
                driver.close()
    
    def _format_graph_results(self, results: List[Dict[str, Any]]) -> str:
        """Format graph results as text context."""
        if not results:
            return ""
        
        context_lines = ["Knowledge Graph Information:"]
        for i, result in enumerate(results[:5], 1):  # Limit to 5
            # Extract key information
            parts = []
            for key, value in result.items():
                if value is not None:
                    parts.append(f"{key}: {value}")
            context_lines.append(f"{i}. {', '.join(parts)}")
        
        return "\n".join(context_lines)
    
    def _get_visual_context(self, question: Question, video_path: str) -> str:
        """Get visual context from video using Gemini."""
        if not self.gemini_enabled:
            return ""
        
        try:
            # Extract frames
            frames = extract_video_clip(
                video_path,
                start_time=question.timestamp_start,
                end_time=question.timestamp_end,
                max_frames=Config.CLIP_MAX_FRAMES,
                sample_rate=Config.CLIP_FRAME_SAMPLE_RATE
            )
            
            if not frames:
                return ""
            
            # Convert to PIL
            pil_frames = [Image.fromarray(frame) for frame in frames]
            
            # Ask Gemini to describe what it sees (not answer the question yet)
            prompt = f"Describe what you see in these {len(pil_frames)} video frames. Focus on objects, actions, and context."
            
            content = pil_frames + [prompt]
            response = self.model.generate_content(
                content,
                safety_settings=self.safety_settings,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=512,
                )
            )
            
            return f"Visual Analysis: {response.text}" if response.text else ""
        
        except Exception as e:
            logger.error(f"Visual context error: {e}")
            return ""
    
    def _generate_answer(self, question: Question, graph_context: str, visual_context: str) -> str:
        """Generate final answer by combining all contexts."""
        # If we have Gemini, use it to synthesize
        if self.gemini_enabled:
            try:
                prompt = f"""You are answering a question about a video using multiple information sources.

Question: {question.question_text}

{graph_context if graph_context else "No graph information available."}

{visual_context if visual_context else "No visual information available."}

Based on all available information above, provide a clear, concise answer to the question.
If information is missing or conflicting, indicate your uncertainty.

Answer:"""
                
                response = self.model.generate_content(
                    prompt,
                    safety_settings=self.safety_settings,
                    generation_config=genai.GenerationConfig(
                        temperature=Config.GEMINI_TEMPERATURE,
                        max_output_tokens=Config.GEMINI_MAX_TOKENS,
                    )
                )
                
                return response.text if response.text else "Unable to generate answer"
            
            except Exception as e:
                logger.error(f"Answer generation error: {e}")
        
        # Fallback: simple combination
        if graph_context and visual_context:
            return f"Based on the knowledge graph and visual analysis: {graph_context.split(':')[1] if ':' in graph_context else 'Information available'}"
        elif graph_context:
            return graph_context.split('\n')[1] if '\n' in graph_context else "Based on knowledge graph"
        elif visual_context:
            return visual_context.replace("Visual Analysis:", "").strip()
        else:
            return "Insufficient information to answer the question"


if __name__ == "__main__":
    # Test agents
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Part 3: Agent Implementations")
    print("=" * 80)
    print()
    
    # Check availability
    print("Agent Availability:")
    print(f"  Agent A (Gemini Baseline): {'✓' if GEMINI_AVAILABLE else '✗ (google-generativeai required)'}")
    print(f"  Agent B (Heuristic Graph): {'✓' if NEO4J_AVAILABLE else '✗ (neo4j required)'}")
    print(f"  Agent C (Augmented SOTA): {'✓' if (GEMINI_AVAILABLE or NEO4J_AVAILABLE) else '✗'}")
    print()
    
    print("Agents implemented:")
    print("  ✓ Agent A: Gemini Baseline (vision-only)")
    print("  ✓ Agent B: Heuristic Graph Baseline (graph-only)")
    print("  ✓ Agent C: Augmented SOTA (vision + graph + reasoning)")
