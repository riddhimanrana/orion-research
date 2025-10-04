"""
Part 3: Query & Evaluation Engine
===================================

This module implements the query and evaluation system for video understanding.
It provides three agents with different approaches and two evaluation benchmarks.

Architecture:
    Agent A: Gemini Baseline (vision-only, no graph)
    Agent B: Heuristic Graph Baseline (graph-only, no LLM query planning)
    Agent C: Augmented SOTA (vision + graph + LLM reasoning)

Benchmarks:
    EC-15: Episodic Comprehension (15 question types)
    LOT-Q: Long-form Open-ended Temporal Queries

Author: Orion Research Team
Date: January 2025
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum

# Third-party imports
import numpy as np

# Neo4j for graph queries
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("neo4j not available")

# Google Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("google-generativeai not available. Install with: pip install google-generativeai")

# OpenCV for video processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("opencv-python not available")

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for Part 3: Query & Evaluation Engine"""
    
    # Gemini API
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", None)
    GEMINI_MODEL: str = "gemini-2.0-flash-exp"  # Latest model
    GEMINI_TEMPERATURE: float = 0.3
    GEMINI_MAX_TOKENS: int = 2048
    
    # Neo4j connection
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    
    # Video clip extraction
    CLIP_MAX_FRAMES: int = 10  # Max frames per clip for Gemini
    CLIP_FRAME_SAMPLE_RATE: int = 5  # Sample every Nth frame
    
    # Agent C (Augmented SOTA) configuration
    USE_VISION_CONTEXT: bool = True  # Use Gemini for visual context
    USE_GRAPH_CONTEXT: bool = True  # Use Neo4j graph
    MAX_GRAPH_RESULTS: int = 20  # Max entities/events to retrieve
    RERANK_RESULTS: bool = True  # Re-rank graph results by relevance
    
    # Evaluation
    EVAL_OUTPUT_DIR: str = "data/evaluation"
    EC15_QUESTION_COUNT: int = 15  # Questions per video
    LOTQ_QUESTION_COUNT: int = 5  # Questions per video
    
    # Answer comparison
    SIMILARITY_THRESHOLD: float = 0.75  # For semantic similarity
    TEMPORAL_TOLERANCE_SECONDS: float = 2.0  # Tolerance for time matching


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class QuestionType(Enum):
    """Question types for EC-15 benchmark"""
    WHAT_ACTION = "what_action"
    WHAT_OBJECT = "what_object"
    WHEN_START = "when_start"
    WHEN_END = "when_end"
    WHERE_LOCATION = "where_location"
    WHERE_MOVEMENT = "where_movement"
    WHO_IDENTITY = "who_identity"
    WHO_COUNT = "who_count"
    WHY_REASON = "why_reason"
    WHY_GOAL = "why_goal"
    HOW_METHOD = "how_method"
    HOW_MANNER = "how_manner"
    TEMPORAL_ORDER = "temporal_order"
    SPATIAL_RELATION = "spatial_relation"
    STATE_CHANGE = "state_change"


@dataclass
class Question:
    """Represents a question for evaluation"""
    id: str
    question_text: str
    question_type: QuestionType
    video_path: str
    timestamp_start: Optional[float] = None  # Relevant time range
    timestamp_end: Optional[float] = None
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'question_text': self.question_text,
            'question_type': self.question_type.value if isinstance(self.question_type, QuestionType) else self.question_type,
            'video_path': self.video_path,
            'timestamp_start': self.timestamp_start,
            'timestamp_end': self.timestamp_end,
            'ground_truth': self.ground_truth,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Question':
        """Create from dictionary"""
        question_type = data['question_type']
        if isinstance(question_type, str):
            question_type = QuestionType(question_type)
        
        return cls(
            id=data['id'],
            question_text=data['question_text'],
            question_type=question_type,
            video_path=data['video_path'],
            timestamp_start=data.get('timestamp_start'),
            timestamp_end=data.get('timestamp_end'),
            ground_truth=data.get('ground_truth'),
            metadata=data.get('metadata', {})
        )


@dataclass
class Answer:
    """Represents an answer from an agent"""
    question_id: str
    agent_name: str
    answer_text: str
    confidence: float = 0.0
    reasoning: str = ""
    timestamp: Optional[float] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Answer':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class EvaluationResult:
    """Results of evaluating an answer"""
    question_id: str
    agent_name: str
    correct: bool
    score: float  # 0.0 to 1.0
    metrics: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_video_clip(
    video_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    max_frames: int = Config.CLIP_MAX_FRAMES,
    sample_rate: int = Config.CLIP_FRAME_SAMPLE_RATE
) -> List[np.ndarray]:
    """
    Extract frames from video for a specific time range.
    
    Args:
        video_path: Path to video file
        start_time: Start time in seconds (None = from beginning)
        end_time: End time in seconds (None = to end)
        max_frames: Maximum number of frames to extract
        sample_rate: Sample every Nth frame
    
    Returns:
        List of frames as numpy arrays (RGB format)
    """
    if not CV2_AVAILABLE:
        logger.error("OpenCV not available, cannot extract video clip")
        return []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame range
    start_frame = int(start_time * fps) if start_time else 0
    end_frame = int(end_time * fps) if end_time else total_frames
    
    # Extract frames
    frames = []
    frame_count = 0
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    while frame_count < max_frames and cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_count += 1
    
    cap.release()
    
    logger.debug(f"Extracted {len(frames)} frames from {video_path}")
    return frames


def format_timestamp(seconds: float) -> str:
    """Format timestamp as MM:SS"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def extract_entities_from_graph(
    driver,
    query: str,
    max_results: int = Config.MAX_GRAPH_RESULTS
) -> List[Dict[str, Any]]:
    """
    Query Neo4j graph for relevant entities.
    
    Args:
        driver: Neo4j driver instance
        query: Natural language query (will be converted to Cypher)
        max_results: Maximum number of results
    
    Returns:
        List of entity/event dictionaries
    """
    # TODO: Implement query-to-Cypher conversion
    # For now, return empty list
    logger.warning("Graph querying not yet implemented")
    return []


# ============================================================================
# AGENT BASE CLASS
# ============================================================================

class Agent:
    """Base class for question-answering agents"""
    
    def __init__(self, name: str):
        self.name = name
        logger.info(f"Initialized agent: {name}")
    
    def answer_question(
        self,
        question: Question,
        video_path: Optional[str] = None,
        neo4j_driver: Optional[Any] = None,
    ) -> Answer:
        """
        Answer a question about a video.
        
        Args:
            question: Question to answer
            video_path: Path to video file (optional, uses question.video_path if not provided)
            neo4j_driver: Neo4j driver for graph queries (optional)
        
        Returns:
            Answer object
        """
        raise NotImplementedError("Subclasses must implement answer_question()")
    
    def batch_answer(
        self,
        questions: List[Question],
        video_path: Optional[str] = None,
        neo4j_driver: Optional[Any] = None,
    ) -> List[Answer]:
        """
        Answer multiple questions.
        
        Args:
            questions: List of questions
            video_path: Path to video file
            neo4j_driver: Neo4j driver
        
        Returns:
            List of answers
        """
        answers = []
        for question in questions:
            answer = self.answer_question(question, video_path, neo4j_driver)
            answers.append(answer)
        return answers


# ============================================================================
# PLACEHOLDER FOR AGENT IMPLEMENTATIONS
# ============================================================================
# These will be implemented in subsequent steps:
# - AgentA_GeminiBaseline
# - AgentB_HeuristicGraph
# - AgentC_AugmentedSOTA


# ============================================================================
# EVALUATION FRAMEWORK
# ============================================================================

class Evaluator:
    """Evaluates agent answers against ground truth"""
    
    def __init__(self):
        self.results: List[EvaluationResult] = []
    
    def evaluate_answer(
        self,
        question: Question,
        answer: Answer,
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate a single answer.
        
        Args:
            question: The question
            answer: Agent's answer
            ground_truth: Ground truth answer (uses question.ground_truth if not provided)
        
        Returns:
            EvaluationResult
        """
        gt = ground_truth or question.ground_truth
        
        if not gt:
            logger.warning(f"No ground truth for question {question.id}")
            return EvaluationResult(
                question_id=question.id,
                agent_name=answer.agent_name,
                correct=False,
                score=0.0,
                explanation="No ground truth available"
            )
        
        # Simple exact match for now
        # TODO: Implement semantic similarity
        correct = answer.answer_text.lower().strip() == gt.lower().strip()
        score = 1.0 if correct else 0.0
        
        result = EvaluationResult(
            question_id=question.id,
            agent_name=answer.agent_name,
            correct=correct,
            score=score,
            explanation=f"Exact match: {correct}"
        )
        
        self.results.append(result)
        return result
    
    def evaluate_batch(
        self,
        questions: List[Question],
        answers: List[Answer]
    ) -> List[EvaluationResult]:
        """Evaluate multiple answers"""
        if len(questions) != len(answers):
            raise ValueError("Number of questions must match number of answers")
        
        results = []
        for question, answer in zip(questions, answers):
            result = self.evaluate_answer(question, answer)
            results.append(result)
        
        return results
    
    def compute_metrics(self, agent_name: Optional[str] = None) -> Dict[str, float]:
        """
        Compute aggregate metrics.
        
        Args:
            agent_name: Filter by agent name (None = all agents)
        
        Returns:
            Dictionary of metrics
        """
        # Filter results
        results = self.results
        if agent_name:
            results = [r for r in results if r.agent_name == agent_name]
        
        if not results:
            return {}
        
        # Compute metrics
        total = len(results)
        correct = sum(1 for r in results if r.correct)
        accuracy = correct / total if total > 0 else 0.0
        avg_score = sum(r.score for r in results) / total if total > 0 else 0.0
        
        return {
            'total_questions': total,
            'correct': correct,
            'incorrect': total - correct,
            'accuracy': accuracy,
            'average_score': avg_score
        }
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate evaluation report.
        
        Args:
            output_path: Path to save report (optional)
        
        Returns:
            Report as string
        """
        # Get unique agents
        agents = list(set(r.agent_name for r in self.results))
        
        report_lines = [
            "=" * 80,
            "EVALUATION REPORT",
            "=" * 80,
            ""
        ]
        
        # Overall metrics
        overall_metrics = self.compute_metrics()
        report_lines.extend([
            "Overall Metrics:",
            f"  Total Questions: {overall_metrics.get('total_questions', 0)}",
            f"  Accuracy: {overall_metrics.get('accuracy', 0):.2%}",
            f"  Average Score: {overall_metrics.get('average_score', 0):.3f}",
            ""
        ])
        
        # Per-agent metrics
        for agent in agents:
            metrics = self.compute_metrics(agent)
            report_lines.extend([
                f"Agent: {agent}",
                f"  Questions: {metrics.get('total_questions', 0)}",
                f"  Correct: {metrics.get('correct', 0)}",
                f"  Incorrect: {metrics.get('incorrect', 0)}",
                f"  Accuracy: {metrics.get('accuracy', 0):.2%}",
                ""
            ])
        
        report = "\n".join(report_lines)
        
        # Save if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to: {output_path}")
        
        return report


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def run_evaluation(
    questions: List[Question],
    agents: List[Agent],
    video_path: Optional[str] = None,
    neo4j_driver: Optional[Any] = None,
    output_dir: str = Config.EVAL_OUTPUT_DIR
) -> Dict[str, Any]:
    """
    Run full evaluation with all agents.
    
    Args:
        questions: List of questions to evaluate
        agents: List of agents to evaluate
        video_path: Path to video file
        neo4j_driver: Neo4j driver for graph queries
        output_dir: Directory to save results
    
    Returns:
        Results dictionary
    """
    logger.info(f"Running evaluation with {len(agents)} agents on {len(questions)} questions")
    
    evaluator = Evaluator()
    all_answers = {}
    
    # Run each agent
    for agent in agents:
        logger.info(f"Running agent: {agent.name}")
        answers = agent.batch_answer(questions, video_path, neo4j_driver)
        all_answers[agent.name] = answers
        
        # Evaluate
        results = evaluator.evaluate_batch(questions, answers)
        logger.info(f"  Completed {len(results)} evaluations")
    
    # Generate report
    report = evaluator.generate_report(
        output_path=os.path.join(output_dir, "evaluation_report.txt")
    )
    
    # Save detailed results
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'num_questions': len(questions),
        'num_agents': len(agents),
        'agents': [agent.name for agent in agents],
        'metrics': {
            agent.name: evaluator.compute_metrics(agent.name)
            for agent in agents
        },
        'questions': [q.to_dict() for q in questions],
        'answers': {
            agent_name: [a.to_dict() for a in answers]
            for agent_name, answers in all_answers.items()
        },
        'evaluation_results': [r.to_dict() for r in evaluator.results]
    }
    
    results_path = os.path.join(output_dir, "evaluation_results.json")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"Results saved to: {results_path}")
    
    return results_data


if __name__ == "__main__":
    # Test basic functionality
    logging.basicConfig(level=logging.INFO)
    
    print("Part 3: Query & Evaluation Engine")
    print("=" * 80)
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    print(f"  Neo4j driver: {'✓' if NEO4J_AVAILABLE else '✗'}")
    print(f"  Google Gemini: {'✓' if GEMINI_AVAILABLE else '✗'}")
    print(f"  OpenCV: {'✓' if CV2_AVAILABLE else '✗'}")
    print()
    
    # Create sample question
    question = Question(
        id="test_001",
        question_text="What is happening in the video?",
        question_type=QuestionType.WHAT_ACTION,
        video_path="data/testing/sample_video.mp4",
        ground_truth="A person is walking"
    )
    
    print("Sample Question:")
    print(f"  {question.question_text}")
    print(f"  Type: {question.question_type.value}")
    print(f"  Ground Truth: {question.ground_truth}")
    print()
    
    print("✓ Part 3 core module initialized")
    print("  Next: Implement agents A, B, C")
