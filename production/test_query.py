"""
Part 3 Testing Infrastructure
==============================

Comprehensive testing suite for the Query & Evaluation Engine.
Validates all three agents, benchmarks, and evaluation framework.

Author: Orion Research Team
Date: January 2025
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from query_evaluation import (
    Question,
    QuestionType,
    Config,
    Agent,
    Evaluator,
    run_evaluation,
)
from agents import (
    AgentA_GeminiBaseline,
    AgentB_HeuristicGraph,
    AgentC_AugmentedSOTA,
)
from query_config import (
    apply_config,
    get_preset,
    BALANCED_CONFIG,
    FAST_CONFIG,
)


# ============================================================================
# TEST DATA GENERATION
# ============================================================================

def generate_sample_questions(video_path: str = "test_video.mp4") -> List[Question]:
    """
    Generate sample questions for testing all 15 question types.
    
    Args:
        video_path: Path to test video
    
    Returns:
        List of test questions
    """
    questions = [
        # WHAT questions
        Question(
            id="q1",
            question_text="What action is the person performing?",
            question_type=QuestionType.WHAT_ACTION,
            video_path=video_path,
            timestamp_start=5.0,
            timestamp_end=10.0,
            ground_truth="walking",
            metadata={"difficulty": "easy"}
        ),
        Question(
            id="q2",
            question_text="What object is the person holding?",
            question_type=QuestionType.WHAT_OBJECT,
            video_path=video_path,
            timestamp_start=8.0,
            timestamp_end=12.0,
            ground_truth="red ball",
            metadata={"difficulty": "medium"}
        ),
        
        # WHEN questions
        Question(
            id="q3",
            question_text="When does the person start running?",
            question_type=QuestionType.WHEN_START,
            video_path=video_path,
            timestamp_start=0.0,
            timestamp_end=30.0,
            ground_truth="15.5",
            metadata={"difficulty": "medium"}
        ),
        Question(
            id="q4",
            question_text="When does the person stop moving?",
            question_type=QuestionType.WHEN_END,
            video_path=video_path,
            timestamp_start=0.0,
            timestamp_end=30.0,
            ground_truth="28.0",
            metadata={"difficulty": "medium"}
        ),
        
        # WHERE questions
        Question(
            id="q5",
            question_text="Where is the person located?",
            question_type=QuestionType.WHERE_LOCATION,
            video_path=video_path,
            timestamp_start=10.0,
            timestamp_end=15.0,
            ground_truth="park",
            metadata={"difficulty": "easy"}
        ),
        Question(
            id="q6",
            question_text="Where does the person move from the bench?",
            question_type=QuestionType.WHERE_MOVEMENT,
            video_path=video_path,
            timestamp_start=5.0,
            timestamp_end=20.0,
            ground_truth="towards the fountain",
            metadata={"difficulty": "hard"}
        ),
        
        # WHO questions
        Question(
            id="q7",
            question_text="Who is wearing a red shirt?",
            question_type=QuestionType.WHO_IDENTITY,
            video_path=video_path,
            timestamp_start=0.0,
            timestamp_end=10.0,
            ground_truth="the child",
            metadata={"difficulty": "medium"}
        ),
        Question(
            id="q8",
            question_text="How many people are in the scene?",
            question_type=QuestionType.WHO_COUNT,
            video_path=video_path,
            timestamp_start=5.0,
            timestamp_end=15.0,
            ground_truth="3",
            metadata={"difficulty": "easy"}
        ),
        
        # WHY questions
        Question(
            id="q9",
            question_text="Why did the person stop?",
            question_type=QuestionType.WHY_REASON,
            video_path=video_path,
            timestamp_start=10.0,
            timestamp_end=20.0,
            ground_truth="to pick up the ball",
            metadata={"difficulty": "hard"}
        ),
        Question(
            id="q10",
            question_text="What is the person's goal?",
            question_type=QuestionType.WHY_GOAL,
            video_path=video_path,
            timestamp_start=0.0,
            timestamp_end=30.0,
            ground_truth="playing with the dog",
            metadata={"difficulty": "hard"}
        ),
        
        # HOW questions
        Question(
            id="q11",
            question_text="How does the person open the door?",
            question_type=QuestionType.HOW_METHOD,
            video_path=video_path,
            timestamp_start=12.0,
            timestamp_end=18.0,
            ground_truth="by turning the handle",
            metadata={"difficulty": "medium"}
        ),
        Question(
            id="q12",
            question_text="How quickly is the person moving?",
            question_type=QuestionType.HOW_MANNER,
            video_path=video_path,
            timestamp_start=15.0,
            timestamp_end=20.0,
            ground_truth="slowly",
            metadata={"difficulty": "easy"}
        ),
        
        # Complex questions
        Question(
            id="q13",
            question_text="What is the order of events: picking up ball, running, stopping?",
            question_type=QuestionType.TEMPORAL_ORDER,
            video_path=video_path,
            timestamp_start=0.0,
            timestamp_end=30.0,
            ground_truth="running -> stopping -> picking up ball",
            metadata={"difficulty": "hard"}
        ),
        Question(
            id="q14",
            question_text="What is the spatial relationship between the person and the tree?",
            question_type=QuestionType.SPATIAL_RELATION,
            video_path=video_path,
            timestamp_start=10.0,
            timestamp_end=15.0,
            ground_truth="the person is to the left of the tree",
            metadata={"difficulty": "medium"}
        ),
        Question(
            id="q15",
            question_text="How does the door's state change?",
            question_type=QuestionType.STATE_CHANGE,
            video_path=video_path,
            timestamp_start=10.0,
            timestamp_end=20.0,
            ground_truth="closed -> opening -> open",
            metadata={"difficulty": "hard"}
        ),
    ]
    
    return questions


def generate_ec15_benchmark(
    video_path: str,
    count: int = 15
) -> List[Question]:
    """
    Generate EC-15 benchmark (Egocentric Comprehension 15 types).
    
    Args:
        video_path: Path to video file
        count: Number of questions to generate
    
    Returns:
        List of EC-15 questions
    """
    questions = generate_sample_questions(video_path)
    
    # Ensure we have all 15 types
    assert len(questions) == 15, "EC-15 must have exactly 15 questions (one per type)"
    
    # Duplicate if count > 15
    if count > 15:
        repetitions = (count // 15) + 1
        questions = (questions * repetitions)[:count]
    
    return questions[:count]


def generate_lotq_benchmark(
    video_path: str,
    count: int = 5
) -> List[Question]:
    """
    Generate LOT-Q benchmark (Long-form Temporal Questions).
    
    Args:
        video_path: Path to video file
        count: Number of questions to generate
    
    Returns:
        List of LOT-Q questions
    """
    questions = [
        Question(
            id="lotq1",
            question_text="Describe the complete sequence of actions from start to finish.",
            question_type=QuestionType.TEMPORAL_ORDER,
            video_path=video_path,
            timestamp_start=0.0,
            timestamp_end=60.0,
            ground_truth="person enters -> walks to bench -> sits down -> reads book -> stands up -> exits",
            metadata={"complexity": "high", "temporal_span": "long"}
        ),
        Question(
            id="lotq2",
            question_text="What state changes occur to the objects in the scene over time?",
            question_type=QuestionType.STATE_CHANGE,
            video_path=video_path,
            timestamp_start=0.0,
            timestamp_end=60.0,
            ground_truth="door: closed->open, book: closed->open->closed, lights: off->on",
            metadata={"complexity": "high", "temporal_span": "long"}
        ),
        Question(
            id="lotq3",
            question_text="How do the spatial relationships between entities change during the video?",
            question_type=QuestionType.SPATIAL_RELATION,
            video_path=video_path,
            timestamp_start=0.0,
            timestamp_end=60.0,
            ground_truth="person moves from door (far) -> bench (near) -> fountain (far)",
            metadata={"complexity": "high", "temporal_span": "long"}
        ),
        Question(
            id="lotq4",
            question_text="What are the goals and sub-goals demonstrated in this activity?",
            question_type=QuestionType.WHY_GOAL,
            video_path=video_path,
            timestamp_start=0.0,
            timestamp_end=60.0,
            ground_truth="main goal: relaxation, sub-goals: finding seat, reading, enjoying outdoors",
            metadata={"complexity": "high", "temporal_span": "long"}
        ),
        Question(
            id="lotq5",
            question_text="Identify all interactions between the person and objects, in order.",
            question_type=QuestionType.TEMPORAL_ORDER,
            video_path=video_path,
            timestamp_start=0.0,
            timestamp_end=60.0,
            ground_truth="door handle -> bench -> book -> book (close) -> bench (stand)",
            metadata={"complexity": "high", "temporal_span": "long"}
        ),
    ]
    
    return questions[:count]


# ============================================================================
# AGENT INITIALIZATION
# ============================================================================

def init_agent_a(api_key: Optional[str] = None) -> Optional[AgentA_GeminiBaseline]:
    """Initialize Agent A (Gemini Baseline)."""
    try:
        api_key = api_key or os.getenv('GEMINI_API_KEY') or Config.GEMINI_API_KEY
        if not api_key:
            print("⚠️  Warning: No Gemini API key provided. Agent A will be skipped.")
            return None
        
        agent = AgentA_GeminiBaseline(api_key=api_key)
        print("✓ Agent A (Gemini Baseline) initialized")
        return agent
    except Exception as e:
        print(f"✗ Failed to initialize Agent A: {e}")
        return None


def init_agent_b(
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None
) -> Optional[AgentB_HeuristicGraph]:
    """Initialize Agent B (Heuristic Graph)."""
    try:
        uri = neo4j_uri or Config.NEO4J_URI
        user = neo4j_user or Config.NEO4J_USER
        password = neo4j_password or Config.NEO4J_PASSWORD
        
        agent = AgentB_HeuristicGraph(
            neo4j_uri=uri,
            neo4j_user=user,
            neo4j_password=password
        )
        print("✓ Agent B (Heuristic Graph) initialized")
        return agent
    except Exception as e:
        print(f"✗ Failed to initialize Agent B: {e}")
        return None


def init_agent_c(
    api_key: Optional[str] = None,
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None
) -> Optional[AgentC_AugmentedSOTA]:
    """Initialize Agent C (Augmented SOTA)."""
    try:
        api_key = api_key or os.getenv('GEMINI_API_KEY') or Config.GEMINI_API_KEY
        uri = neo4j_uri or Config.NEO4J_URI
        user = neo4j_user or Config.NEO4J_USER
        password = neo4j_password or Config.NEO4J_PASSWORD
        
        if not api_key:
            print("⚠️  Warning: No Gemini API key provided. Agent C will be skipped.")
            return None
        
        agent = AgentC_AugmentedSOTA(
            api_key=api_key,
            neo4j_uri=uri,
            neo4j_user=user,
            neo4j_password=password
        )
        print("✓ Agent C (Augmented SOTA) initialized")
        return agent
    except Exception as e:
        print(f"✗ Failed to initialize Agent C: {e}")
        return None


def init_all_agents(
    api_key: Optional[str] = None,
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None
) -> List[Agent]:
    """Initialize all agents that are available."""
    agents = []
    
    print("\nInitializing agents...")
    print("=" * 60)
    
    # Agent A
    agent_a = init_agent_a(api_key)
    if agent_a:
        agents.append(agent_a)
    
    # Agent B
    agent_b = init_agent_b(neo4j_uri, neo4j_user, neo4j_password)
    if agent_b:
        agents.append(agent_b)
    
    # Agent C
    agent_c = init_agent_c(api_key, neo4j_uri, neo4j_user, neo4j_password)
    if agent_c:
        agents.append(agent_c)
    
    print("=" * 60)
    print(f"✓ Initialized {len(agents)}/3 agents\n")
    
    return agents


# ============================================================================
# TEST CASES
# ============================================================================

def test_single_question(
    question: Question,
    agents: List[Agent],
    video_path: str
) -> None:
    """
    Test a single question with all agents.
    
    Args:
        question: Question to test
        agents: List of agents
        video_path: Path to video file
    """
    print(f"\nTesting Question: {question.id}")
    print("=" * 80)
    print(f"Question: {question.question_text}")
    print(f"Type: {question.question_type.value}")
    print(f"Ground Truth: {question.ground_truth}")
    print(f"Timestamp: {question.timestamp_start}s - {question.timestamp_end}s")
    print()
    
    for agent in agents:
        print(f"\n{agent.name}")
        print("-" * 80)
        
        try:
            start_time = time.time()
            answer = agent.answer_question(question, video_path, None)
            elapsed = time.time() - start_time
            
            print(f"Answer: {answer.answer_text}")
            print(f"Confidence: {answer.confidence:.2f}")
            print(f"Processing Time: {elapsed:.2f}s")
            
            if answer.reasoning:
                print(f"Reasoning: {answer.reasoning[:200]}...")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\n" + "=" * 80)


def test_batch_questions(
    questions: List[Question],
    agents: List[Agent],
    video_path: str,
    output_dir: str = "test_results"
) -> None:
    """
    Test a batch of questions with all agents.
    
    Args:
        questions: List of questions
        agents: List of agents
        video_path: Path to video file
        output_dir: Directory to save results
    """
    print(f"\nBatch Testing: {len(questions)} questions")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run evaluation
    result = run_evaluation(
        questions=questions,
        agents=agents,
        video_path=video_path,
        neo4j_driver=None,  # Agents handle their own connections
        output_dir=output_dir
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    
    if isinstance(result, dict):
        # Result is a metrics dictionary
        for agent in agents:
            agent_metrics = result.get(agent.name, {})
            if agent_metrics:
                print(f"\n{agent.name}:")
                print(f"  Accuracy: {agent_metrics.get('accuracy', 0):.2%}")
                print(f"  Avg Confidence: {agent_metrics.get('avg_confidence', 0):.2f}")
                print(f"  Avg Processing Time: {agent_metrics.get('avg_processing_time', 0):.2f}s")
                print(f"  Total Questions: {agent_metrics.get('total_questions', 0)}")
                print(f"  Correct Answers: {agent_metrics.get('correct_answers', 0)}")
    else:
        # Result is an Evaluator object
        for agent in agents:
            metrics = result.compute_metrics(agent.name)
            print(f"\n{agent.name}:")
            print(f"  Accuracy: {metrics['accuracy']:.2%}")
            print(f"  Avg Confidence: {metrics['avg_confidence']:.2f}")
            print(f"  Avg Processing Time: {metrics['avg_processing_time']:.2f}s")
            print(f"  Total Questions: {metrics['total_questions']}")
            print(f"  Correct Answers: {metrics['correct_answers']}")
        
        # Generate report
        report_path = os.path.join(output_dir, "evaluation_report.json")
        result.generate_report(report_path)
        print(f"\n✓ Report saved to: {report_path}")


def test_question_types(
    agents: List[Agent],
    video_path: str
) -> None:
    """
    Test all 15 question types.
    
    Args:
        agents: List of agents
        video_path: Path to video file
    """
    print("\nTesting All Question Types (EC-15)")
    print("=" * 80)
    
    questions = generate_sample_questions(video_path)
    
    # Test each type
    for question in questions:
        test_single_question(question, agents, video_path)
        print("\n" + "-" * 80 + "\n")


def test_ec15_benchmark(
    agents: List[Agent],
    video_path: str,
    output_dir: str = "test_results/ec15"
) -> None:
    """
    Test EC-15 benchmark.
    
    Args:
        agents: List of agents
        video_path: Path to video file
        output_dir: Directory to save results
    """
    print("\nRunning EC-15 Benchmark")
    print("=" * 80)
    
    questions = generate_ec15_benchmark(video_path, count=Config.EC15_QUESTION_COUNT)
    test_batch_questions(questions, agents, video_path, output_dir)


def test_lotq_benchmark(
    agents: List[Agent],
    video_path: str,
    output_dir: str = "test_results/lotq"
) -> None:
    """
    Test LOT-Q benchmark.
    
    Args:
        agents: List of agents
        video_path: Path to video file
        output_dir: Directory to save results
    """
    print("\nRunning LOT-Q Benchmark")
    print("=" * 80)
    
    questions = generate_lotq_benchmark(video_path, count=Config.LOTQ_QUESTION_COUNT)
    test_batch_questions(questions, agents, video_path, output_dir)


def test_integration_with_parts12(
    parts12_output_dir: str = "output_integrated",
    output_dir: str = "test_results/integrated"
) -> None:
    """
    Test integration with Parts 1+2 output.
    
    Args:
        parts12_output_dir: Directory with Parts 1+2 output
        output_dir: Directory to save Part 3 results
    """
    print("\nTesting Integration with Parts 1+2")
    print("=" * 80)
    
    # Check for Parts 1+2 output
    metadata_path = os.path.join(parts12_output_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"✗ Parts 1+2 output not found at: {parts12_output_dir}")
        print("  Run integrated_pipeline.py first!")
        return
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    video_path = metadata.get('video_path', 'test_video.mp4')
    graph_path = metadata.get('graph_path', 'knowledge_graph.json')
    
    print(f"✓ Found Parts 1+2 output")
    print(f"  Video: {video_path}")
    print(f"  Graph: {graph_path}")
    print()
    
    # Initialize agents
    agents = init_all_agents()
    
    if not agents:
        print("✗ No agents available")
        return
    
    # Generate questions based on the video
    questions = generate_sample_questions(video_path)
    
    # Run evaluation
    test_batch_questions(questions, agents, video_path, output_dir)


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests(
    video_path: str = "test_video.mp4",
    config_preset: str = "balanced"
) -> None:
    """
    Run all test cases.
    
    Args:
        video_path: Path to test video
        config_preset: Configuration preset to use
    """
    print("\n" + "=" * 80)
    print("Part 3 Testing Infrastructure")
    print("=" * 80)
    print(f"Video: {video_path}")
    print(f"Config: {config_preset}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)
    
    # Apply configuration
    config = get_preset(config_preset)
    apply_config(config)
    
    # Initialize agents
    agents = init_all_agents()
    
    if not agents:
        print("\n✗ No agents available. Please configure API keys and Neo4j.")
        return
    
    # Run tests
    print("\n" + "=" * 80)
    print("Starting Tests")
    print("=" * 80)
    
    # 1. Single question test
    print("\n\n1. SINGLE QUESTION TEST")
    print("=" * 80)
    questions = generate_sample_questions(video_path)
    test_single_question(questions[0], agents, video_path)
    
    # 2. Question types test
    print("\n\n2. QUESTION TYPES TEST")
    print("=" * 80)
    test_question_types(agents, video_path)
    
    # 3. EC-15 benchmark
    print("\n\n3. EC-15 BENCHMARK")
    print("=" * 80)
    test_ec15_benchmark(agents, video_path)
    
    # 4. LOT-Q benchmark
    print("\n\n4. LOT-Q BENCHMARK")
    print("=" * 80)
    test_lotq_benchmark(agents, video_path)
    
    print("\n" + "=" * 80)
    print("All Tests Complete!")
    print("=" * 80)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Part 3 Testing Infrastructure")
    parser.add_argument(
        '--video',
        type=str,
        default='test_video.mp4',
        help='Path to test video'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='balanced',
        choices=['baseline', 'balanced', 'high_quality', 'fast'],
        help='Configuration preset'
    )
    parser.add_argument(
        '--test',
        type=str,
        choices=['single', 'types', 'ec15', 'lotq', 'integration', 'all'],
        default='all',
        help='Test to run'
    )
    parser.add_argument(
        '--gemini-key',
        type=str,
        help='Google Gemini API key'
    )
    parser.add_argument(
        '--neo4j-uri',
        type=str,
        default='bolt://localhost:7687',
        help='Neo4j URI'
    )
    parser.add_argument(
        '--neo4j-user',
        type=str,
        default='neo4j',
        help='Neo4j username'
    )
    parser.add_argument(
        '--neo4j-password',
        type=str,
        help='Neo4j password'
    )
    
    args = parser.parse_args()
    
    # Apply configuration
    config = get_preset(args.config)
    apply_config(config)
    
    # Initialize agents
    agents = init_all_agents(
        api_key=args.gemini_key,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password
    )
    
    if not agents:
        print("\n✗ No agents available. Please provide API keys and Neo4j credentials.")
        sys.exit(1)
    
    # Run selected test
    if args.test == 'single':
        questions = generate_sample_questions(args.video)
        test_single_question(questions[0], agents, args.video)
    
    elif args.test == 'types':
        test_question_types(agents, args.video)
    
    elif args.test == 'ec15':
        test_ec15_benchmark(agents, args.video)
    
    elif args.test == 'lotq':
        test_lotq_benchmark(agents, args.video)
    
    elif args.test == 'integration':
        test_integration_with_parts12()
    
    elif args.test == 'all':
        run_all_tests(args.video, args.config)
