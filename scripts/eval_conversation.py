#!/usr/bin/env python3
"""
Conversational Q&A Evaluation with Gemini
==========================================

Tests Orion's Stage 6 reasoning capabilities through a series of 
conversational queries, validated by Gemini watching the same video.

This script:
1. Runs a conversation with Orion about a video
2. Sends the same questions + video to Gemini
3. Compares answers to detect hallucinations and inaccuracies
4. Generates a detailed evaluation report

Usage:
    python scripts/eval_conversation.py --video data/examples/video.mp4 --episode conv_eval_001
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Conversational test scenarios
CONVERSATION_SCENARIOS = [
    {
        "name": "Basic Object Discovery",
        "questions": [
            "What objects are visible in this video?",
            "How many times does the person appear?",
            "What electronic devices can you see?",
        ],
    },
    {
        "name": "Spatial Reasoning",
        "questions": [
            "What objects were near the person?",
            "Were any objects on top of other objects?",
            "What was the spatial layout of objects?",
        ],
    },
    {
        "name": "Temporal Understanding",
        "questions": [
            "What happened at the beginning of the video?",
            "When did objects first appear?",
            "What changed throughout the video?",
        ],
    },
    {
        "name": "Interaction Analysis",
        "questions": [
            "What did the person interact with?",
            "Were any objects picked up or moved?",
            "Describe the activities shown in the video.",
        ],
    },
    {
        "name": "Follow-up Reasoning",
        "questions": [
            "Based on what you know, what room is this?",
            "What was the person likely doing?",
            "Were there any objects that appeared together frequently?",
        ],
    },
]


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    question: str
    orion_answer: str
    orion_latency_ms: float
    gemini_answer: Optional[str] = None
    gemini_verdict: str = "pending"  # correct, partial, incorrect, hallucination
    notes: str = ""


@dataclass
class ConversationResult:
    """Result of a conversation scenario."""
    scenario_name: str
    turns: List[ConversationTurn] = field(default_factory=list)
    accuracy_score: float = 0.0
    hallucination_count: int = 0
    avg_latency_ms: float = 0.0


def load_dotenv():
    """Load .env file."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


def setup_gemini():
    """Initialize Gemini API."""
    load_dotenv()
    
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("No API key found - Gemini validation disabled")
        return None
    
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except ImportError:
        logger.warning("google-genai not installed")
        return None


def run_orion_conversation(
    rag,
    questions: List[str],
) -> List[ConversationTurn]:
    """Run a conversation with Orion."""
    turns = []
    
    for question in questions:
        start_time = time.time()
        
        try:
            result = rag.query(question, use_llm=True)
            latency = (time.time() - start_time) * 1000
            
            turns.append(ConversationTurn(
                question=question,
                orion_answer=result.answer,
                orion_latency_ms=latency,
            ))
            
            logger.info(f"Q: {question[:50]}...")
            logger.info(f"A: {result.answer[:80]}... ({latency:.0f}ms)")
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            turns.append(ConversationTurn(
                question=question,
                orion_answer=f"Error: {str(e)}",
                orion_latency_ms=0,
            ))
    
    return turns


def validate_conversation_with_gemini(
    client,
    video_path: Path,
    conversation: List[ConversationTurn],
    model_name: str = "gemini-2.5-flash",
) -> List[ConversationTurn]:
    """
    Send conversation + video to Gemini for validation.
    
    Returns updated conversation turns with Gemini verdicts.
    """
    if client is None:
        return conversation
    
    from google.genai import types
    
    logger.info("Uploading video to Gemini...")
    
    try:
        video_file = client.files.upload(file=video_path)
        
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)
        
        if video_file.state.name != "ACTIVE":
            logger.error(f"Video processing failed: {video_file.state.name}")
            return conversation
        
        logger.info("Video ready, validating answers...")
        
        # Format Q&A for validation
        qa_pairs = []
        for turn in conversation:
            qa_pairs.append({
                "question": turn.question,
                "orion_answer": turn.orion_answer,
            })
        
        prompt = f"""Watch this video carefully, then evaluate these Q&A pairs.

For each question, I'll show you the question and Orion's answer.
Rate each answer based on the actual video content.

Q&A Pairs to Evaluate:
{json.dumps(qa_pairs, indent=2)}

For each Q&A pair, respond with:
- verdict: CORRECT (answer matches video), PARTIAL (partly correct), INCORRECT (wrong), HALLUCINATION (mentions things not in video)
- your_answer: What you would answer based on the video
- notes: Brief explanation of your rating

Respond in JSON format:
{{
    "evaluations": [
        {{
            "question": "...",
            "verdict": "CORRECT|PARTIAL|INCORRECT|HALLUCINATION",
            "your_answer": "...",
            "notes": "..."
        }}
    ]
}}
"""
        
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(file_uri=video_file.uri, mime_type=video_file.mime_type),
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ],
        )
        
        # Parse response
        text = response.text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        
        result = json.loads(text)
        evaluations = result.get("evaluations", [])
        
        # Update conversation turns
        for i, eval_item in enumerate(evaluations):
            if i < len(conversation):
                conversation[i].gemini_answer = eval_item.get("your_answer", "")
                conversation[i].gemini_verdict = eval_item.get("verdict", "unknown")
                conversation[i].notes = eval_item.get("notes", "")
        
        return conversation
        
    except Exception as e:
        logger.error(f"Gemini validation failed: {e}")
        return conversation


def run_scenario(
    rag,
    gemini_client,
    video_path: Path,
    scenario: Dict,
) -> ConversationResult:
    """Run a single conversation scenario."""
    logger.info(f"\n{'='*60}")
    logger.info(f"SCENARIO: {scenario['name']}")
    logger.info(f"{'='*60}")
    
    # Run Orion conversation
    turns = run_orion_conversation(rag, scenario["questions"])
    
    # Validate with Gemini
    if gemini_client:
        turns = validate_conversation_with_gemini(
            gemini_client,
            video_path,
            turns,
        )
    
    # Calculate metrics
    total_latency = sum(t.orion_latency_ms for t in turns)
    avg_latency = total_latency / len(turns) if turns else 0
    
    verdicts = [t.gemini_verdict for t in turns]
    correct = verdicts.count("CORRECT") + verdicts.count("correct")
    partial = verdicts.count("PARTIAL") + verdicts.count("partial")
    hallucinations = verdicts.count("HALLUCINATION") + verdicts.count("hallucination")
    
    accuracy = (correct + 0.5 * partial) / len(turns) if turns else 0
    
    return ConversationResult(
        scenario_name=scenario["name"],
        turns=turns,
        accuracy_score=accuracy,
        hallucination_count=hallucinations,
        avg_latency_ms=avg_latency,
    )


def print_results(results: List[ConversationResult]):
    """Print evaluation results."""
    print("\n" + "=" * 70)
    print("CONVERSATION EVALUATION RESULTS")
    print("=" * 70)
    
    total_questions = 0
    total_correct = 0
    total_hallucinations = 0
    total_latency = 0
    
    for result in results:
        print(f"\n📋 {result.scenario_name}")
        print(f"   Accuracy: {result.accuracy_score:.1%}")
        print(f"   Hallucinations: {result.hallucination_count}")
        print(f"   Avg Latency: {result.avg_latency_ms:.0f}ms")
        
        for turn in result.turns:
            verdict_emoji = {
                "CORRECT": "✓",
                "correct": "✓",
                "PARTIAL": "◐",
                "partial": "◐",
                "INCORRECT": "✗",
                "incorrect": "✗",
                "HALLUCINATION": "⚠️",
                "hallucination": "⚠️",
            }.get(turn.gemini_verdict, "?")
            
            print(f"\n   {verdict_emoji} Q: {turn.question[:50]}...")
            print(f"      Orion: {turn.orion_answer[:60]}...")
            if turn.notes:
                print(f"      Note: {turn.notes[:60]}...")
        
        total_questions += len(result.turns)
        total_correct += sum(1 for t in result.turns if t.gemini_verdict in ["CORRECT", "correct"])
        total_hallucinations += result.hallucination_count
        total_latency += result.avg_latency_ms * len(result.turns)
    
    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"Total Questions: {total_questions}")
    print(f"Correct Answers: {total_correct} ({total_correct/total_questions:.1%})" if total_questions else "N/A")
    print(f"Hallucinations: {total_hallucinations}")
    print(f"Avg Latency: {total_latency/total_questions:.0f}ms" if total_questions else "N/A")
    
    # Verdict
    if total_hallucinations == 0 and total_correct / total_questions > 0.8:
        print("\n🏆 VERDICT: EXCELLENT - High accuracy, no hallucinations")
    elif total_hallucinations <= 2 and total_correct / total_questions > 0.6:
        print("\n👍 VERDICT: GOOD - Acceptable accuracy with minor issues")
    elif total_hallucinations <= 5:
        print("\n⚠️ VERDICT: FAIR - Needs improvement")
    else:
        print("\n❌ VERDICT: POOR - Significant hallucination issues")
    
    print("=" * 70)


def save_results(results: List[ConversationResult], output_dir: Path):
    """Save results to JSON."""
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scenarios": [
            {
                "name": r.scenario_name,
                "accuracy_score": r.accuracy_score,
                "hallucination_count": r.hallucination_count,
                "avg_latency_ms": r.avg_latency_ms,
                "turns": [
                    {
                        "question": t.question,
                        "orion_answer": t.orion_answer,
                        "gemini_answer": t.gemini_answer,
                        "verdict": t.gemini_verdict,
                        "notes": t.notes,
                        "latency_ms": t.orion_latency_ms,
                    }
                    for t in r.turns
                ],
            }
            for r in results
        ],
    }
    
    output_path = output_dir / f"conversation_eval_{int(time.time())}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Results saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Conversational Q&A Evaluation")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--episode", required=True, help="Episode ID with perception data")
    parser.add_argument("--host", default="127.0.0.1", help="Memgraph host")
    parser.add_argument("--no-gemini", action="store_true", help="Skip Gemini validation")
    parser.add_argument("--scenarios", help="JSON file with custom scenarios")
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)
    
    # Load custom scenarios if provided
    scenarios = CONVERSATION_SCENARIOS
    if args.scenarios:
        with open(args.scenarios) as f:
            scenarios = json.load(f)
    
    # Initialize RAG
    try:
        from orion.query.rag_v2 import OrionRAG
        from orion.config import ensure_results_dir
        
        results_dir = ensure_results_dir(args.episode)
        
        logger.info(f"Connecting to Memgraph at {args.host}...")
        rag = OrionRAG(host=args.host, port=7687, enable_llm=True)
        
        stats = rag.get_stats()
        if stats['entities'] == 0:
            logger.error("No entities in database! Run perception pipeline first.")
            sys.exit(1)
        
        logger.info(f"Database: {stats['entities']} entities, {stats['frames']} frames")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        sys.exit(1)
    
    # Initialize Gemini
    gemini_client = None
    if not args.no_gemini:
        gemini_client = setup_gemini()
        if gemini_client:
            logger.info("Gemini validation enabled")
        else:
            logger.warning("Gemini validation disabled")
    
    # Run scenarios
    results = []
    for scenario in scenarios:
        result = run_scenario(rag, gemini_client, video_path, scenario)
        results.append(result)
    
    # Print and save results
    print_results(results)
    save_results(results, results_dir)
    
    rag.close()


if __name__ == "__main__":
    main()
