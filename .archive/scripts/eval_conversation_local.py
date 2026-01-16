#!/usr/bin/env python3
"""
Conversational Q&A Evaluation with Local VLM (Qwen3-VL)
========================================================

Tests Orion's Stage 6 reasoning capabilities through a series of 
conversational queries, validated by Qwen3-VL running locally via Ollama.

This is fully local - no API keys needed!

Key differences from Gemini version:
- Uses Qwen3-VL:8b locally (6.1GB, fits on 24GB A10)
- Processes video frames instead of uploading video
- Faster iteration, no rate limits
- Works offline

Usage:
    python scripts/eval_conversation_local.py \
        --video data/examples/video.mp4 \
        --episode stage6_eval
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator, Tuple

import cv2
import numpy as np
from PIL import Image

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
    vlm_answer: Optional[str] = None
    vlm_verdict: str = "pending"  # correct, partial, incorrect, hallucination
    notes: str = ""


@dataclass
class ConversationResult:
    """Result of a conversation scenario."""
    scenario_name: str
    turns: List[ConversationTurn] = field(default_factory=list)
    accuracy_score: float = 0.0
    hallucination_count: int = 0
    avg_latency_ms: float = 0.0


class LocalVLMValidator:
    """
    Validates answers using Qwen3-VL locally via Ollama.
    
    Instead of uploading video, we:
    1. Extract key frames from video
    2. Send frames + questions to Qwen3-VL
    3. Get ground truth answers for comparison
    """
    
    def __init__(
        self,
        model: str = "qwen3-vl:8b",
        base_url: str = "http://localhost:11434",
        num_frames: int = 8,  # Number of frames to sample
    ):
        self.model = model
        self.base_url = base_url
        self.num_frames = num_frames
        
        try:
            import httpx
            self.client = httpx.Client(base_url=base_url, timeout=120.0)
        except ImportError:
            raise ImportError("httpx required: pip install httpx")
        
        # Verify model is available
        self._verify_model()
    
    def _verify_model(self) -> bool:
        """Check if model is available."""
        try:
            response = self.client.get("/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                available = [m["name"] for m in models]
                
                if self.model in available or any(self.model.split(":")[0] in m for m in available):
                    logger.info(f"✓ VLM model {self.model} available")
                    return True
                else:
                    logger.warning(f"Model {self.model} not found. Available: {available}")
                    logger.info(f"Pull with: ollama pull {self.model}")
                    return False
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
    
    def extract_frames(self, video_path: Path) -> List[np.ndarray]:
        """Extract evenly-spaced frames from video."""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError(f"Could not read video: {video_path}")
        
        # Sample frames evenly
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize to reasonable size for VLM
                h, w = frame.shape[:2]
                max_dim = 512
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                frames.append(frame)
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from video")
        return frames
    
    def frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert OpenCV frame to base64 string."""
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        
        # Compress to JPEG
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode()
    
    def query_vlm(
        self,
        question: str,
        frames: List[np.ndarray],
        context: str = "",
    ) -> str:
        """
        Query Qwen3-VL with frames + question.
        
        Uses Ollama's multimodal API.
        """
        # Build image list
        images = [self.frame_to_base64(f) for f in frames]
        
        prompt = f"""You are analyzing a video through {len(images)} sampled frames.
The frames are shown in temporal order from start to end.

{f"Context: {context}" if context else ""}

Question: {question}

Analyze the frames carefully and provide a detailed, accurate answer based only on what you can see.
"""
        
        try:
            response = self.client.post(
                "/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": images,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 512,
                    },
                },
                timeout=120.0,
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"VLM request failed: {response.text}")
                return f"Error: {response.status_code}"
                
        except Exception as e:
            logger.error(f"VLM query failed: {e}")
            return f"Error: {str(e)}"
    
    def validate_answer(
        self,
        question: str,
        orion_answer: str,
        frames: List[np.ndarray],
    ) -> Tuple[str, str, str]:
        """
        Validate Orion's answer against VLM ground truth.
        
        Returns: (vlm_answer, verdict, notes)
        """
        # First, get VLM's own answer
        vlm_answer = self.query_vlm(question, frames)
        
        # Then ask VLM to compare
        comparison_prompt = f"""Compare these two answers about a video:

QUESTION: {question}

ANSWER A (System being tested): {orion_answer}

ANSWER B (Your answer): {vlm_answer}

Rate Answer A based on accuracy:
- CORRECT: Answer A is accurate and matches what's in the video
- PARTIAL: Answer A is partly correct but missing details or has minor errors  
- INCORRECT: Answer A is wrong about what's in the video
- HALLUCINATION: Answer A mentions things that are NOT in the video

Respond in this exact JSON format:
{{"verdict": "CORRECT|PARTIAL|INCORRECT|HALLUCINATION", "notes": "brief explanation"}}
"""
        
        try:
            response = self.client.post(
                "/api/generate",
                json={
                    "model": self.model,
                    "prompt": comparison_prompt,
                    "images": [self.frame_to_base64(frames[len(frames)//2])],  # Middle frame for reference
                    "stream": False,
                    "options": {
                        "temperature": 0.0,  # Deterministic comparison
                        "num_predict": 128,
                    },
                },
                timeout=60.0,
            )
            
            if response.status_code == 200:
                result_text = response.json().get("response", "")
                
                # Parse JSON from response
                try:
                    # Find JSON in response
                    start = result_text.find("{")
                    end = result_text.rfind("}") + 1
                    if start >= 0 and end > start:
                        result = json.loads(result_text[start:end])
                        return (
                            vlm_answer,
                            result.get("verdict", "unknown"),
                            result.get("notes", ""),
                        )
                except json.JSONDecodeError:
                    pass
                
                # Fallback: look for verdict keywords
                result_lower = result_text.lower()
                if "hallucination" in result_lower:
                    verdict = "HALLUCINATION"
                elif "incorrect" in result_lower:
                    verdict = "INCORRECT"
                elif "partial" in result_lower:
                    verdict = "PARTIAL"
                elif "correct" in result_lower:
                    verdict = "CORRECT"
                else:
                    verdict = "unknown"
                
                return vlm_answer, verdict, result_text[:100]
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
        
        return vlm_answer, "unknown", "Validation error"
    
    def close(self):
        """Close HTTP client."""
        self.client.close()


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


def validate_with_local_vlm(
    validator: LocalVLMValidator,
    frames: List[np.ndarray],
    conversation: List[ConversationTurn],
) -> List[ConversationTurn]:
    """Validate conversation using local VLM."""
    logger.info(f"Validating {len(conversation)} turns with local VLM...")
    
    for i, turn in enumerate(conversation):
        logger.info(f"  Validating Q{i+1}: {turn.question[:40]}...")
        
        vlm_answer, verdict, notes = validator.validate_answer(
            turn.question,
            turn.orion_answer,
            frames,
        )
        
        turn.vlm_answer = vlm_answer
        turn.vlm_verdict = verdict
        turn.notes = notes
        
        logger.info(f"    Verdict: {verdict}")
    
    return conversation


def run_scenario(
    rag,
    validator: Optional[LocalVLMValidator],
    frames: List[np.ndarray],
    scenario: Dict,
) -> ConversationResult:
    """Run a single conversation scenario."""
    logger.info(f"\n{'='*60}")
    logger.info(f"SCENARIO: {scenario['name']}")
    logger.info(f"{'='*60}")
    
    # Run Orion conversation
    turns = run_orion_conversation(rag, scenario["questions"])
    
    # Validate with local VLM
    if validator:
        turns = validate_with_local_vlm(validator, frames, turns)
    
    # Calculate metrics
    total_latency = sum(t.orion_latency_ms for t in turns)
    avg_latency = total_latency / len(turns) if turns else 0
    
    verdicts = [t.vlm_verdict.upper() for t in turns]
    correct = verdicts.count("CORRECT")
    partial = verdicts.count("PARTIAL")
    hallucinations = verdicts.count("HALLUCINATION")
    
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
    print("CONVERSATION EVALUATION RESULTS (Local VLM Validation)")
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
                "PARTIAL": "◐",
                "INCORRECT": "✗",
                "HALLUCINATION": "⚠️",
            }.get(turn.vlm_verdict.upper(), "?")
            
            print(f"\n   {verdict_emoji} Q: {turn.question[:50]}...")
            print(f"      Orion: {turn.orion_answer[:60]}...")
            if turn.notes:
                print(f"      Note: {turn.notes[:60]}...")
        
        total_questions += len(result.turns)
        total_correct += sum(1 for t in result.turns if t.vlm_verdict.upper() == "CORRECT")
        total_hallucinations += result.hallucination_count
        total_latency += result.avg_latency_ms * len(result.turns)
    
    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"Total Questions: {total_questions}")
    
    if total_questions > 0:
        accuracy = total_correct / total_questions
        avg_lat = total_latency / total_questions
        print(f"Correct Answers: {total_correct} ({accuracy:.1%})")
        print(f"Hallucinations: {total_hallucinations}")
        print(f"Avg Latency: {avg_lat:.0f}ms")
        
        # Verdict
        if total_hallucinations == 0 and accuracy > 0.8:
            print("\n🏆 VERDICT: EXCELLENT - High accuracy, no hallucinations")
        elif total_hallucinations <= 2 and accuracy > 0.6:
            print("\n👍 VERDICT: GOOD - Acceptable accuracy with minor issues")
        elif total_hallucinations <= 5:
            print("\n⚠️ VERDICT: FAIR - Needs improvement")
        else:
            print("\n❌ VERDICT: POOR - Significant hallucination issues")
    
    print("=" * 70)
    
    return {
        "total_questions": total_questions,
        "correct": total_correct,
        "hallucinations": total_hallucinations,
        "accuracy": total_correct / total_questions if total_questions else 0,
    }


def save_results(results: List[ConversationResult], output_dir: Path) -> Path:
    """Save results to JSON."""
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "validator": "qwen3-vl:8b (local)",
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
                        "vlm_answer": t.vlm_answer,
                        "verdict": t.vlm_verdict,
                        "notes": t.notes,
                        "latency_ms": t.orion_latency_ms,
                    }
                    for t in r.turns
                ],
            }
            for r in results
        ],
    }
    
    output_path = output_dir / f"conversation_eval_local_{int(time.time())}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Results saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Conversational Q&A Evaluation with Local VLM (Qwen3-VL)"
    )
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--episode", required=True, help="Episode ID with perception data")
    parser.add_argument("--host", default="127.0.0.1", help="Memgraph host")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--vlm-model", default="qwen3-vl:8b", help="VLM model for validation")
    parser.add_argument("--num-frames", type=int, default=8, help="Frames to sample for VLM")
    parser.add_argument("--no-validate", action="store_true", help="Skip VLM validation")
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
        rag = OrionRAG(
            host=args.host,
            port=7687,
            enable_llm=True,
            ollama_url=args.ollama_url,
        )
        
        stats = rag.get_stats()
        if stats['entities'] == 0:
            logger.error("No entities in database! Run perception pipeline first.")
            logger.info("Run: python -m orion.cli.run_showcase --episode <name> --video <path> --memgraph")
            sys.exit(1)
        
        logger.info(f"Database: {stats['entities']} entities, {stats['frames']} frames")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG: {e}")
        sys.exit(1)
    
    # Initialize local VLM validator
    validator = None
    frames = []
    
    if not args.no_validate:
        try:
            logger.info(f"Initializing local VLM validator ({args.vlm_model})...")
            validator = LocalVLMValidator(
                model=args.vlm_model,
                base_url=args.ollama_url,
                num_frames=args.num_frames,
            )
            
            logger.info("Extracting video frames for validation...")
            frames = validator.extract_frames(video_path)
            
        except Exception as e:
            logger.warning(f"VLM validation disabled: {e}")
            validator = None
    
    # Run scenarios
    results = []
    for scenario in scenarios:
        result = run_scenario(rag, validator, frames, scenario)
        results.append(result)
    
    # Print and save results
    summary = print_results(results)
    output_path = save_results(results, results_dir)
    
    # Cleanup
    rag.close()
    if validator:
        validator.close()
    
    return summary


if __name__ == "__main__":
    main()
