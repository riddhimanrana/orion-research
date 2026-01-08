#!/usr/bin/env python3
"""
Iterative Development Loop for Orion Stage 6
=============================================

This script automates the evaluation-iteration cycle:
1. Run full pipeline on test videos
2. Evaluate with local VLM (Qwen3-VL)
3. Analyze failures and identify improvements
4. Log results and track progress over iterations

Usage:
    python scripts/iterative_eval.py --video data/examples/video.mp4
"""

import argparse
import json
import logging
import os
import subprocess
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


@dataclass
class IterationResult:
    """Result of a single evaluation iteration."""
    iteration: int
    timestamp: str
    accuracy: float
    hallucinations: int
    total_questions: int
    avg_latency_ms: float
    issues: List[str] = field(default_factory=list)
    improvements_made: List[str] = field(default_factory=list)


class IterativeEvaluator:
    """
    Runs iterative evaluation and tracks improvement over time.
    """
    
    def __init__(
        self,
        video_path: Path,
        episode_prefix: str = "iter_eval",
        memgraph_host: str = "127.0.0.1",
        ollama_url: str = "http://localhost:11434",
        vlm_model: str = "qwen3-vl:8b",
        reasoning_model: str = "qwen2.5:14b-instruct-q8_0",
    ):
        self.video_path = video_path
        self.episode_prefix = episode_prefix
        self.memgraph_host = memgraph_host
        self.ollama_url = ollama_url
        self.vlm_model = vlm_model
        self.reasoning_model = reasoning_model
        
        self.results_history: List[IterationResult] = []
        self.iteration_count = 0
        
        # Track what we've tried
        self.attempted_improvements: List[str] = []
    
    def run_perception_pipeline(self, episode: str) -> bool:
        """Run the perception pipeline with Memgraph export."""
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING PERCEPTION PIPELINE: {episode}")
        logger.info(f"{'='*60}")
        
        cmd = [
            sys.executable, "-m", "orion.cli.run_showcase",
            "--episode", episode,
            "--video", str(self.video_path),
            "--memgraph",
            "--memgraph-host", self.memgraph_host,
        ]
        
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Pipeline failed:\n{result.stderr}")
                return False
            
            logger.info("✓ Perception pipeline completed")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Pipeline timed out after 10 minutes")
            return False
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return False
    
    def run_conversation_eval(self, episode: str) -> Optional[Dict]:
        """Run conversational evaluation with local VLM."""
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING CONVERSATION EVAL: {episode}")
        logger.info(f"{'='*60}")
        
        # Import and run directly for better error handling
        try:
            from scripts.eval_conversation_local import (
                LocalVLMValidator,
                run_scenario,
                print_results,
                save_results,
                CONVERSATION_SCENARIOS,
            )
            from orion.query.rag_v2 import OrionRAG
            from orion.config import ensure_results_dir
            
            results_dir = ensure_results_dir(episode)
            
            # Initialize RAG
            rag = OrionRAG(
                host=self.memgraph_host,
                port=7687,
                enable_llm=True,
                ollama_url=self.ollama_url,
                llm_model=self.reasoning_model,
            )
            
            stats = rag.get_stats()
            logger.info(f"Database: {stats['entities']} entities, {stats['frames']} frames")
            
            if stats['entities'] == 0:
                logger.error("No entities in database!")
                rag.close()
                return None
            
            # Initialize VLM validator
            validator = LocalVLMValidator(
                model=self.vlm_model,
                base_url=self.ollama_url,
                num_frames=8,
            )
            
            frames = validator.extract_frames(self.video_path)
            
            # Run scenarios
            results = []
            for scenario in CONVERSATION_SCENARIOS:
                result = run_scenario(rag, validator, frames, scenario)
                results.append(result)
            
            # Get summary
            summary = print_results(results)
            save_results(results, results_dir)
            
            # Cleanup
            rag.close()
            validator.close()
            
            return summary
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_failures(self, eval_results: Dict) -> List[str]:
        """Analyze evaluation results and identify issues."""
        issues = []
        
        accuracy = eval_results.get("accuracy", 0)
        hallucinations = eval_results.get("hallucinations", 0)
        
        if accuracy < 0.5:
            issues.append("LOW_ACCURACY: Less than 50% correct answers")
        elif accuracy < 0.7:
            issues.append("MEDIUM_ACCURACY: 50-70% correct, needs improvement")
        
        if hallucinations > 3:
            issues.append(f"HIGH_HALLUCINATION: {hallucinations} hallucinated answers")
        elif hallucinations > 0:
            issues.append(f"SOME_HALLUCINATION: {hallucinations} hallucinated answers")
        
        return issues
    
    def suggest_improvements(self, issues: List[str]) -> List[str]:
        """Suggest improvements based on identified issues."""
        suggestions = []
        
        for issue in issues:
            if "HALLUCINATION" in issue:
                if "Increase confidence threshold" not in self.attempted_improvements:
                    suggestions.append("Increase confidence threshold for answers")
                if "Add explicit uncertainty" not in self.attempted_improvements:
                    suggestions.append("Add explicit uncertainty when evidence is weak")
            
            if "LOW_ACCURACY" in issue:
                if "Improve query routing" not in self.attempted_improvements:
                    suggestions.append("Improve query routing accuracy")
                if "Expand spatial predicates" not in self.attempted_improvements:
                    suggestions.append("Expand spatial predicates (on, under, beside)")
            
            if "MEDIUM_ACCURACY" in issue:
                if "Better temporal understanding" not in self.attempted_improvements:
                    suggestions.append("Better temporal understanding")
        
        return suggestions
    
    def run_iteration(self) -> IterationResult:
        """Run a single evaluation iteration."""
        self.iteration_count += 1
        episode = f"{self.episode_prefix}_{self.iteration_count:03d}"
        
        logger.info(f"\n{'#'*70}")
        logger.info(f"# ITERATION {self.iteration_count}")
        logger.info(f"{'#'*70}")
        
        # Step 1: Run perception pipeline
        if not self.run_perception_pipeline(episode):
            return IterationResult(
                iteration=self.iteration_count,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                accuracy=0.0,
                hallucinations=0,
                total_questions=0,
                avg_latency_ms=0,
                issues=["PIPELINE_FAILED"],
            )
        
        # Step 2: Run conversation evaluation
        eval_results = self.run_conversation_eval(episode)
        
        if eval_results is None:
            return IterationResult(
                iteration=self.iteration_count,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                accuracy=0.0,
                hallucinations=0,
                total_questions=0,
                avg_latency_ms=0,
                issues=["EVALUATION_FAILED"],
            )
        
        # Step 3: Analyze results
        issues = self.analyze_failures(eval_results)
        suggestions = self.suggest_improvements(issues)
        
        result = IterationResult(
            iteration=self.iteration_count,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            accuracy=eval_results.get("accuracy", 0),
            hallucinations=eval_results.get("hallucinations", 0),
            total_questions=eval_results.get("total_questions", 0),
            avg_latency_ms=0,  # TODO: track from eval
            issues=issues,
            improvements_made=suggestions,
        )
        
        self.results_history.append(result)
        
        return result
    
    def print_iteration_summary(self, result: IterationResult):
        """Print summary of iteration."""
        print(f"\n{'='*60}")
        print(f"ITERATION {result.iteration} SUMMARY")
        print(f"{'='*60}")
        print(f"Timestamp: {result.timestamp}")
        print(f"Accuracy: {result.accuracy:.1%}")
        print(f"Hallucinations: {result.hallucinations}")
        print(f"Total Questions: {result.total_questions}")
        
        if result.issues:
            print(f"\nIssues Identified:")
            for issue in result.issues:
                print(f"  - {issue}")
        
        if result.improvements_made:
            print(f"\nSuggested Improvements:")
            for imp in result.improvements_made:
                print(f"  → {imp}")
        
        # Compare with previous
        if len(self.results_history) > 1:
            prev = self.results_history[-2]
            delta = result.accuracy - prev.accuracy
            print(f"\nChange from previous: {delta:+.1%}")
    
    def print_progress_report(self):
        """Print overall progress across iterations."""
        print(f"\n{'='*70}")
        print("PROGRESS REPORT")
        print(f"{'='*70}")
        
        if not self.results_history:
            print("No iterations completed yet.")
            return
        
        print(f"{'Iter':<6} {'Accuracy':<10} {'Halluc':<8} {'Issues':<30}")
        print("-" * 60)
        
        for r in self.results_history:
            issues_str = ", ".join(r.issues[:2]) if r.issues else "None"
            print(f"{r.iteration:<6} {r.accuracy:<10.1%} {r.hallucinations:<8} {issues_str:<30}")
        
        # Best result
        best = max(self.results_history, key=lambda x: x.accuracy)
        print(f"\nBest iteration: {best.iteration} (accuracy: {best.accuracy:.1%})")
        
        # Trend
        if len(self.results_history) >= 2:
            first_acc = self.results_history[0].accuracy
            last_acc = self.results_history[-1].accuracy
            trend = "↑ Improving" if last_acc > first_acc else "↓ Declining" if last_acc < first_acc else "→ Stable"
            print(f"Trend: {trend} ({first_acc:.1%} → {last_acc:.1%})")
    
    def save_progress(self, output_path: Path):
        """Save progress to JSON."""
        data = {
            "video": str(self.video_path),
            "total_iterations": self.iteration_count,
            "attempted_improvements": self.attempted_improvements,
            "iterations": [
                {
                    "iteration": r.iteration,
                    "timestamp": r.timestamp,
                    "accuracy": r.accuracy,
                    "hallucinations": r.hallucinations,
                    "total_questions": r.total_questions,
                    "issues": r.issues,
                    "improvements_made": r.improvements_made,
                }
                for r in self.results_history
            ],
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Progress saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Iterative Evaluation Loop")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations")
    parser.add_argument("--episode-prefix", default="iter_eval", help="Episode name prefix")
    parser.add_argument("--host", default="127.0.0.1", help="Memgraph host")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--vlm-model", default="qwen3-vl:8b", help="VLM model")
    parser.add_argument("--reasoning-model", default="qwen2.5:14b-instruct-q8_0", help="Reasoning model")
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)
    
    evaluator = IterativeEvaluator(
        video_path=video_path,
        episode_prefix=args.episode_prefix,
        memgraph_host=args.host,
        ollama_url=args.ollama_url,
        vlm_model=args.vlm_model,
        reasoning_model=args.reasoning_model,
    )
    
    # Run iterations
    for i in range(args.iterations):
        result = evaluator.run_iteration()
        evaluator.print_iteration_summary(result)
        
        # Check for good enough result
        if result.accuracy >= 0.9 and result.hallucinations == 0:
            logger.info("\n🏆 Excellent results achieved! Stopping early.")
            break
        
        # Pause between iterations if doing multiple
        if i < args.iterations - 1:
            logger.info("\nPausing before next iteration...")
            time.sleep(2)
    
    # Final report
    evaluator.print_progress_report()
    
    # Save progress
    progress_path = Path(f"results/{args.episode_prefix}_progress.json")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator.save_progress(progress_path)


if __name__ == "__main__":
    main()
