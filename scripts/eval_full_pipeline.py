#!/usr/bin/env python3
"""
Orion V2 Full Pipeline Evaluation
==================================

Comprehensive evaluation of all 6 stages with Gemini validation.

Stages Tested:
1. Detection (YOLO-World + SemanticFilterV2)
2. Tracking (Hungarian + IoU)
3. Re-ID (V-JEPA2 embeddings)
4. Scene Graph (Spatial + Interaction relations)
5. Memory (Memgraph ingest)
6. Reasoning (Ollama Q&A)

Validation:
- Gemini video analysis for ground truth
- Automated query testing
- Response quality scoring
- Hallucination detection

Usage:
    python scripts/eval_full_pipeline.py --video data/examples/video.mp4 --episode eval_001
    
    # Skip stages that have already run
    python scripts/eval_full_pipeline.py --video data/examples/video.mp4 --episode eval_001 --skip-perception
    
    # Custom test questions
    python scripts/eval_full_pipeline.py --video data/examples/video.mp4 --episode eval_001 --questions questions.json
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Result of a single stage evaluation."""
    stage: str
    success: bool
    duration_seconds: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class QueryEvaluation:
    """Evaluation of a single query."""
    question: str
    expected_type: str
    orion_answer: str
    gemini_verdict: str  # correct, partial, incorrect, hallucination
    latency_ms: float
    evidence_count: int
    notes: str = ""


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    episode_id: str
    video_path: str
    timestamp: str
    stages: List[StageResult] = field(default_factory=list)
    perception_metrics: Dict[str, Any] = field(default_factory=dict)
    query_evaluations: List[QueryEvaluation] = field(default_factory=list)
    gemini_validation: Dict[str, Any] = field(default_factory=dict)
    overall_scores: Dict[str, float] = field(default_factory=dict)


def load_dotenv():
    """Load environment variables from .env file."""
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
        logger.warning("GOOGLE_API_KEY not found - Gemini validation disabled")
        return None
    
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return client
    except ImportError:
        logger.warning("google-genai not installed - Gemini validation disabled")
        return None


# ==============================================================================
# STAGE 1-4: Perception Pipeline
# ==============================================================================

def run_perception_pipeline(
    video_path: Path,
    episode_id: str,
    skip_if_exists: bool = True,
) -> StageResult:
    """Run full perception pipeline (Stages 1-4)."""
    logger.info("=" * 60)
    logger.info("STAGES 1-4: Perception Pipeline")
    logger.info("=" * 60)
    
    from orion.config import ensure_results_dir
    results_dir = ensure_results_dir(episode_id)
    tracks_path = results_dir / "tracks.jsonl"
    
    if skip_if_exists and tracks_path.exists():
        logger.info(f"Reusing existing tracks in {results_dir}")
        metrics: Dict[str, Any] = {"skipped": True}

        try:
            with open(tracks_path) as f:
                metrics["track_count"] = sum(1 for _ in f)
        except Exception:
            metrics["track_count"] = None

        metadata_path = results_dir / "run_metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
                metrics.update(metadata)
            except Exception:
                pass

        return StageResult(
            stage="perception",
            success=True,
            duration_seconds=0.0,
            metrics=metrics,
        )
    
    start_time = time.time()
    
    # Run the showcase pipeline with YOLO-World + SemanticFilterV2
    cmd = [
        sys.executable, "-m", "orion.cli.run_showcase",
        "--episode", episode_id,
        "--video", str(video_path),
        "--no-overlay",  # Skip overlay for speed
        # YOLO-World is default in v2
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        # Always persist logs for post-mortem debugging.
        stdout_path = results_dir / "perception_stdout.log"
        stderr_path = results_dir / "perception_stderr.log"
        try:
            stdout_path.write_text(result.stdout or "")
            stderr_path.write_text(result.stderr or "")
        except Exception as e:
            logger.warning(f"Failed to write perception logs: {e}")
        
        duration = time.time() - start_time
        success = result.returncode == 0
        
        errors = []
        if not success:
            errors.append(f"Exit code: {result.returncode}")
            errors.append(f"stdout_log: {stdout_path}")
            errors.append(f"stderr_log: {stderr_path}")
            if result.stderr:
                tail = result.stderr[-4000:]
                errors.append(f"stderr_tail:\n{tail}")
            else:
                errors.append("No stderr")
        
        # Load metrics from results
        metrics = {"duration": duration}
        metrics["perception_stdout_log"] = str(results_dir / "perception_stdout.log")
        metrics["perception_stderr_log"] = str(results_dir / "perception_stderr.log")
        if tracks_path.exists():
            with open(tracks_path) as f:
                track_count = sum(1 for _ in f)
            metrics["track_count"] = track_count
        
        metadata_path = results_dir / "run_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            metrics.update(metadata)
        
        return StageResult(
            stage="perception",
            success=success,
            duration_seconds=duration,
            metrics=metrics,
            errors=errors,
        )
        
    except subprocess.TimeoutExpired:
        return StageResult(
            stage="perception",
            success=False,
            duration_seconds=600.0,
            errors=["Timeout after 10 minutes"],
        )
    except Exception as e:
        return StageResult(
            stage="perception",
            success=False,
            duration_seconds=time.time() - start_time,
            errors=[str(e)],
        )


# ==============================================================================
# STAGE 5: Memgraph Ingest
# ==============================================================================

def run_memgraph_ingest(
    episode_id: str,
    memgraph_host: str = "127.0.0.1",
) -> StageResult:
    """Ingest perception results into Memgraph."""
    logger.info("=" * 60)
    logger.info("STAGE 5: Memgraph Ingest")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        from orion.config import ensure_results_dir
        from orion.graph.backends.exporter import export_results_to_memgraph
        
        results_dir = ensure_results_dir(episode_id)
        
        result = export_results_to_memgraph(
            results_dir=results_dir,
            host=memgraph_host,
            clear_existing=True,
        )
        
        duration = time.time() - start_time
        
        # MemgraphExportResult fields:
        #   observations_written, relations_written, entities_indexed, scene_graph_edges, cis_edges_written
        metrics: Dict[str, Any] = {
            "entities": getattr(result, "entities_indexed", None),
            "observations": getattr(result, "observations_written", None),
            "relationships": getattr(result, "relations_written", None),
            "scene_graph_edges": getattr(result, "scene_graph_edges", None),
            "cis_edges_written": getattr(result, "cis_edges_written", 0),
            "host": getattr(result, "output_host", memgraph_host),
            "port": getattr(result, "output_port", 7687),
        }

        # Best-effort frame count from graph summary (if present)
        try:
            summary_path = results_dir / "graph_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                metrics["frames"] = summary.get("total_frames")
        except Exception:
            pass

        return StageResult(
            stage="memgraph_ingest",
            success=True,
            duration_seconds=duration,
            metrics=metrics,
        )
        
    except Exception as e:
        return StageResult(
            stage="memgraph_ingest",
            success=False,
            duration_seconds=time.time() - start_time,
            errors=[str(e)],
        )


# ==============================================================================
# STAGE 6: Query Evaluation
# ==============================================================================

DEFAULT_TEST_QUESTIONS = [
    # Basic retrieval
    {"question": "What objects are in the video?", "type": "all_objects"},
    {"question": "What did the person interact with?", "type": "interactions"},
    
    # Spatial queries
    {"question": "What was near the laptop?", "type": "spatial"},
    {"question": "What objects were close to each other?", "type": "spatial"},
    
    # Temporal queries
    {"question": "What happened at 10 seconds?", "type": "temporal"},
    {"question": "When did objects first appear?", "type": "temporal"},
    
    # Object-specific
    {"question": "Where did the book appear?", "type": "location"},
    {"question": "How long was the person visible?", "type": "duration"},
    
    # Reasoning (requires LLM)
    {"question": "Describe the main activities in this video", "type": "reasoning"},
    {"question": "What is the person doing with the objects?", "type": "reasoning"},
]


def run_query_evaluation(
    episode_id: str,
    test_questions: Optional[List[Dict]] = None,
    memgraph_host: str = "127.0.0.1",
    use_llm: bool = True,
) -> StageResult:
    """Evaluate Stage 6 query capabilities."""
    logger.info("=" * 60)
    logger.info("STAGE 6: Query Evaluation")
    logger.info("=" * 60)
    
    start_time = time.time()
    questions = test_questions or DEFAULT_TEST_QUESTIONS
    
    try:
        from orion.query.rag_v2 import OrionRAG
        
        rag = OrionRAG(
            host=memgraph_host,
            port=7687,
            enable_llm=use_llm,
        )
        
        # Get stats
        stats = rag.get_stats()
        logger.info(f"Database: {stats['entities']} entities, {stats['frames']} frames")
        
        results = []
        total_latency = 0.0
        
        for q in questions:
            question = q["question"]
            expected_type = q["type"]
            
            logger.info(f"Testing: {question[:50]}...")
            
            result = rag.query(question)
            
            results.append({
                "question": question,
                "expected_type": expected_type,
                "actual_type": result.query_type,
                "answer": result.answer,
                "confidence": result.confidence,
                "latency_ms": result.latency_ms,
                "evidence_count": len(result.evidence),
                "llm_used": result.llm_used,
            })
            
            total_latency += result.latency_ms
            
            logger.info(f"  → {result.query_type}: {result.confidence:.2f} ({result.latency_ms:.0f}ms)")
        
        rag.close()
        
        # Calculate metrics
        avg_latency = total_latency / len(results) if results else 0
        avg_confidence = sum(r["confidence"] for r in results) / len(results) if results else 0
        llm_rate = sum(1 for r in results if r["llm_used"]) / len(results) if results else 0
        
        return StageResult(
            stage="query_evaluation",
            success=True,
            duration_seconds=time.time() - start_time,
            metrics={
                "questions_tested": len(results),
                "avg_latency_ms": avg_latency,
                "avg_confidence": avg_confidence,
                "llm_usage_rate": llm_rate,
                "results": results,
                "db_stats": stats,
            },
        )
        
    except Exception as e:
        logger.exception("Query evaluation failed")
        return StageResult(
            stage="query_evaluation",
            success=False,
            duration_seconds=time.time() - start_time,
            errors=[str(e)],
        )


# ==============================================================================
# GEMINI VALIDATION
# ==============================================================================

def validate_with_gemini(
    client,
    video_path: Path,
    perception_result: StageResult,
    query_result: StageResult,
    model_name: str = "gemini-3-flash-preview",
) -> Dict[str, Any]:
    """
    Comprehensive Gemini validation of Orion results.
    
    Sends:
    1. Full video
    2. Perception metrics
    3. Query answers
    
    Returns:
    - Ground truth objects
    - Precision/recall estimates
    - Query answer verification
    """
    if client is None:
        return {"error": "Gemini client not available"}
    
    from google.genai import types
    
    logger.info("=" * 60)
    logger.info("GEMINI VALIDATION")
    logger.info("=" * 60)
    
    # Prepare perception summary
    track_summary = perception_result.metrics.get("track_count", "unknown")
    
    # Prepare query answers for validation
    query_answers = []
    if query_result.success:
        for r in query_result.metrics.get("results", []):
            query_answers.append({
                "question": r["question"],
                "answer": r["answer"][:500],  # Truncate long answers
            })
    
    # Upload video
    logger.info(f"Uploading video: {video_path.name}")
    start_time = time.time()
    
    try:
        video_file = client.files.upload(file=video_path)
        
        # Wait for processing
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)
        
        if video_file.state.name != "ACTIVE":
            return {"error": f"Video processing failed: {video_file.state.name}"}
        
        logger.info(f"Video ready ({time.time() - start_time:.1f}s)")
        
        # Build comprehensive prompt
        prompt = f"""Analyze this video and evaluate our Orion video understanding system.

## PART 1: Perception Evaluation
Orion detected approximately {track_summary} object instances.

Tasks:
1. List all MAIN objects visible in the video
2. Identify any objects Orion likely MISSED (false negatives)
3. Note any objects that seem incorrectly detected (false positives)

## PART 2: Query Answer Evaluation
Orion answered these questions about the video. Rate each answer:

{json.dumps(query_answers, indent=2)}

For each answer, rate as:
- CORRECT: Answer accurately reflects video content
- PARTIAL: Partially correct but missing key details
- INCORRECT: Answer doesn't match video content
- HALLUCINATION: Answer mentions things NOT in the video

## OUTPUT FORMAT (JSON):
{{
    "video_summary": "<brief description>",
    "video_duration_seconds": <number>,
    
    "ground_truth_objects": ["list of main objects"],
    "likely_false_positives": ["objects detected but not present"],
    "likely_false_negatives": ["objects present but not detected"],
    
    "perception_precision_estimate": <0.0-1.0>,
    "perception_recall_estimate": <0.0-1.0>,
    
    "query_evaluations": [
        {{
            "question": "...",
            "verdict": "CORRECT|PARTIAL|INCORRECT|HALLUCINATION",
            "notes": "explanation"
        }}
    ],
    
    "query_accuracy_score": <0.0-1.0>,
    "hallucination_count": <number>,
    
    "overall_verdict": "EXCELLENT|GOOD|FAIR|POOR",
    "key_strengths": ["list"],
    "key_weaknesses": ["list"],
    "recommendations": ["list"]
}}
"""
        
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(
                            file_uri=video_file.uri,
                            mime_type=video_file.mime_type
                        ),
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ],
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Gemini analysis complete ({elapsed:.1f}s)")
        
        # Parse response
        text = response.text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        
        try:
            result = json.loads(text)
            result["processing_time_seconds"] = elapsed
            return result
        except json.JSONDecodeError:
            return {
                "raw_response": text,
                "processing_time_seconds": elapsed,
            }
            
    except Exception as e:
        logger.exception("Gemini validation failed")
        return {"error": str(e)}


# ==============================================================================
# REPORT GENERATION
# ==============================================================================

def generate_report(
    episode_id: str,
    video_path: Path,
    stage_results: List[StageResult],
    gemini_validation: Dict[str, Any],
) -> EvaluationReport:
    """Generate comprehensive evaluation report."""
    
    report = EvaluationReport(
        episode_id=episode_id,
        video_path=str(video_path),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        stages=stage_results,
        gemini_validation=gemini_validation,
    )
    
    # Calculate overall scores
    perception_result = next((s for s in stage_results if s.stage == "perception"), None)
    query_result = next((s for s in stage_results if s.stage == "query_evaluation"), None)
    
    if perception_result and perception_result.success:
        report.perception_metrics = perception_result.metrics
    
    if query_result and query_result.success:
        report.overall_scores["avg_query_latency_ms"] = query_result.metrics.get("avg_latency_ms", 0)
        report.overall_scores["avg_confidence"] = query_result.metrics.get("avg_confidence", 0)
    
    if gemini_validation and "error" not in gemini_validation:
        report.overall_scores["perception_precision"] = gemini_validation.get("perception_precision_estimate", 0)
        report.overall_scores["perception_recall"] = gemini_validation.get("perception_recall_estimate", 0)
        report.overall_scores["query_accuracy"] = gemini_validation.get("query_accuracy_score", 0)
        report.overall_scores["hallucination_count"] = gemini_validation.get("hallucination_count", 0)
    
    # Pipeline success rate
    successful_stages = sum(1 for s in stage_results if s.success)
    report.overall_scores["pipeline_success_rate"] = successful_stages / len(stage_results) if stage_results else 0
    
    return report


def print_report(report: EvaluationReport):
    """Print formatted evaluation report."""
    print("\n" + "=" * 70)
    print("ORION V2 EVALUATION REPORT")
    print("=" * 70)
    
    print(f"\nEpisode: {report.episode_id}")
    print(f"Video: {report.video_path}")
    print(f"Timestamp: {report.timestamp}")
    
    print("\n" + "-" * 70)
    print("STAGE RESULTS")
    print("-" * 70)
    
    for stage in report.stages:
        status = "✓" if stage.success else "✗"
        print(f"\n{status} {stage.stage.upper()}")
        print(f"  Duration: {stage.duration_seconds:.1f}s")
        if stage.errors:
            for err in stage.errors[:2]:
                print(f"  Error: {err[:80]}")
        if stage.metrics:
            for k, v in stage.metrics.items():
                if k != "results" and not isinstance(v, (list, dict)):
                    print(f"  {k}: {v}")
    
    print("\n" + "-" * 70)
    print("OVERALL SCORES")
    print("-" * 70)
    
    for k, v in report.overall_scores.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    
    if report.gemini_validation and "error" not in report.gemini_validation:
        print("\n" + "-" * 70)
        print("GEMINI VALIDATION")
        print("-" * 70)
        
        gv = report.gemini_validation
        print(f"\n  Video: {gv.get('video_summary', 'N/A')}")
        print(f"  Verdict: {gv.get('overall_verdict', 'N/A')}")
        
        print("\n  Ground Truth Objects:")
        for obj in gv.get("ground_truth_objects", [])[:10]:
            print(f"    • {obj}")
        
        if gv.get("likely_false_positives"):
            print("\n  ⚠️ False Positives:")
            for obj in gv.get("likely_false_positives", []):
                print(f"    • {obj}")
        
        if gv.get("likely_false_negatives"):
            print("\n  ⚠️ Missed Objects:")
            for obj in gv.get("likely_false_negatives", []):
                print(f"    • {obj}")
        
        if gv.get("recommendations"):
            print("\n  💡 Recommendations:")
            for rec in gv.get("recommendations", []):
                print(f"    → {rec}")
    
    print("\n" + "=" * 70)


def save_report(report: EvaluationReport, output_dir: Path):
    """Save report to JSON file."""
    output_path = output_dir / f"eval_report_{int(time.time())}.json"
    
    # Convert dataclass to dict
    report_dict = {
        "episode_id": report.episode_id,
        "video_path": report.video_path,
        "timestamp": report.timestamp,
        "stages": [
            {
                "stage": s.stage,
                "success": s.success,
                "duration_seconds": s.duration_seconds,
                "metrics": s.metrics,
                "errors": s.errors,
            }
            for s in report.stages
        ],
        "perception_metrics": report.perception_metrics,
        "gemini_validation": report.gemini_validation,
        "overall_scores": report.overall_scores,
    }
    
    with open(output_path, "w") as f:
        json.dump(report_dict, f, indent=2, default=str)
    
    logger.info(f"Report saved to: {output_path}")
    return output_path


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Orion V2 Full Pipeline Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--episode", required=True, help="Episode ID for results")
    
    # Stage control
    parser.add_argument("--skip-perception", action="store_true", help="Skip perception (use existing)")
    parser.add_argument("--skip-memgraph", action="store_true", help="Skip Memgraph ingest")
    parser.add_argument("--skip-queries", action="store_true", help="Skip query evaluation")
    parser.add_argument("--skip-gemini", action="store_true", help="Skip Gemini validation")
    
    # Options
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM for queries")
    parser.add_argument("--questions", help="JSON file with custom test questions")
    parser.add_argument("--memgraph-host", default="127.0.0.1", help="Memgraph host")
    parser.add_argument("--gemini-model", default="gemini-3-flash-preview", help="Gemini model")
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)
    
    # Load custom questions if provided
    test_questions = None
    if args.questions:
        with open(args.questions) as f:
            test_questions = json.load(f)
    
    stage_results = []
    
    # Stage 1-4: Perception
    if not args.skip_perception:
        result = run_perception_pipeline(
            video_path=video_path,
            episode_id=args.episode,
            skip_if_exists=False,
        )
        stage_results.append(result)
        if not result.success:
            logger.error("Perception pipeline failed!")
    else:
        result = run_perception_pipeline(
            video_path=video_path,
            episode_id=args.episode,
            skip_if_exists=True,
        )
        stage_results.append(result)
    
    # Stage 5: Memgraph
    if not args.skip_memgraph:
        result = run_memgraph_ingest(
            episode_id=args.episode,
            memgraph_host=args.memgraph_host,
        )
        stage_results.append(result)
        if not result.success:
            logger.warning("Memgraph ingest failed - queries may not work")
    
    # Stage 6: Queries
    query_result = None
    if not args.skip_queries:
        query_result = run_query_evaluation(
            episode_id=args.episode,
            test_questions=test_questions,
            memgraph_host=args.memgraph_host,
            use_llm=not args.no_llm,
        )
        stage_results.append(query_result)
    
    # Gemini Validation
    gemini_validation = {}
    if not args.skip_gemini:
        gemini_client = setup_gemini()
        if gemini_client:
            perception_result = next((s for s in stage_results if s.stage == "perception"), None)
            gemini_validation = validate_with_gemini(
                client=gemini_client,
                video_path=video_path,
                perception_result=perception_result or StageResult("perception", False, 0),
                query_result=query_result or StageResult("query_evaluation", False, 0),
                model_name=args.gemini_model,
            )
    
    # Generate report
    report = generate_report(
        episode_id=args.episode,
        video_path=video_path,
        stage_results=stage_results,
        gemini_validation=gemini_validation,
    )
    
    # Print and save
    print_report(report)
    
    from orion.config import ensure_results_dir
    output_dir = ensure_results_dir(args.episode)
    save_report(report, output_dir)


if __name__ == "__main__":
    main()
