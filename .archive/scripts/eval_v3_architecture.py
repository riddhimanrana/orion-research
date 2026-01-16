#!/usr/bin/env python3
"""
Orion v3 Architecture Evaluation with Gemini Validation
=========================================================

Tests the Neural Cypher RAG and validates against Gemini-3-Flash-Preview
with full video input for ground truth.

Key improvements over v2:
1. Neural Cypher RAG: LLM generates Cypher queries dynamically
2. No redundant VLM validation loop (FastVLM for perception, Qwen for reasoning only)
3. Gemini ground truth for accuracy measurement

Usage:
    python scripts/eval_v3_architecture.py \
        --video data/examples/video.mp4 \
        --episode eval_v3_test
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


DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"


# Test questions covering different query types
TEST_QUESTIONS = [
    # Basic discovery
    {"q": "What objects were detected in this video?", "type": "discovery"},
    {"q": "How many different types of objects are there?", "type": "discovery"},
    
    # Spatial reasoning
    {"q": "What objects were near the person?", "type": "spatial"},
    {"q": "Describe the spatial relationships between objects.", "type": "spatial"},
    
    # Temporal understanding
    {"q": "What happened at the beginning of the video?", "type": "temporal"},
    {"q": "When did the book first appear?", "type": "temporal"},
    {"q": "What objects were visible around 30 seconds?", "type": "temporal"},
    
    # Interaction analysis
    {"q": "What did the person interact with?", "type": "interaction"},
    {"q": "Were any objects being held or moved?", "type": "interaction"},
    
    # Complex reasoning
    {"q": "Based on the objects present, what room is this likely?", "type": "reasoning"},
    {"q": "What activities were happening in this video?", "type": "reasoning"},
    {"q": "Which objects appeared together most frequently?", "type": "reasoning"},
]


@dataclass
class QueryResult:
    """Result of a single query evaluation."""
    question: str
    query_type: str
    orion_answer: str
    orion_latency_ms: float
    cypher_used: Optional[str] = None
    evidence_count: int = 0
    neural_cypher_used: bool = False
    gemini_verdict: str = "pending"
    gemini_notes: str = ""


@dataclass
class EvaluationReport:
    """Full evaluation report."""
    video_path: str
    episode_id: str
    timestamp: str
    gemini_model: str = "disabled"
    total_questions: int = 0
    correct_count: int = 0
    partial_count: int = 0
    incorrect_count: int = 0
    avg_latency_ms: float = 0.0
    neural_cypher_success_rate: float = 0.0
    top_detected_classes: List[Dict[str, Any]] = field(default_factory=list)
    class_audit: Optional[Dict[str, Any]] = None
    results: List[QueryResult] = field(default_factory=list)

def setup_gemini(model_name: str) -> Optional[Any]:
    """Initialize Gemini API for validation."""
    try:
        from orion.utils.gemini_client import get_gemini_model
        model = get_gemini_model(model_name)
        logger.info("✓ Gemini validation enabled (%s)", model_name)
        return model
    except Exception as e:
        logger.warning("Gemini setup failed (%s): %s", model_name, e)
        return None


def _guess_video_mime(video_path: Path) -> str:
    ext = video_path.suffix.lower()
    if ext == ".mp4":
        return "video/mp4"
    if ext == ".mov":
        return "video/quicktime"
    return "application/octet-stream"


def prepare_video_payload(video_path: Path) -> Dict[str, Any]:
    """Prepare a reusable video payload for Gemini (sent with each request)."""
    return {
        "mime_type": _guess_video_mime(video_path),
        "data": video_path.read_bytes(),
    }


def validate_with_gemini(
    gemini_model,
    video_payload: Dict[str, Any],
    question: str,
    orion_answer: str,
) -> tuple:
    """
    Validate Orion's answer against Gemini with full video context.
    
    Returns: (verdict, notes)
    """
    if gemini_model is None:
        return "skipped", "Gemini not available"
    
    try:
        prompt = f"""Watch this video carefully and evaluate the following answer.

QUESTION: {question}

SYSTEM ANSWER: {orion_answer}

Evaluate the answer based on what you can see in the video:
- CORRECT: The answer accurately describes what's in the video
- PARTIAL: The answer is partially correct but missing details
- INCORRECT: The answer is wrong or describes things not in the video
- HALLUCINATION: The answer invents objects or events not present

Respond with ONLY a JSON object:
{{"verdict": "CORRECT|PARTIAL|INCORRECT|HALLUCINATION", "notes": "brief explanation"}}
"""
        
        response = gemini_model.generate_content([video_payload, prompt])
        result_text = response.text.strip()
        
        # Parse JSON
        try:
            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(result_text[start:end])
                return result.get("verdict", "unknown"), result.get("notes", "")
        except json.JSONDecodeError:
            pass
        
        # Fallback parsing
        result_lower = result_text.lower()
        if "hallucination" in result_lower:
            return "HALLUCINATION", result_text[:100]
        elif "incorrect" in result_lower:
            return "INCORRECT", result_text[:100]
        elif "partial" in result_lower:
            return "PARTIAL", result_text[:100]
        elif "correct" in result_lower:
            return "CORRECT", result_text[:100]
        
        return "unknown", result_text[:100]
        
    except Exception as e:
        logger.error(f"Gemini validation failed: {e}")
        return "error", str(e)


def fetch_top_detected_classes(rag, limit: int = 20) -> List[Dict[str, Any]]:
    """Read top detected classes from Memgraph to audit semantic filtering quality."""
    cypher = f"""
        MATCH (e:Entity)-[r:OBSERVED_IN]->(f:Frame)
        RETURN e.class_name AS class_name, count(r) AS observations
        ORDER BY observations DESC
        LIMIT {int(limit)}
    """
    cur = rag.backend.connection.cursor()
    cur.execute(cypher)
    rows = cur.fetchall() or []
    out: List[Dict[str, Any]] = []
    for row in rows:
        # Memgraph cursor may return tuples
        if isinstance(row, dict):
            out.append(row)
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            out.append({"class_name": row[0], "observations": row[1]})
        else:
            out.append({"row": row})
    return out


def audit_classes_with_gemini(
    gemini_model,
    video_payload: Dict[str, Any],
    top_classes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Ask Gemini to confirm whether top detected classes truly appear in the video."""
    if gemini_model is None:
        return {"verdict": "skipped", "notes": "Gemini not available"}

    class_list = [str(it.get("class_name")) for it in top_classes if it.get("class_name")]
    class_list = class_list[:20]

    prompt = f"""Watch this video and verify whether each object class truly appears at any time.

Classes to verify (from an object detector + tracker):
{json.dumps(class_list)}

Return ONLY JSON in this exact schema:
{{
  "present": ["class", ...],
  "absent": ["class", ...],
  "unsure": ["class", ...],
  "notes": "brief overall notes"
}}
"""

    try:
        resp = gemini_model.generate_content([video_payload, prompt])
        text = (resp.text or "").strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return {"present": [], "absent": [], "unsure": [], "notes": text[:200]}
    except Exception as e:
        return {"present": [], "absent": [], "unsure": [], "notes": f"error: {e}"}


def maybe_run_showcase(
    *,
    episode_id: str,
    video_path: Path,
    host: str,
    port: int,
    fps: float,
    yolo_model: str,
    device: str,
    memgraph_clear: bool,
) -> None:
    """Run the end-to-end pipeline to populate results + Memgraph for evaluation."""
    cmd = [
        sys.executable,
        "-m",
        "orion.cli.run_showcase",
        "--episode",
        episode_id,
        "--video",
        str(video_path),
        "--fps",
        str(fps),
        "--yolo-model",
        str(yolo_model),
        "--device",
        str(device),
        "--no-overlay",
        "--memgraph",
        "--memgraph-host",
        str(host),
        "--memgraph-port",
        str(port),
    ]
    if memgraph_clear:
        cmd.append("--memgraph-clear")

    logger.info("[Showcase] Running end-to-end pipeline to populate Memgraph...")
    subprocess.run(cmd, check=True)


def run_evaluation(
    video_path: Path,
    episode_id: str,
    host: str = "127.0.0.1",
    port: int = 7687,
    use_gemini: bool = True,
    gemini_model_name: str = DEFAULT_GEMINI_MODEL,
    run_showcase_first: bool = False,
    showcase_fps: float = 4.0,
    showcase_yolo_model: str = "yolo11m",
    showcase_device: str = "mps",
    showcase_memgraph_clear: bool = False,
) -> EvaluationReport:
    """Run full evaluation of Orion v3 architecture."""
    
    from orion.query.rag_v2 import OrionRAG
    
    if run_showcase_first:
        maybe_run_showcase(
            episode_id=episode_id,
            video_path=video_path,
            host=host,
            port=port,
            fps=showcase_fps,
            yolo_model=showcase_yolo_model,
            device=showcase_device,
            memgraph_clear=showcase_memgraph_clear,
        )

    logger.info(f"Connecting to Memgraph at {host}:{port}...")
    rag = OrionRAG(
        host=host,
        port=port,
        enable_llm=True,
        llm_model="qwen2.5:14b-instruct-q8_0",
    )
    
    # Show database stats
    stats = rag.get_stats()
    logger.info(f"Database: {stats['entities']} entities, {stats['frames']} frames")
    logger.info(f"LLM: {stats['llm_model'] or 'disabled'}")
    
    # Setup Gemini if requested
    gemini_model = setup_gemini(gemini_model_name) if use_gemini else None
    video_payload = prepare_video_payload(video_path) if gemini_model else None
    
    # Run evaluation
    report = EvaluationReport(
        video_path=str(video_path),
        episode_id=episode_id,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        gemini_model=(gemini_model_name if gemini_model else "disabled"),
        total_questions=len(TEST_QUESTIONS),
    )

    # Audit top detected classes for false positives (semantic filtering quality)
    try:
        top_classes = fetch_top_detected_classes(rag, limit=20)
        report.top_detected_classes = top_classes
        if gemini_model and video_payload is not None:
            logger.info("\n[Audit] Verifying top detected classes with Gemini...")
            report.class_audit = audit_classes_with_gemini(gemini_model, video_payload, top_classes)
            absent = report.class_audit.get("absent", []) if isinstance(report.class_audit, dict) else []
            if absent:
                logger.info("[Audit] Gemini marked absent: %s", ", ".join(map(str, absent[:10])))
    except Exception as e:
        logger.warning("Class audit skipped (failed): %s", e)
    
    total_latency = 0.0
    neural_cypher_count = 0
    
    logger.info(f"\nEvaluating {len(TEST_QUESTIONS)} questions...")
    logger.info("=" * 60)
    
    for i, q_data in enumerate(TEST_QUESTIONS, 1):
        question = q_data["q"]
        q_type = q_data["type"]
        
        logger.info(f"\n[{i}/{len(TEST_QUESTIONS)}] {question}")
        
        start_time = time.time()
        result = rag.query(question, use_llm=True, use_neural_cypher=True)
        latency = (time.time() - start_time) * 1000
        
        total_latency += latency
        
        # Check if neural cypher was used
        neural_used = result.query_type == "neural_cypher"
        if neural_used:
            neural_cypher_count += 1
        
        logger.info(f"  Answer: {result.answer[:100]}...")
        logger.info(f"  Latency: {latency:.0f}ms | Neural Cypher: {neural_used}")
        
        # Validate with Gemini
        verdict, notes = "skipped", ""
        if gemini_model and video_payload is not None:
            logger.info("  Validating with Gemini...")
            verdict, notes = validate_with_gemini(
                gemini_model, video_payload, question, result.answer
            )
            logger.info(f"  Verdict: {verdict}")
            
            if verdict == "CORRECT":
                report.correct_count += 1
            elif verdict == "PARTIAL":
                report.partial_count += 1
            elif verdict in ("INCORRECT", "HALLUCINATION"):
                report.incorrect_count += 1
        
        report.results.append(QueryResult(
            question=question,
            query_type=q_type,
            orion_answer=result.answer,
            orion_latency_ms=latency,
            cypher_used=result.cypher_query,
            evidence_count=len(result.evidence),
            neural_cypher_used=neural_used,
            gemini_verdict=verdict,
            gemini_notes=notes,
        ))
    
    # Calculate stats
    report.avg_latency_ms = total_latency / len(TEST_QUESTIONS)
    report.neural_cypher_success_rate = neural_cypher_count / len(TEST_QUESTIONS)
    
    rag.close()
    return report


def print_report(report: EvaluationReport):
    """Print evaluation report."""
    print("\n" + "=" * 70)
    print("ORION v3 ARCHITECTURE EVALUATION REPORT")
    print("=" * 70)
    print(f"Video: {report.video_path}")
    print(f"Episode: {report.episode_id}")
    print(f"Timestamp: {report.timestamp}")
    print(f"Gemini: {report.gemini_model}")
    print()
    
    print("📊 PERFORMANCE METRICS")
    print("-" * 40)
    print(f"  Total Questions: {report.total_questions}")
    print(f"  Avg Latency: {report.avg_latency_ms:.0f}ms")
    print(f"  Neural Cypher Success: {report.neural_cypher_success_rate*100:.1f}%")
    print()

    if report.top_detected_classes:
        print("🔎 DETECTION QUALITY (Top Classes)")
        print("-" * 40)
        for it in report.top_detected_classes[:10]:
            print(f"  - {it.get('class_name')}: {it.get('observations')}")
        if report.class_audit and isinstance(report.class_audit, dict):
            absent = report.class_audit.get("absent", [])
            unsure = report.class_audit.get("unsure", [])
            if absent:
                print(f"  Gemini absent (likely false positives): {', '.join(map(str, absent[:10]))}")
            if unsure:
                print(f"  Gemini unsure: {', '.join(map(str, unsure[:10]))}")
            notes = report.class_audit.get("notes")
            if notes:
                print(f"  Notes: {str(notes)[:120]}")
        print()
    
    if report.correct_count + report.partial_count + report.incorrect_count > 0:
        print("🎯 ACCURACY (Gemini Validated)")
        print("-" * 40)
        total_validated = report.correct_count + report.partial_count + report.incorrect_count
        print(f"  Correct: {report.correct_count} ({report.correct_count/total_validated*100:.1f}%)")
        print(f"  Partial: {report.partial_count} ({report.partial_count/total_validated*100:.1f}%)")
        print(f"  Incorrect: {report.incorrect_count} ({report.incorrect_count/total_validated*100:.1f}%)")
        print()
    
    print("📝 DETAILED RESULTS")
    print("-" * 40)
    for r in report.results:
        status = "✓" if r.gemini_verdict == "CORRECT" else "◐" if r.gemini_verdict == "PARTIAL" else "✗"
        cypher_tag = "[NC]" if r.neural_cypher_used else "[TQ]"
        print(f"  {status} {cypher_tag} {r.question[:50]}...")
        print(f"      → {r.orion_answer[:60]}...")
        if r.gemini_notes:
            print(f"      Note: {r.gemini_notes[:50]}...")
        print()
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Orion v3 Architecture Evaluation")
    parser.add_argument("--video", required=True, help="Video file to evaluate")
    parser.add_argument("--episode", required=True, help="Episode ID (for Memgraph lookup)")
    parser.add_argument("--host", default="127.0.0.1", help="Memgraph host")
    parser.add_argument("--port", type=int, default=7687, help="Memgraph port")
    parser.add_argument("--no-gemini", action="store_true", help="Skip Gemini validation")
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL, help="Gemini model name")
    parser.add_argument(
        "--run-showcase",
        action="store_true",
        help="Run end-to-end showcase (FastVLM + semantic filtering + Memgraph export) before evaluation",
    )
    parser.add_argument("--showcase-fps", type=float, default=4.0, help="FPS used when --run-showcase")
    parser.add_argument("--showcase-yolo-model", default="yolo11m", help="YOLO model used when --run-showcase")
    parser.add_argument("--showcase-device", default="mps", choices=["cuda", "mps", "cpu"], help="Device used when --run-showcase")
    parser.add_argument("--showcase-memgraph-clear", action="store_true", help="Clear Memgraph before ingest when --run-showcase")
    parser.add_argument("--output", help="Save report to JSON file")
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)
    
    report = run_evaluation(
        video_path=video_path,
        episode_id=args.episode,
        host=args.host,
        port=args.port,
        use_gemini=not args.no_gemini,
        gemini_model_name=args.gemini_model,
        run_showcase_first=args.run_showcase,
        showcase_fps=args.showcase_fps,
        showcase_yolo_model=args.showcase_yolo_model,
        showcase_device=args.showcase_device,
        showcase_memgraph_clear=args.showcase_memgraph_clear,
    )
    
    print_report(report)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict for JSON serialization
        report_dict = {
            "video_path": report.video_path,
            "episode_id": report.episode_id,
            "timestamp": report.timestamp,
            "gemini_model": report.gemini_model,
            "total_questions": report.total_questions,
            "correct_count": report.correct_count,
            "partial_count": report.partial_count,
            "incorrect_count": report.incorrect_count,
            "avg_latency_ms": report.avg_latency_ms,
            "neural_cypher_success_rate": report.neural_cypher_success_rate,
            "top_detected_classes": report.top_detected_classes,
            "class_audit": report.class_audit,
            "results": [
                {
                    "question": r.question,
                    "query_type": r.query_type,
                    "orion_answer": r.orion_answer,
                    "orion_latency_ms": r.orion_latency_ms,
                    "cypher_used": r.cypher_used,
                    "evidence_count": r.evidence_count,
                    "neural_cypher_used": r.neural_cypher_used,
                    "gemini_verdict": r.gemini_verdict,
                    "gemini_notes": r.gemini_notes,
                }
                for r in report.results
            ],
        }
        
        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        logger.info(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
