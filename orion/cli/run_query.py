#!/usr/bin/env python3
"""
Orion Stage 6 Query CLI
========================

Interactive natural language query interface over video memory.

Features:
- Conversational Q&A with context tracking
- Streaming responses
- Memgraph-backed retrieval
- Ollama-powered reasoning

Usage:
    # Interactive mode (default)
    python -m orion.cli.run_query --episode my_episode
    
    # Single question
    python -m orion.cli.run_query --episode my_episode -q "What objects are in the video?"
    
    # Without LLM (template answers only)
    python -m orion.cli.run_query --episode my_episode --no-llm
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.config import ensure_results_dir

logger = logging.getLogger("orion.query")


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )
    # Reduce noise from other modules
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def print_banner():
    """Print welcome banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                    🔮 ORION QUERY (Stage 6)                   ║
║                                                               ║
║  Natural Language Video Understanding                         ║
║  Powered by Memgraph + Ollama                                 ║
╚═══════════════════════════════════════════════════════════════╝
""")


def print_help():
    """Print help for interactive mode."""
    print("""
📖 Available Commands:
  <question>     Ask any question about the video
  stats          Show database statistics
  clear          Clear conversation history
  export         Export conversation to JSON
  help           Show this help
  exit/quit      Exit the interface

💡 Example Questions:
  • What objects are in the video?
  • Where did the book appear?
  • What was near the laptop?
  • What did the person interact with?
  • What happened at 25 seconds?
  • Find objects similar to the person
  • How long was the laptop visible?
  • Describe the interactions in this video
""")


def verify_memgraph_data(rag) -> bool:
    """Verify Memgraph has data to query."""
    try:
        stats = rag.get_stats()
        if stats['entities'] == 0:
            print("\n⚠️  Warning: No entities found in Memgraph!")
            print("   Run the perception pipeline first:")
            print("   python -m orion.cli.run_showcase --episode <id> --video <path> --memgraph")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to verify Memgraph data: {e}")
        return False


def run_interactive(
    rag,
    episode_id: str,
    results_dir: Path,
):
    """Run interactive query session."""
    conversation_log = []
    
    print_banner()
    
    # Show stats
    stats = rag.get_stats()
    print(f"📊 Episode: {episode_id}")
    print(f"   Entities: {stats['entities']} | Frames: {stats['frames']}")
    print(f"   Observations: {stats['observations']} | Relations: {stats['near_relationships'] + stats['held_by_relationships']}")
    if stats['llm_enabled']:
        print(f"   🤖 LLM: {stats['llm_model']}")
    else:
        print(f"   🤖 LLM: Disabled (using template answers)")
    
    print("\nType 'help' for commands, 'exit' to quit.\n")
    
    while True:
        try:
            question = input("❓ ").strip()
            if not question:
                continue
            
            # Handle commands
            if question.lower() in ("exit", "quit", "q"):
                break
            
            if question.lower() == "help":
                print_help()
                continue
            
            if question.lower() == "stats":
                stats = rag.get_stats()
                print("\n📊 Database Statistics:")
                for k, v in stats.items():
                    print(f"   {k}: {v}")
                print()
                continue
            
            if question.lower() == "clear":
                if rag.reasoning_model:
                    rag.reasoning_model.clear_history()
                conversation_log.clear()
                print("✓ Conversation cleared\n")
                continue
            
            if question.lower() == "export":
                export_path = results_dir / f"conversation_{int(time.time())}.json"
                with open(export_path, "w") as f:
                    json.dump(conversation_log, f, indent=2, default=str)
                print(f"✓ Exported to {export_path}\n")
                continue
            
            # Process question
            start_time = time.time()
            
            if rag.llm_enabled:
                # Streaming response
                print("💡 ", end="", flush=True)
                full_answer = ""
                for token in rag.stream_query(question):
                    print(token, end="", flush=True)
                    full_answer += token
                print()
            else:
                # Non-streaming
                result = rag.query(question, use_llm=False)
                full_answer = result.answer
                print(f"💡 {full_answer}")
            
            elapsed = (time.time() - start_time) * 1000
            print(f"   ⏱️ {elapsed:.0f}ms\n")
            
            # Log conversation
            conversation_log.append({
                "timestamp": time.time(),
                "question": question,
                "answer": full_answer,
                "latency_ms": elapsed,
            })
            
        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")
            logger.exception("Query error")
    
    # Save conversation on exit
    if conversation_log:
        export_path = results_dir / f"conversation_{int(time.time())}.json"
        with open(export_path, "w") as f:
            json.dump(conversation_log, f, indent=2, default=str)
        print(f"✓ Conversation saved to {export_path}")
    
    print("\n👋 Goodbye!")


def run_single_query(
    rag,
    question: str,
    use_llm: bool = True,
) -> dict:
    """Run a single query and return result."""
    start_time = time.time()
    
    result = rag.query(question, use_llm=use_llm)
    
    return {
        "question": result.question,
        "answer": result.answer,
        "query_type": result.query_type,
        "confidence": result.confidence,
        "latency_ms": result.latency_ms,
        "evidence_count": len(result.evidence),
        "evidence": result.evidence[:5],  # First 5 items
        "llm_used": result.llm_used,
        "cypher": result.cypher_query,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Orion Stage 6 Query Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python -m orion.cli.run_query --episode test_demo

  # Single question
  python -m orion.cli.run_query --episode test_demo -q "What objects are in the video?"

  # Export results as JSON
  python -m orion.cli.run_query --episode test_demo -q "..." --json
""",
    )
    
    # Required arguments
    parser.add_argument(
        "--episode", "-e",
        required=True,
        help="Episode ID to query"
    )
    
    # Query options
    parser.add_argument(
        "--question", "-q",
        help="Single question (skips interactive mode)"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM synthesis (use template answers)"
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:14b-instruct-q8_0",
        help="LLM model for reasoning (default: qwen2.5:14b-instruct-q8_0)"
    )
    
    # Memgraph connection
    parser.add_argument(
        "--memgraph-host",
        default="127.0.0.1",
        help="Memgraph host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--memgraph-port",
        type=int,
        default=7687,
        help="Memgraph port (default: 7687)"
    )
    
    # Output options
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Initialize results directory
    results_dir = ensure_results_dir(args.episode)
    
    # Initialize RAG
    try:
        from orion.query.rag_v2 import OrionRAG
        
        logger.info(f"Connecting to Memgraph at {args.memgraph_host}:{args.memgraph_port}")
        
        rag = OrionRAG(
            host=args.memgraph_host,
            port=args.memgraph_port,
            enable_llm=not args.no_llm,
            llm_model=args.model,
        )
    except ImportError as e:
        print(f"❌ Failed to import RAG module: {e}")
        print("   Make sure pymgclient is installed: pip install pymgclient")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to connect to Memgraph: {e}")
        print("   Make sure Memgraph is running: docker compose up -d")
        sys.exit(1)
    
    # Verify data exists
    if not verify_memgraph_data(rag):
        rag.close()
        sys.exit(1)
    
    try:
        if args.question:
            # Single question mode
            result = run_single_query(rag, args.question, use_llm=not args.no_llm)
            
            if args.json:
                print(json.dumps(result, indent=2, default=str))
            else:
                print(f"\n❓ Q: {result['question']}")
                print(f"💡 A: {result['answer']}")
                print(f"\n📊 Type: {result['query_type']}")
                print(f"📈 Confidence: {result['confidence']:.2f}")
                print(f"⏱️  Latency: {result['latency_ms']:.0f}ms")
                print(f"🔍 Evidence: {result['evidence_count']} items")
                if result['llm_used']:
                    print(f"🤖 LLM: Yes")
                if result['cypher']:
                    print(f"\n📝 Cypher: {result['cypher'][:100]}...")
        else:
            # Interactive mode
            run_interactive(rag, args.episode, results_dir)
    
    finally:
        rag.close()


if __name__ == "__main__":
    main()
