"""
Test Script for Part 2: Semantic Uplift Engine
===============================================

This script tests the semantic uplift pipeline with sample data.

Usage:
    python production/test_semantic.py --perception-log path/to/perception_log.json
    python production/test_semantic.py --use-part1-output  # Use Part 1 test output
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_uplift import (
    run_semantic_uplift,
    Config,
    logger
)


def validate_perception_log(perception_log: list) -> bool:
    """
    Validate perception log structure
    
    Args:
        perception_log: List of perception objects
        
    Returns:
        True if valid
    """
    logger.info("Validating perception log structure...")
    
    if not perception_log:
        logger.error("Perception log is empty")
        return False
    
    required_fields = [
        'timestamp', 'frame_number', 'bounding_box',
        'visual_embedding', 'object_class', 'rich_description'
    ]
    
    errors = []
    for i, obj in enumerate(perception_log[:5]):  # Check first 5
        for field in required_fields:
            if field not in obj:
                errors.append(f"Object {i} missing field: {field}")
            elif field == 'visual_embedding' and len(obj.get(field, [])) == 0:
                errors.append(f"Object {i} has empty visual_embedding")
    
    if errors:
        logger.error("Validation errors:")
        for error in errors:
            logger.error(f"  {error}")
        return False
    
    logger.info(f"✓ Perception log validated ({len(perception_log)} objects)")
    return True


def check_neo4j_connection() -> bool:
    """
    Check if Neo4j is running and accessible
    
    Returns:
        True if Neo4j is accessible
    """
    logger.info("Checking Neo4j connection...")
    
    try:
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
        )
        
        driver.verify_connectivity()
        driver.close()
        
        logger.info("✓ Neo4j connection successful")
        return True
    
    except Exception as e:
        logger.error(f"✗ Neo4j connection failed: {e}")
        logger.error("\nTo fix:")
        logger.error("1. Install Neo4j Desktop or Docker")
        logger.error("2. Start Neo4j server")
        logger.error("3. Update credentials in Config if needed")
        logger.error(f"   Current URI: {Config.NEO4J_URI}")
        return False


def check_ollama_connection() -> bool:
    """
    Check if Ollama is running
    
    Returns:
        True if Ollama is accessible
    """
    logger.info("Checking Ollama connection...")
    
    try:
        import requests
        
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            logger.info(f"✓ Ollama running with {len(models)} models")
            
            # Check if required model is available
            model_names = [m['name'] for m in models]
            if not any(Config.OLLAMA_MODEL in name for name in model_names):
                logger.warning(f"⚠ Model '{Config.OLLAMA_MODEL}' not found")
                logger.info("Install with: ollama pull llama3")
            else:
                logger.info(f"✓ Model '{Config.OLLAMA_MODEL}' available")
            
            return True
        else:
            logger.warning("✗ Ollama responded with unexpected status")
            return False
    
    except Exception as e:
        logger.warning(f"✗ Ollama not accessible: {e}")
        logger.warning("\nOllama is optional but recommended for event composition")
        logger.warning("To install:")
        logger.warning("  macOS: brew install ollama")
        logger.warning("  Then: ollama serve")
        logger.warning("  And: ollama pull llama3")
        return False


def visualize_graph_stats(stats: dict):
    """
    Visualize graph statistics
    
    Args:
        stats: Dictionary of graph statistics
    """
    print("\n" + "="*80)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("="*80)
    
    print("\n┌─────────────────────────────────────────┐")
    print("│  Graph Nodes                            │")
    print("├─────────────────────────────────────────┤")
    print(f"│  Entity Nodes:    {stats.get('entity_nodes', 0):>6}              │")
    print(f"│  Event Nodes:     {stats.get('event_nodes', 0):>6}              │")
    print("└─────────────────────────────────────────┘")
    
    print("\n┌─────────────────────────────────────────┐")
    print("│  Graph Relationships                    │")
    print("├─────────────────────────────────────────┤")
    print(f"│  Total:           {stats.get('relationships', 0):>6}              │")
    print("└─────────────────────────────────────────┘")
    
    print("\n" + "="*80 + "\n")


def query_sample_data(neo4j_uri: str = None):
    """
    Query and display sample data from the graph
    
    Args:
        neo4j_uri: Neo4j URI
    """
    try:
        from neo4j import GraphDatabase
        
        uri = neo4j_uri or Config.NEO4J_URI
        driver = GraphDatabase.driver(
            uri,
            auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
        )
        
        print("\n" + "="*80)
        print("SAMPLE GRAPH DATA")
        print("="*80)
        
        with driver.session() as session:
            # Get sample entities
            print("\nSample Entities (first 5):")
            print("-" * 80)
            result = session.run("""
                MATCH (e:Entity)
                RETURN e.id, e.label, e.appearance_count, e.first_seen
                LIMIT 5
            """)
            
            for record in result:
                print(f"  {record['e.id']}")
                print(f"    Class: {record['e.label']}")
                print(f"    Appearances: {record['e.appearance_count']}")
                print(f"    First seen: {record['e.first_seen']:.2f}s")
                print()
            
            # Get sample events
            print("\nSample Events (first 5):")
            print("-" * 80)
            result = session.run("""
                MATCH (ev:Event)
                RETURN ev.id, ev.type, ev.timestamp, ev.description
                LIMIT 5
            """)
            
            count = 0
            for record in result:
                print(f"  {record.get('ev.id', 'N/A')}")
                print(f"    Type: {record.get('ev.type', 'N/A')}")
                print(f"    Time: {record.get('ev.timestamp', 'N/A')}")
                print(f"    Description: {record.get('ev.description', 'N/A')[:80]}...")
                print()
                count += 1
            
            if count == 0:
                print("  No events created yet")
            
            # Get sample relationships
            print("\nSample Relationships (first 5):")
            print("-" * 80)
            result = session.run("""
                MATCH (e:Entity)-[r]->(ev)
                RETURN e.id, type(r), labels(ev), ev.type
                LIMIT 5
            """)
            
            count = 0
            for record in result:
                print(f"  {record['e.id']} -{record['type(r)']}-> {record['labels(ev)'][0]}")
                print(f"    Event type: {record.get('ev.type', 'N/A')}")
                print()
                count += 1
            
            if count == 0:
                print("  No relationships created yet")
        
        driver.close()
        print("="*80 + "\n")
    
    except Exception as e:
        logger.error(f"Failed to query sample data: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Test the Semantic Uplift Engine'
    )
    parser.add_argument(
        '--perception-log',
        type=str,
        help='Path to perception log JSON file'
    )
    parser.add_argument(
        '--use-part1-output',
        action='store_true',
        help='Use output from Part 1 test script'
    )
    parser.add_argument(
        '--neo4j-uri',
        type=str,
        default=Config.NEO4J_URI,
        help=f'Neo4j URI (default: {Config.NEO4J_URI})'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip pre-flight validation checks'
    )
    
    args = parser.parse_args()
    
    # Determine perception log path
    if args.use_part1_output:
        perception_log_path = "data/testing/perception_log.json"
    elif args.perception_log:
        perception_log_path = args.perception_log
    else:
        logger.error("Please provide --perception-log or --use-part1-output")
        parser.print_help()
        sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(perception_log_path):
        logger.error(f"Perception log not found: {perception_log_path}")
        logger.info("\nTo generate a perception log:")
        logger.info("  python production/test_part1.py --generate-sample")
        sys.exit(1)
    
    # Load perception log
    logger.info(f"Loading perception log from: {perception_log_path}")
    with open(perception_log_path, 'r') as f:
        perception_log = json.load(f)
    
    logger.info(f"Loaded {len(perception_log)} perception objects")
    
    # Pre-flight checks
    if not args.skip_validation:
        print("\n" + "="*80)
        print("PRE-FLIGHT CHECKS")
        print("="*80 + "\n")
        
        # Validate perception log
        if not validate_perception_log(perception_log):
            logger.error("Perception log validation failed")
            sys.exit(1)
        
        # Check Neo4j
        if not check_neo4j_connection():
            logger.error("Neo4j is required for semantic uplift")
            sys.exit(1)
        
        # Check Ollama (optional)
        check_ollama_connection()
        
        print("\n" + "="*80)
        print("All checks passed! Starting semantic uplift...")
        print("="*80 + "\n")
    
    # Run semantic uplift
    try:
        results = run_semantic_uplift(
            perception_log,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=Config.NEO4J_USER,
            neo4j_password=Config.NEO4J_PASSWORD
        )
        
        # Save results
        output_dir = os.path.dirname(perception_log_path) or '.'
        results_path = os.path.join(output_dir, 'uplift_results.json')
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_path}")
        
        # Display results
        print("\n" + "="*80)
        print("SEMANTIC UPLIFT RESULTS")
        print("="*80)
        print(f"\nEntities tracked:       {results['num_entities']}")
        print(f"State changes detected: {results['num_state_changes']}")
        print(f"Temporal windows:       {results['num_windows']}")
        print(f"Cypher queries:         {results['num_queries']}")
        
        if results['graph_stats']:
            visualize_graph_stats(results['graph_stats'])
        
        # Query sample data
        if results['success']:
            query_sample_data(args.neo4j_uri)
            
            print("✅ Semantic uplift completed successfully!")
            print(f"📁 Results saved to: {results_path}")
            print("\nNext steps:")
            print("  1. Explore the graph in Neo4j Browser: http://localhost:7474")
            print("  2. Run Part 3 for query & evaluation")
        else:
            print("⚠️ Semantic uplift completed with errors")
            print("Check the logs above for details")
    
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
