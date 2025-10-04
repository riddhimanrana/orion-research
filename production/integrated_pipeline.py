"""
Integrated Pipeline: Parts 1 + 2
=================================

This script runs the complete "From Moments to Memory" pipeline:
1. Part 1: Extract rich perception data from video
2. Part 2: Build knowledge graph in Neo4j

Author: Orion Research Team
Date: October 3, 2025
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add production directory to path
sys.path.insert(0, str(Path(__file__).parent))

from production.part1_perception_engine import run_perception_engine
from production.part2_semantic_uplift import run_semantic_uplift

# Optional: Use configurations
try:
    from production.part1_config import apply_config as apply_part1_config, BALANCED_CONFIG as PART1_BALANCED
    from production.part2_config import apply_config as apply_part2_config, BALANCED_CONFIG as PART2_BALANCED
    CONFIGS_AVAILABLE = True
except ImportError:
    CONFIGS_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('integrated_pipeline.log')
    ]
)
logger = logging.getLogger('IntegratedPipeline')


def check_prerequisites() -> Dict[str, bool]:
    """
    Check if all required dependencies and services are available.
    
    Returns:
        Dictionary with status of each prerequisite
    """
    logger.info("Checking prerequisites...")
    
    status = {
        'torch': False,
        'transformers': False,
        'ultralytics': False,
        'hdbscan': False,
        'sentence_transformers': False,
        'neo4j': False,
        'neo4j_connection': False,
        'ollama': False,
    }
    
    # Check Python packages
    try:
        import torch
        status['torch'] = True
        logger.info(f"✓ PyTorch {torch.__version__} available")
    except ImportError:
        logger.warning("✗ PyTorch not available")
    
    try:
        import transformers
        status['transformers'] = True
        logger.info(f"✓ Transformers {transformers.__version__} available")
    except ImportError:
        logger.warning("✗ Transformers not available (required for FastVLM)")
    
    try:
        import ultralytics
        status['ultralytics'] = True
        logger.info(f"✓ Ultralytics available")
    except ImportError:
        logger.warning("✗ Ultralytics not available (required for YOLO)")
    
    try:
        import hdbscan
        status['hdbscan'] = True
        logger.info(f"✓ HDBSCAN available")
    except ImportError:
        logger.warning("✗ HDBSCAN not available (required for entity tracking)")
    
    try:
        import sentence_transformers
        status['sentence_transformers'] = True
        logger.info(f"✓ Sentence Transformers available")
    except ImportError:
        logger.warning("✗ Sentence Transformers not available (required for state detection)")
    
    try:
        import neo4j
        status['neo4j'] = True
        logger.info(f"✓ Neo4j driver available")
    except ImportError:
        logger.warning("✗ Neo4j driver not available")
    
    # Check Neo4j connection
    if status['neo4j']:
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                "bolt://localhost:7687",
                auth=("neo4j", "password")
            )
            driver.verify_connectivity()
            driver.close()
            status['neo4j_connection'] = True
            logger.info("✓ Neo4j database connection successful")
        except Exception as e:
            logger.warning(f"✗ Neo4j database not accessible: {e}")
    
    # Check Ollama (optional)
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            status['ollama'] = True
            logger.info("✓ Ollama service available")
    except Exception:
        logger.info("○ Ollama not available (optional, will use fallback)")
    
    return status


def print_results_summary(results: Dict[str, Any]):
    """
    Print a formatted summary of pipeline results.
    
    Args:
        results: Results dictionary from run_integrated_pipeline()
    """
    print(f"Video: {results['video_path']}")
    print(f"Timestamp: {results['timestamp']}")
    print()
    
    # Part 1 Results
    if results['part1']:
        part1 = results['part1']
        print("PART 1: Perception Engine")
        print(f"  Status: {'✓ Success' if part1.get('success') else '✗ Failed'}")
        print(f"  Duration: {part1.get('duration_seconds', 0):.2f}s")
        print(f"  Objects Detected: {part1.get('num_objects', 0)}")
        if 'output_file' in part1:
            print(f"  Output: {part1['output_file']}")
        print()
    
    # Part 2 Results
    if results['part2']:
        part2 = results['part2']
        print("PART 2: Semantic Uplift")
        print(f"  Status: {'✓ Success' if part2.get('success') else '✗ Failed'}")
        print(f"  Duration: {part2.get('duration_seconds', 0):.2f}s")
        print(f"  Entities Tracked: {part2.get('num_entities', 0)}")
        print(f"  State Changes: {part2.get('num_state_changes', 0)}")
        print(f"  Temporal Windows: {part2.get('num_windows', 0)}")
        if 'graph_stats' in part2:
            stats = part2['graph_stats']
            print(f"  Graph Nodes: {stats.get('entity_nodes', 0)} entities, {stats.get('event_nodes', 0)} events")
            print(f"  Graph Relationships: {stats.get('relationships', 0)}")
        print()
    
    # Overall Status
    total_time = 0
    if results['part1']:
        total_time += results['part1'].get('duration_seconds', 0)
    if results['part2']:
        total_time += results['part2'].get('duration_seconds', 0)
    
    print(f"Total Pipeline Time: {total_time:.2f}s")
    
    if results['errors']:
        print(f"\nErrors encountered:")
        for error in results['errors']:
            print(f"  - {error}")


def run_integrated_pipeline(
    video_path: str,
    output_dir: str = "data/testing",
    use_fastvlm: bool = True,
    part1_config: str = "balanced",
    part2_config: str = "balanced",
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    skip_part1: bool = False,
    skip_part2: bool = False,
) -> Dict[str, Any]:
    """
    Run the complete integrated pipeline.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save outputs
        use_fastvlm: Whether to use real FastVLM (vs placeholder)
        part1_config: Configuration preset for Part 1
        part2_config: Configuration preset for Part 2
        neo4j_uri: Neo4j database URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        skip_part1: Skip Part 1 and use existing perception log
        skip_part2: Skip Part 2 (only run perception)
    
    Returns:
        Results dictionary with statistics from both parts
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filenames
    video_name = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    perception_log_path = output_dir / f"perception_log_{video_name}_{timestamp}.json"
    
    results = {
        'video_path': str(video_path),
        'timestamp': timestamp,
        'part1': None,
        'part2': None,
        'success': False,
        'errors': []
    }
    
    # =======================================================================
    # PART 1: PERCEPTION ENGINE
    # =======================================================================
    
    if not skip_part1:
        logger.info("="*80)
        logger.info("PART 1: ASYNCHRONOUS PERCEPTION ENGINE")
        logger.info("="*80)
        
        # Apply configuration if available
        if CONFIGS_AVAILABLE:
            config_map = {
                'fast': 'FAST_CONFIG',
                'balanced': 'BALANCED_CONFIG',
                'accurate': 'ACCURATE_CONFIG',
            }
            if part1_config in config_map:
                try:
                    from production.part1_config import FAST_CONFIG, BALANCED_CONFIG, ACCURATE_CONFIG
                    config_dict = {
                        'fast': FAST_CONFIG,
                        'balanced': BALANCED_CONFIG,
                        'accurate': ACCURATE_CONFIG
                    }
                    apply_part1_config(config_dict[part1_config])
                    logger.info(f"Applied Part 1 config: {part1_config}")
                except Exception as e:
                    logger.warning(f"Failed to apply Part 1 config: {e}")
        
        # Update config for FastVLM usage
        try:
            from production.part1_perception_engine import Config
            # Note: We can't change Config values here since they're class attributes
            # The use_fastvlm parameter is passed to generate_rich_description
            logger.info(f"FastVLM mode: {'ENABLED' if use_fastvlm else 'DISABLED (using placeholders)'}")
        except Exception as e:
            logger.warning(f"Could not configure FastVLM: {e}")
        
        try:
            logger.info(f"Processing video: {video_path}")
            start_time = time.time()
            
            # Run Part 1
            perception_log = run_perception_engine(
                video_path=video_path,
                output_path=str(perception_log_path)
            )
            
            part1_time = time.time() - start_time
            
            # Part 1 results
            results['part1'] = {
                'duration_seconds': part1_time,
                'num_objects': len(perception_log),
                'output_file': str(perception_log_path),
                'success': True
            }
            
            logger.info(f"✓ Part 1 completed in {part1_time:.2f}s")
            logger.info(f"✓ Processed {len(perception_log)} objects")
            logger.info(f"✓ Saved to: {perception_log_path}")
            
        except Exception as e:
            error_msg = f"Part 1 failed: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            results['part1'] = {'success': False, 'error': str(e)}
            return results
    
    else:
        # Load existing perception log
        logger.info("Skipping Part 1, loading existing perception log...")
        
        # Find most recent perception log
        perception_logs = list(output_dir.glob(f"perception_log_*.json"))
        if not perception_logs:
            error_msg = "No perception log found in output directory"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            return results
        
        perception_log_path = max(perception_logs, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using: {perception_log_path}")
        
        with open(perception_log_path, 'r') as f:
            perception_log = json.load(f)
        
        results['part1'] = {
            'skipped': True,
            'num_objects': len(perception_log),
            'output_file': str(perception_log_path)
        }
    
    # =======================================================================
    # PART 2: SEMANTIC UPLIFT ENGINE
    # =======================================================================
    
    if not skip_part2:
        logger.info("")
        logger.info("="*80)
        logger.info("PART 2: SEMANTIC UPLIFT ENGINE")
        logger.info("="*80)
        
        # Apply configuration if available
        if CONFIGS_AVAILABLE:
            config_map = {
                'fast': 'FAST_CONFIG',
                'balanced': 'BALANCED_CONFIG',
                'accurate': 'ACCURATE_CONFIG',
            }
            if part2_config in config_map:
                try:
                    from production.part2_config import FAST_CONFIG, BALANCED_CONFIG, ACCURATE_CONFIG
                    config_dict = {
                        'fast': FAST_CONFIG,
                        'balanced': BALANCED_CONFIG,
                        'accurate': ACCURATE_CONFIG
                    }
                    apply_part2_config(config_dict[part2_config])
                    logger.info(f"Applied Part 2 config: {part2_config}")
                except Exception as e:
                    logger.warning(f"Failed to apply Part 2 config: {e}")
        
        try:
            logger.info(f"Building knowledge graph from {len(perception_log)} objects...")
            start_time = time.time()
            
            # Run Part 2
            part2_results = run_semantic_uplift(
                perception_log=perception_log,
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password
            )
            
            part2_time = time.time() - start_time
            
            # Add timing to results
            part2_results['duration_seconds'] = part2_time
            results['part2'] = part2_results
            
            logger.info(f"✓ Part 2 completed in {part2_time:.2f}s")
            logger.info(f"✓ Tracked {part2_results.get('num_entities', 0)} entities")
            logger.info(f"✓ Detected {part2_results.get('num_state_changes', 0)} state changes")
            logger.info(f"✓ Created {part2_results.get('graph_stats', {}).get('entity_nodes', 0)} graph nodes")
            
        except Exception as e:
            error_msg = f"Part 2 failed: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            results['part2'] = {'success': False, 'error': str(e)}
            return results
    
    else:
        logger.info("Skipping Part 2 (semantic uplift)")
        results['part2'] = {'skipped': True}
    
    # =======================================================================
    # FINAL SUMMARY
    # =======================================================================
    
    logger.info("")
    logger.info("="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    
    if results['part1'] and results['part1'].get('success'):
        logger.info(f"✓ Part 1: {results['part1']['num_objects']} objects in {results['part1']['duration_seconds']:.2f}s")
    
    if results['part2'] and results['part2'].get('success'):
        logger.info(f"✓ Part 2: {results['part2']['num_entities']} entities in {results['part2']['duration_seconds']:.2f}s")
    
    if results['errors']:
        logger.warning(f"⚠ Pipeline completed with {len(results['errors'])} error(s)")
        for error in results['errors']:
            logger.warning(f"  - {error}")
    else:
        results['success'] = True
        logger.info("✓ Pipeline completed successfully!")
    
    # Save results summary
    results_path = output_dir / f"pipeline_results_{video_name}_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_path}")
    
    return results


def main():
    """Command-line interface for integrated pipeline"""
    
    parser = argparse.ArgumentParser(
        description="Run integrated Part 1 + Part 2 pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on video
  python production/integrated_pipeline.py video.mp4
  
  # Use accurate configs
  python production/integrated_pipeline.py video.mp4 --part1-config accurate --part2-config accurate
  
  # Use placeholder descriptions (faster, no FastVLM)
  python production/integrated_pipeline.py video.mp4 --no-fastvlm
  
  # Only run Part 1
  python production/integrated_pipeline.py video.mp4 --skip-part2
  
  # Only run Part 2 (use existing perception log)
  python production/integrated_pipeline.py video.mp4 --skip-part1
  
  # Custom Neo4j connection
  python production/integrated_pipeline.py video.mp4 \\
      --neo4j-uri bolt://localhost:7687 \\
      --neo4j-user neo4j \\
      --neo4j-password mypassword
        """
    )
    
    # Required arguments
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to input video file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/testing',
        help='Output directory for results (default: data/testing)'
    )
    
    parser.add_argument(
        '--no-fastvlm',
        action='store_true',
        help='Disable FastVLM and use placeholder descriptions'
    )
    
    parser.add_argument(
        '--part1-config',
        type=str,
        choices=['fast', 'balanced', 'accurate'],
        default='balanced',
        help='Configuration preset for Part 1 (default: balanced)'
    )
    
    parser.add_argument(
        '--part2-config',
        type=str,
        choices=['fast', 'balanced', 'accurate'],
        default='balanced',
        help='Configuration preset for Part 2 (default: balanced)'
    )
    
    parser.add_argument(
        '--neo4j-uri',
        type=str,
        default='bolt://localhost:7687',
        help='Neo4j database URI (default: bolt://localhost:7687)'
    )
    
    parser.add_argument(
        '--neo4j-user',
        type=str,
        default='neo4j',
        help='Neo4j username (default: neo4j)'
    )
    
    parser.add_argument(
        '--neo4j-password',
        type=str,
        default='password',
        help='Neo4j password (default: password)'
    )
    
    parser.add_argument(
        '--skip-part1',
        action='store_true',
        help='Skip Part 1 and use existing perception log'
    )
    
    parser.add_argument(
        '--skip-part2',
        action='store_true',
        help='Skip Part 2 (only run perception engine)'
    )
    
    parser.add_argument(
        '--check-prereqs',
        action='store_true',
        help='Check prerequisites and exit'
    )
    
    args = parser.parse_args()
    
    # Check prerequisites if requested
    if args.check_prereqs:
        status = check_prerequisites()
        print("\nPrerequisite Summary:")
        all_ok = all(status.values())
        print(f"Status: {'✓ All prerequisites met' if all_ok else '⚠ Some prerequisites missing'}")
        return 0 if all_ok else 1
    
    # Validate video file
    if not args.skip_part1 and not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    # Check prerequisites before running
    logger.info("Running prerequisite check...")
    status = check_prerequisites()
    
    # Warn about missing critical components
    critical = ['torch', 'ultralytics']
    if not args.skip_part2:
        critical.extend(['neo4j', 'neo4j_connection'])
    
    missing_critical = [k for k in critical if not status.get(k, False)]
    if missing_critical:
        logger.error(f"Missing critical prerequisites: {', '.join(missing_critical)}")
        logger.error("Please install required packages and start necessary services")
        return 1
    
    # Run pipeline
    try:
        results = run_integrated_pipeline(
            video_path=args.video_path,
            output_dir=args.output_dir,
            use_fastvlm=not args.no_fastvlm,
            part1_config=args.part1_config,
            part2_config=args.part2_config,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            skip_part1=args.skip_part1,
            skip_part2=args.skip_part2,
        )
        
        return 0 if results['success'] else 1
        
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
