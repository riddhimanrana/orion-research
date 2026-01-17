#!/usr/bin/env python3
"""
PVSG + ActionGenome Setup & Testing Script
============================================

This script:
1. Validates dataset setup
2. Tests PVSG evaluator
3. Tests ActionGenome SGA evaluator
4. Benchmarks scene graph generation pipeline
5. Provides setup instructions

Usage:
    python scripts/setup_evaluation_datasets.py --check-datasets
    python scripts/setup_evaluation_datasets.py --test-evaluators
    python scripts/setup_evaluation_datasets.py --benchmark-pipeline
"""

import logging
import argparse
from pathlib import Path
from typing import List

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dataset_setup():
    """Check if PVSG and ActionGenome are properly set up."""
    logger.info("=" * 60)
    logger.info("Checking Dataset Setup")
    logger.info("=" * 60)
    
    pvsg_root = Path("/Users/yogeshatluru/orion-research/datasets/PVSG")
    action_genome_root = Path("/Users/yogeshatluru/orion-research/datasets/ActionGenome")
    
    # Check PVSG
    logger.info("\n📊 PVSG Dataset:")
    if pvsg_root.exists():
        logger.info(f"  ✓ PVSG directory found: {pvsg_root}")
        
        pvsg_json = pvsg_root / "pvsg.json"
        if pvsg_json.exists():
            logger.info(f"  ✓ pvsg.json found ({pvsg_json.stat().st_size / 1e6:.1f} MB)")
        else:
            logger.warning(f"  ✗ pvsg.json not found at {pvsg_json}")
        
        # Check video subdirectories
        for subdir in ['Ego4D', 'EpicKitchen', 'VidOR']:
            subdir_path = pvsg_root / subdir
            if subdir_path.exists():
                video_count = len(list(subdir_path.glob('**/*.mp4')))
                logger.info(f"  ✓ {subdir}: {video_count} videos")
            else:
                logger.warning(f"  ✗ {subdir} not found")
    else:
        logger.error(f"  ✗ PVSG not found at {pvsg_root}")
        logger.info("\nTo download PVSG, run:")
        logger.info("  git clone https://huggingface.co/datasets/Jingkang/PVSG datasets/PVSG")
    
    # Check ActionGenome
    logger.info("\n📊 ActionGenome Dataset:")
    if action_genome_root.exists():
        logger.info(f"  ✓ ActionGenome directory found: {action_genome_root}")
    else:
        logger.warning(f"  ✗ ActionGenome not found at {action_genome_root}")
        logger.info("\nTo set up ActionGenome:")
        logger.info("  mkdir -p datasets/ActionGenome")
        logger.info("  # Download from: https://github.com/jingkang50/OpenPVSG")


def test_pvsg_evaluator():
    """Test PVSG evaluator functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing PVSG Evaluator")
    logger.info("=" * 60)
    
    try:
        from orion.evaluation.pvsg_evaluator import PVSGEvaluator, SceneGraphTriplet
        
        evaluator = PVSGEvaluator()
        logger.info(f"✓ PVSG evaluator initialized")
        logger.info(f"  - {len(evaluator.videos)} videos in dataset")
        
        # Test on first video
        if evaluator.videos:
            video_id = evaluator.videos[0]
            sgs = evaluator.get_video_scene_graphs(video_id)
            logger.info(f"\n✓ Loaded {video_id}:")
            logger.info(f"  - Frames with scene graphs: {len(sgs)}")
            
            if sgs:
                sample_frame = list(sgs.keys())[0]
                sample_sgs = sgs[sample_frame]
                logger.info(f"  - Frame {sample_frame}: {len(sample_sgs)} triplets")
                
                for i, triplet in enumerate(sample_sgs[:3]):
                    logger.info(f"    {i+1}. ({triplet.subject}, {triplet.predicate}, {triplet.object})")
                
                # Test evaluation
                if len(sample_sgs) > 0:
                    # Create mock predictions
                    predicted = sample_sgs[:max(1, len(sample_sgs)//2)]  # 50% correct
                    result = evaluator.evaluate_predictions(video_id, sample_frame, predicted)
                    
                    logger.info(f"\n✓ Mock evaluation:")
                    logger.info(f"  - Recall@1: {result.recall_at_k[1]:.3f}")
                    logger.info(f"  - Recall@5: {result.recall_at_k[5]:.3f}")
                    logger.info(f"  - Recall@10: {result.recall_at_k[10]:.3f}")
                    logger.info(f"  - Precision: {result.precision:.3f}")
                    logger.info(f"  - F1-Score: {result.f1_score:.3f}")
    
    except Exception as e:
        logger.error(f"✗ PVSG evaluator test failed: {e}", exc_info=True)


def test_sga_evaluator():
    """Test ActionGenome SGA evaluator."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing ActionGenome SGA Evaluator")
    logger.info("=" * 60)
    
    try:
        from orion.evaluation.sga_evaluator import SGAEvaluator
        
        evaluator = SGAEvaluator()
        logger.info("✓ ActionGenome SGA evaluator initialized")
        
        if not evaluator.videos:
            logger.warning("  No ActionGenome videos loaded (dataset may not be set up)")
            logger.info("  This is expected if ActionGenome is not downloaded yet")
    
    except Exception as e:
        logger.error(f"✗ SGA evaluator test failed: {e}", exc_info=True)


def test_gemini_vlm():
    """Test Gemini VLM integration."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Gemini VLM Backend")
    logger.info("=" * 60)
    
    try:
        from orion.backends.gemini_vlm import create_vlm_backend
        
        # Try to create Gemini backend
        try:
            vlm = create_vlm_backend("gemini")
            logger.info("✓ Gemini VLM backend initialized")
            logger.info(f"  Model: {vlm.model}")
        except ImportError as e:
            logger.warning(f"⚠ google-generativeai not installed: {e}")
            logger.info("  Install with: pip install google-generativeai")
            
            # Fall back to FastVLM
            vlm = create_vlm_backend("fastvlm")
            logger.info("✓ Fallback to FastVLM backend")
        except ValueError as e:
            logger.warning(f"⚠ {e}")
            vlm = create_vlm_backend("fastvlm")
            logger.info("✓ Using FastVLM backend")
    
    except Exception as e:
        logger.error(f"✗ VLM backend test failed: {e}", exc_info=True)


def benchmark_scene_graph_pipeline():
    """Benchmark scene graph generation pipeline."""
    logger.info("\n" + "=" * 60)
    logger.info("Benchmarking Scene Graph Pipeline")
    logger.info("=" * 60)
    
    try:
        from orion.evaluation.scene_graph_pipeline import create_pipeline, PipelineMode
        
        # Test lightweight pipeline (faster)
        logger.info("\n📈 Lightweight Pipeline (YOLO-World + FastVLM):")
        pipeline = create_pipeline(PipelineMode.LIGHTWEIGHT)
        logger.info("✓ Pipeline initialized")
        
        # Test paper pipeline
        logger.info("\n📈 Paper Pipeline (DINOv3 + Gemini):")
        try:
            pipeline = create_pipeline(PipelineMode.PAPER)
            logger.info("✓ Pipeline initialized")
        except Exception as e:
            logger.warning(f"⚠ Paper pipeline not available: {e}")
    
    except Exception as e:
        logger.error(f"✗ Pipeline benchmark failed: {e}", exc_info=True)


def print_setup_instructions():
    """Print setup instructions for next steps."""
    logger.info("\n" + "=" * 60)
    logger.info("📋 Setup Instructions")
    logger.info("=" * 60)
    
    instructions = """
1. Install Dependencies:
   pip install google-generativeai  # For Gemini VLM
   pip install pillow  # For image processing

2. PVSG Dataset (Already Downloaded):
   ✓ Location: datasets/PVSG
   ✓ Contains: Ego4D, EpicKitchen, VidOR

3. ActionGenome Dataset (To Download):
   mkdir -p datasets/ActionGenome
   git clone https://github.com/jingkang50/OpenPVSG datasets/ActionGenome

4. Gemini API Setup:
   - Get API key from: https://aistudio.google.com/apikey
   - Set environment variable: export GOOGLE_API_KEY=your_key_here

5. Testing:
   python scripts/setup_evaluation_datasets.py --test-evaluators
   
6. Running Pipeline:
   from orion.evaluation.scene_graph_pipeline import create_pipeline, PipelineMode
   
   # Paper version (stronger results)
   pipeline = create_pipeline(PipelineMode.PAPER)
   
   # Lightweight version (faster)
   pipeline = create_pipeline(PipelineMode.LIGHTWEIGHT)
   
   # Process video
   video_sgs = pipeline.process_video("path/to/video.mp4")
   
   # Evaluate on PVSG
   metrics = pipeline.evaluate_on_pvsg(video_sgs)

7. Evaluation Metrics Used:
   - Recall@K: Standard metric used by HyperGLM
   - Precision: % of predictions that are correct
   - F1-Score: Harmonic mean of precision/recall

Next Steps:
1. Download ActionGenome dataset
2. Set up Gemini API key
3. Run evaluation benchmarks on PVSG
4. Compare against HyperGLM baseline
"""
    
    logger.info(instructions)


def main():
    parser = argparse.ArgumentParser(description="Setup and test evaluation datasets")
    parser.add_argument("--check-datasets", action="store_true", help="Check dataset setup")
    parser.add_argument("--test-evaluators", action="store_true", help="Test evaluators")
    parser.add_argument("--benchmark-pipeline", action="store_true", help="Benchmark pipeline")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if not any([args.check_datasets, args.test_evaluators, args.benchmark_pipeline, args.all]):
        args.all = True
    
    if args.check_datasets or args.all:
        check_dataset_setup()
    
    if args.test_evaluators or args.all:
        test_pvsg_evaluator()
        test_sga_evaluator()
        test_gemini_vlm()
    
    if args.benchmark_pipeline or args.all:
        benchmark_scene_graph_pipeline()
    
    print_setup_instructions()


if __name__ == "__main__":
    main()
