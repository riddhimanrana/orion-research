#!/usr/bin/env python3
"""
Extract Ground Truth from VSGR ASPIRe Dataset

Extracts causal relationships from VSGR ASPIRe annotations for CIS optimization.
Uses the reasoning text in annotations to identify object interactions.

Usage:
    python scripts/extract_vsgr_ground_truth.py \
        --aspire-root data \
        --output data/ground_truth/vsgr_aspire_train.json \
        --split train \
        --max-videos 50
"""

import argparse
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from orion.evaluation.benchmarks.vsgr_aspire_loader import VSGRASpireLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Extract causal ground truth from VSGR ASPIRe dataset"
    )
    parser.add_argument(
        "--aspire-root",
        type=str,
        required=True,
        help="Root directory containing aspire_train.json and aspire_test.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path for ground truth"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to process"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum number of videos to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Paths
    aspire_root = Path(args.aspire_root)
    output_path = Path(args.output)
    
    # Validate paths
    if not aspire_root.exists():
        logger.error(f"ASPIRe root not found: {aspire_root}")
        return 1
    
    # Check for annotation files
    ann_file = aspire_root / f"aspire_{args.split}.json"
    if not ann_file.exists():
        logger.error(f"Annotation file not found: {ann_file}")
        logger.info("Expected file format: aspire_train.json or aspire_test.json")
        return 1
    
    logger.info("=" * 70)
    logger.info("VSGR ASPIRe Ground Truth Extraction")
    logger.info("=" * 70)
    logger.info(f"ASPIRe root: {aspire_root}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Output: {output_path}")
    if args.max_videos:
        logger.info(f"Max videos: {args.max_videos}")
    logger.info("")
    
    # Initialize loader
    logger.info("Loading VSGR ASPIRe dataset...")
    loader = VSGRASpireLoader(aspire_root)
    
    # Load dataset
    try:
        data = loader.load_split(args.split)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return 1
    
    # Get statistics
    num_videos = len(data.get('videos', []))
    num_images = len(data.get('images', []))
    num_annotations = len(data.get('annotations', []))
    
    logger.info(f"Dataset loaded: {num_videos} videos, {num_images} frames, {num_annotations} annotations")
    logger.info("")
    
    # Extract causal relationships
    logger.info("Extracting causal relationships from reasoning text...")
    causal_pairs = loader.extract_causal_relationships(
        split=args.split,
        max_videos=args.max_videos
    )
    
    if not causal_pairs:
        logger.warning("No causal pairs extracted!")
        return 1
    
    # Statistics
    logger.info("")
    logger.info("=" * 70)
    logger.info("Extraction Results")
    logger.info("=" * 70)
    logger.info(f"Total causal pairs: {len(causal_pairs)}")
    
    # Count unique videos
    unique_videos = len(set(p.video_id for p in causal_pairs))
    logger.info(f"Videos with causal pairs: {unique_videos}")
    logger.info(f"Average pairs per video: {len(causal_pairs) / unique_videos:.1f}")
    
    # Count by interaction type
    interaction_counts = {}
    for pair in causal_pairs:
        interaction_counts[pair.interaction_type] = interaction_counts.get(pair.interaction_type, 0) + 1
    
    logger.info(f"\nTop interaction types:")
    for interaction, count in sorted(interaction_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  {interaction}: {count}")
    
    # Average confidence
    avg_confidence = sum(p.confidence for p in causal_pairs) / len(causal_pairs)
    logger.info(f"\nAverage confidence: {avg_confidence:.3f}")
    
    # Export to CIS format
    logger.info("")
    logger.info(f"Exporting to {output_path}...")
    loader.export_to_cis_format(causal_pairs, output_path)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✓ Ground truth extraction complete!")
    logger.info("=" * 70)
    logger.info(f"Output: {output_path}")
    logger.info(f"Total pairs: {len(causal_pairs)}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Run Orion pipeline on VSGR videos")
    logger.info("  2. Run CIS optimization:")
    logger.info(f"     python scripts/run_cis_hpo.py \\")
    logger.info(f"       --ground-truth {output_path} \\")
    logger.info(f"       --perception-logs data/orion_predictions/vsgr/ \\")
    logger.info(f"       --trials 200")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
