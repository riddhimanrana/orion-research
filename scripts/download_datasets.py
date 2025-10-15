#!/usr/bin/env python3
"""
Dataset Downloader and Preprocessor
====================================

Downloads and prepares all benchmark datasets for evaluation.

Supported datasets:
- Action Genome
- VSGR (if publicly available)
- PVSG
- ASPIRe (if publicly available)
- Custom datasets

Author: Orion Research Team
Date: October 2025
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlretrieve

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DatasetDownloader")


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: Path, desc: str = "Downloading"):
    """Download a file with progress bar"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=desc) as t:
        urlretrieve(url, filename=output_path, reporthook=t.update_to)


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> int:
    """Run a shell command"""
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


class DatasetDownloader:
    """Manages downloading and preparing datasets"""
    
    def __init__(self, data_root: str = "data/benchmarks"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Data root: {self.data_root.absolute()}")
    
    def download_action_genome(self):
        """Download Action Genome dataset"""
        logger.info("="*80)
        logger.info("Downloading Action Genome")
        logger.info("="*80)
        
        ag_dir = self.data_root / "action_genome"
        ag_dir.mkdir(exist_ok=True)
        
        # Note: Action Genome requires manual download from GitHub
        # https://github.com/JingweiJ/ActionGenome
        
        logger.info("""
Action Genome requires manual download:

1. Visit: https://github.com/JingweiJ/ActionGenome
2. Download the following files:
   - person_bbox.pkl
   - object_bbox_and_relationship.pkl
   - Videos (if available)

3. Place them in: {ag_dir}
   Structure:
   {ag_dir}/
   ├── annotations/
   │   ├── person_bbox.pkl
   │   └── object_bbox_and_relationship.pkl
   └── videos/
       ├── clip_001.mp4
       └── ...

4. After manual download, run:
   python scripts/prepare_action_genome.py

Alternatively, if you have access to the dataset ZIP:
   Place it in {ag_dir}/ and we'll extract it.
""".format(ag_dir=ag_dir))
        
        # Check if ZIP exists
        zip_path = ag_dir / "action_genome.zip"
        if zip_path.exists():
            logger.info(f"Found ZIP at {zip_path}, extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(ag_dir)
            logger.info("Extraction complete!")
        else:
            logger.warning("No ZIP found. Please download manually.")
        
        return ag_dir
    
    def download_pvsg(self):
        """Download PVSG dataset"""
        logger.info("="*80)
        logger.info("Downloading PVSG (Panoptic Video Scene Graph)")
        logger.info("="*80)
        
        pvsg_dir = self.data_root / "pvsg"
        pvsg_dir.mkdir(exist_ok=True)
        
        # PVSG download info
        logger.info("""
PVSG dataset download:

1. Visit: https://pvsg-dataset.github.io/
2. Request access and download:
   - PVSG annotations
   - VidOR videos (base dataset)

3. Place in: {pvsg_dir}
   Structure:
   {pvsg_dir}/
   ├── annotations/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── videos/
       └── ...

4. After download, the dataset will be automatically detected.
""".format(pvsg_dir=pvsg_dir))
        
        return pvsg_dir
    
    def download_sample_dataset(self):
        """Download a small sample dataset for testing"""
        logger.info("="*80)
        logger.info("Creating sample dataset for testing")
        logger.info("="*80)
        
        sample_dir = self.data_root / "sample"
        sample_dir.mkdir(exist_ok=True)
        
        # Create sample data structure
        (sample_dir / "videos").mkdir(exist_ok=True)
        (sample_dir / "annotations").mkdir(exist_ok=True)
        
        # Create sample annotation
        sample_annotation = {
            "video_id": "sample_001",
            "objects": [
                {
                    "id": "person_1",
                    "class": "person",
                    "bbox": [100, 100, 200, 300],
                    "frame_id": 0
                },
                {
                    "id": "cup_1",
                    "class": "cup",
                    "bbox": [250, 200, 300, 250],
                    "frame_id": 0
                }
            ],
            "relationships": [
                {
                    "subject_id": "person_1",
                    "object_id": "cup_1",
                    "predicate": "holding",
                    "frame_id": 0
                }
            ],
            "actions": [
                {
                    "id": "action_1",
                    "class": "pick_up",
                    "person_id": "person_1",
                    "start_frame": 0,
                    "end_frame": 30,
                    "objects": ["cup_1"]
                }
            ]
        }
        
        annotation_path = sample_dir / "annotations" / "sample_001.json"
        with open(annotation_path, 'w') as f:
            json.dump(sample_annotation, f, indent=2)
        
        logger.info(f"Created sample dataset at: {sample_dir}")
        logger.info("""
Sample dataset structure:
{sample_dir}/
├── annotations/
│   └── sample_001.json
└── videos/
    └── (place your test video here as sample_001.mp4)

You can now test the evaluation pipeline with:
    python scripts/run_evaluation.py \\
        --mode benchmark \\
        --benchmark sample \\
        --dataset-path {sample_dir}
""".format(sample_dir=sample_dir))
        
        return sample_dir
    
    def create_validation_split(
        self,
        dataset_name: str,
        split_ratio: float = 0.15
    ):
        """Create validation split from a dataset"""
        logger.info(f"Creating validation split for {dataset_name}...")
        
        dataset_dir = self.data_root / dataset_name
        if not dataset_dir.exists():
            logger.error(f"Dataset {dataset_name} not found at {dataset_dir}")
            return None
        
        # Load all clips
        annotations_dir = dataset_dir / "annotations"
        all_clips = list(annotations_dir.glob("*.json"))
        
        # Shuffle and split
        import random
        random.shuffle(all_clips)
        
        split_idx = int(len(all_clips) * (1 - split_ratio))
        train_clips = all_clips[:split_idx]
        val_clips = all_clips[split_idx:]
        
        # Create validation directory
        val_dir = self.data_root / f"{dataset_name}_validation"
        val_dir.mkdir(exist_ok=True)
        (val_dir / "annotations").mkdir(exist_ok=True)
        
        # Copy validation clips
        import shutil
        for clip in val_clips:
            dest = val_dir / "annotations" / clip.name
            shutil.copy(clip, dest)
        
        logger.info(f"Created validation split:")
        logger.info(f"  Train: {len(train_clips)} clips")
        logger.info(f"  Val: {len(val_clips)} clips")
        logger.info(f"  Saved to: {val_dir}")
        
        return val_dir
    
    def verify_datasets(self) -> Dict[str, bool]:
        """Verify which datasets are properly set up"""
        logger.info("="*80)
        logger.info("Verifying datasets...")
        logger.info("="*80)
        
        datasets = {
            'action_genome': self.data_root / 'action_genome',
            'pvsg': self.data_root / 'pvsg',
            'sample': self.data_root / 'sample',
        }
        
        status = {}
        
        for name, path in datasets.items():
            if not path.exists():
                logger.warning(f"❌ {name}: Not found at {path}")
                status[name] = False
                continue
            
            # Check for required subdirectories
            has_annotations = (path / 'annotations').exists()
            has_videos = (path / 'videos').exists()
            
            if has_annotations and has_videos:
                num_annotations = len(list((path / 'annotations').glob('*.json')))
                logger.info(f"✓ {name}: Found ({num_annotations} annotations)")
                status[name] = True
            else:
                logger.warning(f"⚠ {name}: Incomplete (missing annotations or videos)")
                status[name] = False
        
        logger.info("="*80)
        return status


def main():
    parser = argparse.ArgumentParser(description="Download and prepare datasets")
    parser.add_argument(
        "--dataset",
        choices=["action_genome", "pvsg", "sample", "all"],
        default="sample",
        help="Dataset to download"
    )
    parser.add_argument(
        "--data-root",
        default="data/benchmarks",
        help="Root directory for datasets"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing datasets"
    )
    parser.add_argument(
        "--create-val-split",
        type=str,
        help="Create validation split for specified dataset"
    )
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_root)
    
    if args.verify_only:
        downloader.verify_datasets()
        return
    
    if args.create_val_split:
        downloader.create_validation_split(args.create_val_split)
        return
    
    # Download requested dataset
    if args.dataset == "action_genome" or args.dataset == "all":
        downloader.download_action_genome()
    
    if args.dataset == "pvsg" or args.dataset == "all":
        downloader.download_pvsg()
    
    if args.dataset == "sample" or args.dataset == "all":
        downloader.download_sample_dataset()
    
    # Verify
    downloader.verify_datasets()
    
    print("\n" + "="*80)
    print("Dataset download complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Verify datasets are properly set up")
    print("2. Create validation splits if needed:")
    print("   python scripts/download_datasets.py --create-val-split action_genome")
    print("3. Run hyperparameter tuning:")
    print("   python -m orion.evaluation.hyperparameter_tuning --method grid")
    print("4. Run evaluation:")
    print("   python scripts/run_evaluation.py --mode benchmark --benchmark action_genome")
    print("="*80)


if __name__ == "__main__":
    main()
