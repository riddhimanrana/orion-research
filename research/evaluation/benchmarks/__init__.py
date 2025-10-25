"""
Benchmark Support Package
==========================

This package provides loaders for various video scene graph benchmarks.

Author: Orion Research Team
Date: October 2025
"""

from .vsgr_loader import VSGRBenchmark, VSGRDataset
from .action_genome_loader import ActionGenomeBenchmark, ActionGenomeDataset

__all__ = [
    "VSGRBenchmark",
    "VSGRDataset",
    "ActionGenomeBenchmark",
    "ActionGenomeDataset",
]
