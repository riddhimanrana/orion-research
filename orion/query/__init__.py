"""
Visual Query System
===================

Query-driven semantic enrichment for processed videos.

Author: Orion Research Team
Date: November 2025
"""

from .index import VideoIndex
from .query_engine import QueryEngine

__all__ = ['VideoIndex', 'QueryEngine']
