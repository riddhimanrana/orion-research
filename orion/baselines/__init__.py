"""
Baseline implementations for comparing against Orion

Includes:
- VoT (Video-of-Thought) style: LLM-only caption reasoning
- Other future baselines for comparison
"""

from .vot_baseline import VOTBaseline, VOTConfig

__all__ = ['VOTBaseline', 'VOTConfig']
