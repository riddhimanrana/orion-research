"""Lightweight baseline models for VidSGG experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def build_pair_feature(
    boxes: np.ndarray,
    labels: np.ndarray,
    subject_idx: int,
    object_idx: int,
) -> np.ndarray:
    """Concatenate subject/object boxes + labels into a flat feature vector."""

    if subject_idx >= len(boxes) or object_idx >= len(boxes):
        raise IndexError("Pair indices out of range")
    subj_box = boxes[subject_idx]
    obj_box = boxes[object_idx]
    subj_label = labels[subject_idx]
    obj_label = labels[object_idx]
    feature = np.concatenate([
        subj_box.astype(np.float32) / 1000.0,
        obj_box.astype(np.float32) / 1000.0,
        np.array([subj_label, obj_label], dtype=np.float32) / 100.0,
    ])
    return feature


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class PairwiseRelationModel(nn.Module):
    """Simple MLP over handcrafted pair features."""

    def __init__(
        self,
        feature_dim: int,
        num_predicates: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_predicates),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)


class TemporalRelationForecaster(nn.Module):
    """GRU-based forecaster for Scene Graph Anticipation."""

    def __init__(
        self,
        feature_dim: int,
        num_predicates: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, num_predicates)

    def forward(self, sequence_features: torch.Tensor) -> torch.Tensor:
        """sequence_features: [B, T, D]; returns logits [B, num_predicates]."""

        _, hidden = self.rnn(sequence_features)
        last_hidden = hidden[-1]
        return self.head(last_hidden)


__all__ = [
    "build_pair_feature",
    "PairwiseRelationModel",
    "TemporalRelationForecaster",
]
