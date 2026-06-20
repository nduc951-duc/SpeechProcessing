"""BiGRU with temporal attention for 100-speaker identification."""

from __future__ import annotations

import torch
from torch import nn


class Attention(nn.Module):
    """Computes one normalized attention score for each time frame."""

    def __init__(self, feature_size: int) -> None:
        super().__init__()
        self.attention = nn.Linear(feature_size, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(sequence), dim=1)
        return torch.sum(weights * sequence, dim=1)


class BiGRUAttention(nn.Module):
    """Checkpoint-compatible model used by ``best_bigru_attention_100.pth``."""

    def __init__(
        self,
        input_size: int = 120,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_speakers: int = 100,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = Attention(hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_speakers),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.bigru(features)
        context = self.attention(sequence)
        return self.classifier(context)
