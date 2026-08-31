"""
Residual Cross-Lingual Semantic Projector (TSSA 2.0)
Maps source sentence representations into the teacher semantic manifold
via a non-linear residual projection without modifying the internal
syntactic representations of the student encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualSemanticProjector(nn.Module):
    def __init__(self, d_model: int = 1024, hidden_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] or [B, S, D] Source token/sentence representations.
        Returns:
            Projected representation with residual connection [B, D] or [B, S, D].
        """
        return self.layer_norm(x + self.net(x))
