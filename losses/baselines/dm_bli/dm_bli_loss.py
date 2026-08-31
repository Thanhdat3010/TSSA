"""
DM-BLI Subspace Loss (ACL 2024 Long Paper)
Bilingual Lexicon Induction via Dynamic Manifold Subspace Alignment

Projects representations into low-dimensional dynamic subspaces (d=64)
and aligns them using normalized manifold cosine similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DMBLISubspaceLoss(nn.Module):
    def __init__(self, hidden_dim: int = 1024, subspace_dim: int = 64, mu: float = 0.5):
        super().__init__()
        self.subspace_src = nn.Sequential(
            nn.Linear(hidden_dim, subspace_dim),
            nn.LayerNorm(subspace_dim)
        )
        self.subspace_tgt = nn.Sequential(
            nn.Linear(hidden_dim, subspace_dim),
            nn.LayerNorm(subspace_dim)
        )
        self.mu = mu

    def forward(self, src_h: torch.Tensor, tgt_h: torch.Tensor,
                align_matrix: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            src_h: [B, S, D]
            tgt_h: [B, T, D]
            align_matrix: [B, S, T] or [B, T, S]
        """
        if align_matrix.size(1) == tgt_h.size(1) and align_matrix.size(2) == src_h.size(1):
            A = align_matrix.transpose(1, 2) # [B, S, T]
        else:
            A = align_matrix

        S_min = min(src_h.size(1), A.size(1))
        T_min = min(tgt_h.size(1), A.size(2))

        src_cut = src_h[:, :S_min, :]
        tgt_cut = tgt_h[:, :T_min, :]
        A_cut = A[:, :S_min, :T_min].float().detach()

        # Project to subspace
        sub_src = F.normalize(self.subspace_src(src_cut), p=2, dim=-1) # [B, S_min, d]
        sub_tgt = F.normalize(self.subspace_tgt(tgt_cut), p=2, dim=-1) # [B, T_min, d]

        # Target subspace barycenter
        row_sum = A_cut.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        A_norm = A_cut / row_sum
        tgt_barycenter = torch.bmm(A_norm, sub_tgt) # [B, S_min, d]

        # Manifold Cosine Distance Loss
        cos_sim = (sub_src * tgt_barycenter).sum(dim=-1)
        loss = (1.0 - cos_sim).mean()

        return self.mu * loss
