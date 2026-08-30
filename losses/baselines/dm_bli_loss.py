"""
DM-BLI Subspace Loss (Dynamic Multiple Subspaces Alignment, ACL 2024 Long)
Aligns representations within low-dimensional Grassmannian projection subspaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DMBLISubspaceLoss(nn.Module):
    def __init__(self, hidden_dim: int = 768, subspace_dim: int = 64, mu: float = 0.5):
        super().__init__()
        self.subspace_dim = subspace_dim
        self.mu = mu
        self.proj_s = nn.Linear(hidden_dim, subspace_dim, bias=False)
        self.proj_t = nn.Linear(hidden_dim, subspace_dim, bias=False)

    def forward(self, src_h: torch.Tensor, tgt_h: torch.Tensor, align_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src_h: [B, S, D]
            tgt_h: [B, T, D]
            align_matrix: [B, S, T]
        """
        # Compute orthonormal basis using QR decomposition
        u_s, _ = torch.linalg.qr(self.proj_s.weight) # [D, k]
        u_t, _ = torch.linalg.qr(self.proj_t.weight) # [D, k]

        # Projection operators: P = U U^T [D, D]
        p_s = torch.matmul(u_s, u_s.t())
        p_t = torch.matmul(u_t, u_t.t())

        # Subspace distance on Grassmannian manifold
        subspace_dist = 0.5 * (torch.norm(p_s - p_t, p='fro') ** 2)

        # Project representations into subspaces
        p_src_h = torch.matmul(src_h, p_s) # [B, S, D]
        p_tgt_h = torch.matmul(tgt_h, p_t) # [B, T, D]

        # Pairwise distance between projected instances
        dist_matrix = torch.cdist(p_src_h, p_tgt_h, p=2) ** 2 # [B, S, T]

        S_min = min(dist_matrix.size(1), align_matrix.size(1))
        T_min = min(dist_matrix.size(2), align_matrix.size(2))

        dist_cut = dist_matrix[:, :S_min, :T_min]
        align_cut = align_matrix[:, :S_min, :T_min].float()

        instance_dist = (dist_cut * align_cut).sum() / (align_cut.sum() + 1e-8)

        return subspace_dist + (self.mu * instance_dist)
