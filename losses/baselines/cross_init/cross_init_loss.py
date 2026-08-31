"""
CrossInit Loss (Ai & Huang, Findings of ACL 2024)
CrossInit: Subword Alignment for Cross-Lingual Transfer

Forces source subwords to align with target subwords via an orthogonal linear map W
such that W^T W = I (Orthogonality constraint) and minimizes MSE alignment distance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossInitLoss(nn.Module):
    def __init__(self, embed_dim: int = 1024, lambda_orth: float = 0.01):
        super().__init__()
        self.embed_dim = embed_dim
        self.lambda_orth = lambda_orth
        self.linear_map = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.orthogonal_(self.linear_map.weight)

    def forward(self, src_embeddings: torch.Tensor, tgt_embeddings: torch.Tensor,
                align_matrix: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src_embeddings: [B, S, D] Source subword representations
            tgt_embeddings: [B, T, D] Target subword representations
            align_matrix: [B, S, T] or [B, T, S] Subword alignment prior matrix A
        Returns:
            Scalar CrossInit loss = L_align + lambda_orth * L_orth
        """
        if align_matrix.size(1) == tgt_embeddings.size(1) and align_matrix.size(2) == src_embeddings.size(1):
            A = align_matrix.transpose(1, 2) # [B, T, S] -> [B, S, T]
        else:
            A = align_matrix # [B, S, T]

        S_min = min(src_embeddings.size(1), A.size(1))
        T_min = min(tgt_embeddings.size(1), A.size(2))

        src_cut = src_embeddings[:, :S_min, :]
        tgt_cut = tgt_embeddings[:, :T_min, :]
        A_cut = A[:, :S_min, :T_min].float().detach()

        # 1. Project source subwords through orthogonal linear map W
        proj_src = self.linear_map(src_cut) # [B, S_min, D]

        # 2. Target alignment barycenter: T_bar = A * T_emb / sum(A)
        row_sum = A_cut.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        A_norm = A_cut / row_sum
        tgt_barycenter = torch.bmm(A_norm, tgt_cut) # [B, S_min, D]

        # Alignment MSE Loss
        loss_align = F.mse_loss(proj_src, tgt_barycenter)

        # 3. Orthogonality Constraint: || W^T W - I ||_F^2
        W = self.linear_map.weight
        I = torch.eye(self.embed_dim, device=W.device)
        loss_orth = F.mse_loss(torch.mm(W.t(), W), I)

        total_loss = loss_align + (self.lambda_orth * loss_orth)
        return total_loss
