"""
CrossInit Loss (Ai & Huang, Findings of ACL 2024)
Rotates and aligns cross-lingual token embeddings using an orthogonal projection matrix W
with Frobenius norm regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossInitLoss(nn.Module):
    def __init__(self, embed_dim: int = 768, lambda_orth: float = 0.01):
        super().__init__()
        self.embed_dim = embed_dim
        self.lambda_orth = lambda_orth
        self.W = nn.Parameter(torch.eye(embed_dim))

    def forward(self, src_embeddings: torch.Tensor, tgt_embeddings: torch.Tensor,
                alignment_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src_embeddings: [B, S, D]
            tgt_embeddings: [B, T, D]
            alignment_matrix: [B, S, T]
        """
        projected_src = torch.matmul(src_embeddings, self.W)

        norm_src = F.normalize(projected_src, p=2, dim=-1)
        norm_tgt = F.normalize(tgt_embeddings, p=2, dim=-1)

        # Cosine similarity matrix [B, S, T]
        cos_sim = torch.bmm(norm_src, norm_tgt.transpose(1, 2))
        cosine_distance = 1.0 - cos_sim

        # Match dimensions if needed
        S_min = min(cosine_distance.size(1), alignment_matrix.size(1))
        T_min = min(cosine_distance.size(2), alignment_matrix.size(2))

        dist_cut = cosine_distance[:, :S_min, :T_min]
        align_cut = alignment_matrix[:, :S_min, :T_min].float()

        aligned_loss = (dist_cut * align_cut).sum() / (align_cut.sum() + 1e-8)

        wt_w = torch.matmul(self.W.t(), self.W)
        identity = torch.eye(self.embed_dim, device=self.W.device)
        orth_loss = torch.norm(wt_w - identity, p='fro') ** 2

        total_loss = aligned_loss + (self.lambda_orth * orth_loss)
        return total_loss
