"""
AWESOME-align Loss (Dou & Neubig, EACL 2021)
Symmetric Embedding Alignment Loss aligning source and target representation spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AwesomeAlignLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, src_embeddings: torch.Tensor, tgt_embeddings: torch.Tensor, align_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src_embeddings: [B, S, D] Source token embeddings/states.
            tgt_embeddings: [B, T, D] Target token embeddings/states.
            align_matrix: [B, S, T] Ground-truth word alignments.
        """
        # Cosine similarity matrix: [B, S, T]
        src_norm = F.normalize(src_embeddings, dim=-1)
        tgt_norm = F.normalize(tgt_embeddings, dim=-1)
        sim_matrix = torch.bmm(src_norm, tgt_norm.transpose(1, 2)) / self.temperature

        # Softmax forward (over target tokens) and backward (over source tokens)
        p_fwd = F.softmax(sim_matrix, dim=-1)             # [B, S, T]
        p_bwd = F.softmax(sim_matrix.transpose(1, 2), dim=-1) # [B, T, S]

        align_fwd = align_matrix.detach()
        align_bwd = align_matrix.transpose(1, 2).detach()

        # Symmetric alignment loss
        loss_fwd = - torch.sum(align_fwd * torch.log(p_fwd + self.eps)) / align_fwd.sum().clamp(min=1.0)
        loss_bwd = - torch.sum(align_bwd * torch.log(p_bwd + self.eps)) / align_bwd.sum().clamp(min=1.0)

        return 0.5 * (loss_fwd + loss_bwd)
