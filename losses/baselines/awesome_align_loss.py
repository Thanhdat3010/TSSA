"""
AWESOME-align Loss (Dou & Neubig, EACL 2021)
Combines Self-Optimization (SO) bidirectional contrastive loss
and Consistency Optimization (CO) masked MSE loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AwesomeAlignLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, lambda_co: float = 1.0):
        super().__init__()
        self.temp = temperature
        self.lambda_co = lambda_co
        self.mse_loss = nn.MSELoss()

    def forward(self, src_h: torch.Tensor, tgt_h: torch.Tensor,
                align_matrix: torch.Tensor,
                src_h_masked: torch.Tensor = None, tgt_h_masked: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src_h: [B, S, D] Source contextual embeddings
            tgt_h: [B, T, D] Target contextual embeddings
            align_matrix: [B, S, T] Ground-truth / posterior alignment
            src_h_masked: Optional [B, S, D] Masked representations for CO
            tgt_h_masked: Optional [B, T, D] Masked representations for CO
        """
        src_norm = F.normalize(src_h, p=2, dim=-1)
        tgt_norm = F.normalize(tgt_h, p=2, dim=-1)

        # Cosine similarity matrix [B, S, T]
        sim_matrix = torch.bmm(src_norm, tgt_norm.transpose(1, 2)) / self.temp

        S_min = min(sim_matrix.size(1), align_matrix.size(1))
        T_min = min(sim_matrix.size(2), align_matrix.size(2))

        sim_cut = sim_matrix[:, :S_min, :T_min]
        align_cut = align_matrix[:, :S_min, :T_min].float()

        # Direction 1: Source to Target
        log_prob_src = F.log_softmax(sim_cut, dim=-1)
        loss_src_to_tgt = -(log_prob_src * align_cut).sum() / (align_cut.sum() + 1e-8)

        # Direction 2: Target to Source
        log_prob_tgt = F.log_softmax(sim_cut.transpose(1, 2), dim=-1)
        loss_tgt_to_src = -(log_prob_tgt * align_cut.transpose(1, 2)).sum() / (align_cut.sum() + 1e-8)

        loss_so = 0.5 * (loss_src_to_tgt + loss_tgt_to_src)

        # Consistency Optimization (if masked views are provided)
        if src_h_masked is not None and tgt_h_masked is not None:
            loss_co = self.mse_loss(src_h, src_h_masked) + self.mse_loss(tgt_h, tgt_h_masked)
        else:
            loss_co = torch.tensor(0.0, device=src_h.device)

        return loss_so + (self.lambda_co * loss_co)
