"""
AWESOME-align Loss (Dou & Neubig, EACL 2021 / CMU)
Word Alignment by Fine-tuning Embeddings on Parallel Corpora

Implements exact objectives from Section 2.2:
1. Self-training Objective (L_SO, Eq. 4): Bidirectional alignment probability maximization
2. Consistency Optimization (L_CO, Eq. 6): Negative trace of alignment matrix product -trace(S_xy^T S_yx) / min(m, n)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AwesomeAlignLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, lambda_co: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.temp = temperature
        self.lambda_co = lambda_co
        self.eps = eps

    def forward(self, src_h: torch.Tensor, tgt_h: torch.Tensor,
                align_matrix: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            src_h: [B, S, D] Source contextual embeddings H_x
            tgt_h: [B, T, D] Target contextual embeddings H_y
            align_matrix: [B, S, T] Subword alignment prior matrix A
        Returns:
            Scalar tensor: L_SO + beta * L_CO
        """
        # Stop-gradient on alignment matrix
        align_matrix = align_matrix.detach().float()

        src_norm = F.normalize(src_h, p=2, dim=-1) # [B, S, D]
        tgt_norm = F.normalize(tgt_h, p=2, dim=-1) # [B, T, D]

        # Dot-product similarity matrix [B, S, T]
        sim_matrix = torch.bmm(src_norm, tgt_norm.transpose(1, 2)) / self.temp

        B, S, T = sim_matrix.shape
        S_min = min(S, align_matrix.size(1))
        T_min = min(T, align_matrix.size(2))

        sim_cut = sim_matrix[:, :S_min, :T_min]
        A_cut = align_matrix[:, :S_min, :T_min] # [B, S_min, T_min]

        # 1. Softmax Alignment Probability Matrices S_xy and S_yx
        # S_xy: [B, S, T] Softmax along target dimension (Source to Target)
        S_xy = F.softmax(sim_cut, dim=-1)
        # S_yx: [B, T, S] Softmax along source dimension (Target to Source)
        S_yx = F.softmax(sim_cut.transpose(1, 2), dim=-1)

        # 2. Self-training Objective (Eq. 4 in Dou & Neubig EACL 2021)
        # L_SO = - sum A_ij * 0.5 * (S_xy_ij / T + S_yx_ji / S)
        S_yx_t = S_yx.transpose(1, 2) # [B, S_min, T_min]
        so_term = 0.5 * ((S_xy / max(1, T_min)) + (S_yx_t / max(1, S_min)))
        loss_so = - (A_cut * so_term).sum(dim=(-2, -1)).mean()

        # 3. Consistency Optimization (Eq. 6 in Dou & Neubig EACL 2021)
        # L_CO = - trace(S_xy^T S_yx) / min(S, T) = - sum_ij (S_xy_ij * S_yx_ji) / min(S, T)
        trace_val = (S_xy * S_yx_t).sum(dim=(-2, -1)) # [B]
        loss_co = - (trace_val / max(1, min(S_min, T_min))).mean()

        total_loss = loss_so + (self.lambda_co * loss_co)
        return total_loss
