"""
Structural Supervision Loss (Li et al., Findings of ACL 2022)
Structural Supervision for Word Alignment and Machine Translation

Directly supervises decoder cross-attention distributions using gold/silver alignment matrices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class StructuralSupervisionLoss(nn.Module):
    def __init__(self, lambda_struct: float = 0.3):
        super().__init__()
        self.lambda_struct = lambda_struct

    def forward(self, cross_attentions: tuple, align_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cross_attentions: Tuple of [B, H, T, S] tensors from decoder layers
            align_matrix: [B, T, S] or [B, S, T] Ground-truth word alignment
        """
        if cross_attentions is None or len(cross_attentions) == 0:
            return torch.tensor(0.0, device=align_matrix.device)

        align_matrix = align_matrix.detach().float()
        
        # Take top decoder layer cross-attentions
        last_attn = cross_attentions[-1] # [B, H, T, S]
        if last_attn is None:
            return torch.tensor(0.0, device=align_matrix.device)

        # Average across heads: [B, T, S]
        attn_avg = last_attn.mean(dim=1)

        # Format align_matrix dimensions
        if align_matrix.size(1) == attn_avg.size(2) and align_matrix.size(2) == attn_avg.size(1):
            A = align_matrix.transpose(1, 2)
        else:
            A = align_matrix

        T_min = min(attn_avg.size(1), A.size(1))
        S_min = min(attn_avg.size(2), A.size(2))

        attn_cut = attn_avg[:, :T_min, :S_min]
        A_cut = A[:, :T_min, :S_min]

        # Row-normalize A
        row_sum = A_cut.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        A_norm = A_cut / row_sum

        loss = F.smooth_l1_loss(attn_cut, A_norm)
        return self.lambda_struct * loss
