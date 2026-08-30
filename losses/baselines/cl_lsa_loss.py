"""
Cross-Lingual InfoNCE (CL-LSA / Contrastive Alignment, ACL 2024 / EMNLP 2023)
Aligns bilingual token and sentence features with in-batch InfoNCE contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossLingualInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, src_features: torch.Tensor, tgt_features: torch.Tensor,
                align_matrix: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src_features: [B, S, D] or [B, D]
            tgt_features: [B, T, D] or [B, D]
            align_matrix: Optional [B, S, T]
        """
        if src_features.dim() == 2 and tgt_features.dim() == 2:
            # Sentence-level InfoNCE
            s_norm = F.normalize(src_features, p=2, dim=-1)
            t_norm = F.normalize(tgt_features, p=2, dim=-1)
            sim_matrix = torch.matmul(s_norm, t_norm.t()) / self.temperature
            labels = torch.arange(src_features.size(0), device=src_features.device)
            loss = 0.5 * (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.t(), labels))
            return loss

        # Token-level InfoNCE
        src_norm = F.normalize(src_features, p=2, dim=-1)
        tgt_norm = F.normalize(tgt_features, p=2, dim=-1)

        B, S, D = src_norm.shape
        _, T, _ = tgt_norm.shape

        sim_matrix = torch.bmm(src_norm, tgt_norm.transpose(1, 2)) / self.temperature # [B, S, T]
        log_prob = F.log_softmax(sim_matrix, dim=-1)

        if align_matrix is not None:
            S_min = min(log_prob.size(1), align_matrix.size(1))
            T_min = min(log_prob.size(2), align_matrix.size(2))
            pos_mask = align_matrix[:, :S_min, :T_min].float()
            log_prob_cut = log_prob[:, :S_min, :T_min]
            loss = -(log_prob_cut * pos_mask).sum() / (pos_mask.sum() + 1e-8)
        else:
            loss = -log_prob.mean()

        return loss
