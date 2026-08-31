"""
CrossInit Loss (Ai & Huang, Findings of ACL 2024)
Zero-shot Cross-lingual Alignment for Embedding Initialization

Exact formulation from Section 2.3, Eq. (1) & Figure 2:
- Positive pairs: Aligned source and target subword embeddings (E_{V_{span}^{L_i}}, E_{V_{span}^{L_j}})
- Negative pairs: Unaligned / out-of-span target subword embeddings (E_{V_{span}^{L_i}}, E_{V_{notin span}^{L_j}})
- Eq. (1): L_CrossInit = - log P(1 | E_{src} E_{tgt}^+) - log P(0 | E_{src} E_{tgt}^-)
          where P(1 | u, v) = sigmoid( u^T v / tau )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossInitLoss(nn.Module):
    def __init__(self, embed_dim: int = 1024, temperature: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.embed_dim = embed_dim
        self.temp = temperature
        self.eps = eps

    def forward(self, src_embeddings: torch.Tensor, tgt_embeddings: torch.Tensor,
                align_matrix: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src_embeddings: [B, S, D] Source subword representations E_{src}
            tgt_embeddings: [B, T, D] Target subword representations E_{tgt}
            align_matrix: [B, S, T] or [B, T, S] Subword alignment prior matrix A
        Returns:
            Scalar CrossInit contrastive loss (Eq. 1)
        """
        if align_matrix.size(1) == tgt_embeddings.size(1) and align_matrix.size(2) == src_embeddings.size(1):
            A = align_matrix.transpose(1, 2) # [B, S, T]
        else:
            A = align_matrix

        S_min = min(src_embeddings.size(1), A.size(1))
        T_min = min(tgt_embeddings.size(1), A.size(2))

        src_cut = F.normalize(src_embeddings[:, :S_min, :], p=2, dim=-1) # [B, S_min, D]
        tgt_cut = F.normalize(tgt_embeddings[:, :T_min, :], p=2, dim=-1) # [B, T_min, D]
        A_cut = A[:, :S_min, :T_min].float().detach()

        # Positive target embeddings: weighted sum according to alignment span (Eq. 1)
        row_sum = A_cut.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        A_norm = A_cut / row_sum
        tgt_pos = torch.bmm(A_norm, tgt_cut) # [B, S_min, D]

        # Negative target embeddings: in-batch roll / out-of-span tokens
        tgt_neg = torch.roll(tgt_cut, shifts=1, dims=0)[:, :S_min, :] # [B, S_min, D]

        # Dot products
        pos_sim = (src_cut * tgt_pos).sum(dim=-1) / self.temp # [B, S_min]
        neg_sim = (src_cut * tgt_neg).sum(dim=-1) / self.temp # [B, S_min]

        # Eq. (1): - log sigmoid(pos_sim) - log (1 - sigmoid(neg_sim)) = - log_sigmoid(pos_sim) - log_sigmoid(-neg_sim)
        loss_pos = - F.logsigmoid(pos_sim)
        loss_neg = - F.logsigmoid(-neg_sim)

        loss = (loss_pos + loss_neg).mean()
        return loss
