"""
Cross-Lingual InfoNCE Contrastive Loss (CL-LSA, ACL 2024 / EMNLP 2023)
Aligns subword representations using token-level in-batch InfoNCE contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossLingualInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, src_h: torch.Tensor, tgt_h: torch.Tensor,
                align_matrix: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Args:
            src_h: [B, S, D] Source contextual embeddings
            tgt_h: [B, T, D] Target contextual embeddings
            align_matrix: Optional [B, S, T] alignment matrix
        """
        src_norm = F.normalize(src_h, p=2, dim=-1) # [B, S, D]
        tgt_norm = F.normalize(tgt_h, p=2, dim=-1) # [B, T, D]

        # [B, S, D] x [B, D, T] -> [B, S, T]
        logits = torch.bmm(src_norm, tgt_norm.transpose(1, 2)) / self.temperature

        B, S, T = logits.shape
        
        if align_matrix is not None:
            if align_matrix.size(1) == T and align_matrix.size(2) == S:
                A = align_matrix.transpose(1, 2)
            else:
                A = align_matrix
            S_min = min(S, A.size(1))
            T_min = min(T, A.size(2))

            logits_cut = logits[:, :S_min, :T_min]
            A_cut = A[:, :S_min, :T_min].float().detach()

            # Target position with highest alignment weight
            target_labels = torch.argmax(A_cut, dim=-1) # [B, S_min]
            loss = self.cross_entropy(logits_cut.view(-1, T_min), target_labels.view(-1))
        else:
            # Fallback to diagonal
            min_len = min(S, T)
            target_labels = torch.arange(min_len, device=src_h.device).unsqueeze(0).expand(B, min_len)
            loss = self.cross_entropy(logits[:, :min_len, :min_len].reshape(-1, min_len), target_labels.reshape(-1))

        return loss
