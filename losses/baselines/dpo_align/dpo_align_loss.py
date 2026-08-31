"""
Alignment as Preference (DPO-Align, Wu et al., EMNLP 2024)
Formulates cross-lingual semantic alignment using direct preference optimization margins.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignmentDPOLoss(nn.Module):
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(self, src_h: torch.Tensor, tgt_h: torch.Tensor,
                negative_h: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Args:
            src_h: [B, S, D] Contextual source representations
            tgt_h: [B, T, D] Positive target representations
            negative_h: Optional [B, T, D] Negative target representations
        """
        src_sent = src_h.mean(dim=1)
        tgt_sent = tgt_h.mean(dim=1)

        if negative_h is not None:
            neg_sent = negative_h.mean(dim=1)
            pos_sim = F.cosine_similarity(src_sent, tgt_sent)
            neg_sim = F.cosine_similarity(src_sent, neg_sent)
            # DPO Margin Loss
            loss = - F.logsigmoid(self.beta * (pos_sim - neg_sim)).mean()
        else:
            # In standard batch forward, uses in-batch negative roll
            pos_sim = F.cosine_similarity(src_sent, tgt_sent)
            neg_sent = torch.roll(tgt_sent, shifts=1, dims=0)
            neg_sim = F.cosine_similarity(src_sent, neg_sent)
            loss = - F.logsigmoid(self.beta * (pos_sim - neg_sim)).mean()

        return loss
