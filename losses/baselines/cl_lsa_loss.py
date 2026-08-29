"""
Cross-Lingual InfoNCE Loss (CL-LSA / Contrastive NMT Baseline, ACL 2024)
Sentence-level representation alignment with bidirectional InfoNCE loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossLingualInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, src_sent_vec: torch.Tensor, tgt_sent_vec: torch.Tensor) -> torch.Tensor:
        """
        Bidirectional contrastive loss: L = 0.5 * (L(src->tgt) + L(tgt->src))
        """
        s_norm = F.normalize(src_sent_vec, dim=-1)
        t_norm = F.normalize(tgt_sent_vec, dim=-1)

        sim_matrix = torch.matmul(s_norm, t_norm.transpose(0, 1)) / self.temperature
        batch_size = src_sent_vec.size(0)
        labels = torch.arange(batch_size, device=src_sent_vec.device)

        loss_s2t = self.cross_entropy(sim_matrix, labels)
        loss_t2s = self.cross_entropy(sim_matrix.transpose(0, 1), labels)

        return 0.5 * (loss_s2t + loss_t2s)
