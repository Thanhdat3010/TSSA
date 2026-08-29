"""
Joint-Align Loss (Garg et al., EMNLP 2019 / Fairseq)
Computes alignment logits between Decoder representations and Encoder representations
and optimizes with Binary Cross-Entropy with Logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class JointAlignHead(nn.Module):
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_k = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, dec_hidden: torch.Tensor, enc_hidden: torch.Tensor) -> torch.Tensor:
        """
        dec_hidden: [B, T, D]
        enc_hidden: [B, S, D]
        returns logits: [B, T, S]
        """
        q = self.proj_q(dec_hidden) # [B, T, D]
        k = self.proj_k(enc_hidden) # [B, S, D]
        logits = torch.bmm(q, k.transpose(1, 2)) / (dec_hidden.size(-1) ** 0.5) # [B, T, S]
        return logits

class JointAlignLoss(nn.Module):
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.align_head = JointAlignHead(embed_dim)

    def forward(self, dec_hidden: torch.Tensor, enc_hidden: torch.Tensor, align_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dec_hidden: [B, T, D] Decoder top layer states.
            enc_hidden: [B, S, D] Encoder top layer states.
            align_matrix: [B, S, T] Ground-truth word alignments.
        """
        logits = self.align_head(dec_hidden, enc_hidden) # [B, T, S]
        align_target = align_matrix.transpose(1, 2).detach() # [B, T, S]

        # Match dimensions
        T_min = min(logits.size(1), align_target.size(1))
        S_min = min(logits.size(2), align_target.size(2))

        logits_cut = logits[:, :T_min, :S_min]
        target_cut = align_target[:, :T_min, :S_min]

        loss = F.binary_cross_entropy_with_logits(logits_cut, target_cut, reduction="mean")
        return loss
