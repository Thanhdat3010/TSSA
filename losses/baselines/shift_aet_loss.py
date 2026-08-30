"""
Shift-AET (Alignment-Enhanced Transformer, Chen et al., EMNLP 2020)
Aligns representations at step i+1 when the target word has entered the decoder as an input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ShiftAETLoss(nn.Module):
    def __init__(self, hidden_dim: int = 768, alignment_dim: int = 128):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, alignment_dim)
        self.k_proj = nn.Linear(hidden_dim, alignment_dim)
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, decoder_hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor,
                target_align_matrix: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            decoder_hidden_states: [B, T, D]
            encoder_hidden_states: [B, S, D]
            target_align_matrix: [B, S, T] Ground-truth word alignment
            mask: Optional [B, T-1, S] mask
        """
        # Shift step: take decoder states from step 1 onward [B, T-1, D]
        if decoder_hidden_states.size(1) > 1:
            shifted_dec_states = decoder_hidden_states[:, 1:, :]
        else:
            shifted_dec_states = decoder_hidden_states

        queries = self.q_proj(shifted_dec_states)    # [B, T-1, align_dim]
        keys = self.k_proj(encoder_hidden_states)     # [B, S, align_dim]

        scores = torch.bmm(queries, keys.transpose(1, 2)) / (queries.size(-1) ** 0.5) # [B, T-1, S]
        align_probs = torch.sigmoid(scores)

        # Transpose target_align_matrix from [B, S, T] to [B, T-1, S]
        if target_align_matrix.size(1) != align_probs.size(1):
            align_target = target_align_matrix.transpose(1, 2) # [B, T, S]
        else:
            align_target = target_align_matrix

        T_min = min(align_probs.size(1), align_target.size(1))
        S_min = min(align_probs.size(2), align_target.size(2))

        probs_cut = align_probs[:, :T_min, :S_min]
        target_cut = align_target[:, :T_min, :S_min].float()

        loss = self.bce_loss(probs_cut, target_cut)

        if mask is not None:
            mask_cut = mask[:, :T_min, :S_min].float()
            loss = (loss * mask_cut).sum() / (mask_cut.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss
