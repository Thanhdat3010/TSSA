"""
Shift-AET: Accurate Word Alignment Induced from Transformers (Chen et al., EMNLP 2020)
Aligns source tokens with shifted decoder states s_{t+1} (after target word y_t has entered the decoder).

Formulation from Section 3 of Chen et al. (EMNLP 2020):
- Shift step: Decoder state s_{t+1} at step t+1 captures target word y_t context
- Alignment score: alpha_{ts} = Sigmoid( (W_d s_{t+1})^T (W_e h_s^E) / sqrt(d_a) )
- Supervised Loss: Binary Cross-Entropy against alignment prior matrix A
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ShiftAETLoss(nn.Module):
    def __init__(self, hidden_dim: int = 1024, alignment_dim: int = 128):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, alignment_dim)
        self.k_proj = nn.Linear(hidden_dim, alignment_dim)
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, decoder_hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor,
                target_align_matrix: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            decoder_hidden_states: [B, T, D] Student decoder states
            encoder_hidden_states: [B, S, D] Student encoder states
            target_align_matrix: [B, T, S] or [B, S, T] Target-Source alignment matrix
            mask: Optional [B, T-1, S] sequence mask
        Returns:
            Scalar Shift-AET alignment loss
        """
        # 1. Shift step: take decoder states s_{t+1} (step 1 onward) [B, T-1, D]
        if decoder_hidden_states.size(1) > 1:
            shifted_dec_states = decoder_hidden_states[:, 1:, :] # s_{1}, s_{2}, ..., s_{T-1}
        else:
            shifted_dec_states = decoder_hidden_states

        # 2. Linear alignment projections into d_a = 128
        queries = self.q_proj(shifted_dec_states) # [B, T-1, d_a]
        keys = self.k_proj(encoder_hidden_states)  # [B, S, d_a]

        # 3. Scaled dot-product alignment probabilities alpha_{ts}
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (queries.size(-1) ** 0.5) # [B, T-1, S]
        align_probs = torch.sigmoid(scores) # [B, T-1, S]

        # 4. Dimension alignment for target matrix [B, T, S]
        if target_align_matrix.size(1) == align_probs.size(2) and target_align_matrix.size(2) != align_probs.size(2):
            align_target = target_align_matrix.transpose(1, 2) # [B, S, T] -> [B, T, S]
        else:
            align_target = target_align_matrix # [B, T, S]

        # Slicing to match [B, T-1, S]
        T_curr = min(align_probs.size(1), align_target.size(1))
        S_curr = min(align_probs.size(2), align_target.size(2))

        probs_cut = align_probs[:, :T_curr, :S_curr]
        target_cut = align_target[:, :T_curr, :S_curr].float().detach()

        # 5. Supervised BCE Alignment Loss
        loss = self.bce_loss(probs_cut, target_cut)

        if mask is not None:
            mask_cut = mask[:, :T_curr, :S_curr].float()
            loss = (loss * mask_cut).sum() / (mask_cut.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss
