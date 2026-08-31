"""
Shift-AET: Accurate Word Alignment Induced from Transformers (Chen et al., EMNLP 2020)
Word alignment induction method using an Alignment-Enhanced Transformer (AET).

Exact formulation from Section 3.2, Eq. (5) & Eq. (6) and Figure 2:
- Inputs: Decoder state z_i^{l_b} at step i (corresponding to target token y_{i-1}) and Encoder outputs h
- Eq. (5): S_{i-1} = (1/N) * sum_{n=1}^N Softmax( (z_i G_n^Q) (h G_n^K)^T / sqrt(d_k) )
- Eq. (6): L_a = - (1/|y|) * sum_{i=1}^{|y|} sum_{j=1}^{|x|} (A_hat_{i,j}^P * log S_{i,j})
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ShiftAETLoss(nn.Module):
    def __init__(self, hidden_dim: int = 1024, n_heads: int = 16, eps: float = 1e-8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads # d_k = 64
        self.eps = eps

        # Multi-Head Key and Query Projections (G_n^Q, G_n^K)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, decoder_hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor,
                target_align_matrix: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            decoder_hidden_states: [B, T, D] Decoder hidden states z_i
            encoder_hidden_states: [B, S, D] Encoder hidden states h
            target_align_matrix: [B, T, S] or [B, S, T] Reference alignment matrix A
        Returns:
            Scalar alignment loss L_a (Eq. 6)
        """
        # 1. Shift step: Take decoder states from step 1 onward (z_1, z_2, ..., z_T)
        # Corresponding to input target tokens y_0, y_1, ..., y_{T-2}
        if decoder_hidden_states.size(1) > 1:
            shifted_dec_states = decoder_hidden_states[:, 1:, :] # [B, T-1, D]
        else:
            shifted_dec_states = decoder_hidden_states

        B, T_shift, D = shifted_dec_states.shape
        S = encoder_hidden_states.size(1)
        H = self.n_heads
        d_k = self.head_dim

        # 2. Multi-Head Projections (Eq. 5)
        # q: [B, H, T_shift, d_k], k: [B, H, S, d_k]
        q = self.q_proj(shifted_dec_states).view(B, T_shift, H, d_k).transpose(1, 2)
        k = self.k_proj(encoder_hidden_states).view(B, S, H, d_k).transpose(1, 2)

        # Scaled dot product across source tokens [B, H, T_shift, S]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        attn_heads = F.softmax(scores, dim=-1) # Softmax along source dimension

        # Average across all N heads -> S_{i-1} [B, T_shift, S] (Eq. 5)
        S_matrix = attn_heads.mean(dim=1)

        # 3. Format Reference Alignment Matrix A_hat^P [B, T_shift, S]
        if target_align_matrix.size(1) == S and target_align_matrix.size(2) != S:
            align_ref = target_align_matrix.transpose(1, 2) # [B, S, T] -> [B, T, S]
        else:
            align_ref = target_align_matrix

        T_curr = min(T_shift, align_ref.size(1))
        S_curr = min(S, align_ref.size(2))

        S_cut = S_matrix[:, :T_curr, :S_curr] # [B, T_curr, S_curr]
        A_cut = align_ref[:, :T_curr, :S_curr].float().detach()

        # Row-normalize A_hat^P for target tokens aligned to at least one source token (Footnote 2)
        row_sum = A_cut.sum(dim=-1, keepdim=True) # [B, T_curr, 1]
        has_align = (row_sum > 0).float()
        A_norm = (A_cut / row_sum.clamp(min=1.0)) * has_align

        # 4. Supervised Cross-Entropy Alignment Loss L_a (Eq. 6)
        # L_a = - (1/|y|) sum (A_hat^P * log S)
        log_S = torch.log(S_cut + self.eps)
        loss_per_sent = - (A_norm * log_S).sum(dim=-1) # sum over source tokens: [B, T_curr]
        
        # Average over active target sequence length |y|
        valid_targets = has_align.sum(dim=1).clamp(min=1.0) # [B, 1]
        loss_a = (loss_per_sent.sum(dim=1, keepdim=True) / valid_targets).mean()

        return loss_a
