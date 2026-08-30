"""
Structural Supervision for Word Alignment and Machine Translation (Li et al., Findings of ACL 2022)
Supervises cross-attention weights with the target-source syntactic adjacency structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class StructuralSupervisionLoss(nn.Module):
    def __init__(self, lambda_struct: float = 0.3):
        super().__init__()
        self.lambda_struct = lambda_struct

    def forward(self, cross_attention_weights: torch.Tensor, syntactic_adj_matrix: torch.Tensor,
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            cross_attention_weights: [B, H, T, S] or Tuple of layers from Decoder.
            syntactic_adj_matrix: [B, S, T] Ground-truth word alignment / syntactic matrix.
            padding_mask: [B, T, S] Optional 2D mask.
        Returns:
            Weighted structural loss scalar.
        """
        if isinstance(cross_attention_weights, (tuple, list)):
            cross_attention_weights = cross_attention_weights[-1] # topmost layer

        # Average over attention heads -> [B, T, S]
        mean_attn = torch.mean(cross_attention_weights, dim=1)

        # Transpose align/adj matrix from [B, S, T] to [B, T, S] if needed
        if syntactic_adj_matrix.size(1) != mean_attn.size(1) and syntactic_adj_matrix.size(2) == mean_attn.size(1):
            adj_target = syntactic_adj_matrix.transpose(1, 2)
        else:
            adj_target = syntactic_adj_matrix

        # Match dimensions
        T_min = min(mean_attn.size(1), adj_target.size(1))
        S_min = min(mean_attn.size(2), adj_target.size(2))

        attn_cut = mean_attn[:, :T_min, :S_min]
        adj_cut = adj_target[:, :T_min, :S_min].float()

        diff = (attn_cut - adj_cut) ** 2

        if padding_mask is not None:
            mask_cut = padding_mask[:, :T_min, :S_min].float()
            diff = diff * mask_cut
            loss = diff.sum() / (mask_cut.sum() + 1e-8)
        else:
            loss = diff.mean()

        return self.lambda_struct * loss
