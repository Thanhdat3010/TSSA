"""
Guided Cross-Attention Loss (Chen et al., ACL 2016 / OpenNMT-py)
Directly supervises the Decoder Cross-Attention distribution to match
the external word alignment matrix A.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GuidedAttentionLoss(nn.Module):
    def __init__(self, mode: str = "mse", eps: float = 1e-8):
        super().__init__()
        self.mode = mode # "mse" or "ce"
        self.eps = eps

    def forward(self, cross_attentions: tuple, align_matrix: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Args:
            cross_attentions: Tuple of [B, H, T, S] Cross-Attention weights per decoder layer.
            align_matrix: [B, S, T] Ground-truth word alignment matrix from SimAlign.
            layer_idx: Which decoder layer attention to supervise (-1 for topmost).
        Returns:
            Scalar loss tensor.
        """
        if cross_attentions is None or len(cross_attentions) == 0:
            return torch.tensor(0.0, device=align_matrix.device)

        # Select layer attention: [B, H, T, S]
        attn = cross_attentions[layer_idx]
        
        # Average over attention heads: [B, T, S]
        attn_avg = attn.mean(dim=1)
        
        # Transpose align_matrix from [B, S, T] to [B, T, S]
        align_target = align_matrix.transpose(1, 2).detach()

        # Match dimensions if sequence lengths differ
        B, T_a, S_a = align_target.shape
        _, T_m, S_m = attn_avg.shape
        
        T_min = min(T_a, T_m)
        S_min = min(S_a, S_m)
        
        attn_cut = attn_avg[:, :T_min, :S_min]
        align_cut = align_target[:, :T_min, :S_min]

        if self.mode == "ce":
            # Negative log probability on aligned positions
            loss = - torch.sum(align_cut * torch.log(attn_cut + self.eps)) / align_cut.sum().clamp(min=1.0)
        else:
            # Mean Squared Error (Standard OpenNMT default)
            loss = F.mse_loss(attn_cut, align_cut, reduction="mean")

        return loss
