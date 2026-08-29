"""
RouteLoss: Head-Wise Router Supervision Loss (TSSA)
Supervises the Cross-Attention Head Gating MLP to route information
through reliable Anchor Heads via Soft Binary Cross-Entropy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RouteLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, router_gates: torch.Tensor, teacher_target: torch.Tensor, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            router_gates: [B, L, H, T] or [B, T, L, H] Gate activations in [0, 1].
            teacher_target: Same shape as router_gates, continuous target in [0, 1].
            tgt_mask: [B, T] Target sequence valid mask.
        Returns:
            Scalar Soft BCE loss.
        """
        # Ensure stop-gradient on target supervision
        teacher_target = teacher_target.detach()

        # Compute element-wise Binary Cross Entropy with soft labels
        bce = - (teacher_target * torch.log(router_gates + self.eps) + 
                 (1.0 - teacher_target) * torch.log(1.0 - router_gates + self.eps))

        if tgt_mask is not None:
            # Reshape tgt_mask to match bce dimensions [B, 1, 1, T]
            while tgt_mask.dim() < bce.dim():
                tgt_mask = tgt_mask.unsqueeze(1)
            bce = bce * tgt_mask.float()
            return bce.sum() / tgt_mask.sum().clamp(min=1.0)
            
        return bce.mean()
