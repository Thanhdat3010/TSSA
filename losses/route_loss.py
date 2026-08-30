"""
RouteLoss: Head-Wise Router Supervision Loss (TSSA)
Supervises the Cross-Attention Head Gating MLP to route information
through reliable Anchor Heads via Normalized Soft Binary Cross-Entropy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RouteLoss(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, router_gates: torch.Tensor, teacher_target: torch.Tensor, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            router_gates: [B, L, H, T, 1] or [B, L, H, T] Gate activations in [0, 1].
            teacher_target: Same shape as router_gates, continuous target in [0, 1].
            tgt_mask: [B, T] Target sequence valid mask.
        Returns:
            Properly normalized scalar Soft BCE loss in [0, 1].
        """
        teacher_target = teacher_target.detach()

        # Numerical clamping to prevent log(0)
        gates_clamped = router_gates.clamp(min=self.eps, max=1.0 - self.eps)
        
        # Element-wise Binary Cross Entropy with soft labels
        bce = - (teacher_target * torch.log(gates_clamped) + 
                 (1.0 - teacher_target) * torch.log(1.0 - gates_clamped))

        if tgt_mask is not None:
            mask = tgt_mask.float()
            while mask.dim() < bce.dim():
                mask = mask.unsqueeze(1)
            # Expand mask to full tensor dimensions to ensure strict per-element normalization
            mask_expanded = mask.expand_as(bce)
            return (bce * mask_expanded).sum() / mask_expanded.sum().clamp(min=1.0)

        return bce.mean()
