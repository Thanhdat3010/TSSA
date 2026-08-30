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

        # Squeeze trailing singleton dimension if present: [B, L, H, T, 1] -> [B, L, H, T]
        if router_gates.dim() == 5 and router_gates.size(-1) == 1:
            router_gates = router_gates.squeeze(-1)
        if teacher_target.dim() == 5 and teacher_target.size(-1) == 1:
            teacher_target = teacher_target.squeeze(-1)

        # Numerical clamping to prevent log(0)
        gates_clamped = router_gates.clamp(min=self.eps, max=1.0 - self.eps)
        
        # Element-wise Binary Cross Entropy with soft labels: [B, L, H, T]
        bce = - (teacher_target * torch.log(gates_clamped) + 
                 (1.0 - teacher_target) * torch.log(1.0 - gates_clamped))

        if tgt_mask is not None:
            # tgt_mask is [B, T]. Reshape to [B, 1, 1, T] matching bce [B, L, H, T]
            B = tgt_mask.size(0)
            T_mask = tgt_mask.size(-1)
            mask = tgt_mask.float().view(B, 1, 1, T_mask)
            
            T_min = min(bce.size(-1), mask.size(-1))
            bce_cut = bce[..., :T_min]
            mask_cut = mask[..., :T_min]
            mask_expanded = mask_cut.expand_as(bce_cut)
            return (bce_cut * mask_expanded).sum() / mask_expanded.sum().clamp(min=1.0)

        return bce.mean()
