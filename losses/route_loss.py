"""
RouteLoss: Head-Wise Router Supervision Loss (TSSA 2.1)
Supervises the Cross-Attention Head Gating MLP to specialize a subset of Anchor Heads (~25%)
via Capacity Budget Penalty and Decisive Binarization Entropy, avoiding head over-constraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RouteLoss(nn.Module):
    def __init__(self, target_budget: float = 0.333, target_usage: float = None, entropy_weight: float = 0.1, eps: float = 1e-7, **kwargs):
        super().__init__()
        self.target_budget = target_usage if target_usage is not None else target_budget
        self.entropy_weight = entropy_weight
        self.eps = eps

    def forward(self, router_gates: torch.Tensor, teacher_target: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            router_gates: [B, L, H, T, 1] or [B, L, H, T] Gate activations in [0, 1].
            teacher_target: Optional soft target tensor for backward compatibility.
            tgt_mask: [B, T] Target sequence valid mask.
        Returns:
            Scalar Head Capacity Budget & Sparsity loss.
        """
        # Squeeze trailing singleton dimension if present: [B, L, H, T, 1] -> [B, L, H, T]
        if router_gates.dim() == 5 and router_gates.size(-1) == 1:
            router_gates = router_gates.squeeze(-1)

        # 1. Capacity Budget Penalty: Mean activation across heads should align with target_budget (e.g. 0.333 = 33.3% heads)
        # Mean across head dimension (dim=2): [B, L, T]
        mean_head_act = router_gates.mean(dim=2)
        target_val = self.target_budget if teacher_target is None else (teacher_target.mean().item() if isinstance(teacher_target, torch.Tensor) else self.target_budget)
        budget_loss = F.mse_loss(mean_head_act, torch.full_like(mean_head_act, target_val), reduction="none")

        # 2. Decisive Binarization Entropy: Encourages gates to be sharp (bimodal near 0 or 1) rather than uniform
        gates_clamped = router_gates.clamp(min=self.eps, max=1.0 - self.eps)
        entropy = - (gates_clamped * torch.log(gates_clamped) + (1.0 - gates_clamped) * torch.log(1.0 - gates_clamped))
        entropy_loss = entropy.mean(dim=2) # [B, L, T]

        loss_per_token = budget_loss + self.entropy_weight * entropy_loss # [B, L, T]

        if tgt_mask is not None:
            B = tgt_mask.size(0)
            T_mask = tgt_mask.size(-1)
            mask = tgt_mask.float().view(B, 1, T_mask)
            T_min = min(loss_per_token.size(-1), mask.size(-1))
            loss_cut = loss_per_token[..., :T_min]
            mask_cut = mask[..., :T_min].expand_as(loss_cut)
            return (loss_cut * mask_cut).sum() / mask_cut.sum().clamp(min=1.0)

        return loss_per_token.mean()
