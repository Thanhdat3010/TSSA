"""
losses/gira_criterion.py
Gradient-Isolated Residual Anchor (GIRA) Criterion.

Formulation:
1. Subword Cross-Lingual Affinity via Normalized Einsum:
   sim = einsum("bsd, btd -> bst", z, t) / tau_align
2. Target-Padding Masking & Soft Alignment Posterior (Stop-Gradient):
   A = softmax(sim.masked_fill(~tgt_mask, -inf), dim=-1).detach()
3. Target Spherical Barycenter Vector:
   c = Normalize(einsum("bst, btd -> bsd", A, t), dim=-1)
4. Length-Normalized Information Entropy Gate:
   H_norm = (- sum A ln A) / ln(max(2, T_valid)) in [0, 1]
   w = exp(- H_norm / tau_H)
5. Weighted Cosine Distance Loss:
   L_struct = sum(w * (1 - cos) * m) / sum(m)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GIRACriterion(nn.Module):
    def __init__(self, lambda_struct: float = 1.0, tau_align: float = 0.10, tau_H: float = 0.50):
        super().__init__()
        self.lambda_struct = lambda_struct
        self.tau_align = tau_align
        self.entropy_tau = tau_H

    def compute_struct_loss(self, z_src: torch.Tensor, h_teacher: torch.Tensor,
                            src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> tuple:
        """Computes isolated barycentric anchor loss."""
        # Ensure float32 for numerical stability under FP16 training
        z = F.normalize(z_src.float(), dim=-1)
        t = F.normalize(h_teacher.detach().float(), dim=-1)

        # 1. Similarity matrix
        sim = torch.einsum("bsd,btd->bst", z, t) / self.tau_align

        # 2. Mask target padding
        if tgt_mask is not None:
            mask_t = tgt_mask.bool().unsqueeze(1) # [B, 1, T]
            sim = sim.masked_fill(~mask_t, float("-inf"))

        # 3. Soft alignment posterior with stop-gradient
        A = torch.softmax(sim, dim=-1).detach() # [B, S, T]

        # 4. Spherical target barycenter
        c = F.normalize(torch.einsum("bst,btd->bsd", A, t), dim=-1)
        cos = (z * c).sum(-1) # [B, S] in [-1, 1]

        # 5. Length-normalized information entropy gate
        if tgt_mask is not None:
            T_valid = tgt_mask.sum(-1, keepdim=True).clamp(min=2).float() # [B, 1]
        else:
            T_valid = torch.full((z.size(0), 1), max(2.0, float(t.size(1))), device=z.device)

        H = -(A * A.clamp_min(1e-8).log()).sum(-1) # [B, S]
        H_norm = (H / torch.log(T_valid)).clamp(0.0, 1.0) # [B, S] in [0, 1]
        w = torch.exp(-H_norm / self.entropy_tau)          # [B, S] in (0, 1]

        # 6. Mask source padding
        if src_mask is not None:
            m = src_mask.bool()
        else:
            m = torch.ones((z.size(0), z.size(1)), dtype=torch.bool, device=z.device)

        valid_count = m.float().sum().clamp(min=1.0)
        l_struct = (w * (1.0 - cos) * m.float()).sum() / valid_count

        diag = {
            "mean_w": round(w[m].mean().item(), 4) if m.any() else 0.0,
            "mean_H": round(H_norm[m].mean().item(), 4) if m.any() else 0.0
        }
        return l_struct, diag

    def forward(self, loss_mt: torch.Tensor, student_outputs: dict, batch: dict = None, **kwargs) -> dict:
        z_src = student_outputs.get("z_src")
        h_teacher = student_outputs.get("h_teacher")
        src_mask = student_outputs.get("src_mask")
        tgt_mask = student_outputs.get("tgt_mask")

        if z_src is None or h_teacher is None:
            return {
                "loss": loss_mt,
                "log_dict": {
                    "loss_mt": round(loss_mt.item(), 4),
                    "loss_total": round(loss_mt.item(), 4)
                }
            }

        l_struct, diag = self.compute_struct_loss(z_src, h_teacher, src_mask, tgt_mask)
        total_loss = loss_mt + (self.lambda_struct * l_struct)

        alpha_val = student_outputs.get("alpha")
        alpha_scalar = round(alpha_val.item(), 4) if isinstance(alpha_val, torch.Tensor) else 0.0

        log_dict = {
            "loss_mt": round(loss_mt.item(), 4),
            "loss_struct": round(l_struct.item(), 4),
            "loss_total": round(total_loss.item(), 4),
            "mean_w": diag["mean_w"],
            "mean_H": diag["mean_H"],
            "alpha": alpha_scalar
        }
        return {"loss": total_loss, "log_dict": log_dict}
