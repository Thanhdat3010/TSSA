"""
StructLoss: Confidence-Weighted Barycentric Semantic Anchoring (TSSA)
Formulation:
    A_tilde_ij = A_ij / (sum_k A_ik + eps)
    h_bar_i^T = sum_j A_tilde_ij * sg(h_j^T)
    c_i = max_j A_ij
    L_struct = (1 / |S|) * sum_{i} I(c_i >= c_th) * c_i * SmoothL1(h_i^S, h_bar_i^T)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class StructLoss(nn.Module):
    def __init__(self, conf_threshold: float = 0.2, eps: float = 1e-8, loss_type: str = "smooth_l1"):
        super().__init__()
        self.conf_threshold = conf_threshold
        self.eps = eps
        self.loss_type = loss_type

    def forward(self, student_enc_states: torch.Tensor, teacher_enc_states: torch.Tensor,
                align_matrix: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            student_enc_states: [B, S, D] Student encoder token representations.
            teacher_enc_states: [B, T, D] Teacher encoder token representations (Frozen with Stop-Gradient).
            align_matrix: [B, S, T] Subword alignment probabilities in [0, 1].
            src_mask: [B, S] Source attention mask (1 for valid token, 0 for pad).
        Returns:
            Scalar tensor representing L_struct.
        """
        # 1. Ensure stop-gradient on Teacher features (Stop representation drift)
        teacher_enc_states = teacher_enc_states.detach()

        # 2. Normalize alignment matrix rows to sum to 1 (Barycentric coefficients)
        row_sums = align_matrix.sum(dim=-1, keepdim=True) + self.eps # [B, S, 1]
        norm_align = align_matrix / row_sums                         # [B, S, T]

        # 3. Compute Target-Side Barycentric Anchor vector h_bar_i^T
        # [B, S, T] x [B, T, D] -> [B, S, D]
        target_barycenter = torch.bmm(norm_align, teacher_enc_states)

        # 4. Compute Confidence weight c_i = max_j A_ij
        c_i, _ = torch.max(align_matrix, dim=-1) # [B, S]

        # 5. Build mask for valid and confident alignments
        conf_mask = (c_i >= self.conf_threshold).float() # [B, S]
        if src_mask is not None:
            conf_mask = conf_mask * src_mask.float()

        # 6. Compute distance (Smooth L1 or MSE)
        if self.loss_type == "smooth_l1":
            diff = F.smooth_l1_loss(student_enc_states, target_barycenter, reduction="none").mean(dim=-1) # [B, S]
        elif self.loss_type == "cosine":
            student_norm = F.normalize(student_enc_states, dim=-1)
            bary_norm = F.normalize(target_barycenter, dim=-1)
            diff = 1.0 - (student_norm * bary_norm).sum(dim=-1) # [B, S]
        else:
            diff = F.mse_loss(student_enc_states, target_barycenter, reduction="none").mean(dim=-1)

        # Weighted loss
        weighted_loss = diff * c_i * conf_mask
        total_valid = conf_mask.sum().clamp(min=1.0)
        
        return weighted_loss.sum() / total_valid
