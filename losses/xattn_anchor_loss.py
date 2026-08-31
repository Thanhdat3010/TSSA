"""
CrossAttentionAnchorLoss: Target-Guided Cross-Attention Anchoring (TSSA 2.0)
Supervises the Decoder Cross-Attention maps in the top decoder layers
to align with the target-to-source semantic alignment prior,
directly mitigating attention dispersion during decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionAnchorLoss(nn.Module):
    def __init__(self, top_k_layers: int = 3, conf_threshold: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.top_k_layers = top_k_layers
        self.conf_threshold = conf_threshold
        self.eps = eps

    def forward(self, cross_attentions: tuple, align_matrix: torch.Tensor,
                tgt_mask: torch.Tensor = None, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            cross_attentions: Tuple of [B, H, T, S] tensors from decoder layers.
            align_matrix: [B, T, S] Target-to-Source subword alignment matrix in [0, 1].
            tgt_mask: [B, T] Target sequence mask (1 for valid token, 0 for pad).
            src_mask: [B, S] Source sequence mask.
        Returns:
            Scalar tensor representing normalized L_xattn.
        """
        if cross_attentions is None or len(cross_attentions) == 0 or align_matrix is None:
            return torch.tensor(0.0, device=align_matrix.device if align_matrix is not None else "cpu")

        # Select top-k decoder layers
        num_layers = len(cross_attentions)
        selected_layers = range(max(0, num_layers - self.top_k_layers), num_layers)

        # Stop-gradient on alignment prior
        align_matrix = align_matrix.detach() # [B, T, S]
        
        # Row-normalize alignment prior so sum over source tokens is 1 (where target has alignment)
        row_sum = align_matrix.sum(dim=-1, keepdim=True) + self.eps
        align_norm = align_matrix / row_sum # [B, T, S]

        # Target token alignment confidence c_t = max_s A_ts in [0, 1]
        c_t, _ = torch.max(align_matrix, dim=-1) # [B, T]
        conf_filter = (c_t >= self.conf_threshold).float() # [B, T]

        # Build 2D joint mask [B, T, S]
        B, T, S = align_matrix.shape
        joint_mask = conf_filter.unsqueeze(-1).expand(B, T, S) # [B, T, S]

        if tgt_mask is not None:
            joint_mask = joint_mask * tgt_mask.unsqueeze(-1).float()
        if src_mask is not None:
            joint_mask = joint_mask * src_mask.unsqueeze(1).float()

        # Expand alignment target and mask to match [B, H, T, S]
        align_target = align_norm.unsqueeze(1) # [B, 1, T, S]
        mask_expanded = joint_mask.unsqueeze(1) # [B, 1, T, S]
        conf_weights = c_t.unsqueeze(1).unsqueeze(-1) # [B, 1, T, 1]

        layer_losses = []
        for l_idx in selected_layers:
            attn_layer = cross_attentions[l_idx] # [B, H, T_attn, S_attn]
            if attn_layer is None:
                continue
            
            # Slice to match dimensions if needed
            T_curr = min(attn_layer.size(2), T)
            S_curr = min(attn_layer.size(3), S)

            attn_cut = attn_layer[:, :, :T_curr, :S_curr]
            target_cut = align_target[:, :, :T_curr, :S_curr].expand_as(attn_cut)
            mask_cut = mask_expanded[:, :, :T_curr, :S_curr].expand_as(attn_cut)
            conf_cut = conf_weights[:, :, :T_curr, :].expand_as(attn_cut)

            # SmoothL1 distance between Cross-Attention and Alignment Prior
            diff = F.smooth_l1_loss(attn_cut, target_cut, reduction="none") # [B, H, T, S]
            weighted_diff = diff * conf_cut * mask_cut

            norm_factor = mask_cut.sum().clamp(min=1.0)
            layer_losses.append(weighted_diff.sum() / norm_factor)

        if len(layer_losses) == 0:
            return torch.tensor(0.0, device=align_matrix.device)

        return torch.stack(layer_losses).mean()
