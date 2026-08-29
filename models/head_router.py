"""
HeadWiseRouter Module for TSSA
Computes dynamic gating coefficients g_lht in [0, 1] for each Cross-Attention head
to route information specifically through reliable Anchor Heads.
Also provides Head-Pruning interfaces for Causal Ablation studies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HeadWiseRouter(nn.Module):
    def __init__(self, d_model: int = 768, n_heads: int = 12, n_layers: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # 2-layer MLP for gating: Input = [Decoder Query state ; Context vector] (dim = 2 * d_model)
        # Output = n_heads gate scalars in [0, 1] per layer
        self.router_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * d_model, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_heads),
                nn.Sigmoid()
            ) for _ in range(n_layers)
        ])

        # Head pruning mask: [n_layers, n_heads] (1 = keep, 0 = prune)
        self.register_buffer("pruning_mask", torch.ones((n_layers, n_heads), dtype=torch.float32))

    def forward(self, dec_states: torch.Tensor, context_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Args:
            dec_states: [B, T, D] Query states of Decoder at layer_idx.
            context_states: [B, T, D] Cross-Attention context vectors.
            layer_idx: Index of decoder layer (0 <= layer_idx < n_layers).
        Returns:
            gates: [B, n_heads, T, 1] Gating multiplier applied to attention heads.
        """
        # Concatenate query and context
        combined = torch.cat([dec_states, context_states], dim=-1) # [B, T, 2*D]
        
        # Compute raw sigmoid gates: [B, T, n_heads]
        raw_gates = self.router_mlps[layer_idx](combined)
        
        # Transpose to [B, n_heads, T]
        gates = raw_gates.transpose(1, 2)
        
        # Apply pruning mask for Causal Ablation
        mask = self.pruning_mask[layer_idx].view(1, self.n_heads, 1).to(gates.device)
        gates = gates * mask

        return gates.unsqueeze(-1) # [B, n_heads, T, 1]

    def reset_pruning_mask(self):
        """Resets all heads to active (1.0)."""
        self.pruning_mask.fill_(1.0)

    def prune_heads(self, pruned_indices: list):
        """
        Prunes specified heads.
        pruned_indices: list of tuple (layer_idx, head_idx) or flat integer indices.
        """
        for item in pruned_indices:
            if isinstance(item, tuple):
                l, h = item
            else:
                l = item // self.n_heads
                h = item % self.n_heads
            if 0 <= l < self.n_layers and 0 <= h < self.n_heads:
                self.pruning_mask[l, h] = 0.0
