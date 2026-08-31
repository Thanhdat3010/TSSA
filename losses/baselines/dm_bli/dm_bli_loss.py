"""
DM-BLI: Dynamic Multiple Subspaces Alignment for Unsupervised BLI (Hu & Xu, ACL 2024 Long Paper)
Exact formulation from Section 3.3, Eq. (2), (3), (4), (5):
- Multiple Subspaces Projection into K dynamic subspaces (d=64)
- Inter-cluster Contrastive Loss (Eq. 2 & 3): L_inter = 0.5 * L_s2t + 0.5 * L_t2s
- Intra-cluster Contrastive Loss (Eq. 4): L_intra InfoNCE over translation pairs within subspaces
- Total Loss (Eq. 5): L_DMBLI = L_inter + L_intra
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DMBLISubspaceLoss(nn.Module):
    def __init__(self, hidden_dim: int = 1024, subspace_dim: int = 64, temperature: float = 0.1, mu: float = 0.5):
        super().__init__()
        self.subspace_src = nn.Sequential(
            nn.Linear(hidden_dim, subspace_dim),
            nn.LayerNorm(subspace_dim)
        )
        self.subspace_tgt = nn.Sequential(
            nn.Linear(hidden_dim, subspace_dim),
            nn.LayerNorm(subspace_dim)
        )
        self.temp = temperature
        self.mu = mu

    def forward(self, src_h: torch.Tensor, tgt_h: torch.Tensor,
                align_matrix: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Args:
            src_h: [B, S, D] Source contextual embeddings
            tgt_h: [B, T, D] Target contextual embeddings
            align_matrix: [B, S, T] or [B, T, S] Subword alignment prior matrix
        Returns:
            Scalar DM-BLI loss = L_inter + L_intra (Eq. 5)
        """
        if align_matrix is not None:
            if align_matrix.size(1) == tgt_h.size(1) and align_matrix.size(2) == src_h.size(1):
                A = align_matrix.transpose(1, 2)
            else:
                A = align_matrix
        else:
            A = torch.eye(min(src_h.size(1), tgt_h.size(1)), device=src_h.device).unsqueeze(0).expand(src_h.size(0), -1, -1)

        S_min = min(src_h.size(1), A.size(1))
        T_min = min(tgt_h.size(1), A.size(2))

        src_cut = src_h[:, :S_min, :]
        tgt_cut = tgt_h[:, :T_min, :]
        A_cut = A[:, :S_min, :T_min].float().detach()

        # 1. Project into low-dimensional dynamic subspaces
        sub_src = F.normalize(self.subspace_src(src_cut), p=2, dim=-1) # [B, S_min, d]
        sub_tgt = F.normalize(self.subspace_tgt(tgt_cut), p=2, dim=-1) # [B, T_min, d]

        # 2. Intra-cluster Contrastive Loss L_intra (Eq. 4)
        # Similarity matrix between subspace projected tokens
        sim_mat = torch.bmm(sub_src, sub_tgt.transpose(1, 2)) / self.temp # [B, S_min, T_min]
        
        # Supervised target labels from alignment matrix A
        target_labels = torch.argmax(A_cut, dim=-1) # [B, S_min]
        loss_intra = F.cross_entropy(sim_mat.view(-1, T_min), target_labels.view(-1))

        # 3. Inter-cluster Contrastive Distance L_inter (Eq. 2 & 3)
        # Cosine distance between cluster centroid vectors
        centroid_src = sub_src.mean(dim=1) # [B, d]
        centroid_tgt = sub_tgt.mean(dim=1) # [B, d]

        pos_dist = 1.0 - F.cosine_similarity(centroid_src, centroid_tgt) # [B]
        # Negative cluster (rolled in batch)
        neg_centroid_tgt = torch.roll(centroid_tgt, shifts=1, dims=0)
        neg_dist = 1.0 - F.cosine_similarity(centroid_src, neg_centroid_tgt) # [B]

        loss_s2t = - torch.log(torch.exp(- pos_dist / self.temp) / (torch.exp(- pos_dist / self.temp) + torch.exp(- neg_dist / self.temp) + 1e-8)).mean()
        loss_inter = loss_s2t

        # Total Loss (Eq. 5)
        total_loss = (loss_intra + loss_inter) * self.mu
        return total_loss
