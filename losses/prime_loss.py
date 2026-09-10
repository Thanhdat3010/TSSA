"""
PrimeLoss: In-Batch InfoNCE Semantic Priming (TSSA)
Aligns whole-sentence representations between minority source sentence
and Vietnamese teacher sentence in a contrastive semantic space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PrimeLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, student_sent_vec: torch.Tensor, teacher_sent_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_sent_vec: [B, D] Normalized or raw source sentence vector.
            teacher_sent_vec: [B, D] Normalized or raw target teacher sentence vector.
        Returns:
            Scalar In-Batch InfoNCE loss.
        """
        # Ensure stop gradient on teacher
        teacher_sent_vec = teacher_sent_vec.detach()

        # Normalize representations to unit sphere
        s_norm = F.normalize(student_sent_vec, dim=-1) # [B, D]
        t_norm = F.normalize(teacher_sent_vec, dim=-1) # [B, D]

        # Compute cosine similarity matrix for all pairs in batch
        # [B, D] x [D, B] -> [B, B]
        logits = torch.matmul(s_norm, t_norm.transpose(0, 1)) / self.temperature

        # Ground truth labels: diagonal elements (batch index 0->0, 1->1, ..., B-1->B-1)
        batch_size = student_sent_vec.size(0)
        labels = torch.arange(batch_size, device=student_sent_vec.device)

        return self.cross_entropy(logits, labels)
