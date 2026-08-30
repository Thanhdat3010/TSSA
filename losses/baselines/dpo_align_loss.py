"""
Alignment as Preference (DPO Alignment, Wu et al., EMNLP 2024)
Direct Preference Optimization on translation hypotheses guided by word alignment rewards.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignmentDPOLoss(nn.Module):
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def _get_batch_logps(self, logits: torch.Tensor, labels: torch.Tensor, pad_index: int = -100) -> torch.Tensor:
        """
        Computes per-sequence log probabilities.
        logits: [B, T, V]
        labels: [B, T]
        """
        log_probs = F.log_softmax(logits, dim=-1)
        valid_labels = labels.clone()
        valid_labels[valid_labels == pad_index] = 0
        per_token_logps = torch.gather(log_probs, dim=2, index=valid_labels.unsqueeze(2)).squeeze(2)
        mask = (labels != pad_index).float()
        return (per_token_logps * mask).sum(dim=-1)

    def forward(self, policy_win_logits: torch.Tensor, policy_lose_logits: torch.Tensor,
                ref_win_logits: torch.Tensor, ref_lose_logits: torch.Tensor,
                win_labels: torch.Tensor, lose_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            policy_win_logits: Logits of preferred translation (y_w) from current model.
            policy_lose_logits: Logits of dispreferred translation (y_l) from current model.
            ref_win_logits: Logits of y_w from reference (frozen) model.
            ref_lose_logits: Logits of y_l from reference (frozen) model.
            win_labels: Token IDs for y_w.
            lose_labels: Token IDs for y_l.
        """
        policy_win_logp = self._get_batch_logps(policy_win_logits, win_labels)
        policy_lose_logp = self._get_batch_logps(policy_lose_logits, lose_labels)

        ref_win_logp = self._get_batch_logps(ref_win_logits, win_labels)
        ref_lose_logp = self._get_batch_logps(ref_lose_logits, lose_labels)

        win_log_ratio = policy_win_logp - ref_win_logp
        lose_log_ratio = policy_lose_logp - ref_lose_logp

        logits_diff = self.beta * (win_log_ratio - lose_log_ratio)
        loss = -F.logsigmoid(logits_diff).mean()

        return loss
