"""
Align-to-Distill (A2D) Loss (Jin et al., LREC-COLING 2024 / NCSOFT)
Combines Cross-Entropy Loss, Softened Logit Knowledge Distillation Loss,
and Attention Distillation Loss with a Time-Decay Factor (gamma^step).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignToDistillLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 1.0, decay: float = 0.9, temperature: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.decay = decay
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                student_attns: tuple, teacher_attns: tuple, targets: torch.Tensor, step: int = 0) -> torch.Tensor:
        """
        Args:
            student_logits: [B, T, V] Student output logits.
            teacher_logits: [B, T, V] Teacher output logits.
            student_attns: Tuple of [B, H, T, S] cross-attentions per student decoder layer.
            teacher_attns: Tuple of [B, H, T, S] cross-attentions per teacher decoder layer.
            targets: [B, T] Ground-truth token ids.
            step: Current global training step (for exponential time-decay gamma^step).
        Returns:
            Scalar total A2D loss.
        """
        # 1. Standard Cross-Entropy Loss
        loss_ce = self.ce_loss(student_logits.view(-1, student_logits.size(-1)), targets.view(-1))

        # 2. Logit Knowledge Distillation Loss (KL Divergence with temperature scaling)
        if teacher_logits is not None:
            p_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
            p_student = F.log_softmax(student_logits / self.temperature, dim=-1)
            loss_kd = F.kl_div(p_student, p_teacher, reduction='batchmean') * (self.temperature ** 2)
        else:
            loss_kd = torch.tensor(0.0, device=student_logits.device)

        # 3. Attention Map Distillation Loss
        loss_attn = torch.tensor(0.0, device=student_logits.device)
        if student_attns is not None and teacher_attns is not None and len(student_attns) > 0 and len(teacher_attns) > 0:
            num_layers = min(len(student_attns), len(teacher_attns))
            layer_losses = []
            for l in range(num_layers):
                s_attn = torch.clamp(student_attns[l], min=1e-8)
                t_attn = torch.clamp(teacher_attns[l], min=1e-8)
                
                H_min = min(s_attn.size(1), t_attn.size(1))
                T_min = min(s_attn.size(2), t_attn.size(2))
                S_min = min(s_attn.size(3), t_attn.size(3))
                
                s_cut = s_attn[:, :H_min, :T_min, :S_min]
                t_cut = t_attn[:, :H_min, :T_min, :S_min]
                
                s_prob = s_cut / s_cut.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                t_prob = t_cut / t_cut.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                
                kl = F.kl_div(torch.log(s_prob + 1e-8), t_prob, reduction='batchmean')
                layer_losses.append(kl)
            if len(layer_losses) > 0:
                loss_attn = torch.stack(layer_losses).mean()

        decay_factor = (self.decay ** (step // 100)) if step > 0 else 1.0
        total_loss = (self.alpha * loss_ce) + ((1.0 - self.alpha) * loss_kd) + (self.beta * decay_factor * loss_attn)
        return total_loss
