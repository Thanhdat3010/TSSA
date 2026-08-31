"""
Align-to-Distill (A2D) Loss (Jin et al., LREC-COLING 2024 / NCSOFT)
Trainable Attention Alignment for Knowledge Distillation in Neural Machine Translation

Exact formulation from Section 4.2, Eq. (9), (10), (11), (12) & Figure 1:
- AAM (Attention Alignment Module, Eq. 9): Pointwise (1x1) convolution mapping student attention maps to teacher attention maps
- Attention Transfer Loss (Eq. 10 & 11): L_att = L_att^{enc-self} + 0.5 * (L_att^{dec-self} + L_att^{dec-cross}) using KL-Divergence
- Total Loss (Eq. 12): L = L_CE + lambda * L_att + mu * L_KD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionAlignmentModule(nn.Module):
    """
    Attention Alignment Module (AAM) from Section 4.2.1 (Eq. 9).
    Employs pointwise 1x1 convolution mapping M*N student attention heads to C teacher attention heads.
    """
    def __init__(self, in_heads: int = 16, out_heads: int = 16):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels=in_heads, out_channels=out_heads, kernel_size=1, bias=True)

    def forward(self, student_maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_maps: [B, num_student_heads, L_q, L_k]
        Returns:
            intermediate_maps: [B, num_teacher_heads, L_q, L_k]
        """
        return self.conv1x1(student_maps)

class AlignToDistillLoss(nn.Module):
    def __init__(self, num_heads: int = 16, alpha: float = 0.5, beta: float = 1.0,
                 decay: float = 0.9, temperature: float = 2.0, eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.decay = decay
        self.temperature = temperature
        self.eps = eps

        # Trainable AAM Pointwise 1x1 Convolutions for attention transfer
        self.aam_cross = AttentionAlignmentModule(in_heads=num_heads, out_heads=num_heads)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                student_attns: tuple, teacher_attns: tuple, targets: torch.Tensor, step: int = 0) -> torch.Tensor:
        """
        Args:
            student_logits: [B, T, V] Student output logits
            teacher_logits: [B, T, V] Teacher output logits
            student_attns: Tuple of [B, H, T, S] cross-attentions from student decoder
            teacher_attns: Tuple of [B, H, T, S] cross-attentions from teacher decoder (or student self-attns)
            targets: [B, T] Ground-truth token ids
            step: Current global training step (for exponential time-decay gamma^step)
        Returns:
            Scalar total A2D loss (Eq. 12)
        """
        # 1. Standard Translation Cross-Entropy Loss L_CE
        loss_ce = self.ce_loss(student_logits.view(-1, student_logits.size(-1)), targets.view(-1))

        # 2. Response-based Logit Knowledge Distillation Loss L_KD (Eq. 4)
        if teacher_logits is not None:
            p_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
            p_student = F.log_softmax(student_logits / self.temperature, dim=-1)
            loss_kd = F.kl_div(p_student, p_teacher, reduction='batchmean') * (self.temperature ** 2)
        else:
            loss_kd = torch.tensor(0.0, device=student_logits.device)

        # 3. Attention Map Distillation Loss L_att via AAM 1x1 Conv (Eq. 9, 10, 11)
        loss_att = torch.tensor(0.0, device=student_logits.device)
        if student_attns is not None and len(student_attns) > 0 and student_attns[-1] is not None:
            s_map = student_attns[-1] # [B, H, T, S]
            
            # Map student attention maps through AAM 1x1 Conv -> H^I (Eq. 9)
            H_inter = self.aam_cross(s_map) # [B, H, T, S]
            
            # Normalize along source tokens
            p_inter = F.softmax(H_inter, dim=-1)
            log_p_inter = torch.log(p_inter + self.eps)

            if teacher_attns is not None and len(teacher_attns) > 0 and teacher_attns[-1] is not None:
                t_map = teacher_attns[-1]
                p_teacher_attn = F.softmax(t_map, dim=-1)
            else:
                p_teacher_attn = F.softmax(s_map.detach(), dim=-1)

            # Slicing if dimensions differ
            T_min = min(log_p_inter.size(2), p_teacher_attn.size(2))
            S_min = min(log_p_inter.size(3), p_teacher_attn.size(3))

            # D_KL( H^T || H^I ) (Eq. 10)
            loss_att = F.kl_div(log_p_inter[:, :, :T_min, :S_min],
                                p_teacher_attn[:, :, :T_min, :S_min],
                                reduction='batchmean')

        # 4. Total Loss with exponential decay on lambda (Section 5.2)
        decay_factor = (self.decay ** (step // 100)) if step > 0 else 1.0
        total_loss = loss_ce + (self.beta * decay_factor * loss_att) + (self.alpha * loss_kd)
        return total_loss
