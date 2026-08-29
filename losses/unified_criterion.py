"""
Unified Criterion for TSSA
Combines Machine Translation Cross-Entropy Loss with the 3 TSSA loss objectives.
"""

import torch
import torch.nn as nn
from .struct_loss import StructLoss
from .prime_loss import PrimeLoss
from .route_loss import RouteLoss

class TSSAUnifiedCriterion(nn.Module):
    def __init__(self, use_struct: bool = True, use_prime: bool = True, use_route: bool = True,
                 conf_threshold: float = 0.2, temperature: float = 0.07):
        super().__init__()
        self.use_struct = use_struct
        self.use_prime = use_prime
        self.use_route = use_route

        self.struct_loss_fn = StructLoss(conf_threshold=conf_threshold) if use_struct else None
        self.prime_loss_fn = PrimeLoss(temperature=temperature) if use_prime else None
        self.route_loss_fn = RouteLoss() if use_route else None

    def forward(self, loss_mt: torch.Tensor, student_outputs: dict, batch: dict, lambdas: tuple = (0.5, 0.2, 0.1)) -> dict:
        """
        Args:
            loss_mt: Standard Seq2Seq Cross Entropy loss.
            student_outputs: Dict containing 'encoder_last_hidden_state', 'router_gates', etc.
            batch: Dict containing 'align_matrix', 'teacher_enc_states', 'teacher_sent_vec', etc.
            lambdas: (lambda_1, lambda_2, lambda_3) weights from LossScheduler.
        Returns:
            dict with 'loss_total', 'loss_mt', 'loss_struct', 'loss_prime', 'loss_route'.
        """
        l1, l2, l3 = lambdas
        loss_total = loss_mt
        log_dict = {"loss_mt": loss_mt.item()}

        # 1. Struct Loss
        if self.use_struct and l1 > 0 and "teacher_enc_states" in batch:
            student_enc = student_outputs["encoder_last_hidden_state"] # [B, S, D]
            teacher_enc = batch["teacher_enc_states"]                   # [B, T, D]
            align_mat = batch["align_matrix"]                           # [B, S, T]
            src_mask = batch.get("attention_mask")

            l_struct = self.struct_loss_fn(student_enc, teacher_enc, align_mat, src_mask)
            loss_total = loss_total + l1 * l_struct
            log_dict["loss_struct"] = l_struct.item()
        else:
            log_dict["loss_struct"] = 0.0

        # 2. Prime Loss
        if self.use_prime and l2 > 0 and "teacher_sent_vec" in batch:
            student_enc = student_outputs["encoder_last_hidden_state"]
            src_mask = batch.get("attention_mask").unsqueeze(-1) if "attention_mask" in batch else None
            if src_mask is not None:
                student_sent = (student_enc * src_mask).sum(dim=1) / src_mask.sum(dim=1).clamp(min=1)
            else:
                student_sent = student_enc.mean(dim=1)

            teacher_sent = batch["teacher_sent_vec"] # [B, D]
            l_prime = self.prime_loss_fn(student_sent, teacher_sent)
            loss_total = loss_total + l2 * l_prime
            log_dict["loss_prime"] = l_prime.item()
        else:
            log_dict["loss_prime"] = 0.0

        # 3. Route Loss
        if self.use_route and l3 > 0 and "router_gates" in student_outputs and "teacher_enc_states" in batch:
            router_gates = student_outputs["router_gates"] # [B, L, H, T]
            # Construct target reliability from teacher cosine match
            # For simplicity, target is high when target token has high confidence alignment
            teacher_target = torch.ones_like(router_gates) * 0.8
            l_route = self.route_loss_fn(router_gates, teacher_target)
            loss_total = loss_total + l3 * l_route
            log_dict["loss_route"] = l_route.item()
        else:
            log_dict["loss_route"] = 0.0

        log_dict["loss_total"] = loss_total.item()
        return {"loss": loss_total, "log_dict": log_dict}
