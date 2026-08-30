"""
Unified Criterion for TSSA
Combines Machine Translation Cross-Entropy Loss with the 3 TSSA loss objectives:
1. L_struct (Confidence-Weighted Token Barycenter Anchoring)
2. L_prime (In-batch InfoNCE Sentence Semantic Priming)
3. L_route (Anchor-Consistent Head-Wise Decoder Routing)
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

    def forward(self, loss_mt: torch.Tensor, student_outputs: dict, batch: dict = None,
                lambdas: tuple = (0.5, 0.2, 0.1)) -> dict:
        """
        Args:
            loss_mt: Standard Seq2Seq Cross Entropy loss.
            student_outputs: Dict containing 'encoder_last_hidden_state', 'router_gates',
                             'teacher_enc_states', 'teacher_sent_vec', 'align_matrix'.
            batch: Optional dict containing batch tensors.
            lambdas: (lambda_1, lambda_2, lambda_3) weights from LossScheduler.
        Returns:
            dict with 'loss', 'loss_mt', 'loss_struct', 'loss_prime', 'loss_route'.
        """
        batch = batch or {}
        l1, l2, l3 = lambdas
        loss_total = loss_mt
        log_dict = {"loss_mt": loss_mt.item()}

        teacher_enc = batch.get("teacher_enc_states") if "teacher_enc_states" in batch else student_outputs.get("teacher_enc_states")
        teacher_sent = batch.get("teacher_sent_vec") if "teacher_sent_vec" in batch else student_outputs.get("teacher_sent_vec")
        align_mat = batch.get("align_matrix") if "align_matrix" in batch else student_outputs.get("align_matrix")
        src_mask = batch.get("attention_mask")

        # 1. Struct Loss (Token Barycenter Semantic Anchoring)
        if self.use_struct and l1 > 0 and teacher_enc is not None and align_mat is not None:
            student_enc = student_outputs["encoder_last_hidden_state"] # [B, S, D]
            l_struct = self.struct_loss_fn(student_enc, teacher_enc, align_mat, src_mask)
            loss_total = loss_total + l1 * l_struct
            log_dict["loss_struct"] = l_struct.item()
        else:
            log_dict["loss_struct"] = 0.0

        # 2. Prime Loss (Sentence InfoNCE Semantic Priming)
        if self.use_prime and l2 > 0 and teacher_sent is not None:
            student_enc = student_outputs["encoder_last_hidden_state"]
            if src_mask is not None:
                src_mask_exp = src_mask.unsqueeze(-1).float()
                student_sent = (student_enc * src_mask_exp).sum(dim=1) / src_mask_exp.sum(dim=1).clamp(min=1.0)
            else:
                student_sent = student_enc.mean(dim=1)

            l_prime = self.prime_loss_fn(student_sent, teacher_sent)
            loss_total = loss_total + l2 * l_prime
            log_dict["loss_prime"] = l_prime.item()
        else:
            log_dict["loss_prime"] = 0.0

        # 3. Route Loss (Head-Wise Gate Reliability Supervision)
        if self.use_route and l3 > 0 and student_outputs.get("router_gates") is not None:
            router_gates = student_outputs["router_gates"] # [B, L, H, T, 1]
            teacher_target = torch.ones_like(router_gates) * 0.8
            tgt_mask = batch.get("decoder_attention_mask")
            l_route = self.route_loss_fn(router_gates, teacher_target, tgt_mask)
            loss_total = loss_total + l3 * l_route
            log_dict["loss_route"] = l_route.item()
        else:
            log_dict["loss_route"] = 0.0

        log_dict["loss_total"] = loss_total.item()
        return {"loss": loss_total, "log_dict": log_dict}
