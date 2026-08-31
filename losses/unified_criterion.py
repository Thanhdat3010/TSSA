"""
Unified Criterion for TSSA 2.0
Combines Machine Translation Cross-Entropy Loss with the 3 Core TSSA 2.0 Objectives:
1. L_xattn (Target-Guided Cross-Attention Anchoring with Confidence Weighting)
2. L_prime (Residual Cross-Lingual Sentence InfoNCE Priming)
3. L_route (Anchor-Consistent Head-Wise Attention Router Supervision)
"""

import torch
import torch.nn as nn
from .xattn_anchor_loss import CrossAttentionAnchorLoss
from .prime_loss import PrimeLoss
from .route_loss import RouteLoss

class TSSAUnifiedCriterion(nn.Module):
    def __init__(self, use_struct: bool = True, use_prime: bool = True, use_route: bool = True,
                 top_k_layers: int = 3, conf_threshold: float = 0.1, temperature: float = 0.07):
        super().__init__()
        self.use_struct = use_struct
        self.use_prime = use_prime
        self.use_route = use_route

        # 1. Target-Guided Cross-Attention Anchoring Loss (TSSA 2.0)
        self.xattn_loss_fn = CrossAttentionAnchorLoss(top_k_layers=top_k_layers, conf_threshold=conf_threshold) if use_struct else None
        
        # 2. Residual Projector Sentence Priming InfoNCE Loss
        self.prime_loss_fn = PrimeLoss(temperature=temperature) if use_prime else None
        
        # 3. Dynamic Head-Wise Router Supervision Loss
        self.route_loss_fn = RouteLoss() if use_route else None

    def forward(self, loss_mt: torch.Tensor, student_outputs: dict, batch: dict = None,
                lambdas: tuple = (0.10, 0.05, 0.10)) -> dict:
        """
        Args:
            loss_mt: Standard Seq2Seq Cross Entropy loss.
            student_outputs: Dict containing 'cross_attentions', 'student_projected_sent',
                             'teacher_sent_vec', 'align_matrix_ts', 'router_gates'.
            batch: Optional dict containing batch masks and tensors.
            lambdas: (lambda_xattn, lambda_prime, lambda_route) weights from LossScheduler.
        Returns:
            dict with 'loss', 'loss_mt', 'loss_struct' (L_xattn), 'loss_prime', 'loss_route'.
        """
        batch = batch or {}
        l1, l2, l3 = lambdas
        loss_total = loss_mt
        log_dict = {"loss_mt": loss_mt.item()}

        src_mask = batch.get("attention_mask")
        tgt_mask = batch.get("decoder_attention_mask")

        # 1. Target-Guided Cross-Attention Anchoring Loss (L_xattn)
        cross_attns = student_outputs.get("cross_attentions")
        align_mat_ts = student_outputs.get("align_matrix_ts")
        if self.use_struct and l1 > 0 and cross_attns is not None and align_mat_ts is not None:
            l_xattn = self.xattn_loss_fn(cross_attns, align_mat_ts, tgt_mask=tgt_mask, src_mask=src_mask)
            loss_total = loss_total + l1 * l_xattn
            log_dict["loss_struct"] = l_xattn.item()
        else:
            log_dict["loss_struct"] = 0.0

        # 2. Residual Projector Sentence InfoNCE Priming Loss (L_prime)
        student_proj_sent = student_outputs.get("student_projected_sent")
        teacher_sent = student_outputs.get("teacher_sent_vec")
        if self.use_prime and l2 > 0 and student_proj_sent is not None and teacher_sent is not None:
            l_prime = self.prime_loss_fn(student_proj_sent, teacher_sent)
            loss_total = loss_total + l2 * l_prime
            log_dict["loss_prime"] = l_prime.item()
        else:
            log_dict["loss_prime"] = 0.0

        # 3. Dynamic Head-Wise Router Supervision Loss (L_route)
        router_gates = student_outputs.get("router_gates")
        if self.use_route and l3 > 0 and router_gates is not None:
            teacher_target = torch.ones_like(router_gates) * 0.8
            l_route = self.route_loss_fn(router_gates, teacher_target, tgt_mask)
            loss_total = loss_total + l3 * l_route
            log_dict["loss_route"] = l_route.item()
        else:
            log_dict["loss_route"] = 0.0

        log_dict["loss_total"] = loss_total.item()
        return {"loss": loss_total, "log_dict": log_dict}
