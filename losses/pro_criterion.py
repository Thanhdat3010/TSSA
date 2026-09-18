"""
losses/pro_criterion.py
TSSA-Pro Universal Mathematical Criterion:
100% Continuous, Typology-Aware Mathematical Formulation (Zero Language-Specific Heuristics / Zero IF-ELSE):
1. L_struct: Confidence-Weighted Latent Barycentric Semantic Anchoring
   with Dynamic Information-Theoretic Entropy Filtering w_s = exp(- H_s / tau_H)
2. L_prime: Residual Cross-Lingual Sentence InfoNCE Priming
   with Continuous Gaussian Subword Fertility Attenuation:
   fertility_factor(kappa) = exp(- (kappa - 1.0)^2 / (2 * sigma_kappa^2))
3. L_route: Head-Wise Closed Capacity Budget Regularization under rho*(H) = min(0.333, 4/H)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .prime_loss import PrimeLoss
from .route_loss import RouteLoss

class TSSAProCriterion(nn.Module):
    def __init__(self, 
                 use_struct: bool = True, 
                 use_prime: bool = True, 
                 use_route: bool = True,
                 conf_threshold: float = 0.20,
                 temperature: float = 0.07,
                 align_tau: float = 0.10,
                 entropy_tau: float = 1.5,
                 target_budget: float = 0.250,
                 kappa: float = 1.20,
                 sigma_kappa: float = 0.75,
                 eps: float = 1e-8):
        super().__init__()
        self.use_struct = use_struct
        self.use_prime = use_prime
        self.use_route = use_route
        self.conf_threshold = conf_threshold
        self.temperature = temperature
        self.align_tau = align_tau
        self.entropy_tau = entropy_tau
        self.target_budget = target_budget
        self.kappa = float(kappa) if kappa is not None else 1.20
        self.sigma_kappa = float(sigma_kappa)
        self.eps = eps

        # 1. Residual Projector Sentence InfoNCE Priming
        self.prime_loss_fn = PrimeLoss(temperature=temperature) if use_prime else None

        # 2. Dynamic Head-Wise Router Supervision Loss
        self.route_loss_fn = RouteLoss(target_budget=target_budget) if use_route else None

    def compute_latent_barycenter_loss(self, 
                                      student_enc: torch.Tensor, 
                                      teacher_enc: torch.Tensor, 
                                      src_mask: torch.Tensor = None, 
                                      tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes L_struct via Latent Barycentric Anchoring in the representation space.
        Applies Dynamic Entropy Filtering w_s = exp(- H_s / tau_H) dynamically on every token.
        Pure mathematical formulation without discrete language rules.
        """
        # 1. Stop-gradient on Teacher features (prevent representation drift)
        teacher_enc = teacher_enc.detach()
        B, S, D = student_enc.shape
        T = teacher_enc.size(1)

        # 2. Compute Cosine Affinity Matrix in representation space [B, S, T]
        s_norm = F.normalize(student_enc, p=2, dim=-1) # [B, S, D]
        t_norm = F.normalize(teacher_enc, p=2, dim=-1) # [B, T, D]
        sim_st = torch.bmm(s_norm, t_norm.transpose(1, 2)) / self.align_tau # [B, S, T]

        if tgt_mask is not None:
            # Mask out target padding positions from softmax
            mask_t = (1.0 - tgt_mask.unsqueeze(1).float()) * -1e4
            sim_st = sim_st + mask_t

        # 3. Soft alignment posterior A_{s, t} over target tokens
        align_st = F.softmax(sim_st, dim=-1) # [B, S, T]

        # 4. Target Barycenter vector: h̄_s^T = ∑_t A_{s,t} h_t^T
        target_barycenter = torch.bmm(align_st, teacher_enc) # [B, S, D]

        # 5. Alignment confidence c_s = max_t A_{s,t}
        c_s, _ = torch.max(align_st, dim=-1) # [B, S]

        # 6. Dynamic Entropy Filter w_s = exp(- H_s / tau_H)
        # Sharp lexical alignment -> Low entropy -> w_s ~ 1.0
        # Dispersed affix noise -> High entropy -> w_s -> 0.0
        p_clamped = align_st.clamp(min=self.eps)
        entropy_s = - (p_clamped * torch.log(p_clamped)).sum(dim=-1) # [B, S]
        w_s = torch.exp(- entropy_s / self.entropy_tau) # [B, S]

        # 7. Confidence & valid token gating
        conf_gate = (c_s >= self.conf_threshold).float() # [B, S]
        effective_weights = c_s * w_s * conf_gate # [B, S]

        if src_mask is not None:
            effective_weights = effective_weights * src_mask.float()

        # 8. Distance in latent representation space (Smooth L1)
        diff = F.smooth_l1_loss(student_enc, target_barycenter, reduction="none").mean(dim=-1) # [B, S]
        weighted_loss = diff * effective_weights

        normalizer = effective_weights.sum().clamp(min=1.0)
        return weighted_loss.sum() / normalizer

    def forward(self, 
                loss_mt: torch.Tensor, 
                student_outputs: dict, 
                batch: dict = None,
                lambdas: tuple = (0.20, 0.08, 0.05)) -> dict:
        """
        Forward pass for TSSA-Pro loss computation.
        Args:
            loss_mt: Standard Seq2Seq Cross Entropy loss.
            student_outputs: Dict from TSSASeq2SeqModel or TSSAViT5Model.
            batch: Dict with attention masks.
            lambdas: (l_struct, l_prime, l_route).
        Returns:
            dict containing total loss and individual loss components.
        """
        batch = batch or {}
        l1, l2, l3 = lambdas
        loss_total = loss_mt
        log_dict = {"loss_mt": loss_mt.item()}

        src_mask = batch.get("attention_mask")
        tgt_mask = batch.get("decoder_attention_mask")

        # 1. Latent Barycentric Anchoring Loss (L_struct)
        student_enc = student_outputs.get("encoder_last_hidden_state")
        teacher_enc = student_outputs.get("teacher_enc_states")
        
        if self.use_struct and l1 > 0 and student_enc is not None and teacher_enc is not None:
            l_struct = self.compute_latent_barycenter_loss(
                student_enc, teacher_enc, src_mask=src_mask, tgt_mask=tgt_mask
            )
            loss_total = loss_total + l1 * l_struct
            log_dict["loss_struct"] = l_struct.item()
        else:
            log_dict["loss_struct"] = 0.0

        # 2. Residual Projector Sentence InfoNCE Priming (L_prime)
        # Continuous Gaussian Subword Fertility Attenuation (Zero IF-ELSE):
        # fertility_factor(kappa) = exp(- (kappa - 1.0)^2 / (2 * sigma_kappa^2))
        fertility_factor = math.exp(- ((max(1.0, self.kappa) - 1.0) ** 2) / (2.0 * (self.sigma_kappa ** 2)))
        eff_l2 = l2 * fertility_factor

        student_proj_sent = student_outputs.get("student_projected_sent")
        teacher_sent = student_outputs.get("teacher_sent_vec")

        if self.use_prime and eff_l2 > 0 and student_proj_sent is not None and teacher_sent is not None:
            l_prime = self.prime_loss_fn(student_proj_sent, teacher_sent)
            loss_total = loss_total + eff_l2 * l_prime
            log_dict["loss_prime"] = l_prime.item()
        else:
            log_dict["loss_prime"] = 0.0

        # 3. Dynamic Head Router Supervision / Capacity Budget (L_route)
        router_gates = student_outputs.get("router_gates")
        if self.use_route and l3 > 0 and router_gates is not None and self.route_loss_fn is not None:
            l_route = self.route_loss_fn(router_gates, tgt_mask=tgt_mask)
            loss_total = loss_total + l3 * l_route
            log_dict["loss_route"] = l_route.item()
        else:
            log_dict["loss_route"] = 0.0

        log_dict["loss_total"] = loss_total.item()

        return {
            "loss": loss_total,
            "loss_mt": loss_mt,
            "loss_struct": log_dict["loss_struct"],
            "loss_prime": log_dict["loss_prime"],
            "loss_route": log_dict["loss_route"],
            "log_dict": log_dict
        }
