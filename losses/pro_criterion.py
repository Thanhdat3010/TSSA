"""
losses/pro_criterion.py
TSSA-Pro Universal Mathematical Criterion:
100% Continuous, Typology-Aware Mathematical Formulation (Zero Language-Specific Heuristics / Zero IF-ELSE):
1. L_struct: Confidence-Weighted Latent Barycentric Semantic Anchoring
   - Batch-Centering before L2 Normalization (eliminates representation anisotropy)
   - Unit Hypersphere S^{D-1} Scale-Invariance across LayerNorm & RMSNorm
   - Stop-gradient on alignment posterior A_{s,t} with target padding masked
   - Length-Normalized Dynamic Information Entropy Gate:
     H_norm = (- sum A ln A) / ln(max(2, T_valid)) in [0, 1]
     w_s = exp(- H_norm / tau_H) with tau_H = 0.50
   - Continuous Gaussian Fertility Protection for High-Fertility Edge Cases (Ba Na kappa ~ 3.5)
2. L_prime: Residual Cross-Lingual Sentence InfoNCE Priming
   with Continuous Gaussian Subword Fertility Attenuation:
   fertility_factor(kappa) = exp(- (max(1.0, kappa) - 1.0)^2 / (2 * sigma_kappa^2))
3. L_route: Disabled by default to preserve 100% autoregressive decoder generation capacity.
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
                 use_route: bool = False,
                 use_centering: bool = True,
                 use_gate: bool = True,
                 shuffle_teacher: bool = False,
                 protect_struct_fertility: bool = True,
                 conf_threshold: float = 0.20,
                 temperature: float = 0.07,
                 align_tau: float = 0.10,
                 entropy_tau: float = 0.50,
                 target_budget: float = 0.250,
                 kappa: float = 1.20,
                 sigma_kappa: float = 0.75,
                 eps: float = 1e-8):
        super().__init__()
        self.use_struct = use_struct
        self.use_prime = use_prime
        self.use_route = use_route
        self.use_centering = use_centering
        self.use_gate = use_gate
        self.shuffle_teacher = shuffle_teacher
        self.protect_struct_fertility = protect_struct_fertility
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

        # 2. Dynamic Head-Wise Router Supervision Loss (Disabled by default to avoid decoder interference)
        self.route_loss_fn = RouteLoss(target_budget=target_budget) if use_route else None

    def compute_latent_barycenter_loss(self, 
                                      student_enc: torch.Tensor, 
                                      teacher_enc: torch.Tensor, 
                                      src_mask: torch.Tensor = None, 
                                      tgt_mask: torch.Tensor = None) -> tuple:
        """
        Scale-Invariant Latent Hypersphere Anchoring Loss:
        - Float32 precision casting to prevent FP16 underflow/overflow.
        - Centering across valid batch tokens before L2 normalization (eliminates representation anisotropy).
        - Unit Hypersphere L2 normalization S^{D-1} to eliminate LayerNorm vs RMSNorm scale divergence.
        - Subword cross-lingual affinity & soft alignment posterior with stop-gradient on alignment.
        - Spherical barycenter vector projected to S^{D-1}.
        - Length-Normalized Dynamic Information Entropy Gate:
          H_norm = (- sum A ln A) / ln(max(2, T_valid)) in [0, 1]
          w_s = exp(- H_norm / tau_H) with tau_H = 0.50
        - Scale-invariant Cosine distance (1 - cos) strictly bounded in [0, 2].
        - Normalized by actual valid token count sum(src_mask).
        Returns:
            (loss_struct, mean_H_norm, mean_w_s, random_cosine)
        """
        # 1. Cast to float32 for stable numerical operations under FP16 training
        student_enc_fp32 = student_enc.float()
        teacher_enc_fp32 = teacher_enc.detach().float()
        
        B, S, D = student_enc_fp32.shape
        T = teacher_enc_fp32.size(1)

        # 2. Centering before L2 normalization to remove Anisotropy
        if self.use_centering:
            if src_mask is not None and src_mask.bool().any():
                s_mean = student_enc_fp32[src_mask.bool()].mean(dim=0, keepdim=True).detach() # [1, D]
            else:
                s_mean = student_enc_fp32.mean(dim=(0, 1), keepdim=True).detach() # [1, 1, D]
                
            if tgt_mask is not None and tgt_mask.bool().any():
                t_mean = teacher_enc_fp32[tgt_mask.bool()].mean(dim=0, keepdim=True).detach() # [1, D]
            else:
                t_mean = teacher_enc_fp32.mean(dim=(0, 1), keepdim=True).detach() # [1, 1, D]

            s_centered = student_enc_fp32 - s_mean.view(1, 1, D)
            t_centered = teacher_enc_fp32 - t_mean.view(1, 1, D)
        else:
            s_centered = student_enc_fp32
            t_centered = teacher_enc_fp32

        # 3. Project onto Unit Hypersphere S^{D-1} (Scale-Invariance across LayerNorm & RMSNorm)
        s_norm = F.normalize(s_centered, p=2, dim=-1, eps=self.eps) # [B, S, D]
        t_norm = F.normalize(t_centered, p=2, dim=-1, eps=self.eps) # [B, T, D]

        # Control Experiment: Teacher Shuffling to test whether semantic alignment is genuine
        if self.shuffle_teacher and T > 1:
            perm = torch.randperm(T, device=t_norm.device)
            t_norm = t_norm[:, perm, :]

        # 4. Subword Cross-Lingual Affinity Matrix [B, S, T]
        sim_st = torch.bmm(s_norm, t_norm.transpose(1, 2)) / self.align_tau # [B, S, T]

        if tgt_mask is not None:
            mask_t = (1.0 - tgt_mask.unsqueeze(1).float()) * -1e4
            sim_st = sim_st + mask_t

        # 5. Soft Alignment Posterior with Stop-Gradient to prevent student cheating
        align_st = F.softmax(sim_st, dim=-1).detach() # [B, S, T]

        # 6. Spherical Barycentric Target Vector
        target_barycenter_raw = torch.bmm(align_st, t_norm) # [B, S, D]
        target_barycenter = F.normalize(target_barycenter_raw, p=2, dim=-1, eps=self.eps) # [B, S, D]

        # 7. Dynamic Information Entropy Filter with Length Normalization
        if self.use_gate:
            p_clamped = align_st.clamp(min=self.eps)
            entropy_s = - (p_clamped * torch.log(p_clamped)).sum(dim=-1) # [B, S]

            if tgt_mask is not None:
                tgt_valid_len = tgt_mask.sum(dim=-1, keepdim=True).float() # [B, 1]
            else:
                tgt_valid_len = torch.full((B, 1), float(T), device=student_enc.device, dtype=torch.float32)

            # Clamp T_valid >= 2 to prevent ln(1) = 0 division-by-zero
            clamped_tgt_len = tgt_valid_len.clamp(min=2.0)
            max_entropy = torch.log(clamped_tgt_len) # [B, 1] >= ln(2) ~ 0.693

            norm_entropy_s = (entropy_s / max_entropy).clamp(0.0, 1.0) # [B, S] in [0, 1]
            w_s = torch.exp(- norm_entropy_s / self.entropy_tau) # [B, S] in [exp(-1/tau), 1.0]
        else:
            norm_entropy_s = torch.zeros((B, S), device=student_enc.device, dtype=torch.float32)
            w_s = torch.ones((B, S), device=student_enc.device, dtype=torch.float32)

        if src_mask is not None:
            valid_mask = src_mask.float()
        else:
            valid_mask = torch.ones((B, S), device=student_enc.device, dtype=torch.float32)

        # 8. Scale-Invariant Cosine Distance on Hypersphere: 1 - <s_norm, target_barycenter>
        cos_sim = (s_norm * target_barycenter).sum(dim=-1) # [B, S]
        cos_dist = 1.0 - cos_sim # [B, S] in [0, 2]

        weighted_loss = cos_dist * w_s * valid_mask # [B, S]

        # 9. Normalization by actual valid token count
        normalizer = valid_mask.sum().clamp(min=1.0)
        loss_struct = (weighted_loss.sum() / normalizer).to(student_enc.dtype)

        # Diagnostic metrics
        valid_bool = valid_mask.bool()
        if valid_bool.any():
            mean_H_norm = norm_entropy_s[valid_bool].mean().item()
            mean_w_s = w_s[valid_bool].mean().item()
        else:
            mean_H_norm = 0.0
            mean_w_s = 0.0

        # Random token cosine to measure Anisotropy
        with torch.no_grad():
            if B > 0 and S > 1:
                rand_s1 = s_norm[:, 0, :] # [B, D]
                rand_s2 = s_norm[:, 1, :] # [B, D]
                random_cosine = (rand_s1 * rand_s2).sum(dim=-1).mean().item()
            else:
                random_cosine = 0.0

        return loss_struct, mean_H_norm, mean_w_s, random_cosine

    def forward(self, 
                loss_mt: torch.Tensor, 
                student_outputs: dict, 
                batch: dict = None,
                lambdas: tuple = (0.20, 0.08, 0.00)) -> dict:
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
        l1 = lambdas[0] if len(lambdas) > 0 else 0.20
        l2 = lambdas[1] if len(lambdas) > 1 else 0.08
        l3 = lambdas[2] if len(lambdas) > 2 else 0.00

        src_mask = batch.get("attention_mask")
        tgt_mask = batch.get("decoder_attention_mask")

        # Continuous Gaussian Subword Fertility Attenuation:
        fertility_factor = math.exp(- ((max(1.0, self.kappa) - 1.0) ** 2) / (2.0 * (self.sigma_kappa ** 2)))
        
        # Effective weights:
        # High-fertility edge cases (Ba Na kappa ~ 3.5, fertility ~ 0.0038)
        # automatically attenuate both L_struct and L_prime to protect from noise!
        eff_l1 = l1 * (fertility_factor if self.protect_struct_fertility else 1.0)
        eff_l2 = l2 * fertility_factor

        loss_total = loss_mt
        log_dict = {
            "loss_mt": loss_mt.item(),
            "fertility_factor": round(fertility_factor, 4),
            "eff_lambda_struct": round(eff_l1, 4),
            "eff_lambda_prime": round(eff_l2, 4),
        }

        # 1. Scale-Invariant Latent Barycentric Anchoring Loss (L_struct)
        student_enc = student_outputs.get("encoder_last_hidden_state")
        teacher_enc = student_outputs.get("teacher_enc_states")
        
        if self.use_struct and eff_l1 > 0 and student_enc is not None and teacher_enc is not None:
            l_struct, mean_H, mean_w, rand_cos = self.compute_latent_barycenter_loss(
                student_enc, teacher_enc, src_mask=src_mask, tgt_mask=tgt_mask
            )
            loss_total = loss_total + eff_l1 * l_struct
            log_dict["loss_struct"] = l_struct.item()
            log_dict["mean_H_norm"] = round(mean_H, 4)
            log_dict["mean_w_s"] = round(mean_w, 4)
            log_dict["random_cosine"] = round(rand_cos, 4)
        else:
            log_dict["loss_struct"] = 0.0
            log_dict["mean_H_norm"] = 0.0
            log_dict["mean_w_s"] = 0.0
            log_dict["random_cosine"] = 0.0

        # 2. Residual Projector Sentence InfoNCE Priming (L_prime)
        student_proj_sent = student_outputs.get("student_projected_sent")
        teacher_sent = student_outputs.get("teacher_sent_vec")

        if self.use_prime and eff_l2 > 0 and student_proj_sent is not None and teacher_sent is not None:
            l_prime = self.prime_loss_fn(student_proj_sent, teacher_sent)
            loss_total = loss_total + eff_l2 * l_prime
            log_dict["loss_prime"] = l_prime.item()
        else:
            log_dict["loss_prime"] = 0.0

        # 3. Dynamic Head Router Supervision (Disabled by default)
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
