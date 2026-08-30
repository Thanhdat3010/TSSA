"""
Unified Alignment Loss Factory
Centralized factory module instantiating and routing losses for all 8 alignment baselines:
1. align_to_distill (A2D, LREC-COLING 2024)
2. structural_supervision (Findings of ACL 2022)
3. shift_aet (EMNLP 2020)
4. cross_init (Findings of ACL 2024)
5. awesome_align (EACL 2021)
6. dm_bli (ACL 2024 Long)
7. cl_lsa (ACL 2024 / EMNLP 2023)
8. dpo_align (EMNLP 2024)
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from .a2d_loss import AlignToDistillLoss
from .structural_supervision_loss import StructuralSupervisionLoss
from .shift_aet_loss import ShiftAETLoss
from .cross_init_loss import CrossInitLoss
from .awesome_align_loss import AwesomeAlignLoss
from .dm_bli_loss import DMBLISubspaceLoss
from .cl_lsa_loss import CrossLingualInfoNCELoss
from .dpo_align_loss import AlignmentDPOLoss

class UnifiedAlignmentLossFactory(nn.Module):
    """
    Unified Alignment Loss Factory initializing and computing losses
    for all 8 semantic alignment baseline methods.
    """
    def __init__(self, method_name: str, config: Dict[str, Any] = None):
        super().__init__()
        self.method_name = method_name.lower().strip()
        self.config = config or {}

        if self.method_name == "align_to_distill":
            self.loss_fn = AlignToDistillLoss(
                alpha=self.config.get("alpha", 0.5),
                beta=self.config.get("beta", 1.0),
                decay=self.config.get("decay", 0.9),
                temperature=self.config.get("temperature", 2.0)
            )
        elif self.method_name == "structural_supervision":
            self.loss_fn = StructuralSupervisionLoss(
                lambda_struct=self.config.get("lambda_struct", 0.3)
            )
        elif self.method_name == "shift_aet":
            self.loss_fn = ShiftAETLoss(
                hidden_dim=self.config.get("hidden_dim", 768),
                alignment_dim=self.config.get("align_dim", 128)
            )
        elif self.method_name == "cross_init":
            self.loss_fn = CrossInitLoss(
                embed_dim=self.config.get("embed_dim", 768),
                lambda_orth=self.config.get("lambda_orth", 0.01)
            )
        elif self.method_name == "awesome_align":
            self.loss_fn = AwesomeAlignLoss(
                temperature=self.config.get("temperature", 0.1),
                lambda_co=self.config.get("lambda_co", 1.0)
            )
        elif self.method_name == "dm_bli":
            self.loss_fn = DMBLISubspaceLoss(
                hidden_dim=self.config.get("hidden_dim", 768),
                subspace_dim=self.config.get("subspace_dim", 64),
                mu=self.config.get("mu", 0.5)
            )
        elif self.method_name == "cl_lsa":
            self.loss_fn = CrossLingualInfoNCELoss(
                temperature=self.config.get("temperature", 0.07)
            )
        elif self.method_name == "dpo_align":
            self.loss_fn = AlignmentDPOLoss(
                beta=self.config.get("beta", 0.1)
            )
        elif self.method_name in ["bartpho_vanilla", "transformer_scratch", "bart_bahnar"]:
            self.loss_fn = None
        else:
            self.loss_fn = None

    def forward(self, loss_mt: torch.Tensor, model_outputs: Dict[str, Any],
                batch: Dict[str, Any], global_step: int = 0) -> Dict[str, torch.Tensor]:
        """
        Computes total loss combining MT cross-entropy with the active alignment loss.
        """
        if self.loss_fn is None:
            return {"loss_total": loss_mt, "loss_mt": loss_mt.item(), "loss_align": 0.0}

        align_loss = torch.tensor(0.0, device=loss_mt.device)

        if self.method_name == "align_to_distill":
            student_logits = model_outputs.get("logits")
            teacher_logits = batch.get("teacher_logits")
            student_attns = model_outputs.get("cross_attentions")
            teacher_attns = batch.get("teacher_cross_attentions")
            targets = batch.get("labels")
            if student_logits is not None and targets is not None:
                align_loss = self.loss_fn(student_logits, teacher_logits, student_attns, teacher_attns, targets, step=global_step)
                return {"loss_total": align_loss, "loss_mt": loss_mt.item(), "loss_align": align_loss.item()}

        elif self.method_name == "structural_supervision":
            cross_attns = model_outputs.get("cross_attentions")
            align_matrix = batch.get("align_matrix")
            if cross_attns is not None and align_matrix is not None:
                align_loss = self.loss_fn(cross_attns, align_matrix)

        elif self.method_name == "shift_aet":
            dec_hidden = model_outputs.get("decoder_hidden_states")
            if isinstance(dec_hidden, (tuple, list)):
                dec_hidden = dec_hidden[-1]
            enc_hidden = model_outputs.get("encoder_last_hidden_state")
            align_matrix = batch.get("align_matrix")
            if dec_hidden is not None and enc_hidden is not None and align_matrix is not None:
                align_loss = self.loss_fn(dec_hidden, enc_hidden, align_matrix)

        elif self.method_name == "cross_init":
            src_emb = model_outputs.get("encoder_last_hidden_state")
            tgt_emb = batch.get("teacher_enc_states")
            align_matrix = batch.get("align_matrix")
            if src_emb is not None and tgt_emb is not None and align_matrix is not None:
                align_loss = self.loss_fn(src_emb, tgt_emb, align_matrix)

        elif self.method_name == "awesome_align":
            src_h = model_outputs.get("encoder_last_hidden_state")
            tgt_h = batch.get("teacher_enc_states")
            align_matrix = batch.get("align_matrix")
            if src_h is not None and tgt_h is not None and align_matrix is not None:
                align_loss = self.loss_fn(src_h, tgt_h, align_matrix)

        elif self.method_name == "dm_bli":
            src_h = model_outputs.get("encoder_last_hidden_state")
            tgt_h = batch.get("teacher_enc_states")
            align_matrix = batch.get("align_matrix")
            if src_h is not None and tgt_h is not None and align_matrix is not None:
                align_loss = self.loss_fn(src_h, tgt_h, align_matrix)

        elif self.method_name == "cl_lsa":
            src_h = model_outputs.get("encoder_last_hidden_state")
            tgt_h = batch.get("teacher_enc_states")
            align_matrix = batch.get("align_matrix")
            if src_h is not None and tgt_h is not None:
                align_loss = self.loss_fn(src_h, tgt_h, align_matrix)

        elif self.method_name == "dpo_align":
            # In standard sequence forward, uses alignment-weighted margin
            src_h = model_outputs.get("encoder_last_hidden_state")
            tgt_h = batch.get("teacher_enc_states")
            if src_h is not None and tgt_h is not None:
                align_loss = F.mse_loss(src_h.mean(dim=1), tgt_h.mean(dim=1)) * self.config.get("beta", 0.1)

        total_loss = loss_mt + align_loss
        return {"loss_total": total_loss, "loss_mt": loss_mt.item(), "loss_align": align_loss.item()}
