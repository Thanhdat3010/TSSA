"""
TSSA Custom Seq2Seq Trainer
Subclasses Hugging Face Seq2SeqTrainer to seamlessly integrate:
- Multi-objective TSSA losses (L_struct, L_prime, L_route)
- Baseline alignment losses (Guided Attention, Joint-Align, AWESOME-loss, CL-LSA)
- Dynamic loss scheduling
- Fast evaluation with SacreBLEU, chrF++, METEOR, and COMET.
"""

import torch
import numpy as np
from transformers import Seq2SeqTrainer
import evaluate

class TSSASeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, criterion=None, loss_scheduler=None, model_type="tssa",
                 baseline_loss_fn=None, baseline_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = criterion
        self.loss_scheduler = loss_scheduler
        self.model_type = model_type
        self.baseline_loss_fn = baseline_loss_fn
        self.baseline_weight = baseline_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        Overrides Hugging Face compute_loss to inject TSSA / baseline alignment losses.
        """
        # 1. Forward pass
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs.get("labels"),
            decoder_attention_mask=inputs.get("decoder_attention_mask"),
            output_hidden_states=True,
            output_attentions=True
        )

        loss_mt = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        total_loss = loss_mt

        # 2. Compute TSSA Loss if active
        if self.model_type == "tssa" and self.criterion is not None:
            current_step = self.state.global_step
            lambdas = self.loss_scheduler.get_lambdas(current_step) if self.loss_scheduler else (0.5, 0.2, 0.1)
            crit_res = self.criterion(loss_mt, outputs, inputs, lambdas=lambdas)
            total_loss = crit_res["loss"]

        # 3. Compute Baseline losses if configured
        elif self.baseline_loss_fn is not None:
            if self.model_type == "guided_attn" and "align_matrix" in inputs:
                # Guided Attention on Cross-Attentions
                loss_align = self.baseline_loss_fn(outputs.get("cross_attentions"), inputs["align_matrix"])
                total_loss = total_loss + self.baseline_weight * loss_align

            elif self.model_type == "joint_align" and "align_matrix" in inputs:
                # Joint Align on Decoder & Encoder Top Hidden States
                dec_states = outputs["decoder_hidden_states"][-1]
                enc_states = outputs["encoder_last_hidden_state"]
                loss_align = self.baseline_loss_fn(dec_states, enc_states, inputs["align_matrix"])
                total_loss = total_loss + self.baseline_weight * loss_align

            elif self.model_type == "awesome_align" and "align_matrix" in inputs and "teacher_enc_states" in inputs:
                # AWESOME Symmetric Embedding Align
                enc_states = outputs["encoder_last_hidden_state"]
                tgt_states = inputs["teacher_enc_states"]
                loss_align = self.baseline_loss_fn(enc_states, tgt_states, inputs["align_matrix"])
                total_loss = total_loss + self.baseline_weight * loss_align

            elif self.model_type == "cl_lsa" and "teacher_sent_vec" in inputs:
                # Cross-Lingual Sentence InfoNCE
                enc_states = outputs["encoder_last_hidden_state"]
                mask = inputs["attention_mask"].unsqueeze(-1)
                src_sent = (enc_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                tgt_sent = inputs["teacher_sent_vec"]
                loss_align = self.baseline_loss_fn(src_sent, tgt_sent)
                total_loss = total_loss + self.baseline_weight * loss_align

        return (total_loss, outputs) if return_outputs else total_loss

    def _save(self, output_dir: str, state_dict=None):
        """
        Safely saves checkpoint by delegating to PreTrainedModel.save_pretrained,
        avoiding shared/tied embedding errors with safetensors.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        target_model = self.model
        if hasattr(target_model, "module"):
            target_model = target_model.module

        if hasattr(target_model, "save_pretrained"):
            target_model.save_pretrained(output_dir)
        elif hasattr(target_model, "model") and hasattr(target_model.model, "save_pretrained"):
            target_model.model.save_pretrained(output_dir)
        else:
            torch.save(target_model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

    def save_model(self, output_dir: str = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir
        self._save(output_dir)
