"""
TSSA Custom Seq2Seq Trainer
Subclasses Hugging Face Seq2SeqTrainer to seamlessly integrate:
- Multi-objective TSSA losses (L_struct, L_prime, L_route)
- Unified Baseline Alignment Loss Factory (A2D, Structural Supervision, Shift-AET, CrossInit, AWESOME-align, DM-BLI, CL-LSA, DPO Alignment)
- Dynamic loss scheduling
- Fast evaluation with SacreBLEU, chrF++, METEOR, and COMET.
"""

import os
import torch
import numpy as np
from transformers import Seq2SeqTrainer
from losses.baselines.factory import UnifiedAlignmentLossFactory

class TSSASeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, criterion=None, loss_scheduler=None, model_type="tssa",
                 baseline_loss_factory=None, log_tracker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = criterion
        self.loss_scheduler = loss_scheduler
        self.model_type = model_type.lower().strip()
        self.baseline_loss_factory = baseline_loss_factory
        self.log_tracker = log_tracker

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        Overrides Hugging Face compute_loss to inject TSSA / unified baseline alignment losses.
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
            lambdas = self.loss_scheduler.get_lambdas(current_step) if self.loss_scheduler else (0.2, 0.1, 0.05)
            crit_res = self.criterion(loss_mt, outputs, inputs, lambdas=lambdas)
            total_loss = crit_res["loss"]

            # Log step-level dynamics ("Log Lại Hết")
            if self.log_tracker is not None and current_step % 20 == 0:
                current_lr = self.optimizer.param_groups[0]["lr"] if hasattr(self, "optimizer") and self.optimizer else None
                epoch_val = self.state.epoch if self.state.epoch is not None else 0.0
                self.log_tracker.log_step(current_step, epoch_val, crit_res["log_dict"], lr=current_lr)
                router_gates = outputs.get("router_gates")
                if router_gates is not None and current_step % 100 == 0:
                    self.log_tracker.log_gate_activations(int(epoch_val), router_gates)

        # 3. Compute Unified Baseline Alignment losses if configured
        elif self.baseline_loss_factory is not None:
            res = self.baseline_loss_factory(
                loss_mt=loss_mt,
                model_outputs=outputs,
                batch=inputs,
                global_step=self.state.global_step
            )
            total_loss = res["loss_total"]

        return (total_loss, outputs) if return_outputs else total_loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """
        Overrides evaluate to log evaluation trajectory (BLEU, eval_loss, etc.) to log_tracker.
        """
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        if self.log_tracker is not None:
            epoch_val = self.state.epoch if self.state.epoch is not None else 0.0
            self.log_tracker.log_eval_metrics(int(round(epoch_val)), metrics)
            self.log_tracker.save()
        return metrics

    def _save(self, output_dir: str, state_dict=None):
        """
        Safely saves checkpoint by delegating to PreTrainedModel.save_pretrained,
        avoiding shared/tied embedding errors with safetensors.
        """
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
