"""Trainer integration for CF-TSSA's training-only counterfactual gate."""

from __future__ import annotations

import os

from .trainer import TSSASeq2SeqTrainer


class CFTSSATrainer(TSSASeq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                        labels=inputs.get("labels"),
                        decoder_attention_mask=inputs.get("decoder_attention_mask"))
        loss = outputs["loss"]
        if model.training:
            if self.criterion is None:
                raise RuntimeError("CF-TSSA training requires CFTSSACriterion")
            loss = self.criterion(loss, model, outputs, inputs, self.state.global_step)["loss"]
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: str, state_dict=None):
        super()._save(output_dir, state_dict=state_dict)
        if self.criterion is not None:
            self.criterion.save_diagnostics(output_dir)

    def _load_best_model(self):
        checkpoint = self.state.best_model_checkpoint
        if checkpoint:
            target = self.accelerator.unwrap_model(self.model)
            if hasattr(target, "load_pretrained") and os.path.isdir(checkpoint):
                target.load_pretrained(checkpoint)
                return
        super()._load_best_model()
