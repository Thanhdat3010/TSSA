"""Trainer integration for CA-TSSA's encoder-interface surrogate gradient."""

from __future__ import annotations

import os

from .trainer import TSSASeq2SeqTrainer


class CATSSATrainer(TSSASeq2SeqTrainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        **kwargs,
    ):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs.get("labels"),
            decoder_attention_mask=inputs.get("decoder_attention_mask"),
        )
        loss_mt = outputs["loss"]

        if model.training:
            if self.criterion is None:
                raise RuntimeError("CA-TSSA training requires CATSSACriterion")
            result = self.criterion(
                loss_mt=loss_mt,
                model_outputs=outputs,
                global_step=self.state.global_step,
            )
            loss = result["loss"]
            if self.log_tracker is not None and self.state.global_step % 20 == 0:
                current_lr = (
                    self.optimizer.param_groups[0]["lr"]
                    if getattr(self, "optimizer", None) is not None
                    else None
                )
                epoch = self.state.epoch if self.state.epoch is not None else 0.0
                self.log_tracker.log_step(
                    self.state.global_step,
                    epoch,
                    result["log_dict"],
                    lr=current_lr,
                )
        else:
            loss = loss_mt

        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: str, state_dict=None):
        super()._save(output_dir, state_dict=state_dict)
        if self.criterion is not None and hasattr(self.criterion, "save_diagnostics"):
            self.criterion.save_diagnostics(output_dir)

    def _load_best_model(self):
        """Restore both backbone and projector from the selected checkpoint."""
        checkpoint = self.state.best_model_checkpoint
        if not checkpoint:
            return
        target = self.accelerator.unwrap_model(self.model)
        if hasattr(target, "load_pretrained") and os.path.isdir(checkpoint):
            target.load_pretrained(checkpoint)
            return
        super()._load_best_model()
