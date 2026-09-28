"""Training-only teacher wrapper for counterfactual semantic anchoring."""

from __future__ import annotations

import copy
import json
import os
from typing import Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

from .backbone_adapter import Seq2SeqBackboneAdapter
from .ca_tssa_seq2seq import module_fingerprint


class CFTSSASeq2SeqModel(nn.Module):
    _keys_to_ignore_on_save = None
    _keys_to_ignore_on_load_missing = None
    _keys_to_ignore_on_load_unexpected = None

    def __init__(
        self,
        model_name_or_path: str = "vinai/bartpho-syllable",
        backbone_model: Optional[nn.Module] = None,
        special_token_ids: Sequence[int] = (),
    ) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.backbone = Seq2SeqBackboneAdapter(model_name_or_path, backbone_model)
        self.special_token_ids = tuple(int(value) for value in special_token_ids)
        self.teacher_encoder = copy.deepcopy(self.backbone.get_encoder())
        self.teacher_encoder.requires_grad_(False)
        self.teacher_encoder.eval()
        self.teacher_initial_fingerprint = module_fingerprint(self.teacher_encoder)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher_encoder.eval()
        return self

    def _alignment_mask(self, ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.bool()
        for token_id in self.special_token_ids:
            mask = mask & (ids != token_id)
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        source_hidden = self.backbone.encode_source(input_ids, attention_mask).last_hidden_state
        decoder_out = self.backbone.decode_from_hidden(
            source_hidden, attention_mask, labels, decoder_attention_mask
        )
        result = {"loss": decoder_out.loss, "logits": decoder_out.logits}
        if self.training and labels is not None:
            target_ids = labels.detach().clone()
            target_ids[target_ids == -100] = self.backbone.pad_token_id
            target_attention = (target_ids != self.backbone.pad_token_id).long()
            with torch.no_grad():
                teacher_source = self.teacher_encoder(
                    input_ids=input_ids, attention_mask=attention_mask, return_dict=True
                ).last_hidden_state
                teacher_target = self.teacher_encoder(
                    input_ids=target_ids, attention_mask=target_attention, return_dict=True
                ).last_hidden_state
            result.update(
                source_hidden=source_hidden,
                teacher_source=teacher_source.detach(),
                teacher_target=teacher_target.detach(),
                source_mask=self._alignment_mask(input_ids, attention_mask),
                target_mask=self._alignment_mask(target_ids, target_attention),
                target_ids=target_ids,
            )
        return result

    def generate(self, *args, **kwargs):
        return self.backbone.generate(*args, **kwargs)

    def get_encoder(self):
        return self.backbone.get_encoder()

    def get_decoder(self):
        return self.backbone.get_decoder()

    def resize_token_embeddings(self, *args, **kwargs):
        return self.backbone.model.resize_token_embeddings(*args, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.backbone.save_pretrained(save_directory, **kwargs)
        with open(os.path.join(save_directory, "cf_tssa_config.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "format_version": 1,
                    "model_name_or_path": self.model_name_or_path,
                    "special_token_ids": self.special_token_ids,
                    "teacher_fingerprint": self.teacher_initial_fingerprint,
                    "inference_uses_teacher": False,
                },
                handle,
                indent=2,
            )

    def load_pretrained(self, load_directory: str) -> None:
        try:
            loaded = AutoModelForSeq2SeqLM.from_pretrained(load_directory, use_safetensors=True)
        except Exception:
            loaded = AutoModelForSeq2SeqLM.from_pretrained(load_directory)
        self.backbone.model.load_state_dict(loaded.state_dict())
        del loaded

    @property
    def config(self):
        return self.backbone.model.config

    @property
    def generation_config(self):
        return self.backbone.model.generation_config

    @generation_config.setter
    def generation_config(self, value):
        self.backbone.model.generation_config = value

    @property
    def device(self):
        return next(self.backbone.model.parameters()).device

    @property
    def main_input_name(self):
        return getattr(self.backbone.model, "main_input_name", "input_ids")

    @property
    def warnings_issued(self):
        return getattr(self.backbone.model, "warnings_issued", {})

    def can_generate(self):
        return True

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.backbone.model.prepare_inputs_for_generation(*args, **kwargs)
