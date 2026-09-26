"""Backbone-neutral helpers for Hugging Face encoder-decoder models."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput


class Seq2SeqBackboneAdapter(nn.Module):
    """Expose one small interface across BART-, T5-, MBART-, and Marian-style models."""

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if model is None:
            if not model_name_or_path:
                raise ValueError("model_name_or_path is required when model is not provided")
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name_or_path, use_safetensors=True
                )
            except Exception:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

        if not hasattr(model, "get_encoder"):
            raise TypeError("CA-TSSA requires an encoder-decoder model with get_encoder()")

        self.model = model
        self.model_name_or_path = model_name_or_path or getattr(
            model.config, "_name_or_path", "in-memory-model"
        )

    @property
    def hidden_size(self) -> int:
        config = self.model.config
        value = getattr(config, "d_model", getattr(config, "hidden_size", None))
        if value is None:
            raise ValueError("Backbone config does not expose d_model or hidden_size")
        return int(value)

    @property
    def pad_token_id(self) -> int:
        value = getattr(self.model.config, "pad_token_id", None)
        return 0 if value is None else int(value)

    def get_encoder(self) -> nn.Module:
        return self.model.get_encoder()

    def get_decoder(self) -> nn.Module:
        return self.model.get_decoder()

    def encode_source(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_hidden_states: bool = False,
    ):
        return self.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

    @staticmethod
    def as_encoder_outputs(last_hidden_state: torch.Tensor) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=last_hidden_state)

    def decode_from_hidden(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        decoder_attention_mask: Optional[torch.Tensor],
        **kwargs,
    ):
        return self.model(
            encoder_outputs=self.as_encoder_outputs(hidden_state),
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
            **kwargs,
        )

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        return self.model.save_pretrained(save_directory, **kwargs)
