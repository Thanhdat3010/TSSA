"""Backbone-agnostic CA-TSSA sequence-to-sequence model."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import AutoModelForSeq2SeqLM

from .backbone_adapter import Seq2SeqBackboneAdapter


class AnchorProjector(nn.Module):
    """Small training-only projection head with dimensions inferred from the backbone."""

    def __init__(self, d_model: int, d_hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        bottleneck = min(int(d_hidden), int(d_model))
        self.net = nn.Sequential(
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_state)


def module_fingerprint(module: nn.Module) -> str:
    """Cheap deterministic fingerprint suitable for detecting accidental teacher updates."""

    digest = hashlib.sha256()
    with torch.no_grad():
        for name, value in module.state_dict().items():
            tensor = value.detach().float()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(round(tensor.sum().item(), 7)).encode("ascii"))
            digest.update(str(round(tensor.square().sum().item(), 7)).encode("ascii"))
    return digest.hexdigest()


class CATSSASeq2SeqModel(nn.Module):
    """CA-TSSA model whose training-only modules never alter the inference path."""

    _keys_to_ignore_on_save = None
    _keys_to_ignore_on_load_missing = None
    _keys_to_ignore_on_load_unexpected = None

    def __init__(
        self,
        model_name_or_path: str = "vinai/bartpho-syllable",
        d_hidden: int = 256,
        backbone_model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.backbone = Seq2SeqBackboneAdapter(
            model_name_or_path=model_name_or_path,
            model=backbone_model,
        )
        self.d_model = self.backbone.hidden_size
        self.d_hidden = min(int(d_hidden), self.d_model)
        self.projector = AnchorProjector(self.d_model, self.d_hidden)

        # This is a real immutable snapshot, not the evolving student under no_grad().
        self.teacher_encoder = copy.deepcopy(self.backbone.get_encoder())
        self.teacher_encoder.requires_grad_(False)
        self.teacher_encoder.eval()
        self.teacher_initial_fingerprint = module_fingerprint(self.teacher_encoder)

    def train(self, mode: bool = True):
        super().train(mode)
        # Parent train() would otherwise enable teacher dropout.
        self.teacher_encoder.eval()
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        encoder_out = self.backbone.encode_source(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        source_hidden = encoder_out.last_hidden_state

        # Split the graph at the encoder interface. The criterion reconnects it
        # with a zero-valued surrogate carrying the routed gradient.
        if self.training and labels is not None:
            encoder_proxy = source_hidden.detach().requires_grad_(True)
        else:
            encoder_proxy = source_hidden

        decoder_out = self.backbone.decode_from_hidden(
            hidden_state=encoder_proxy,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        projected_source = None
        teacher_hidden = None
        target_mask = None
        if self.training and labels is not None:
            projected_source = self.projector(encoder_proxy)
            target_ids = labels.detach().clone()
            target_ids[target_ids == -100] = self.backbone.pad_token_id
            target_mask = (target_ids != self.backbone.pad_token_id).long()
            with torch.no_grad():
                teacher_out = self.teacher_encoder(
                    input_ids=target_ids,
                    attention_mask=target_mask,
                    return_dict=True,
                )
                teacher_hidden = teacher_out.last_hidden_state.detach()

        return {
            "loss": decoder_out.loss,
            "logits": decoder_out.logits,
            "source_hidden": source_hidden,
            "encoder_proxy": encoder_proxy,
            "projected_source": projected_source,
            "teacher_hidden": teacher_hidden,
            "source_mask": attention_mask,
            "target_mask": target_mask,
        }

    def generate(self, *args, **kwargs):
        """Pure backbone generation: no projector and no target teacher."""
        return self.backbone.generate(*args, **kwargs)

    def get_encoder(self):
        return self.backbone.get_encoder()

    def get_decoder(self):
        return self.backbone.get_decoder()

    def resize_token_embeddings(self, *args, **kwargs):
        return self.backbone.model.resize_token_embeddings(*args, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save deployable backbone plus the training-only projector sidecar."""
        os.makedirs(save_directory, exist_ok=True)
        self.backbone.save_pretrained(save_directory, **kwargs)
        projector_state = {
            key: value.detach().cpu().contiguous()
            for key, value in self.projector.state_dict().items()
        }
        save_file(
            projector_state,
            os.path.join(save_directory, "ca_tssa_projector.safetensors"),
        )
        config = {
            "format_version": 1,
            "model_name_or_path": self.model_name_or_path,
            "d_model": self.d_model,
            "d_hidden": self.d_hidden,
            "teacher_fingerprint": self.teacher_initial_fingerprint,
            "inference_uses_auxiliary_modules": False,
        }
        with open(
            os.path.join(save_directory, "ca_tssa_config.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False)

    def load_pretrained(self, load_directory: str) -> None:
        """Restore a CA-TSSA checkpoint into the existing wrapper."""
        try:
            loaded = AutoModelForSeq2SeqLM.from_pretrained(
                load_directory, use_safetensors=True
            )
        except Exception:
            loaded = AutoModelForSeq2SeqLM.from_pretrained(load_directory)
        self.backbone.model.load_state_dict(loaded.state_dict())
        del loaded

        projector_path = os.path.join(load_directory, "ca_tssa_projector.safetensors")
        if not os.path.exists(projector_path):
            raise FileNotFoundError(
                f"Missing CA-TSSA projector sidecar: {projector_path}"
            )
        self.projector.load_state_dict(load_file(projector_path, device=str(self.device)))

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
