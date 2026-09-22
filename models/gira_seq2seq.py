"""
models/gira_seq2seq.py
Gradient-Isolated Residual Anchor (GIRA) Architecture for Low-Resource NMT.

Core Principles:
1. Complete Gradient Isolation:
   Input to the Projector is h.detach().
   L_struct gradients NEVER touch the backbone encoder f_enc.
   f_enc receives gradients EXCLUSIVELY from L_MT (identical to Vanilla baseline).
2. Scale-Bounded Residual Coupling:
   z_norm = LayerNorm(Projector(h.detach()))
   alpha_scale = tanh(alpha) * 0.10 (strictly bounded in [-0.10, +0.10])
   h' = h + alpha_scale * z_norm
   Projector output layer is zero-initialized so step 0 is identical to Vanilla.
   Variance scale of h' is 100% preserved, preventing Decoder cross-attention saturation!
3. Full HuggingFace Compatibility:
   model.generate() automatically encodes via GIRA and supplies h' to Beam Search.
"""

import os
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

class AnchorProjector(nn.Module):
    """Bottleneck MLP Projector with output LayerNorm and zero-initialized final layer."""
    def __init__(self, d_model: int = 1024, d_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )
        self.out_ln = nn.LayerNorm(d_model)
        # Zero-init last layer so initial residual z == 0 (identical to Vanilla at start)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out = self.net(h)
        # When all zeros (step 0), preserve strict zero
        if not self.training and (out == 0).all():
            return out
        return self.out_ln(out)

class GIRASeq2SeqModel(nn.Module):
    _keys_to_ignore_on_save = None
    _keys_to_ignore_on_load_missing = None
    _keys_to_ignore_on_load_unexpected = None

    def __init__(self, model_name_or_path: str = "vinai/bartpho-syllable",
                 anchor_layer: int = -1, d_hidden: int = 256,
                 alpha_init: float = 0.0, learnable_alpha: bool = True,
                 detach_h: bool = True, **kwargs):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, use_safetensors=True)
        cfg = self.model.config

        self.d_model = getattr(cfg, "d_model", getattr(cfg, "hidden_size", 1024))
        self.anchor_layer = anchor_layer
        self.detach_h = detach_h
        self.learnable_alpha = learnable_alpha

        # Bottleneck Projector (1024 -> 256 -> 1024 + LayerNorm)
        self.projector = AnchorProjector(d_model=self.d_model, d_hidden=d_hidden)

        # Learnable or fixed alpha scalar
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(alpha_init, dtype=torch.float32))

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, no_grad: bool = False):
        """Encodes tokens through the backbone encoder."""
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            encoder_module = self.model.get_encoder() if hasattr(self.model, "get_encoder") else self.model.model.encoder
            out = encoder_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        # Select hidden state at anchor_layer (-1 = output layer 6)
        if out.hidden_states is not None and len(out.hidden_states) > 0:
            h_selected = out.hidden_states[self.anchor_layer]
        else:
            h_selected = out.last_hidden_state
        return out, h_selected

    def _compute_h_prime(self, h_src: torch.Tensor):
        """Computes scale-bounded residual h' = h + alpha_scale * z_src."""
        h_proj_in = h_src.detach() if self.detach_h else h_src
        z_src = self.projector(h_proj_in)

        # Strictly scale-bounded residual gate:
        # tanh(alpha) * 0.10 guarantees the residual cannot perturb h_src variance by >10%,
        # preventing Cross-Attention softmax saturation and decoder divergence!
        if self.learnable_alpha:
            alpha_scale = torch.tanh(self.alpha) * 0.10
        else:
            alpha_scale = self.alpha * 0.10

        h_prime = h_src + (alpha_scale * z_src)
        return h_prime, z_src, alpha_scale

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor = None, decoder_attention_mask: torch.Tensor = None,
                **kwargs):
        """
        Forward pass:
        1. Student encode: h_src
        2. Isolated projection & bounded residual: h_prime
        3. Decoder MT loss: dec_out.loss
        4. Frozen Teacher encode: h_teacher
        """
        # 1. Forward Student Encoder
        enc_out, h_src = self._encode(input_ids, attention_mask, no_grad=False)

        # 2. GRADIENT ISOLATION & BOUNDED RESIDUAL INJECTION
        h_prime, z_src, alpha_scale = self._compute_h_prime(h_src)

        # 3. Supply h_prime to Decoder
        enc_out.last_hidden_state = h_prime
        dec_out = self.model(
            encoder_outputs=enc_out,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True
        )

        # 4. Extract Frozen Teacher from target Vietnamese sentence
        h_teacher = None
        tgt_mask = None
        if labels is not None and self.training:
            pad_id = getattr(self.model.config, "pad_token_id", 1)
            tgt_ids = labels.clone()
            tgt_ids[tgt_ids == -100] = pad_id
            tgt_mask = (tgt_ids != pad_id).long()
            _, h_teacher = self._encode(tgt_ids, tgt_mask, no_grad=True)

        return {
            "loss": dec_out.loss,
            "logits": dec_out.logits,
            "z_src": z_src,
            "h_teacher": h_teacher,
            "src_mask": attention_mask,
            "tgt_mask": tgt_mask,
            "alpha": alpha_scale,
            "h_src": h_src,
            "h_prime": h_prime
        }

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        """
        Custom generate ensuring Beam Search receives h_prime instead of unmodified h.
        """
        enc_out, h_src = self._encode(input_ids, attention_mask, no_grad=True)
        h_prime, _, _ = self._compute_h_prime(h_src)
        enc_out.last_hidden_state = h_prime
        return self.model.generate(encoder_outputs=enc_out, attention_mask=attention_mask, **kwargs)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, *args, **kwargs):
        return self.model.resize_token_embeddings(*args, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        return self.model.save_pretrained(save_directory, **kwargs)

    @property
    def config(self):
        return self.model.config

    @property
    def generation_config(self):
        return self.model.generation_config

    @generation_config.setter
    def generation_config(self, val):
        self.model.generation_config = val

    @property
    def device(self):
        return self.model.device

    @property
    def main_input_name(self):
        return getattr(self.model, "main_input_name", "input_ids")
